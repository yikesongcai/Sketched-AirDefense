"""
Sketched-AirDefense: Gradient Sketching and Trajectory Prediction for Byzantine Defense
with AirComp Physical Layer Simulation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque

# Attack types that require special handling
ADAPTIVE_ATTACKS = ['null_space', 'slow_poison', 'predictor_proxy']
BASIC_ATTACKS = ['omni', 'gaussian']  # Traditional attacks that also need explicit handling
ALL_SUPPORTED_ATTACKS = ADAPTIVE_ATTACKS + BASIC_ATTACKS


from .model_utils import get_model_flattened_dim, flatten_model_updates, unflatten_model_updates


class GradientSketcher:
    """
    Responsible for generating random projection matrix S and compressing
    high-dimensional gradients to low-dimensional sketches.

    Memory-efficient version using Sparse Random Projection for large models.
    """
    def __init__(self, input_dim, sketch_dim, device):
        self.device = device
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        
        # Threshold: 20M elements. For ResNet-18 (11M * 1024), we use sparse.
        self.is_sparse = (input_dim * sketch_dim > 20_000_000)

        if self.is_sparse:
            # Sparse Random Projection: each row has exactly 's' non-zero entries
            s = 8 
            # Vectorized creation to avoid slow Python loops (~11M iterations)
            row_indices = torch.arange(input_dim, device=device).view(-1, 1).repeat(1, s).view(-1)
            col_indices = torch.randint(0, sketch_dim, (input_dim, s), device=device).view(-1)
            indices = torch.stack([row_indices, col_indices])
            
            # Values are +/- 1 / sqrt(s)
            values = (torch.randint(0, 2, (input_dim * s,), device=device).float() * 2 - 1) / (s ** 0.5)
            
            self.S = torch.sparse_coo_tensor(indices, values, (input_dim, sketch_dim)).to(device)
            print(f"[GradientSketcher] Using Sparse Projection Matrix (Vectorized Init, s={s})")
        else:
            self.S = torch.randn(input_dim, sketch_dim).to(device) / (sketch_dim ** 0.5)
            print(f"[GradientSketcher] Using Dense Projection Matrix ({input_dim}x{sketch_dim})")

    def sketch(self, update_vector):
        """
        Compress gradient vector. Handles both sparse and dense S.
        """
        if self.is_sparse:
            # Efficient sparse-dense matrix multiplication
            # S is [d, k], update_vector is [d]
            return torch.sparse.mm(self.S.t(), update_vector.unsqueeze(-1)).squeeze()
        else:
            return torch.matmul(update_vector, self.S)

    def reset_projection(self):
        self.__init__(self.input_dim, self.sketch_dim, self.device)


class TrajectoryPredictor(nn.Module):
    """
    Self-supervised temporal predictor: Input past W rounds of sketches,
    predict the next round's sketch.
    """
    def __init__(self, sketch_dim, hidden_dim, num_layers=1):
        """
        Args:
            sketch_dim: Dimension of sketch vectors
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
        """
        super(TrajectoryPredictor, self).__init__()
        # Using GRU to process temporal data
        self.rnn = nn.GRU(input_size=sketch_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
        # Output layer predicts next round's sketch
        self.fc = nn.Linear(hidden_dim, sketch_dim)

    def forward(self, history):
        """
        Predict next sketch based on history.

        Args:
            history: [batch_size, window_size, sketch_dim] (trajectory history for each Cluster)
        Returns:
            prediction: [batch_size, sketch_dim] predicted sketch
        """
        out, _ = self.rnn(history)
        # Take the output of the last timestep
        last_step = out[:, -1, :]
        prediction = self.fc(last_step)
        return prediction


class DefenseAwarePredictor(nn.Module):
    """
    Defense-Aware Trajectory Predictor using LSTM.
    Designed to resist slow-poisoning attacks through adversarial training.

    Key insight: The predictor is trained to be ROBUST against small drifts,
    so when actual attacks cause drift, the prediction error becomes large,
    triggering defense mechanisms.
    """
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=1):
        """
        Args:
            input_dim: Dimension of input sketch vectors
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
        """
        super(DefenseAwarePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Predict next sketch based on history.

        Args:
            x: [batch_size, seq_len, input_dim] history trajectory
        Returns:
            pred: [batch_size, input_dim] predicted next sketch
        """
        out, (h_n, c_n) = self.lstm(x)
        # Take the output of the last timestep
        pred = self.head(out[:, -1, :])
        return pred


def defense_aware_loss(model, history_batch, true_target, lambda_adv=0.5, device='cuda'):
    """
    Defense-aware loss function combining MSE and adversarial robustness.

    The adversarial loss trains the model to be robust against slow drifting attacks.
    When input history has small cumulative drift, the model should still predict
    the NORMAL (un-drifted) target, not follow the drift.

    Args:
        model: DefenseAwarePredictor or TrajectoryPredictor
        history_batch: [batch, window_size, dim] normal history trajectories
        true_target: [batch, dim] true next-round sketches
        lambda_adv: Weight for adversarial loss (default 0.5)
        device: torch device
    Returns:
        total_loss: Combined loss (L_mse + lambda * L_adv)
        loss_mse: MSE loss component
        loss_adv: Adversarial loss component
    """
    # 1. Normal prediction loss (MSE)
    pred_normal = model(history_batch)
    loss_mse = F.mse_loss(pred_normal, true_target.detach())  # [FIX] Detach target to prevent gradient leakage

    # 2. Generate simulated "slow-poisoning" attack samples
    # Attack method: Add small linear drift to history trajectory
    # Simulates attacker gradually shifting gradients in a random direction

    # Random drift direction (normalized)
    drift_dir = torch.randn_like(true_target).to(device)
    drift_dir = drift_dir / (torch.norm(drift_dir, dim=1, keepdim=True) + 1e-8)

    # Create adversarial history by adding cumulative drift
    adv_history = history_batch.clone()
    seq_len = history_batch.size(1)

    # Drift magnitude increases over time (simulating slow poisoning)
    drift_scale = 0.05  # Small drift per timestep
    for t in range(seq_len):
        # Cumulative drift: (t+1) * drift_scale * direction
        noise = (t + 1) * drift_scale * drift_dir.unsqueeze(1)
        adv_history[:, t, :] = adv_history[:, t, :] + noise.squeeze(1)

    # 3. Compute adversarial loss
    # Goal: Even with drifted input, prediction should stay close to TRUE target
    # This makes the model ROBUST - it won't follow the drift
    # When real attack happens, prediction stays normal but received signal drifts
    # -> Large gap -> Anomaly detected!
    pred_adv = model(adv_history)
    loss_adv = F.mse_loss(pred_adv, true_target.detach())  # [FIX] Detach target

    # Total loss: accuracy + robustness
    total_loss = loss_mse + lambda_adv * loss_adv

    return total_loss, loss_mse, loss_adv


def handle_cluster_switch(client_id, old_cluster_id, new_cluster_id,
                          cluster_states, client_states, warmup_rounds=3):
    """
    Handle client switching from one cluster to another.

    Implements soft-landing mechanism for cluster transitions.
    Note: The predictor runs on the Server side using cluster-level history (self.history_buffer),
    not client-level history. So we only track warmup counters here.

    Args:
        client_id: ID of the client switching clusters
        old_cluster_id: ID of the old cluster
        new_cluster_id: ID of the new cluster
        cluster_states: Dictionary tracking each cluster's state
            {cluster_id: {'centroid_history': tensor, 'members': set}}
        client_states: Dictionary tracking each client's state
            {client_id: {'warmup_counter': int, 'cluster_id': int}}
        warmup_rounds: Number of rounds for soft-landing (default 3)
    """
    # [FIX] Removed dead code: client_states[client_id]['history'] assignment
    # The predictor uses Server-side cluster-level history, not client-level history

    # 1. Set warmup flag for soft-landing
    client_states[client_id]['warmup_counter'] = warmup_rounds

    # 2. Update cluster membership
    client_states[client_id]['cluster_id'] = new_cluster_id

    # 3. Update cluster member sets
    if old_cluster_id in cluster_states:
        cluster_states[old_cluster_id]['members'].discard(client_id)
    if new_cluster_id not in cluster_states:
        cluster_states[new_cluster_id] = {'members': set(), 'centroid_history': None}
    cluster_states[new_cluster_id]['members'].add(client_id)


def compute_warmup_weight(base_weight, warmup_counter, warmup_rounds=3):
    """
    Compute adjusted weight for clients in warm-up phase after cluster switch.

    Weight gradually increases from 0.5 * base_weight to base_weight
    over warmup_rounds rounds.

    Args:
        base_weight: Original weight before warm-up adjustment
        warmup_counter: Remaining warm-up rounds (decreases each round)
        warmup_rounds: Total warm-up rounds
    Returns:
        adjusted_weight: Weight adjusted for warm-up phase
    """
    if warmup_counter <= 0:
        return base_weight

    # Linear warm-up: weight increases from 0.5 to 1.0 over warmup_rounds
    progress = 1.0 - (warmup_counter / warmup_rounds)
    warmup_factor = 0.5 + 0.5 * progress

    return base_weight * warmup_factor


def compute_anomaly_score(predicted_sketch, actual_sketch):
    """
    Compute prediction error as anomaly score.

    Args:
        predicted_sketch: Predicted sketches [num_clusters, sketch_dim]
        actual_sketch: Actual sketches [num_clusters, sketch_dim]
    Returns:
        error: Anomaly scores [num_clusters]
    """
    # MSE Loss: ||s_pred - s_real||^2
    error = torch.norm(predicted_sketch - actual_sketch, dim=1) ** 2
    return error


def compute_trust_weights(anomaly_scores, method='softmax', threshold=0.2, temperature=2.0):
    """
    Compute trust weights from anomaly scores.
    
    Args:
        anomaly_scores: Tensor of anomaly scores [num_clusters]
        method: 'softmax' or 'threshold'
        threshold: For threshold method, top percentage to filter
        temperature: For softmax, controls sensitivity (higher = more uniform)
    Returns:
        weights: Trust weights [num_clusters]
    """
    num_clusters = len(anomaly_scores)

    if method == 'softmax':
        # Normalize scores to prevent numerical issues
        scores = anomaly_scores - anomaly_scores.min()
        
        # Use temperature scaling: higher temperature -> more uniform distribution
        # Temperature of 2.0 means we need 2x the score difference to get the same weight ratio
        if scores.max() > 0:
            # Scale to reasonable range, then apply temperature
            scores = scores / scores.max() * temperature
        
        # Softmax of negative scores: lower anomaly -> higher weight
        weights = F.softmax(-scores, dim=0)
        
        # [FIX] Apply minimum weight floor to prevent any cluster from being completely ignored
        # This is important when there are no attackers - all clusters should contribute
        min_weight = 0.5 / num_clusters  # At least half of uniform weight
        weights = torch.clamp(weights, min=min_weight)
        weights = weights / weights.sum()  # Re-normalize

    elif method == 'threshold':
        k = max(1, int(num_clusters * threshold))
        _, top_indices = torch.topk(anomaly_scores, k)
        weights = torch.ones(num_clusters, device=anomaly_scores.device)
        weights[top_indices] = 0
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(num_clusters, device=anomaly_scores.device) / num_clusters
    else:
        weights = torch.ones(num_clusters, device=anomaly_scores.device) / num_clusters

    return weights


class SketchedAirDefense:
    """
    Main defense module that combines sketching and trajectory prediction
    with AirComp physical layer simulation.

    Enhanced with:
    - Defense-aware trajectory predictor (LSTM)
    - Adversarial training loss
    - Dynamic cluster state tracking
    - Soft-landing mechanism for cluster switching
    """
    def __init__(self, args, model_dim, device):
        """
        Args:
            args: Command line arguments
            model_dim: Total dimension of model parameters
            device: torch device
        """
        self.args = args
        self.device = device
        self.model_dim = model_dim
        self.num_clusters = args.num_cluster
        self.window_size = args.window_size
        self.sketch_dim = args.sketch_dim

        # Physical layer parameters
        self.B = 1e+6  # Bandwidth
        self.N0 = 1e-7 # Noise power spectral density

        # Defense-aware training parameters
        self.lambda_adv = getattr(args, 'lambda_adv', 0.5)  # Adversarial loss weight
        self.warmup_rounds = getattr(args, 'warmup_rounds', 3)  # Soft-landing rounds
        self.use_defense_aware = getattr(args, 'use_defense_aware', True)  # Use defense-aware predictor

        # Initialize sketcher
        self.sketcher = GradientSketcher(model_dim, args.sketch_dim, device)

        # Initialize predictor (choose between defense-aware LSTM or standard GRU)
        if self.use_defense_aware:
            self.predictor = DefenseAwarePredictor(
                input_dim=args.sketch_dim,
                hidden_dim=args.pred_hidden,
                num_layers=1
            ).to(device)
        else:
            self.predictor = TrajectoryPredictor(
                sketch_dim=args.sketch_dim,
                hidden_dim=args.pred_hidden,
                num_layers=1
            ).to(device)

        # Optimizer for predictor
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=args.pred_lr
        )

        # History buffer: stores past window_size rounds of sketches for each cluster
        # Shape of each entry: [num_clusters, sketch_dim]
        self.history_buffer = deque(maxlen=self.window_size)

        # Loss function for self-supervised learning (per-sample loss for selective training)
        self.loss_fn = nn.MSELoss(reduction='none')

        # --- New: State tracking for dynamic clustering ---
        # Cluster states: track each cluster's centroid history and members
        self.cluster_states = {}
        for c in range(self.num_clusters):
            self.cluster_states[c] = {
                'centroid_history': None,  # Will store cluster's aggregated sketch history
                'members': set()  # Set of user IDs in this cluster
            }

        # Client states: track each client's individual state
        self.client_states = {}  # Will be initialized when users are assigned

        # Previous cluster assignments for detecting switches
        self.prev_cluster_assignments = None

        # Attack state for adaptive attacks (persistent across rounds)
        self.attack_state = {}

    def set_user_distances(self, distance):
        """
        Set user distances for location-based clustering.
        Should be called after generate_clients() in main.py.
        
        Args:
            distance: Array of distances for each user [num_users]
        """
        self.user_distances = distance[:-1]  # Exclude server (last user)
    
    def get_cluster_assignments(self, num_users, round_idx, method=None):
        """
        Assign users to clusters using specified method.

        Args:
            num_users: Number of users (including server)
            round_idx: Current round index
            method: Clustering method - 'sequential', 'random', 'location', or None (uses args.clustering)
        Returns:
            cluster_assignments: List of lists, each inner list contains user indices for a cluster
        """
        K = num_users - 1  # Exclude server (last user)
        num_per_cluster = K // self.num_clusters

        # Determine clustering method
        if method is None:
            method = getattr(self.args, 'clustering', 'sequential')
        
        indices = list(range(K))
        
        if method == 'random':
            # Random shuffle
            np.random.shuffle(indices)
            
        elif method == 'location':
            # Sort by distance from base station (closer users first)
            if hasattr(self, 'user_distances') and self.user_distances is not None:
                indices = sorted(indices, key=lambda i: self.user_distances[i])
            else:
                print("[Warning] Location clustering requested but distances not set. Using sequential.")
                
        elif method == 'sequential':
            # Keep original order (indices already in order)
            pass
        
        # Assign sorted/shuffled indices to clusters
        cluster_assignments = []
        for c in range(self.num_clusters):
            start_idx = c * num_per_cluster
            end_idx = start_idx + num_per_cluster if c < self.num_clusters - 1 else K
            cluster_assignments.append(indices[start_idx:end_idx])

        return cluster_assignments

    def compute_cluster_updates_with_aircomp(self, w_locals, w_global, cluster_assignments,
                                              P, G, H, round_idx):
        """
        Compute aggregated update for each cluster with AirComp physical layer simulation.
        Simulates signal superposition with channel gain, transmit power, and noise.

        Args:
            w_locals: List of local model weights
            w_global: Global model weights
            cluster_assignments: Cluster assignments from get_cluster_assignments
            P: Transmit power array [num_users]
            G: Small-scale fading matrix [num_users-1, rounds]
            H: Channel gain matrix [num_users-1, rounds]
            round_idx: Current round index
        Returns:
            cluster_updates: List of aggregated updates for each cluster (with noise)
            cluster_scaling: Scaling factors for each cluster
            real_participating_counts: [FIX] Actual number of users participating per cluster
        """
        cluster_updates = []
        cluster_scaling = []
        real_participating_counts = []  # [FIX] Track actual participating users per cluster
        
        # Step 1: Compute gamma (gradient norm squared) for all users - same as FedSAC
        K = self.args.num_users - 1
        gamma = np.zeros((K, 1))
        for user_idx in range(K):
            for key in sorted(w_global.keys()):
                tmp = w_locals[user_idx][key].float() - w_global[key].float()
                tmp_np = tmp.cpu().numpy()
                gamma[user_idx][0] += (np.linalg.norm(tmp_np)) ** 2
        
        # Step 2: Compute H_norm (max gradient norm) - same as FedSAC's H = sqrt(max(gamma))
        H_norm = np.sqrt(np.max(gamma)) if np.max(gamma) > 0 else 1.0

        for cluster_idx, cluster_users in enumerate(cluster_assignments):
            # Initialize cluster aggregate
            cluster_agg = None

            # Find minimum equivalent channel in this cluster for scaling
            min_h = float('inf')
            valid_users = []

            for user_idx in cluster_users:
                if H[user_idx][round_idx] > 0.1:  # Channel threshold
                    valid_users.append(user_idx)
                    eq_h = np.sqrt(P[user_idx]) * G[user_idx][round_idx]
                    if eq_h < min_h:
                        min_h = eq_h

            if min_h == float('inf'):
                min_h = 1.0  # Fallback

            # [FIX] Counter for actual participating users
            participating_count = 0

            # Compute channel-scaled aggregation
            for user_idx in cluster_users:
                # Compute update: w_local - w_global
                user_update = {}
                for key in sorted(w_global.keys()):
                    user_update[key] = w_locals[user_idx][key].float() - w_global[key].float()

                if cluster_agg is None:
                    cluster_agg = {}
                    for key in sorted(w_global.keys()):
                        cluster_agg[key] = torch.zeros_like(w_global[key], dtype=torch.float32).to(self.device)

                # FIX Bug 1: 简化的 AirComp 模拟 (Perfect Power Control Assumption)
                # 移除除法操作，假设预编码已经抵消了信道差异，直接叠加
                if H[user_idx][round_idx] > 0.1:  # 仅当信道足够好时才参与
                    participating_count += 1  # [FIX] Count actual participants
                    for key in cluster_agg.keys():
                        cluster_agg[key] += user_update[key]  # 直接叠加

            # [FIX] Prevent division by zero (if all users dropped, set to 1)
            real_participating_counts.append(max(1, participating_count))

            # Add AWGN noise (simulating over-the-air transmission)
            # [FIX] Compute zeta following FedSAC formula: zeta = sqrt(P[0]) * K / H_norm * min_h
            # noise_std = sqrt(B*N0/2) * lr / zeta
            zeta = np.sqrt(P[0]) * K / H_norm * min_h if H_norm > 0 else 1.0
            noise_std = np.sqrt(self.B * self.N0 / 2) * self.args.lr / zeta if zeta > 0 else 0.0
            
            for key in cluster_agg.keys():
                noise = torch.randn_like(cluster_agg[key]) * noise_std
                cluster_agg[key] = cluster_agg[key] + noise

            cluster_updates.append(cluster_agg)
            cluster_scaling.append(min_h)

        return cluster_updates, cluster_scaling, real_participating_counts  # [FIX] Return counts

    def extract_sketches(self, cluster_updates, cluster_sizes=None):
        """
        Extract sketches from cluster updates.

        Args:
            cluster_updates: List of cluster update dictionaries
            cluster_sizes: List of cluster sizes for normalization (optional)
                           [FIX] Added to prevent magnitude shift when cluster size changes
        Returns:
            sketches: Tensor [num_clusters, sketch_dim]
        """
        sketches = []
        for idx, update in enumerate(cluster_updates):
            flattened = flatten_model_updates(update, self.device)

            # [FIX] Normalize by cluster size to prevent magnitude shift artifacts
            # When cluster size changes (e.g., 10 -> 11 members), the summed gradient
            # magnitude changes proportionally. This causes the predictor to see
            # a sudden jump and incorrectly flag it as anomaly.
            if cluster_sizes is not None and cluster_sizes[idx] > 0:
                flattened = flattened / cluster_sizes[idx]

            sketch = self.sketcher.sketch(flattened)
            sketches.append(sketch)

        return torch.stack(sketches)

    def predict_and_detect(self, current_sketches):
        """
        Use trajectory predictor to predict and detect anomalies.

        Args:
            current_sketches: Current round's sketches [num_clusters, sketch_dim]
        Returns:
            trust_weights: Trust weights for each cluster [num_clusters]
            anomaly_scores: Anomaly scores [num_clusters]
            predicted_sketches: Predicted sketches (or None if not enough history)
        """
        if len(self.history_buffer) < self.window_size:
            # Not enough history, return uniform weights
            trust_weights = torch.ones(self.num_clusters, device=self.device) / self.num_clusters
            anomaly_scores = torch.zeros(self.num_clusters, device=self.device)
            return trust_weights, anomaly_scores, None

        # Prepare history tensor [num_clusters, window_size, sketch_dim]
        history = torch.stack(list(self.history_buffer), dim=1)

        # Predict next sketches (use eval mode for prediction)
        self.predictor.eval()
        with torch.no_grad():
            predicted_sketches = self.predictor(history)

        # Compute anomaly scores
        anomaly_scores = compute_anomaly_score(predicted_sketches, current_sketches)

        # Compute trust weights
        trust_weights = compute_trust_weights(
            anomaly_scores,
            method='softmax',
            threshold=self.args.anomaly_threshold
        )

        return trust_weights, anomaly_scores, predicted_sketches

    def update_predictor_selective(self, current_sketches, anomaly_scores, trust_weights):
        """
        Update predictor using ONLY normal samples with defense-aware loss.
        Uses adversarial training to make predictor robust against slow drift attacks.

        Args:
            current_sketches: Current round's sketches [num_clusters, sketch_dim]
            anomaly_scores: Anomaly scores [num_clusters]
            trust_weights: Trust weights [num_clusters]
        Returns:
            loss_info: Dictionary containing loss components
        """
        if len(self.history_buffer) < self.window_size:
            return {'total': 0.0, 'mse': 0.0, 'adv': 0.0}

        # Determine which clusters are "normal" (below median anomaly score)
        median_score = torch.median(anomaly_scores)
        normal_mask = anomaly_scores <= median_score

        # Also exclude clusters with very low trust weights
        trust_threshold = 1.0 / (2 * self.num_clusters)  # Below uniform / 2
        normal_mask = normal_mask & (trust_weights > trust_threshold)

        if normal_mask.sum() == 0:
            return {'total': 0.0, 'mse': 0.0, 'adv': 0.0}  # No normal samples to train on

        # Prepare history tensor [num_clusters, window_size, sketch_dim]
        history = torch.stack(list(self.history_buffer), dim=1)

        # Select only normal clusters for training
        normal_indices = torch.where(normal_mask)[0]
        normal_history = history[normal_indices]
        normal_targets = current_sketches[normal_indices]

        # Train predictor with defense-aware loss
        self.predictor.train()
        self.predictor_optimizer.zero_grad()

        # Use defense-aware loss for adversarial robustness training
        total_loss, loss_mse, loss_adv = defense_aware_loss(
            self.predictor,
            normal_history,
            normal_targets,
            lambda_adv=self.lambda_adv,
            device=self.device
        )

        total_loss.backward()
        self.predictor_optimizer.step()

        return {
            'total': total_loss.item(),
            'mse': loss_mse.item(),
            'adv': loss_adv.item()
        }

    def update_history(self, current_sketches):
        """
        Add current sketches to history buffer.

        Args:
            current_sketches: Current round's sketches [num_clusters, sketch_dim]
        """
        self.history_buffer.append(current_sketches.detach().clone())

    def update_reputation(self, reputation, trust_weights, cluster_assignments):
        """
        Update reputation based on trust weights.

        Args:
            reputation: Current reputation array [num_users-1]
            trust_weights: Trust weights for each cluster [num_clusters]
            cluster_assignments: Cluster assignments
        Returns:
            updated_reputation: Updated reputation array
        """
        updated_reputation = reputation.copy()

        for cluster_idx, cluster_users in enumerate(cluster_assignments):
            weight = trust_weights[cluster_idx].item()
            for user_idx in cluster_users:
                # Higher trust weight -> positive reputation update
                # Lower trust weight -> negative reputation update
                reputation_delta = weight - (1.0 / self.num_clusters)
                updated_reputation[user_idx] += reputation_delta

        return updated_reputation

    def apply_adaptive_attacks(self, w_locals, w_global, args):
        """
        Apply attacks to Byzantine users' local models.

        Supports both adaptive and basic attack types:
        Adaptive:
        - null_space: Exploits null space of projection matrix
        - slow_poison: Slow drift attack within predictor tolerance
        - predictor_proxy: Uses local predictor to evade detection

        Basic:
        - omni: Omniscient attack (sends negative gradient)
        - gaussian: Random Gaussian noise attack

        Args:
            w_locals: List of local model weights
            w_global: Current global model weights
            args: Command line arguments
        Returns:
            w_locals_attacked: Modified local weights with attacks applied
        """
        attack_type = args.attack

        # Only process supported attacks
        if attack_type not in ALL_SUPPORTED_ATTACKS:
            return w_locals

        w_locals_attacked = copy.deepcopy(w_locals)
        num_byz = args.num_byz

        # Lazy import for adaptive attacks to avoid circular dependency
        if attack_type in ADAPTIVE_ATTACKS:
            from .adaptive_attacks import (
                NullSpaceAttack, SlowPoisoningAttack, PredictorProxyAttack
            )

        for user_idx in range(num_byz):
            # Get benign gradient (flattened)
            benign_flat = flatten_model_updates(
                {k: w_locals[user_idx][k] - w_global[k] for k in sorted(w_global.keys())},
                self.device
            )

            # ========== Basic Attacks ==========
            if attack_type == 'omni':
                # Omniscient attack: send negative gradient (opposite direction)
                poisoned_flat = -benign_flat

            elif attack_type == 'gaussian':
                # Gaussian attack: send random noise with same norm as benign
                noise = torch.randn_like(benign_flat)
                # Normalize noise to have same magnitude as benign gradient
                benign_norm = torch.norm(benign_flat)
                noise_norm = torch.norm(noise)
                if noise_norm > 1e-8:
                    poisoned_flat = noise * (benign_norm / noise_norm)
                else:
                    poisoned_flat = noise

            # ========== Adaptive Attacks ==========
            elif attack_type == 'null_space':
                # Initialize attacker if needed
                if 'null_space' not in self.attack_state:
                    self.attack_state['null_space'] = NullSpaceAttack(
                        self.sketcher.S,
                        device=self.device,
                        attack_strength=getattr(args, 'attack_strength', 1.0)
                    )
                attacker = self.attack_state['null_space']
                poisoned_flat = attacker.generate_poison(benign_flat)

            elif attack_type == 'slow_poison':
                # Initialize attacker if needed
                if 'slow_poison' not in self.attack_state:
                    self.attack_state['slow_poison'] = SlowPoisoningAttack(
                        target_direction=None,
                        alpha=getattr(args, 'poison_alpha', 0.05),
                        decay_rate=getattr(args, 'poison_decay', 0.99),
                        device=self.device
                    )
                attacker = self.attack_state['slow_poison']
                poisoned_flat = attacker.generate_poison(benign_flat)

            elif attack_type == 'predictor_proxy':
                # Initialize attacker if needed
                if 'predictor_proxy' not in self.attack_state:
                    self.attack_state['predictor_proxy'] = PredictorProxyAttack(
                        sketch_dim=self.sketch_dim,
                        hidden_dim=getattr(args, 'proxy_hidden', 64),
                        window_size=self.window_size,
                        threshold_margin=getattr(args, 'threshold_margin', 0.1),
                        device=self.device
                    )
                attacker = self.attack_state['predictor_proxy']

                # Feed history to attacker's local predictor
                if len(self.history_buffer) >= self.window_size:
                    for sketch in self.history_buffer:
                        attacker.observe_sketch(sketch[0])  # Use first cluster's sketch
                    attacker.train_local_predictor(num_steps=3)

                    history_tensor = torch.stack(list(self.history_buffer)[-self.window_size:])
                    # Use mean across clusters for history
                    history_mean = history_tensor.mean(dim=1)
                    poisoned_flat = attacker.generate_poison(
                        benign_flat, self.sketcher, history_mean
                    )
                else:
                    # Not enough history, use simple negative attack
                    poisoned_flat = -benign_flat

            # Convert back to state_dict format
            poisoned_update = unflatten_model_updates(poisoned_flat, w_global, self.device)

            # Apply poisoned update to get new local model
            for key in sorted(w_global.keys()):
                w_locals_attacked[user_idx][key] = w_global[key].float() + poisoned_update[key]

        return w_locals_attacked


    def aggregate_with_defense(self, w_locals, w_global, args, round_idx,
                                P, G, H, reputation, q):
        """
        Main defense aggregation function with AirComp simulation.

        Enhanced with:
        - Defense-aware predictor training
        - Cluster state tracking
        - Soft-landing for cluster switches

        Args:
            w_locals: List of local model weights
            w_global: Current global model weights
            args: Arguments
            round_idx: Current round index
            P: Transmit power array
            G: Small-scale fading matrix
            H: Channel gain matrix
            reputation: Current reputation array
            q: Queue state array
        Returns:
            w_new: New global model weights
            reputation: Updated reputation
            q: Updated queue state
            defense_info: Dictionary with defense statistics
        """
        num_users = args.num_users - 1  # Exclude server

        # Step 0: Initialize client states if first round
        if len(self.client_states) == 0:
            for user_idx in range(num_users):
                self.client_states[user_idx] = {
                    # [FIX] Removed 'history' field - predictor uses cluster-level history
                    'warmup_counter': 0,
                    'cluster_id': -1  # Will be set during assignment
                }

        # Step 1: Get cluster assignments
        cluster_assignments = self.get_cluster_assignments(args.num_users, round_idx)

        # Step 1.5: Detect cluster switches and handle handover
        cluster_switches = []
        for cluster_idx, cluster_users in enumerate(cluster_assignments):
            for user_idx in cluster_users:
                old_cluster = self.client_states[user_idx]['cluster_id']
                if old_cluster != -1 and old_cluster != cluster_idx:
                    # User switched clusters!
                    cluster_switches.append((user_idx, old_cluster, cluster_idx))
                    handle_cluster_switch(
                        user_idx, old_cluster, cluster_idx,
                        self.cluster_states, self.client_states,
                        warmup_rounds=self.warmup_rounds
                    )
                else:
                    # First assignment or same cluster
                    self.client_states[user_idx]['cluster_id'] = cluster_idx
                    if cluster_idx not in self.cluster_states:
                        self.cluster_states[cluster_idx] = {'members': set(), 'centroid_history': None}
                    self.cluster_states[cluster_idx]['members'].add(user_idx)

        # Step 1.8: Attacks are now applied by MaliciousClient during train()
        # The w_locals passed here already contain poisoned weights for Byzantine users
        # (Previously apply_adaptive_attacks was called here, but that caused double-attack issues)

        # Step 2: Compute cluster updates with AirComp physical layer simulation
        # [FIX] Now also returns real_participating_counts for dynamic normalization
        cluster_updates, cluster_scaling, real_participating_counts = self.compute_cluster_updates_with_aircomp(
            w_locals, w_global, cluster_assignments, P, G, H, round_idx
        )

        # Step 3: Extract sketches (with DYNAMIC cluster size normalization)
        # [FIX] Use real_participating_counts instead of static len(cluster_assignments[c])
        # This prevents false anomaly detection when users drop out due to bad channels
        current_sketches = self.extract_sketches(cluster_updates, real_participating_counts)

        # Step 3.5: Update cluster centroid histories
        for cluster_idx in range(self.num_clusters):
            self.cluster_states[cluster_idx]['centroid_history'] = current_sketches[cluster_idx].detach().clone()

        # Step 4: Predict and detect anomalies
        trust_weights, anomaly_scores, _ = self.predict_and_detect(current_sketches)

        # Step 5: Update predictor with defense-aware loss
        predictor_loss_info = self.update_predictor_selective(
            current_sketches, anomaly_scores, trust_weights
        )

        # Step 6: Update history buffer (AFTER prediction and detection)
        self.update_history(current_sketches)

        # Step 7: Update reputation based on trust weights
        reputation = self.update_reputation(reputation, trust_weights, cluster_assignments)

        # Step 8: Weighted aggregation with soft-landing adjustment
        # [FIX] Use real_participating_counts for normalization:
        # 1. cluster_update is SUM of ACTUAL participating user updates (from AirComp)
        # 2. Divide by real_participating_counts[c] to get AVERAGE update
        # 3. Apply trust_weight for defense-aware weighting
        w_new = copy.deepcopy(w_global)
        for key in sorted(w_new.keys()):
            weighted_update = torch.zeros_like(w_new[key], dtype=torch.float32).to(self.device)
            for c, (cluster_update, weight) in enumerate(zip(cluster_updates, trust_weights)):
                # [FIX] Use dynamic count instead of static cluster size
                num_users_in_cluster = real_participating_counts[c]

                # Apply soft-landing warmup weights for users who recently switched
                effective_weight = weight.item()
                warmup_adjustments = []
                for user_idx in cluster_assignments[c]:
                    warmup_counter = self.client_states[user_idx]['warmup_counter']
                    if warmup_counter > 0:
                        user_warmup_factor = compute_warmup_weight(1.0, warmup_counter, self.warmup_rounds)
                        warmup_adjustments.append(user_warmup_factor)

                # Average warmup factor for cluster (if any users are warming up)
                if warmup_adjustments:
                    avg_warmup = sum(warmup_adjustments) / len(warmup_adjustments)
                    effective_weight *= avg_warmup

                weighted_update += (effective_weight * cluster_update[key]) / num_users_in_cluster
            w_new[key] = w_global[key].float() + weighted_update

        # Step 9: Decrement warmup counters
        for user_idx in range(num_users):
            if self.client_states[user_idx]['warmup_counter'] > 0:
                self.client_states[user_idx]['warmup_counter'] -= 1

        # Step 10: Update q (queue state) - simple update based on trust
        q_new = q.copy()
        for cluster_idx, cluster_users in enumerate(cluster_assignments):
            for user_idx in cluster_users:
                if trust_weights[cluster_idx].item() > 1.0 / self.num_clusters:
                    q_new[user_idx] = max(q_new[user_idx] - 0.1, 0)
                else:
                    q_new[user_idx] = q_new[user_idx] + 0.1

        # Store current assignments for next round comparison
        self.prev_cluster_assignments = cluster_assignments

        # Prepare defense info for logging
        defense_info = {
            'trust_weights': trust_weights.cpu().detach().numpy(),
            'anomaly_scores': anomaly_scores.cpu().detach().numpy(),
            'predictor_loss': predictor_loss_info['total'] if isinstance(predictor_loss_info, dict) else predictor_loss_info,
            'predictor_loss_mse': predictor_loss_info.get('mse', 0.0) if isinstance(predictor_loss_info, dict) else 0.0,
            'predictor_loss_adv': predictor_loss_info.get('adv', 0.0) if isinstance(predictor_loss_info, dict) else 0.0,
            'history_size': len(self.history_buffer),
            'cluster_scaling': cluster_scaling,
            'cluster_switches': len(cluster_switches),  # Number of switches this round
            'warmup_users': sum(1 for u in self.client_states.values() if u['warmup_counter'] > 0)
        }

        return w_new, reputation, q_new, defense_info


def Sketched_Defense_Aggregation(w_locals, args, w_global, defense_module, round_idx,
                                  P, G, H, reputation, q):
    """
    Wrapper function for compatibility with main.py structure.
    Now includes all physical layer parameters and returns reputation/q.

    Args:
        w_locals: List of local model weights
        args: Arguments
        w_global: Current global model weights
        defense_module: SketchedAirDefense instance
        round_idx: Current round index
        P: Transmit power array
        G: Small-scale fading matrix
        H: Channel gain matrix
        reputation: Current reputation array
        q: Queue state array
    Returns:
        w_new: New global model weights
        reputation: Updated reputation
        q: Updated queue state
        defense_info: Defense statistics
    """
    return defense_module.aggregate_with_defense(
        w_locals, w_global, args, round_idx, P, G, H, reputation, q
    )


def apply_adaptive_attacks_standalone(w_locals, w_global, args, attack_state,
                                      sketcher, history_buffer, device):
    """
    在非 Sketched-AirDefense 聚合（如 byzantine / proposed）场景下复用自适应攻击逻辑。

    说明：
    - 支持 ADAPTIVE_ATTACKS 和 BASIC_ATTACKS，与 SketchedAirDefense 内部一致。
    - predictor_proxy 需要历史 sketch；若历史不足则回退为负梯度攻击。
    - history_buffer: deque of sketch tensors，形状 [sketch_dim]；长度不足 window_size 时视为历史不足。
    """
    attack_type = args.attack
    if attack_type not in ALL_SUPPORTED_ATTACKS:
        return w_locals

    w_locals_attacked = copy.deepcopy(w_locals)
    num_byz = args.num_byz

    # 延迟导入自适应攻击类以避免循环依赖
    if attack_type in ADAPTIVE_ATTACKS:
        from .adaptive_attacks import (
            NullSpaceAttack, SlowPoisoningAttack, PredictorProxyAttack
        )

    for user_idx in range(num_byz):
        benign_flat = flatten_model_updates(
            {k: w_locals[user_idx][k] - w_global[k] for k in sorted(w_global.keys())},
            device
        )

        # ===== Basic attacks =====
        if attack_type == 'omni':
            poisoned_flat = -benign_flat

        elif attack_type == 'gaussian':
            noise = torch.randn_like(benign_flat)
            benign_norm = torch.norm(benign_flat)
            noise_norm = torch.norm(noise)
            if noise_norm > 1e-8:
                poisoned_flat = noise * (benign_norm / noise_norm)
            else:
                poisoned_flat = noise

        # ===== Adaptive attacks =====
        elif attack_type == 'null_space':
            if 'null_space' not in attack_state:
                attack_state['null_space'] = NullSpaceAttack(
                    sketcher.S,
                    device=device,
                    attack_strength=getattr(args, 'attack_strength', 1.0)
                )
            attacker = attack_state['null_space']
            poisoned_flat = attacker.generate_poison(benign_flat)

        elif attack_type == 'slow_poison':
            if 'slow_poison' not in attack_state:
                attack_state['slow_poison'] = SlowPoisoningAttack(
                    target_direction=None,
                    alpha=getattr(args, 'poison_alpha', 0.05),
                    decay_rate=getattr(args, 'poison_decay', 0.99),
                    device=device
                )
            attacker = attack_state['slow_poison']
            poisoned_flat = attacker.generate_poison(benign_flat)

        elif attack_type == 'predictor_proxy':
            if 'predictor_proxy' not in attack_state:
                attack_state['predictor_proxy'] = PredictorProxyAttack(
                    sketch_dim=sketcher.sketch_dim,
                    hidden_dim=getattr(args, 'proxy_hidden', 64),
                    window_size=args.window_size,
                    threshold_margin=getattr(args, 'threshold_margin', 0.1),
                    device=device
                )
            attacker = attack_state['predictor_proxy']

            if history_buffer is not None and len(history_buffer) >= args.window_size:
                history_tensor = torch.stack(list(history_buffer)[-args.window_size:])
                poisoned_flat = attacker.generate_poison(
                    benign_flat, sketcher, history_tensor
                )
            else:
                poisoned_flat = -benign_flat

        else:
            poisoned_flat = benign_flat

        poisoned_update = unflatten_model_updates(poisoned_flat, w_global, device)
        for key in sorted(w_global.keys()):
            w_locals_attacked[user_idx][key] = w_global[key].float() + poisoned_update[key]

    return w_locals_attacked
