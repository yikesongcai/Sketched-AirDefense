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


def get_model_flattened_dim(net):
    """
    Calculate the total number of parameters in the model.

    Args:
        net: PyTorch model
    Returns:
        total_dim: Total number of parameters
    """
    total_dim = 0
    for param in net.parameters():
        total_dim += param.numel()
    return total_dim


def flatten_model_updates(w_update, device):
    """
    Flatten model update dictionary to a single vector.
    IMPORTANT: Uses sorted keys to ensure deterministic order across environments.

    Args:
        w_update: Dictionary of model updates (state_dict format)
        device: torch device
    Returns:
        flattened: Flattened tensor of all updates
    """
    flattened = []
    # FIX: Use sorted keys to ensure deterministic order
    for key in sorted(w_update.keys()):
        flattened.append(w_update[key].view(-1).float())
    return torch.cat(flattened).to(device)


def unflatten_model_updates(flattened, reference_dict, device):
    """
    Unflatten a vector back to model update dictionary format.

    Args:
        flattened: Flattened tensor
        reference_dict: Reference state_dict for shape information
        device: torch device
    Returns:
        unflattened: Dictionary with same structure as reference_dict
    """
    unflattened = copy.deepcopy(reference_dict)
    idx = 0
    # FIX: Use sorted keys to ensure deterministic order
    for key in sorted(reference_dict.keys()):
        numel = reference_dict[key].numel()
        shape = reference_dict[key].shape
        unflattened[key] = flattened[idx:idx+numel].view(shape).to(device)
        idx += numel
    return unflattened


class GradientSketcher:
    """
    Responsible for generating random projection matrix S and compressing
    high-dimensional gradients to low-dimensional sketches.

    Physical meaning: Simulates compressed sensing or feature extraction
    in AirComp transmission.
    """
    def __init__(self, input_dim, sketch_dim, device):
        """
        Args:
            input_dim: Original dimension of the gradient vector
            sketch_dim: Target dimension of the sketch
            device: torch device
        """
        self.device = device
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        # Fixed random matrix S, ensuring consistent projection across rounds
        # Using normal distribution initialization, satisfying JL Lemma isometric property
        self.S = torch.randn(input_dim, sketch_dim).to(device) / (sketch_dim ** 0.5)

    def sketch(self, update_vector):
        """
        Compress gradient vector using random projection.

        Args:
            update_vector: Flattened gradient tensor [d]
        Returns:
            sketched_vector: Compressed tensor [sketch_dim]
        """
        # Simple linear projection: s = x * S
        return torch.matmul(update_vector, self.S)

    def reset_projection(self):
        """Reset the random projection matrix (optional, for experiments)"""
        self.S = torch.randn(self.input_dim, self.sketch_dim).to(self.device) / (self.sketch_dim ** 0.5)


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


def compute_trust_weights(anomaly_scores, method='softmax', threshold=0.2):
    """
    Compute trust weights based on anomaly scores.

    Args:
        anomaly_scores: Anomaly scores for each cluster [num_clusters]
        method: 'softmax' or 'threshold'
        threshold: Top percentage to filter (only for 'threshold' method)
    Returns:
        weights: Trust weights for each cluster [num_clusters]
    """
    num_clusters = len(anomaly_scores)

    if method == 'softmax':
        # Softmax of negative scores: lower anomaly -> higher weight
        weights = F.softmax(-anomaly_scores, dim=0)
    elif method == 'threshold':
        # Filter top threshold% anomalous clusters
        k = max(1, int(num_clusters * threshold))
        _, top_indices = torch.topk(anomaly_scores, k)
        weights = torch.ones(num_clusters, device=anomaly_scores.device)
        weights[top_indices] = 0
        # Normalize remaining weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(num_clusters, device=anomaly_scores.device) / num_clusters
    else:
        # Default: uniform weights
        weights = torch.ones(num_clusters, device=anomaly_scores.device) / num_clusters

    return weights


class SketchedAirDefense:
    """
    Main defense module that combines sketching and trajectory prediction
    with AirComp physical layer simulation.
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
        self.N0 = 1e-7  # Noise power spectral density

        # Initialize sketcher
        self.sketcher = GradientSketcher(model_dim, args.sketch_dim, device)

        # Initialize predictor
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

    def get_cluster_assignments(self, num_users, round_idx):
        """
        Assign users to clusters. Currently using sequential assignment.

        Args:
            num_users: Number of users (excluding server)
            round_idx: Current round index
        Returns:
            cluster_assignments: List of lists, each inner list contains user indices for a cluster
        """
        K = num_users - 1  # Exclude server (last user)
        num_per_cluster = K // self.num_clusters

        cluster_assignments = []
        indices = list(range(K))

        # Sequential assignment (can be extended to other methods)
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
        """
        cluster_updates = []
        cluster_scaling = []

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
                    for key in cluster_agg.keys():
                        cluster_agg[key] += user_update[key]  # 直接叠加

            # Add AWGN noise (simulating over-the-air transmission)
            noise_std = np.sqrt(self.B * self.N0 / 2) * self.args.lr / min_h
            for key in cluster_agg.keys():
                noise = torch.randn_like(cluster_agg[key]) * noise_std
                cluster_agg[key] = cluster_agg[key] + noise

            cluster_updates.append(cluster_agg)
            cluster_scaling.append(min_h)

        return cluster_updates, cluster_scaling

    def extract_sketches(self, cluster_updates):
        """
        Extract sketches from cluster updates.

        Args:
            cluster_updates: List of cluster update dictionaries
        Returns:
            sketches: Tensor [num_clusters, sketch_dim]
        """
        sketches = []
        for update in cluster_updates:
            flattened = flatten_model_updates(update, self.device)
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
        Update predictor using ONLY normal samples to prevent attack adaptation.
        FIX: Only train on clusters with low anomaly scores (judged as normal).

        Args:
            current_sketches: Current round's sketches [num_clusters, sketch_dim]
            anomaly_scores: Anomaly scores [num_clusters]
            trust_weights: Trust weights [num_clusters]
        Returns:
            loss_value: The training loss (0 if no training occurred)
        """
        if len(self.history_buffer) < self.window_size:
            return 0.0

        # Determine which clusters are "normal" (below median anomaly score)
        median_score = torch.median(anomaly_scores)
        normal_mask = anomaly_scores <= median_score

        # Also exclude clusters with very low trust weights
        trust_threshold = 1.0 / (2 * self.num_clusters)  # Below uniform / 2
        normal_mask = normal_mask & (trust_weights > trust_threshold)

        if normal_mask.sum() == 0:
            return 0.0  # No normal samples to train on

        # Prepare history tensor
        history = torch.stack(list(self.history_buffer), dim=1)

        # Train predictor only on normal clusters
        self.predictor.train()
        predicted_sketches = self.predictor(history)

        # Compute per-cluster loss
        per_cluster_loss = self.loss_fn(predicted_sketches, current_sketches).mean(dim=1)

        # Only backprop on normal clusters
        masked_loss = (per_cluster_loss * normal_mask.float()).sum() / normal_mask.sum()

        self.predictor_optimizer.zero_grad()
        masked_loss.backward()
        self.predictor_optimizer.step()

        return masked_loss.item()

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

    def aggregate_with_defense(self, w_locals, w_global, args, round_idx,
                                P, G, H, reputation, q):
        """
        Main defense aggregation function with AirComp simulation.

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
        # Step 1: Get cluster assignments
        cluster_assignments = self.get_cluster_assignments(args.num_users, round_idx)

        # Step 2: Compute cluster updates with AirComp physical layer simulation
        cluster_updates, cluster_scaling = self.compute_cluster_updates_with_aircomp(
            w_locals, w_global, cluster_assignments, P, G, H, round_idx
        )

        # Step 3: Extract sketches
        current_sketches = self.extract_sketches(cluster_updates)

        # Step 4: Predict and detect anomalies
        trust_weights, anomaly_scores, _ = self.predict_and_detect(current_sketches)

        # Step 5: Update predictor ONLY with normal samples (prevent attack adaptation)
        predictor_loss = self.update_predictor_selective(
            current_sketches, anomaly_scores, trust_weights
        )

        # Step 6: Update history buffer (AFTER prediction and detection)
        self.update_history(current_sketches)

        # Step 7: Update reputation based on trust weights
        reputation = self.update_reputation(reputation, trust_weights, cluster_assignments)

        # Step 8: Weighted aggregation with physical layer effects
        # FIX Bug 2: 除以 Cluster 的大小进行归一化
        # cluster_update 是多个用户更新的叠加，需要除以用户数
        w_new = copy.deepcopy(w_global)
        for key in sorted(w_new.keys()):
            weighted_update = torch.zeros_like(w_new[key], dtype=torch.float32).to(self.device)
            for c, (cluster_update, weight) in enumerate(zip(cluster_updates, trust_weights)):
                num_users_in_cluster = len(cluster_assignments[c])
                weighted_update += (weight.item() * cluster_update[key]) / num_users_in_cluster
            w_new[key] = w_global[key].float() + weighted_update

        # Update q (queue state) - simple update based on trust
        q_new = q.copy()
        for cluster_idx, cluster_users in enumerate(cluster_assignments):
            for user_idx in cluster_users:
                if trust_weights[cluster_idx].item() > 1.0 / self.num_clusters:
                    q_new[user_idx] = max(q_new[user_idx] - 0.1, 0)
                else:
                    q_new[user_idx] = q_new[user_idx] + 0.1

        # Prepare defense info for logging
        defense_info = {
            'trust_weights': trust_weights.cpu().detach().numpy(),
            'anomaly_scores': anomaly_scores.cpu().detach().numpy(),
            'predictor_loss': predictor_loss,
            'history_size': len(self.history_buffer),
            'cluster_scaling': cluster_scaling
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
