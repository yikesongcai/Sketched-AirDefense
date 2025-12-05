"""
Adaptive Attacks for FedGTP Defense System

This module implements three advanced adaptive attacks specifically designed
to test the robustness of the Sketching + Trajectory Prediction defense mechanism.

Attack Types:
1. NullSpaceAttack: Exploits the null space of projection matrix S
2. SlowPoisoningAttack: Slow drift attack that stays within predictor tolerance
3. PredictorProxyAttack: Uses local predictor to estimate detection threshold

Author: Generated for FedGTP security evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def compute_null_space_projector(S, device='cuda', regularization=1e-6):
    """
    Compute the null space projection matrix for a given projection matrix S.

    Mathematical Background:
    - For a matrix S ∈ R^{m×d} where m < d, there exists a null space
    - Null space contains vectors v where S·v = 0
    - Projection onto null space: P_null = I - S^T (S S^T)^{-1} S

    Args:
        S: Projection matrix [sketch_dim, input_dim] or [input_dim, sketch_dim]
        device: torch device
        regularization: Small value for numerical stability
    Returns:
        P_null: Null space projection matrix [input_dim, input_dim]
    """
    S = S.float().to(device)

    # Ensure S is in shape [m, d] where m < d (rows are projection directions)
    if S.shape[0] > S.shape[1]:
        S = S.T  # Transpose if needed

    m, d = S.shape

    # For large matrices, use pseudo-inverse for memory efficiency
    # P_null = I - S^T (S S^T + λI)^{-1} S
    try:
        # Compute S S^T + regularization * I for stability
        SST = torch.mm(S, S.T) + regularization * torch.eye(m, device=device)

        # Compute (S S^T)^{-1}
        SST_inv = torch.inverse(SST)

        # Compute S^T (S S^T)^{-1} S
        projection_onto_row_space = torch.mm(torch.mm(S.T, SST_inv), S)

        # Null space projector = I - projection onto row space
        P_null = torch.eye(d, device=device) - projection_onto_row_space

    except RuntimeError:
        # Fallback: Use SVD for more stable computation
        # This is slower but works for ill-conditioned matrices
        U, singular_values, Vh = torch.linalg.svd(S, full_matrices=True)

        # Null space is spanned by right singular vectors corresponding to zero singular values
        rank = torch.sum(singular_values > 1e-10).item()
        null_space_basis = Vh[rank:, :].T  # [d, d-rank]

        # P_null = V_null @ V_null^T
        P_null = torch.mm(null_space_basis, null_space_basis.T)

    return P_null


class NullSpaceAttack:
    """
    Null Space Attack (零空间攻击)

    Exploits the mathematical blind spot of the projection matrix S.
    Injects malicious gradients that lie in the null space of S,
    making them invisible to the sketch-based detection.

    Attack Principle:
    - Any vector v in null(S) satisfies: S·v = 0
    - Attacker submits: g_attack = g_benign + γ·v_poison
    - Server sees: S·g_attack = S·g_benign (unchanged sketch!)
    - But model receives the full poisoned gradient

    Defense Implications:
    - Tests if dynamic S matrices or channel noise can disrupt null space
    - Reveals theoretical limits of sketch-based detection
    """

    def __init__(self, projection_matrix_S, device='cuda', attack_strength=1.0):
        """
        Args:
            projection_matrix_S: The sketch projection matrix [input_dim, sketch_dim]
            device: torch device
            attack_strength: Scaling factor γ for the null space poison (default 1.0)
        """
        self.device = device
        self.attack_strength = attack_strength

        # Precompute null space projector (expensive, do once)
        self.P_null = compute_null_space_projector(projection_matrix_S, device)

        # Store for verification
        self.S = projection_matrix_S.to(device)

    def generate_poison(self, benign_gradient, malicious_target=None):
        """
        Generate poisoned gradient that is invisible in sketch space.

        Args:
            benign_gradient: Honest gradient from local training [d]
            malicious_target: Optional target gradient direction [d]
                             If None, uses negative of benign (omniscient attack)
        Returns:
            poisoned_gradient: g_benign + γ·P_null·g_mal
        """
        benign_gradient = benign_gradient.float().to(self.device)

        # Default malicious target: opposite direction (maximize damage)
        if malicious_target is None:
            malicious_target = -benign_gradient
        else:
            malicious_target = malicious_target.float().to(self.device)

        # Project malicious gradient onto null space
        # This ensures S·(P_null·g_mal) ≈ 0
        null_space_poison = torch.mv(self.P_null, malicious_target)

        # Scale the poison
        poisoned_gradient = benign_gradient + self.attack_strength * null_space_poison

        return poisoned_gradient

    def verify_invisibility(self, benign_gradient, poisoned_gradient):
        """
        Verify that the attack is invisible in sketch space.
        Returns the relative difference in sketches.
        """
        with torch.no_grad():
            sketch_benign = torch.mv(self.S.T, benign_gradient)
            sketch_poisoned = torch.mv(self.S.T, poisoned_gradient)

            diff = torch.norm(sketch_poisoned - sketch_benign)
            base = torch.norm(sketch_benign) + 1e-8

            return (diff / base).item()


class SlowPoisoningAttack:
    """
    Slow Poisoning Attack (慢速投毒/温水煮青蛙)

    Exploits the trajectory predictor's tolerance for concept drift.
    Instead of sudden large deviations, the attack makes tiny steps
    toward the malicious goal, staying within detection threshold.

    Attack Principle:
    - Predictor learns "normal" gradient evolution patterns
    - Small deviations are tolerated (natural variance)
    - Attack accumulates many small steps: Σ(ε_t) → large drift

    Defense Implications:
    - Tests if defense-aware training truly detects drift patterns
    - Validates the λ_adv adversarial loss component
    """

    def __init__(self, target_direction, alpha=0.05, decay_rate=0.99, device='cuda'):
        """
        Args:
            target_direction: The malicious direction to slowly drift toward [d]
            alpha: Interpolation factor (how much to mix malicious per round)
            decay_rate: Optional decay for alpha over time (for stealth)
            device: torch device
        """
        self.device = device
        self.target_direction = target_direction.float().to(device) if target_direction is not None else None
        self.alpha = alpha
        self.initial_alpha = alpha
        self.decay_rate = decay_rate
        self.round_count = 0

        # Track cumulative drift for analysis
        self.cumulative_drift = 0.0

    def set_target(self, target_direction):
        """Set or update the malicious target direction."""
        self.target_direction = target_direction.float().to(self.device)

    def generate_poison(self, benign_gradient, target_direction=None):
        """
        Generate slowly drifting poisoned gradient.

        Args:
            benign_gradient: Honest gradient from local training [d]
            target_direction: Optional override for target direction
        Returns:
            poisoned_gradient: (1-α)·g_benign + α·g_target
        """
        benign_gradient = benign_gradient.float().to(self.device)

        # Use provided target or stored target
        target = target_direction if target_direction is not None else self.target_direction
        if target is None:
            # Default: opposite direction (omniscient attack)
            target = -benign_gradient
        else:
            target = target.float().to(self.device)

        # Normalize target to match benign gradient magnitude
        # This makes the interpolation more stable
        target_normalized = target * (torch.norm(benign_gradient) / (torch.norm(target) + 1e-8))

        # Interpolate: slow drift toward target
        current_alpha = self.alpha * (self.decay_rate ** self.round_count)
        poisoned_gradient = (1 - current_alpha) * benign_gradient + current_alpha * target_normalized

        # Track cumulative effect
        drift = torch.norm(poisoned_gradient - benign_gradient).item()
        self.cumulative_drift += drift
        self.round_count += 1

        return poisoned_gradient

    def reset(self):
        """Reset attack state for new experiment."""
        self.alpha = self.initial_alpha
        self.round_count = 0
        self.cumulative_drift = 0.0

    def get_stats(self):
        """Return attack statistics."""
        return {
            'rounds': self.round_count,
            'cumulative_drift': self.cumulative_drift,
            'current_alpha': self.alpha * (self.decay_rate ** self.round_count)
        }


class PredictorProxyAttack:
    """
    Predictor Proxy Attack (预测器代理攻击)

    The most sophisticated adaptive attack. Attacker maintains a local
    copy of the trajectory predictor and uses it to estimate the server's
    detection threshold, then crafts attacks that stay just below it.

    Attack Principle:
    - Attacker trains local LSTM on observed global model history
    - Before submitting, attacker checks: "Will server detect this?"
    - Uses binary search to find maximum attack that passes detection

    Defense Implications:
    - Tests robustness under worst-case (white-box) assumptions
    - If defense survives this, attacker can only achieve minimal damage
    """

    def __init__(self, sketch_dim, hidden_dim=64, window_size=5,
                 threshold_margin=0.1, device='cuda'):
        """
        Args:
            sketch_dim: Dimension of sketch vectors
            hidden_dim: Hidden dimension of local predictor LSTM
            window_size: History window for prediction
            threshold_margin: Safety margin below detection threshold (δ)
            device: torch device
        """
        self.device = device
        self.sketch_dim = sketch_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.threshold_margin = threshold_margin

        # Local predictor (mirrors server's architecture)
        self.local_predictor = nn.LSTM(
            input_size=sketch_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        ).to(device)
        self.predictor_head = nn.Linear(hidden_dim, sketch_dim).to(device)

        # Optimizer for local predictor training
        self.optimizer = torch.optim.Adam(
            list(self.local_predictor.parameters()) +
            list(self.predictor_head.parameters()),
            lr=0.001
        )

        # History buffer for training local predictor
        self.sketch_history = []

        # Estimated detection threshold (learned from observations)
        self.estimated_threshold = 1.0  # Will be updated based on observations

    def observe_sketch(self, sketch):
        """
        Record observed global sketch for training local predictor.

        Args:
            sketch: Observed global sketch [sketch_dim]
        """
        sketch = sketch.detach().clone().to(self.device)
        self.sketch_history.append(sketch)

        # Keep only recent history
        if len(self.sketch_history) > self.window_size * 10:
            self.sketch_history = self.sketch_history[-self.window_size * 10:]

    def train_local_predictor(self, num_steps=10):
        """
        Train local predictor on observed history.

        This allows the attacker to "predict the server's prediction".
        """
        if len(self.sketch_history) < self.window_size + 1:
            return 0.0  # Not enough data

        self.local_predictor.train()
        total_loss = 0.0

        for _ in range(num_steps):
            # Create training batch from history
            idx = np.random.randint(0, len(self.sketch_history) - self.window_size)
            history = torch.stack(self.sketch_history[idx:idx + self.window_size]).unsqueeze(0)
            target = self.sketch_history[idx + self.window_size].unsqueeze(0)

            # Forward pass
            out, _ = self.local_predictor(history)
            pred = self.predictor_head(out[:, -1, :])

            # MSE loss
            loss = F.mse_loss(pred, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_steps

    def predict_sketch(self, history):
        """
        Use local predictor to predict next sketch.

        Args:
            history: Recent sketch history [window_size, sketch_dim]
        Returns:
            predicted_sketch: [sketch_dim]
        """
        self.local_predictor.eval()
        with torch.no_grad():
            history = history.unsqueeze(0).to(self.device)  # Add batch dim
            out, _ = self.local_predictor(history)
            pred = self.predictor_head(out[:, -1, :])
        return pred.squeeze(0)

    def estimate_anomaly_score(self, candidate_sketch, history):
        """
        Estimate what anomaly score the server would compute.

        Args:
            candidate_sketch: The sketch that would result from our attack
            history: Recent sketch history [window_size, sketch_dim]
        Returns:
            estimated_score: Estimated anomaly score
        """
        predicted = self.predict_sketch(history)
        score = torch.norm(candidate_sketch - predicted) ** 2
        return score.item()

    def generate_poison(self, benign_gradient, sketcher, history,
                        malicious_target=None, max_iterations=10):
        """
        Generate maximum-damage poison that stays below detection threshold.

        Uses binary search to find the largest attack that passes detection.

        Args:
            benign_gradient: Honest gradient [d]
            sketcher: GradientSketcher instance (to compute sketches)
            history: Recent sketch history [window_size, sketch_dim]
            malicious_target: Target gradient direction (optional)
            max_iterations: Max binary search iterations
        Returns:
            poisoned_gradient: Attack gradient that evades detection
        """
        benign_gradient = benign_gradient.float().to(self.device)

        if malicious_target is None:
            malicious_target = -benign_gradient
        else:
            malicious_target = malicious_target.float().to(self.device)

        # Compute attack direction
        attack_direction = malicious_target - benign_gradient
        attack_direction = attack_direction / (torch.norm(attack_direction) + 1e-8)

        # Binary search for maximum attack strength
        low, high = 0.0, torch.norm(benign_gradient).item() * 2
        best_poison = benign_gradient.clone()

        # Get benign sketch for threshold estimation
        benign_sketch = sketcher.sketch(benign_gradient)
        benign_score = self.estimate_anomaly_score(benign_sketch, history)

        # Use benign score to estimate acceptable threshold
        acceptable_threshold = benign_score * (1 + self.threshold_margin)

        for _ in range(max_iterations):
            mid = (low + high) / 2

            # Generate candidate poison
            candidate = benign_gradient + mid * attack_direction
            candidate_sketch = sketcher.sketch(candidate)

            # Estimate if server would detect this
            score = self.estimate_anomaly_score(candidate_sketch, history)

            if score < acceptable_threshold:
                # Attack passes detection, try stronger
                best_poison = candidate.clone()
                low = mid
            else:
                # Attack would be detected, reduce strength
                high = mid

        return best_poison

    def update_threshold_estimate(self, observed_scores, detected_flags):
        """
        Update estimated threshold based on observed detection results.

        Args:
            observed_scores: List of anomaly scores
            detected_flags: List of boolean (True = was detected/flagged)
        """
        if len(observed_scores) == 0:
            return

        # Find the boundary between detected and undetected
        detected_scores = [s for s, d in zip(observed_scores, detected_flags) if d]
        undetected_scores = [s for s, d in zip(observed_scores, detected_flags) if not d]

        if detected_scores and undetected_scores:
            # Threshold is somewhere between max undetected and min detected
            self.estimated_threshold = (max(undetected_scores) + min(detected_scores)) / 2


def apply_adaptive_attack(attack_type, benign_update, w_global, args,
                          sketcher=None, history=None, attack_state=None, device='cuda'):
    """
    Unified interface to apply adaptive attacks.

    This function is called from defense.py to apply the selected attack type
    to Byzantine users' gradients.

    Args:
        attack_type: One of ['null_space', 'slow_poison', 'predictor_proxy']
        benign_update: The honest local update (state_dict format)
        w_global: Current global model weights
        args: Command line arguments
        sketcher: GradientSketcher instance (needed for some attacks)
        history: Sketch history buffer (needed for predictor_proxy)
        attack_state: Persistent attack state (for stateful attacks)
        device: torch device
    Returns:
        poisoned_update: Modified update dictionary
        attack_state: Updated attack state (for next round)
    """
    from .defense import flatten_model_updates, unflatten_model_updates

    # Flatten the update for processing
    update_flat = []
    for key in sorted(benign_update.keys()):
        diff = benign_update[key].float() - w_global[key].float()
        update_flat.append(diff.view(-1))
    benign_gradient = torch.cat(update_flat).to(device)

    # Initialize attack state if needed
    if attack_state is None:
        attack_state = {}

    # Apply selected attack
    if attack_type == 'null_space':
        # Initialize NullSpaceAttack if not exists
        if 'null_space_attacker' not in attack_state:
            if sketcher is None:
                raise ValueError("NullSpaceAttack requires sketcher")
            attack_state['null_space_attacker'] = NullSpaceAttack(
                sketcher.S, device=device,
                attack_strength=getattr(args, 'attack_strength', 1.0)
            )

        attacker = attack_state['null_space_attacker']
        poisoned_gradient = attacker.generate_poison(benign_gradient)

    elif attack_type == 'slow_poison':
        # Initialize SlowPoisoningAttack if not exists
        if 'slow_poison_attacker' not in attack_state:
            attack_state['slow_poison_attacker'] = SlowPoisoningAttack(
                target_direction=None,  # Will use -benign as default
                alpha=getattr(args, 'poison_alpha', 0.05),
                decay_rate=getattr(args, 'poison_decay', 0.99),
                device=device
            )

        attacker = attack_state['slow_poison_attacker']
        poisoned_gradient = attacker.generate_poison(benign_gradient)

    elif attack_type == 'predictor_proxy':
        # Initialize PredictorProxyAttack if not exists
        if 'proxy_attacker' not in attack_state:
            if sketcher is None:
                raise ValueError("PredictorProxyAttack requires sketcher")
            attack_state['proxy_attacker'] = PredictorProxyAttack(
                sketch_dim=sketcher.sketch_dim,
                hidden_dim=getattr(args, 'proxy_hidden', 64),
                window_size=args.window_size,
                threshold_margin=getattr(args, 'threshold_margin', 0.1),
                device=device
            )

        attacker = attack_state['proxy_attacker']

        # Train local predictor on observed history
        if history is not None and len(history) > 0:
            for sketch in history:
                attacker.observe_sketch(sketch)
            attacker.train_local_predictor(num_steps=5)

            # Generate attack
            history_tensor = torch.stack(list(history)[-args.window_size:])
            poisoned_gradient = attacker.generate_poison(
                benign_gradient, sketcher, history_tensor
            )
        else:
            # Not enough history, fall back to simple attack
            poisoned_gradient = -benign_gradient
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # Unflatten back to state_dict format
    poisoned_update = {}
    idx = 0
    for key in sorted(w_global.keys()):
        numel = w_global[key].numel()
        shape = w_global[key].shape
        poisoned_update[key] = (
            w_global[key].float() +
            poisoned_gradient[idx:idx+numel].view(shape)
        ).to(device)
        idx += numel

    return poisoned_update, attack_state
