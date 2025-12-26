import copy
import torch
import numpy as np
from .update import ModelUpdate
from .adaptive_attacks import apply_adaptive_attack

class Client:
    def __init__(self, args, dataset, user_idx, dict_users, data_size, p_val, channel_gain):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = set(list(dict_users[user_idx]))
        self.data_size = data_size
        self.p_val = p_val
        self.channel_gain = channel_gain
        self.device = args.device

    def train(self, net_glob, round_num, global_w):
        """
        Perform local training.
        Returns:
            w_local: The local model weights (state_dict)
            loss: The training loss
        """
        # Benign label flip behavior from original code (flag_byz default 1)
        flag = 1
        if self.user_idx <= self.args.num_byz - 1 and self.args.attack == 'label':
             flag = 0

        local = ModelUpdate(
            args=self.args, 
            dataset=self.dataset, 
            idxs=self.idxs, 
            rnds=round_num, 
            data_size=self.data_size, 
            flag_byz=flag
        )
        
        w_local, loss = local.train(
            local_net=copy.deepcopy(net_glob).to(self.device), 
            net=copy.deepcopy(net_glob).to(self.device)
        )
        
        return w_local, loss

class MaliciousClient(Client):
    def __init__(self, args, dataset, user_idx, dict_users, data_size, p_val, channel_gain, attack_state, sketcher=None, history_buffer=None):
        super().__init__(args, dataset, user_idx, dict_users, data_size, p_val, channel_gain)
        self.attack_state = attack_state
        self.sketcher = sketcher
        self.history_buffer = history_buffer

    def train(self, net_glob, round_num, global_w):
        # 1. Perform normal local training first to get 'honest' gradient
        w_local, loss = super().train(net_glob, round_num, global_w)
        
        # 2. Apply Attack
        # args.attack determines which attack to run inside apply_adaptive_attack
        # For 'omni', 'gaussian', etc., they are handled in aggregation usually in the original code,
        # but for consistency we might want to move them here eventually.
        # However, the current refactor plan specifically targets ADAPTIVE_ATTACKS.
        # Non-adaptive attacks (omni, gaussian) are still simple mathematical operations 
        # that might remaining in strategy or be moved here. 
        # For this step, we focus on invoking apply_adaptive_attack for adaptive ones.
        
        if self.args.attack in ['null_space', 'slow_poison', 'predictor_proxy']:
            w_local = apply_adaptive_attack(
                benign_update=w_local,
                w_global=global_w,
                args=self.args,
                user_idx=self.user_idx,
                attack_state=self.attack_state,
                sketcher=self.sketcher,
                history=self.history_buffer,
                device=self.device
            )
        elif self.args.attack == 'omni':
             # Omniscient attack: send negative gradient (opposite direction)
             # But here we are returning WEIGHTS, not gradients.
             # The strategy code did: tmp += (-w_attacked[i][k] + wt[k]) ...
             # meaning it conceptually sent a gradient g = - (w_bad - w_prev).
             # Wait, the strategy calculated: tmp = -w[i][k] + wt[k]
             # If w[i] is the submitted weight, the gradient is (w[i] - wt).
             # We want (w[i] - wt) = - (w_honest - wt)
             # => w[i] - wt = -w_honest + wt
             # => w[i] = 2*wt - w_honest
             
             w_local_attacked = copy.deepcopy(w_local)
             for k in w_local_attacked.keys():
                 w_local_attacked[k] = 2 * global_w[k] - w_local[k]
             w_local = w_local_attacked

        elif self.args.attack == 'gaussian':
            # Gaussian attack: send random noise
            # Strategy: tmp2 = torch.randn_like(tmp) ...
            # We want the update (w[i] - wt) to be noise.
            # w[i] = wt + noise
            w_local_attacked = copy.deepcopy(w_local)
            for k in w_local_attacked.keys():
                noise = torch.randn_like(w_local[k])
                # Scaling logic was in strategy: tmp2 / norm(tmp2) * zeta / ...
                # But here we effectively just want to submit noise.
                # The exact scaling in strategy involved 'zeta' which depends on power control.
                # However, for a generic Gaussian attack at the application layer, we can just strictly follow the strategy logic OR simplifying it.
                # Strategy logic: tmp += torch.true_divide(tmp2, np.linalg.norm(tmp2_np)/pow(2*P[i],0.5)/G[i][j]*zeta)
                # This is physical layer aware.
                # If we want to decouple, the Client should produce a "Model Update" (weight).
                # The physical layer simulation (AirComp) handles P, G, zeta.
                # But 'strategy.py' was mixing these.
                
                # Let's look at strategy.py again. 
                # It seems 'FedAvg_Byzantine' does BOTH attack simulation AND aggregation within the channel loop.
                # If I move attack to Client, Client sends w_bad.
                # Strategy then computes (w_bad - wt) / channel_scale.
                # So if I want the EFFECTIVE received signal to be noise, I should submit a weight w_bad such that (w_bad - wt) is noise.
                # So w_bad = wt + noise.
                # But I should probably normalize the noise to have typical gradient magnitude? 
                # In strategy code: tmp2 = torch.randn_like(tmp) - torch.ones_like(tmp) (Why minus ones? Maybe mean shift?)
                # Then normalized.
                
                # Let's stick to simple Gaussian attack: w_bad = global_w + noise * scale
                # Scale can be norm of benign update.
                
                # Calculate benign update norm
                benign_update = w_local[k] - global_w[k]
                norm = torch.norm(benign_update)
                noise_dir = torch.randn_like(w_local[k])
                noise_norm = torch.norm(noise_dir)
                if noise_norm > 1e-9:
                    scaled_noise = noise_dir * (norm / noise_norm)
                else:
                    scaled_noise = noise_dir
                
                w_local_attacked[k] = global_w[k] + scaled_noise
            w_local = w_local_attacked

        return w_local, loss
