import torch
import copy

def get_model_flattened_dim(net):
    """
    Calculate the total dimension of the model as flattened by state_dict.
    Includes BatchNorm buffers (running_mean, etc) to ensure consistency with flatten_model_updates.
    """
    total_dim = 0
    state_dict = net.state_dict()
    for key in state_dict.keys():
        total_dim += state_dict[key].numel()
    return total_dim

def flatten_model_updates(w_update, device):
    """
    Flatten model update dictionary to a single vector.
    IMPORTANT: Uses sorted keys to ensure deterministic order across environments.
    """
    flattened = []
    # FIX: Use sorted keys to ensure deterministic order
    for key in sorted(w_update.keys()):
        flattened.append(w_update[key].view(-1).float())
    return torch.cat(flattened).to(device)


def unflatten_model_updates(flattened, reference_dict, device):
    """
    Unflatten a vector back to model update dictionary format.
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
