import torch
import copy

def get_model_flattened_dim(net):
    """
    Calculate the total number of parameters in the model.
    """
    total_dim = 0
    for param in net.parameters():
        total_dim += param.numel()
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
