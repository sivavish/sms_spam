import torch


def compute_pos_weight(labels):
    labels = torch.tensor(labels)

    total_samples = len(labels)

    num_positive = (labels == 1).sum().item()  
    num_negative = (labels == 0).sum().item()  

    if num_positive == 0:
        raise ValueError("No positive samples found.")

    pos_weight = num_negative / num_positive

    return torch.tensor(pos_weight, dtype=torch.float32)
