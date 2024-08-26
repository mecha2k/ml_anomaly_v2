import torch


def mse_loss(output, target):
    return torch.nn.functional.mse_loss(output, target)
