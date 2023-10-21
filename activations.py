import numpy as np

import torch
from torch import nn


class GELUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.precomputed_constant = np.sqrt(2 / np.pi)

    def forward(self, input):
        return (
            0.5
            * input
            * (
                1
                + torch.tanh(
                    self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))
                )
            )
        )


class Softmax(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.bool()  # Switch to bool - (BS, T, S)
            be_masked = torch.logical_not(mask)
            if be_masked.sum().item():
                mask_value = torch.finfo(x.dtype).min
                x = x.float().masked_fill(be_masked, mask_value).type_as(x)
        return nn.functional.softmax(x, dim=self.dim)


class ACT2FNBase(dict):
    def __getitem__(self, key):
        val = super().__getitem__(key)
        return val()


ACT2FN = ACT2FNBase(
    {"elu": nn.ELU, "relu": nn.ReLU, "gelu": GELUActivation}
)
