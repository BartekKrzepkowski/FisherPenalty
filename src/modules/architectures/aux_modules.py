from math import sqrt

import torch


class PreAct(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = 2 / sqrt(self.dim) * x
        return x
