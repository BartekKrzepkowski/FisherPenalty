from typing import List

import torch

from src.utils import common


class MLP(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2), common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])]
        )
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, padding='same'), common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.Conv2d(32, 32, 3, padding='same'), common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.MaxPool2d(2, 2))
        self.block2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, padding='same'), common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.Conv2d(64, 64, 3, padding='same'), common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.MaxPool2d(2, 2))
        self.block3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, padding='same'), common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.Conv2d(64, 64, 3, padding='same'), common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.MaxPool2d(2, 2))
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(1024, 128), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Linear(128, 10))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
