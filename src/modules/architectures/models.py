from typing import List, Dict, Any

import torch

from src.modules.architectures import aux_modules
from src.utils.utils_model import infer_flatten_dim
from src.utils import common


class MLP(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2), common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class MLP_scaled(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(aux_modules.PreAct(hidden_dim1), torch.nn.Linear(hidden_dim1, hidden_dim2), common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Sequential(aux_modules.PreAct(layers_dim[-2]), torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

class MLPwithNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, norm_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                common.NORM_LAYER_NAME_MAP[norm_name](hidden_dim2),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x


class SimpleCNNwithNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, norm_name: str):
        super().__init__()
        # self.blocks = torch.nn.ModuleList([
        #     torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding='same'),
        #                         common.NORM_LAYER_NAME_MAP[norm_name]([layer_dim2, 16, 16]),
        #                         common.ACT_NAME_MAP[activation_name](),
        #                         torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding='same'),
        #                         common.NORM_LAYER_NAME_MAP[norm_name](layer_dim2, 32, 32),
        #                         common.ACT_NAME_MAP[activation_name](),
        #                         torch.nn.MaxPool2d(2, 2))
        #     for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        # ])
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, padding='same'),
                                          common.NORM_LAYER_NAME_MAP[norm_name]([32, 32, 32]),
                                          common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.Conv2d(32, 32, 3, padding='same'),
                                          common.NORM_LAYER_NAME_MAP[norm_name]([32, 32, 32]),
                                          common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.MaxPool2d(2, 2))
        self.block2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, padding='same'),
                                          common.NORM_LAYER_NAME_MAP[norm_name]([64, 16, 16]),
                                          common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.Conv2d(64, 64, 3, padding='same'),
                                          common.NORM_LAYER_NAME_MAP[norm_name]([64, 16, 16]),
                                          common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.MaxPool2d(2, 2))
        self.block3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, padding='same'),
                                          common.NORM_LAYER_NAME_MAP[norm_name]([64, 8, 8]),
                                          common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.Conv2d(64, 64, 3, padding='same'),
                                          common.NORM_LAYER_NAME_MAP[norm_name]([64, 8, 8]),
                                          common.ACT_NAME_MAP[activation_name](),
                                          torch.nn.MaxPool2d(2, 2))
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(1024, 128), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Linear(128, 10))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # for block in self.blocks:
        #     x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x

