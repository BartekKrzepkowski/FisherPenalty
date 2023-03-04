from math import sqrt

import torch


def init_with_normal(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.normal_(m.weight, mean=0, std=1/sqrt(m.in_features))
        torch.nn.init.zeros_(m.bias)

def init_with_kaiming_normal_fan_in(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)


class WeightInit():
    def __init__(self, modules_to_init):
        self.modules_to_init = modules_to_init

    def forward(self, m):
        if isinstance(m, torch.nn.Linear):
            INIT_MAP_NAMES[self.modules_to_init['linear']['weight']](m)
            INIT_MAP_NAMES[self.modules_to_init['linear']['bias']](m)

INIT_MAP_NAMES = {}
