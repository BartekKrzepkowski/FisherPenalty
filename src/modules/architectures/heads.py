from typing import Tuple, Union

import torch

class SDNPool(torch.nn.Module):
    def __init__(self, target_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self._alpha = torch.nn.Parameter(torch.rand(1))
        self._max_pool = torch.nn.AdaptiveMaxPool2d(target_size)
        self._avg_pool = torch.nn.AdaptiveAvgPool2d(target_size)

    def forward(self, x):
        avg_p = self._alpha * self._max_pool(x)
        max_p = (1 - self._alpha) * self._avg_pool(x)
        mixed = avg_p + max_p
        return mixed


class StandardHead(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pool_size: int = 4):
        super().__init__()
        self._num_classes = num_classes
        self._pooling = SDNPool(pool_size)
        self._fc = torch.nn.Linear(in_channels * pool_size ** 2, num_classes)

    def forward(self, x: torch.Tensor):
        # x = torch.nn.functional.relu(x)
        x = self._pooling(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x