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


def infer_height_and_width_before_flatten(height, width, whether_pooling, kernels, strides, paddings):
    if len(paddings) != 0:
        height = (height - kernels[0] + 2 * paddings[0]) // strides[0] + 1
        width = (width - kernels[0] + 2 * paddings[0]) // strides[0] + 1
        if whether_pooling[0]:
            height /= 2
            width /= 2
        (height, width) = infer_height_and_width_before_flatten(height, width, whether_pooling[1:],
                                                                kernels[1:], strides[1:], paddings[1:])
    return (height, width)


def infer_flatten_dim(conv_params, out_channels):
    height, width = infer_height_and_width_before_flatten(conv_params['img_height'], conv_params['img_widht'],
                                                          conv_params['whether_pooling'], conv_params['kernels'],
                                                          conv_params['strides'], conv_params['paddings'])
    return int(height * width * out_channels)
