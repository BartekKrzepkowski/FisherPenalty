import torch

from src.data.datasets import get_mnist, get_cifar10, get_cifar100, get_tinyimagenet
from src.models.losses import ClassificationLoss, FisherPenalyLoss
from src.models.models import MLP, SimpleCNN
from src.models.resnets import ResNet18, ResNet34
from src.visualization.tensorboard_pytorch import TensorboardPyTorch
from src.visualization.wandb_logger import WandbLogger

ACT_NAME_MAP = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'tanh': torch.nn.Tanh,
    'sigmoid': torch.nn.Sigmoid,
    'identity': torch.nn.Identity
}

DATASET_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet
}

LOGGERS_NAME_MAP = {
    'tensorboard': TensorboardPyTorch,
    'wandb': WandbLogger
}

LOSS_NAME_MAP = {
    'ce': torch.nn.CrossEntropyLoss,
    'cls': ClassificationLoss,
    'nll': torch.nn.NLLLoss,
    'mse': torch.nn.MSELoss,
    'fp': FisherPenalyLoss
}

MODEL_NAME_MAP = {
    'mlp': MLP,
    'simplecnn': SimpleCNN,
    'resnet18': ResNet18,
    'resnet34': ResNet34
}

OPTIMIZER_NAME_MAP = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}
