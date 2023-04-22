import warnings
from collections import Counter

import torch
from torch.optim.lr_scheduler import LRScheduler


def get_every_but_forbidden_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_every_but_forbidden_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def configure_optimizer(optim_wrapper, model, optim_kwargs):
    weight_decay = optim_kwargs['weight_decay']
    del optim_kwargs['weight_decay']

    decay_parameters = get_every_but_forbidden_parameter_names(model, FORBIDDEN_LAYER_TYPES)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for pn, p in model.named_parameters() if pn in decay_parameters and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for pn, p in model.named_parameters() if pn not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim_wrapper(optimizer_grouped_parameters, **optim_kwargs)
    return optimizer


def clip_grad_norm(clip_grad_wrapper, model, clip_value):
    clip_grad_wrapper(filter(lambda p: p.requires_grad, model.parameters()), clip_value)


FORBIDDEN_LAYER_TYPES = [torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]


class MultiStepwithDoubleLinearWarmup(LRScheduler):
    def __init__(self, optimizer, milestones=[], gamma=1e-1,eta_max=None, eta_medium=0.0, eta_min=0.0, warmup_iters2=0, inter_warmups_iters=0, warmup_iters1=0, last_epoch=-1,
                 verbose=False):
        assert eta_max >= eta_medium >= eta_min >= 0.0, 'sa'
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.eta_max = eta_max
        self.eta_medium = eta_medium
        self.eta_min = eta_min
        self.warmup_iters2 = warmup_iters2
        self.inter_warmups_iters = inter_warmups_iters
        self.warmup_iters1 = warmup_iters1
        if eta_min > 0.0:
            for groups in optimizer.param_groups:
                groups['lr'] = eta_min
        elif eta_medium > 0.0:
            for groups in optimizer.param_groups:
                groups['lr'] = eta_medium
        elif eta_max == 0.0:
            raise ValueError('eta_max must be greater than 0.0')
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.warmup_iters1 < self.last_epoch <= self.warmup_iters1 + self.inter_warmups_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch <= self.warmup_iters1:
            return [self.eta_min + (self.eta_medium - self.eta_min) * self.last_epoch / self.warmup_iters1
                    for _ in self.optimizer.param_groups]
        
        if self.last_epoch <= self.warmup_iters1 + self.inter_warmups_iters + self.warmup_iters2:
            return [self.eta_medium + (self.eta_max - self.eta_medium) * (self.last_epoch-(self.warmup_iters1 + self.inter_warmups_iters)) / self.warmup_iters2
                    for _ in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
