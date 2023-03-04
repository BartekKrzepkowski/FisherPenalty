from torch.utils.data import DataLoader

from src.utils.common import DATASET_NAME_MAP, LOSS_NAME_MAP, MODEL_NAME_MAP, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP
from src.utils.utils_model import init_with_kaiming_normal_fan_in
from src.utils.utils_optim import configure_optimizer


def prepare_model(model_name, model_params):
    model = MODEL_NAME_MAP[model_name](**model_params)
    model.apply(init_with_kaiming_normal_fan_in)
    return model


def prepare_loaders(dataset_name, dataset_params, loader_params):
    train_dataset, _, test_dataset = DATASET_NAME_MAP[dataset_name](**dataset_params)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)
    loaders = {
        'train': train_loader,
        'test': test_loader
    }
    return loaders


def prepare_criterion(loss_name, criterion_params={}):
    criterion = LOSS_NAME_MAP[loss_name](**criterion_params)
    return criterion


def prepare_optim_and_scheduler(model, optim_name, optim_params, scheduler_name=None, scheduler_params=None):
    optim_wrapper = OPTIMIZER_NAME_MAP[optim_name]
    optim = configure_optimizer(optim_wrapper, model, optim_params)
    lr_scheduler = SCHEDULER_NAME_MAP[scheduler_name](optim, **scheduler_params) if scheduler_name is not None else None
    return optim, lr_scheduler
