#!/usr/bin/env python3
import numpy as np
import torch

from src.utils.prepare import prepare_model, prepare_loaders, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification import TrainerClassification
from src.trainer.trainer_context import TrainerContext


def objective(lr, wd):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # model
    NUM_FEATURES = 3
    NUM_CLASSES = 10
    DIMS = [NUM_FEATURES, 32, 64, 128, NUM_CLASSES]
    CONV_PARAMS = {'img_height': 32, 'img_widht': 32, 'kernels': [3,3,3,3], 'strides': [1,1,1,1], 'paddings': [1,1,1,1], 'whether_pooling': [False,True,False,True]}
    # trainer & scheduler
    RANDOM_SEED = 42
    EPOCHS = 300
    GRAD_ACCUM_STEPS = 1
    CLIP_VALUE = 0.0
    FP = 0.0
    EXP_NAME = f'sgd_cifar10_cnn_depth_{2}_fp_{FP}_lr_{lr}_wd_{wd}'
    PROJECT_NAME = 'Critical_Periods_lr'
    ENTITY = 'ideas_cv'

    # prepare params
    type_names = {
        'model': 'simple_cnn',
        'criterion': 'fp',
        'dataset': 'cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    h_params_overall = {
        'model': {'layers_dim': DIMS, 'activation_name': 'relu', 'conv_params': CONV_PARAMS},
        'criterion': {'model': None, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': True, 'fpw': FP},
        'dataset': {'dataset_path': 'data/', 'whether_aug': False},
        'loaders': {'batch_size': 256, 'pin_memory': True, 'num_workers': 8},
        'optim': {'lr': lr, 'momentum': 0.9, 'weight_decay': wd},
        'scheduler': {'eta_min': 1e-6, 'T_max': None},
        'type_names': type_names
    }
    # set seed to reproduce the results in the future
    manual_seed(random_seed=RANDOM_SEED, device=device)
    # prepare model
    model = prepare_model(type_names['model'], model_params=h_params_overall['model']).to(device)
    # prepare criterion
    h_params_overall['criterion']['model'] = model
    criterion = prepare_criterion(type_names['criterion'], h_params_overall['criterion'])
    # prepare loaders
    loaders = prepare_loaders(type_names['dataset'], h_params_overall['dataset'], h_params_overall['loaders'])
    # prepare optimizer & scheduler
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * EPOCHS
    h_params_overall['scheduler']['T_max'] = T_max
    optim, lr_scheduler = prepare_optim_and_scheduler(model, type_names['optim'], h_params_overall['optim'],
                                                      type_names['scheduler'], h_params_overall['scheduler'])

    # prepare trainer
    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
    }
    trainer = TrainerClassification(**params_trainer)

    # prepare run
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    config = TrainerContext(
        epoch_start_at=0,
        epoch_end_at=EPOCHS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        save_multi=T_max // 10,
        log_multi=(T_max // EPOCHS) // 10,
        clip_value=CLIP_VALUE,
        base_path='reports',
        exp_name=EXP_NAME,
        logger_config={'logger_name': 'tensorboard', 'project_name': PROJECT_NAME, 'entity': 'ideas_cv',
                       'hyperparameters': h_params_overall, 'whether_use_wandb': True,
                       'layout': ee_tensorboard_layout(params_names), 'mode': 'online'
                       },
        whether_disable_tqdm=True,
        random_seed=RANDOM_SEED,
        device=device
    )
    trainer.run_exp(config)



if __name__ == "__main__":
    for lr in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]:
        objective(lr, 0.0)
