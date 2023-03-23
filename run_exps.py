import torch

from src.utils.prepare import prepare_model, prepare_loaders, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification import TrainerClassification
from src.trainer.trainer_context import TrainerContext


def objective(hidden_layer_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    NUM_FEATURES = 32 * 32 * 3
    NUM_CLASSES = 10
    DIMS = [NUM_FEATURES] + [512] * hidden_layer_num + [NUM_CLASSES]
    # trainer & scheduler
    FP = 1e-2
    EXP_NAME = f'sgd_cifar10_mlp_bn1d_different_depth_{hidden_layer_num}_fp_{FP}'
    EPOCHS = 300
    GRAD_ACCUM_STEPS = 1
    CLIP_VALUE = 0.0
    RANDOM_SEED = 42

    # prepare params
    type_names = {
        'model': 'mlp_with_norm',
        'criterion': 'fp',
        'dataset': 'cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    h_params_overall = {
        'model': {'layers_dim': DIMS, 'activation_name': 'relu', 'norm_name': 'bn1d'},
        'criterion': {'model': None, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': True, 'fpw': FP},
        'dataset': {'dataset_path': 'data/', 'whether_aug': False},
        'loaders': {'batch_size': 100, 'pin_memory': True, 'num_workers': 4},
        'optim': {'lr': 5e-3, 'momentum': 0.9, 'weight_decay': 1e-2},
        'scheduler': {'eta_min': 1e-6},
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
        log_multi=100,
        clip_value=CLIP_VALUE,
        base_path='reports',
        exp_name=EXP_NAME,
        logger_config={'logger_name': 'tensorboard', 'project_name': 'mlp_different_depth',
                       'hyperparameters': h_params_overall, 'whether_use_wandb': True,
                       'layout': ee_tensorboard_layout(params_names), 'mode': 'online'
                       },
        random_seed=RANDOM_SEED,
        device=device
    )
    trainer.run_exp(config)


if __name__ == "__main__":
    for hidden_layer_num in range(2, 8):
        objective(hidden_layer_num)
