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
    EXP_NAME = f'sgd_cifar10_mlp_different_depth_{hidden_layer_num}'
    EPOCHS = 300
    GRAD_ACCUM_STEPS = 1

    # prepare params
    type_names = {
        'model': 'mlp',
        'criterion': 'fp',
        'dataset': 'cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    h_params_overall = {
        'model': {'layers_dim': DIMS, 'activation_name': 'relu'},
        'criterion': {'model': None, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': True, 'fpw': 1e-2},
        'dataset': {'dataset_path': 'data/', 'whether_aug': False},
        'loaders': {'batch_size': 128, 'pin_memory': True, 'num_workers': 4},
        'optim': {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 0.0},
        'scheduler': None,
        'type_names': type_names
    }
    # set seed to reproduce the results in the future
    manual_seed(random_seed=42, device=device)
    # prepare model
    model = prepare_model(type_names['model'], model_params=h_params_overall['model']).to(device)
    # prepare criterion
    h_params_overall['criterion']['model'] = model
    criterion = prepare_criterion(type_names['criterion'], h_params_overall['criterion'])
    # prepare loaders
    loaders = prepare_loaders(type_names['dataset'], h_params_overall['dataset'], h_params_overall['loaders'])
    # prepare optimizer & scheduler
    T_max = 0  # (len(loaders['train']) // GRAD_ACCUM_STEPS) * EPOCHS
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
        clip_value=0.0,
        base_path='reports',
        exp_name=EXP_NAME,
        logger_config={'logger_name': 'clearml', 'hyperparameters': h_params_overall,
                       'project_name': 'mlp_different_depth',
                       # 'layout': ee_tensorboard_layout(params_names),
                       'access_key': 'N26DK2ZRB0M7O86K5209',
                       'secret_key': 'fvpTeylXmn84mNzGPsnLaK9GMSs08MUpiHRie3sWjJJZBkYwCN'},
        random_seed=42,
        device=device
    )
    trainer.run_exp(config)


if __name__ == "__main__":
    for hidden_layer_num in range(1, 10):
        objective(hidden_layer_num)
