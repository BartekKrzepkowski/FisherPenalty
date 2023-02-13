import torch

from src.utils.prepare import prepare_model, prepare_loaders, prepare_criterion, prepare_optim_and_scheduler
from src.trainer.trainer_classification import TrainerClassification
from src.trainer.trainer_context import TrainerContext


def objective():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model
    NUM_FEATURES = 32 * 32 * 3
    NUM_CLASSES = 10
    DIMS = [NUM_FEATURES, 512, 512, NUM_CLASSES]
    model_params = {'layers_dim':  DIMS, 'activation_name': 'relu'}
    model = prepare_model('simplecnn', model_params=model_params).to(device)

    # prepare criterion
    criterion_params = {'model': model, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                        'whether_record_trace': True, 'fpw': 1e-2}
    criterion = prepare_criterion('fp', criterion_params)

    # prepare loaders
    dataset_params = {'dataset_path': 'data/', 'whether_aug': False}
    loader_params = {'batch_size': 128, 'pin_memory': True, 'num_workers': 4}
    loaders = prepare_loaders('cifar10', dataset_params, loader_params)

    # prepare optimizer & scheduler
    GRAD_ACCUM_STEPS = 1
    EPOCHS = 150
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * EPOCHS
    optim_params = {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 0.0}
    optim, lr_scheduler = prepare_optim_and_scheduler(model, 'sgd', optim_params, scheduler_name=None)

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
    EXP_NAME = 'simple_cnn_cifar10_sgd_fp'
    config = TrainerContext(
        epoch_start_at=0,
        epoch_end_at=EPOCHS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        save_multi=T_max // 10,
        log_multi=100,
        clip_value=0.0,
        base_path='reports',
        exp_name=EXP_NAME,
        logger_config={'logger_name': 'tensorboard'},
        random_seed=42,
        device=device
    )
    trainer.run_exp(config)


if __name__ == "__main__":
    objective()
