import os

import wandb
from torch.utils.tensorboard import SummaryWriter


class TensorboardPyTorch:
    def __init__(self, config):
        self.whether_use_wandb = config.logger_config['whether_use_wandb']
        if self.whether_use_wandb:
            wandb.login(key=os.environ['WANDB_API_KEY'])
            wandb.init(
                entity=os.environ['WANDB_ENTITY'],
                project=config.logger_config['project_name'],
                name=config.exp_name,
                config=config.logger_config['hyperparameters'],
                dir=config.logger_config['log_dir'],
                mode=config.logger_config['mode']
            )
            wandb.tensorboard.patch(root_logdir=config.logger_config['log_dir'], pytorch=True, save=False)
            
        self.writer = SummaryWriter(log_dir=config.logger_config['log_dir'], flush_secs=60)
        if 'layout' in config.logger_config:
            self.writer.add_custom_scalars(config.logger_config['layout'])


    def close(self):
        if self.whether_use_wandb:
            wandb.finish()
        self.writer.close()
        

    def flush(self):
        self.writer.flush()

    def log_graph(self, model, inp, criterion):
        if self.whether_use_wandb:
            wandb.watch(model, log_freq=1000, idx=0, log_graph=True, log='all', criterion=criterion)
        self.writer.add_graph(model, inp)
        self.flush()

    def log_histogram(self, tag, tensor, global_step): # problem with numpy=1.24.0
        self.writer.add_histogram(tag, tensor, global_step=global_step)
        self.flush()

    def log_scalars(self, scalar_dict, global_step):
        for tag in scalar_dict:
            self.writer.add_scalar(tag, scalar_dict[tag], global_step=global_step)
        self.flush()

