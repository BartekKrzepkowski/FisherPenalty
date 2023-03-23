import os

import wandb


class WandbLogger:
    def __init__(self, config):
        self.project = config.logger_config['project_name']
        self.wandb = wandb
        self.wandb.login(key=os.environ['WANDB_API_KEY'])
        if not os.path.isdir(config.logger_config['log_dir']):
            os.makedirs(config.logger_config['log_dir'])
        self.writer = self.wandb.init(
            entity=os.environ['WANDB_ENTITY'],
            project=self.project,
            config=config.logger_config['hyperparameters'],
            name=config.exp_name,
            dir=config.logger_config['log_dir'],
            mode=config.logger_config['mode']
        )

    def close(self):
        self.writer.finish()

    def log_model(self, model, criterion, log_freq, log_graph):
        self.writer.watch(model, criterion, log='all', log_freq=log_freq, log_graph=log_graph)

    def log_histogram(self, tag, tensor, global_step): # problem with numpy=1.24.0
        tensor = tensor.view(-1, 1)
        self.writer.log({tag: wandb.Histogram(tensor)}, step=global_step)

    def log_scalars(self, evaluators, global_step=None):
        self.writer.log(evaluators)


