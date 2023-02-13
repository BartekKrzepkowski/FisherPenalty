import os

import wandb


class WandbLogger:
    def __init__(self, config):
        # self.api_token = "07a2cd842a6d792d578f8e6c0978efeb8dcf7638"
        # self.project = f"DistilHerBERT"
        self.api_token = config.logger_config['api_token']
        self.project = config.logger_config['project']
        self.wandb = wandb
        self.wandb.login(key=self.api_token)
        if not os.path.isdir(config.logger_config['log_dir']):
            os.makedirs(config.logger_config['log_dir'])
        self.writer = self.wandb.init(
            # Set the project where this run will be logged
            project=self.project,
            # Track hyperparameters and run metadata
            config=config.logger_config['hyperparameters'],
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=config.exp_name,
            dir=config.logger_config['log_dir'],
            mode='online'
        )

    def close(self):
        self.writer.finish()

    def log_model(self, model, criterion, log_freq, log_graph):
        self.writer.watch(model, criterion, log='all', log_freq=log_freq, log_graph=log_graph)

    def log_histogram(self, tag, tensor, global_step): # problem with numpy=1.24.0
        tensor = tensor.view(-1, 1)
        self.writer.log({tag: wandb.Histogram(tensor)}, step=global_step)

    def log_scalars(self, evaluators, global_step):
        self.writer.log(evaluators, step=global_step)


