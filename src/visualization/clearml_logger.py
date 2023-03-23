import os

from clearml import Task


class ClearMLLogger:
    def __init__(self, config):
        self.project_name = config.logger_config['project_name']
        self.task_name = config.exp_name
        api = {
            'web_host': 'https://app.clear.ml',
            'api_host': 'https://api.clear.ml',
            'files_host': 'https://files.clear.ml',
            "key": os.environ['CLEARML_ACCESS_KEY'],
            "secret": os.environ['CLEARML_SECRET_KEY']
        }
        if not os.path.isdir(config.logger_config['log_dir']):
            os.makedirs(config.logger_config['log_dir'])
        params_clearml = {
            'project_name': self.project_name,
            'task_name': self.task_name,
            'output_uri': config.logger_config['log_dir']
        }
        Task.set_credentials(**api)
        self.task = Task.init(**params_clearml)
        self.task.connect(config.logger_config['hyperparameters'])
        self.writer = self.task.get_logger()

    def close(self):
        self.task.close()

    def log_model(self, model, criterion, log_freq, log_graph):
        self.writer.watch(model, criterion, log='all', log_freq=log_freq, log_graph=log_graph)

    def log_histogram(self, tag, tensor, global_step):
        tensor = tensor.view(-1, 1)
        title, series = tag.split('/')
        self.writer.report_histogram(title=title, series=series, values=tensor, iteration=global_step)

    def log_scalars(self, scalar_dict, global_step):
        for tag in scalar_dict:
            split = tag.split('/')
            title, series = '/'.join(split[:1]), '/'.join(split[1:])
            self.writer.report_scalar(title=title, series=series, value=scalar_dict[tag], iteration=global_step)
