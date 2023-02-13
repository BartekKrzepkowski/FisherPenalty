from torch.utils.tensorboard import SummaryWriter


class TensorboardPyTorch:
    def __init__(self, config):
        self.writer = SummaryWriter(log_dir=config.logger_config['log_dir'], flush_secs=60)
        if 'layout' in config.logger_config:
            self.writer.add_custom_scalars(config.logger_config['layout'])

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def log_graph(self, model, inp):
        self.writer.add_graph(model, inp)
        self.flush()

    def log_histogram(self, tag, tensor, global_step): # problem with numpy=1.24.0
        self.writer.add_histogram(tag, tensor, global_step=global_step)
        self.flush()

    def log_scalars(self, scalar_dict, global_step):
        for tag in scalar_dict:
            self.writer.add_scalar(tag, scalar_dict[tag], global_step=global_step)
        self.flush()

