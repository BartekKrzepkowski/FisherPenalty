from collections import defaultdict

import torch

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
from src.utils.utils_regularizers import get_desired_parameter_names


class FisherPenaly(torch.nn.Module):
    def __init__(self, model, criterion, num_classes):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.labels = torch.arange(num_classes).to(next(model.parameters()).device)
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)

    def forward(self, y_pred):
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze()
        loss = self.criterion(y_pred, y_sampled)
        params_names, params = zip(*[(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=True)
        traces = defaultdict(float)
        overall_trace = 0.0
        # najlepiej rozdzielić po module
        for param_name, gr in zip(params_names, grads):
            if gr is not None:
                trace_p = (gr**2).sum()
                traces[param_name] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p
        return overall_trace, traces

    def prepare_parameters(self):
        model_module_types = {type(m) for m in self.model.modules() if next(m.children(), None) is None
                              and next(m.parameters(), None) is not None}
        grouped_parameter_names = defaultdict(list)
        for module_type in model_module_types:
            for name, module in self.model.named_modules():
                if isinstance(module, tuple(module_type)):
                    module_specific_parameter_names = [f'{name}.{n}' for n in module._parameters.keys()]
                    grouped_parameter_names[module_type.__name__] += module_specific_parameter_names
