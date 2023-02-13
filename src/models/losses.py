import torch

from src.models.metrics import acc_metric
from src.models.regularizers import FisherPenaly
from src.utils import common


class ClassificationLoss(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        acc = acc_metric(y_pred, y_true)
        evaluators = {
            'loss': loss.item(),
            'acc': acc
        }
        return loss, evaluators


class FisherPenalyLoss(torch.nn.Module):
    def __init__(self, model, general_criterion_name, num_classes, whether_record_trace=False, fpw=0.0):
        super().__init__()
        self.criterion = ClassificationLoss(common.LOSS_NAME_MAP[general_criterion_name]())
        self.regularizer = FisherPenaly(model, common.LOSS_NAME_MAP[general_criterion_name](), num_classes)
        self.whether_record_trace = whether_record_trace
        self.fpw = fpw

    def forward(self, y_pred, y_true):
        loss, evaluators = self.criterion(y_pred, y_true)
        if self.whether_record_trace > 0 and self.regularizer.model.training:
            trace = self.regularizer(y_pred)
            evaluators['trace'] = trace.item()
            if self.fpw:
                loss += self.fpw * trace
        return loss, evaluators
