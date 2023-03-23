import torch


def acc_metric(y_pred, y_true):
    correct = (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
    acc = correct / y_pred.size(0)
    return acc


def prepare_evaluators(y_pred, y_true, loss):
    acc = acc_metric(y_pred, y_true)
    evaluators = {'loss': loss.item(), 'acc': acc}
    return evaluators


class CosineAlignments:
    def __init__(self, model, loader, criterion) -> None:
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.device = next(model.parameters()).device

    def calc_variance(self, n):
        gs = torch.tensor(self.gather_gradients(n))
        gdv = 0.
        for i in range(n):
            for j in range(i+1, n):
                gdv += 1 - torch.dot(gs[i], gs[j]) / torch.norm(gs[i], gs[j])
        gdv /= 2 / (n * (n - 1))
        return gdv


    def gather_gradients(self, n, device):
        gs = []
        for i, (x_true, y_true) in enumerate(self.loader):
            if i >= n: break
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            self.criterion(y_pred, y_true).backward()
            g = [p.grad for p in self.model.parameters() if p.requires_grad]
            gs.append(g)
            self.model.zero_grad()
        return gs

