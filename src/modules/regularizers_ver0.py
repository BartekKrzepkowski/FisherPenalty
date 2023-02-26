import torch


class FisherPenaly(torch.nn.Module):
    def __init__(self, model, criterion, num_classes):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.labels = torch.arange(num_classes).to(next(model.parameters()).device)

    def forward(self, y_pred):
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze()
        loss = self.criterion(y_pred, y_sampled)
        loss.backward(retain_graph=True) # can't use create_graph
        trace = 0.0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                trace += (param.grad ** 2).sum()
        self.model.zero_grad()
        return trace
