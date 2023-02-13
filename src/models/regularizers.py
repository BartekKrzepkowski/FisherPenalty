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
        grads = torch.autograd.grad(
            loss,
            [p for n, p in self.model.named_parameters() if p.requires_grad],
            retain_graph=True,
            create_graph=True)
        gr_norm_sq = 0.0
        for gr in grads:
            if gr is not None:
                gr_norm_sq += (gr**2).sum()
        return gr_norm_sq
