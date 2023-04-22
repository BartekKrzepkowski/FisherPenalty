from copy import deepcopy

import torch
from pyhessian import hessian


def acc_metric(y_pred, y_true):
    correct = (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
    acc = correct / y_pred.size(0)
    return acc


def prepare_evaluators(y_pred, y_true, loss):
    acc = acc_metric(y_pred, y_true)
    evaluators = {'loss': loss.item(), 'acc': acc}
    return evaluators

def entropy_loss(y_pred):
    return -torch.sum(torch.nn.functional.softmax(y_pred, dim=1) * torch.log_softmax(y_pred, dim=1))


class BatchVariance(torch.nn.Module):
    def __init__(self, model, optim, criterion=None, dataloader=None, device=None):
        super().__init__()
        self.model_zero = deepcopy(model)
        self.model = model
        self.optim = optim
        self.criterion = criterion
        # held out % examples from dataloader
        self.dataloader = dataloader
        self.device = device
        self.model_trajectory_length = 0.0

    def forward(self, evaluators, distance_type):
        lr = self.optim.param_groups[-1]['lr']
        norm = self.model_gradient_norm()
        evaluators['model_gradient_norm_squared'] = norm ** 2
        self.model_trajectory_length += lr * norm
        evaluators['model_trajectory_length'] = self.model_trajectory_length
        distance_from_initialization = self.distance_between_models(distance_type)
        evaluators[f'distance_from_initialization_{distance_type}'] = distance_from_initialization
        evaluators['excessive_length'] = evaluators['model_trajectory_length'] - evaluators[f'distance_from_initialization_{distance_type}']
        return evaluators
        

    def model_gradient_norm(self, norm_type=2.0):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        norm = torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in parameters]), norm_type)
        return norm.item()
    
    def distance_between_models(self, distance_type):
        def distance_between_models_l2(parameters1, parameters2, norm_type=2.0):
            """
            Returns the l2 distance between two models.
            """
            distance = torch.norm(torch.stack([torch.norm(p1-p2, norm_type) for p1, p2 in zip(parameters1, parameters2)]), norm_type)
            return distance.item()
        
        def distance_between_models_cosine(parameters1, parameters2):
            """
            Returns the cosine distance between two models.
            """
            distance = 0
            for p1, p2 in zip(parameters1, parameters2):
                distance += 1 - torch.cosine_similarity(p1.flatten(), p2.flatten())
            return distance.item()

        """
        Returns the distance between two models.
        """
        parameters1 = [p for p in self.model_zero.parameters() if p.requires_grad]
        parameters2 = [p for p in self.model.parameters() if p.requires_grad]
        if distance_type == 'l2':
            distance = distance_between_models_l2(parameters1, parameters2)
        elif distance_type == 'cosine':
            distance = distance_between_models_cosine(parameters1, parameters2)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        return distance
    
    def sharpness(self, dataloader, maxIter=100):
        hessian_comp = hessian(self.model, self.criterion, dataloader=dataloader, cuda=self.device.type!='cpu')
        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=maxIter)
        self.model.train()
        return top_eigenvalues[0].item()


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

