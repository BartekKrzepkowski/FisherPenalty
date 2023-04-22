import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from torchvision.transforms import Compose, ToTensor, Normalize, Resize,\
    RandomVerticalFlip, RandomAffine, RandomHorizontalFlip, Pad
from torch.utils.data import DataLoader, TensorDataset
class Loaders(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_vertical_flipped_loader(self, batch_size=128, num_workers=6, is_train=True):
        if self.dataset_name == 'cifar10':
            transform = Compose([ToTensor(), RandomAffine(degrees=0, translate=(1/8, 1/8)), RandomHorizontalFlip(),
                                 RandomVerticalFlip(p=1), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(root=os.environ['CIFAR10_PATH'], train=True, transform=transform, download=True)
        elif self.dataset_name == 'mnist':
            transform = Compose([ToTensor(), Pad(2), Normalize((0.5,), (0.5,))])
            train_dataset = datasets.MNIST(root=os.environ['MNIST_PATH'], train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)
        return train_loader

    def get_random_classes_loader(self, batch_size=128, num_workers=6, is_train=True):
        if self.dataset_name == 'cifar10':
            transform = Compose([ToTensor(), RandomAffine(degrees=0, translate=(1/8, 1/8)), RandomHorizontalFlip(),
                                 Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(root=os.environ['CIFAR10_PATH'], train=True, transform=transform, download=True)
        elif self.dataset_name == 'mnist':
            transform = Compose([ToTensor(), Pad(2), Normalize((0.5,), (0.5,))])
            train_dataset = datasets.MNIST(root=os.environ['MNIST_PATH'], train=True, transform=transform, download=True)
        train_dataset.targets = np.array(train_dataset.targets)[torch.randperm(50000).numpy()].tolist()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)
        return train_loader

    def get_noised_loader(self, batch_size=128, num_workers=6, is_train=True):
        if self.dataset_name == 'cifar10':
            y_train = datasets.CIFAR10(root=os.environ['CIFAR10_PATH'], train=True, download=True).targets
        elif self.dataset_name == 'mnist':
            y_train = datasets.MNIST(root=os.environ['MNIST_PATH'], train=True, download=True).targets
        x_train = torch.rand((50000, 3, 32, 32))
        y_train = torch.tensor(y_train)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)
        return train_loader

    def get_blurred_loader(self, batch_size=128, num_workers=4, is_train=True):
        if self.dataset_name == 'cifar10':
            transform = Compose([ToTensor(), RandomAffine(degrees=0, translate=(1/8, 1/8)), RandomHorizontalFlip(),
                                 Resize(8, interpolation=InterpolationMode.BILINEAR, antialias=None), Resize(32, interpolation=InterpolationMode.BILINEAR, antialias=None), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(root=os.environ['CIFAR10_PATH'], train=True, transform=transform, download=True)
        elif self.dataset_name == 'mnist':
            transform = Compose([ToTensor(), Pad(2), Resize(8, interpolation=InterpolationMode.BILINEAR), Resize(32, interpolation=InterpolationMode.BILINEAR), Normalize((0.5,), (0.5,))])
            train_dataset = datasets.MNIST(root=os.environ['MNIST_PATH'], train=True, transform=transform, download=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)
        return train_loader

    def get_proper_loader(self, batch_size=128, num_workers=4, is_train=True):
        if self.dataset_name == 'cifar10':
            transform = Compose([ToTensor(), RandomAffine(degrees=0, translate=(1/8, 1/8)), RandomHorizontalFlip(),
                                 Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(root=os.environ['CIFAR10_PATH'], train=is_train, transform=transform, download=True)
        elif self.dataset_name == 'mnist':
            transform = Compose([ToTensor(), Pad(2), Normalize((0.5,), (0.5,))])
            train_dataset = datasets.MNIST(root=os.environ['MNIST_PATH'], train=is_train, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)
        return train_loader

    def get_proper_transform(self):
        if self.dataset_name == 'cifar10':
            transform = Compose([ToTensor(), RandomAffine(degrees=0, translate=(1/8, 1/8)), RandomHorizontalFlip(),
                                 Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif self.dataset_name == 'mnist':
            transform = Compose([ToTensor(), Pad(2), Normalize((0.5,), (0.5,))])
        return transform