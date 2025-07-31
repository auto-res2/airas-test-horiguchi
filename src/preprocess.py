import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_dataloader(batch_size=64, train=True):
    """
    Returns a DataLoader for the CIFAR-10 dataset.
    Data is stored/downloaded in the './data' directory.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root=os.path.join('.', 'data'), train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
