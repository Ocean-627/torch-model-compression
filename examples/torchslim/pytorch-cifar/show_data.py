import torchslim
import torchslim.pruning
import torchslim.pruning.resrep as resrep

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

def dataset_generator():
    return trainset, testset

dataset_train,dataset_validation = dataset_generator()

batch_size = 1
test_batch_size = 1
num_workers = 2

trainloader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

valloader = DataLoader(
    dataset_validation,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
)

flag = 0
for step, data in enumerate(trainloader):
    x,y=data
    print(x.size(),y.size())
    flag+=1
    if flag==2:
        break
