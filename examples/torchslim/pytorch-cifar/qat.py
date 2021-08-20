import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torchslim.quantizing.qat as qat

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

qat.QATSolver.print_config()
qat.QATSolver.print_config(help=True)

config = {}
config["devices"] = [0]
config["epoch"] = 100
config["lr"] = 0.001
config["batch_size"] = 128
config["test_batch_size"] = 100
config["task_name"] = "resnet56_qat"
config["log_interval"] = 20


def predict_function(model, data):
    X, _ = data
    prediction = model(X)
    return prediction.view(-1, prediction.size(1))


def calculate_loss(predict, data):
    _, Y = data
    loss = F.cross_entropy(predict, Y)
    return loss


def evaluate(predict, data):
    _, Y = data
    _, predicted = predict.max(1)
    correct = predicted.eq(Y).sum().item()
    return {"acc": correct / predict.size(0)}


def dataset_generator():
    return trainset, testset


config["predict_function"] = predict_function
config["calculate_loss_function"] = calculate_loss
config["evaluate_function"] = evaluate
config["dataset_generator"] = dataset_generator

checkpoint = torch.load("checkpoint/resnet56/ckpt.pth")
model = checkpoint["net"]

solver = qat.QATSolver(model, config)
solver.run()
