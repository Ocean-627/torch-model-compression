import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torchslim.quantizing.qat as qat

from retinanet.dataloader import CocoDataset, Normalizer, Augmenter, Resizer
from retinanet import model


trainset = CocoDataset("/data2/xiongyizhe/coco2014/coco", set_name='train2014',
                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

testset = CocoDataset("/data2/xiongyizhe/coco2014/coco", set_name='val2014',
                            transform=transforms.Compose([Normalizer(), Resizer()]))

qat.QATSolver.print_config()
qat.QATSolver.print_config(help=True)

config = {}
config["devices"] = [0]
config["epoch"] = 100
config["lr"] = 0.001
config["batch_size"] = 128
config["test_batch_size"] = 100
config["task_name"] = "retinanet_qat"
config["log_interval"] = 20


def predict_function(model, data):
    X, _ = data
    prediction = model(X)
    return prediction.view(-1, prediction.size(1))


def calculate_loss(predict, data):
    _, Y = data
    loss = F.cross_entropy(predict, Y)
    return loss

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1))

    res = []
    for k in topk:
        correct_k = correct[:, :k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def evaluate(predict, data):
    _, Y = data
    res = accuracy(predict, Y, (1, 5))
    return {"top1": res[0], "top5": res[1]}


def dataset_generator():
    return trainset, testset


config["predict_function"] = predict_function
config["calculate_loss_function"] = calculate_loss
config["evaluate_function"] = evaluate
config["dataset_generator"] = dataset_generator

retinanet = model.resnet50(num_classes=trainset.num_classes(),)
retinanet = retinanet.cuda()
retinanet.load_state_dict(torch.load("checkpoint/retinanet/model.pt"))

solver = qat.QATSolver(retinanet, config)
solver.run()
