import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

""" test on coco """
dataset_train = CocoDataset("/data2/xiongyizhe/coco2014/coco", set_name='train2014',
                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
dataset_val = CocoDataset("/data2/xiongyizhe/coco2014/coco", set_name='val2014',
                            transform=transforms.Compose([Normalizer(), Resizer()]))

retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet = retinanet.cuda()
retinanet.load_state_dict(torch.load("checkpoint/retinanet/model.pt"))
coco_eval.evaluate_coco(dataset_val, retinanet)