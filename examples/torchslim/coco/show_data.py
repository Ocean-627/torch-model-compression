import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torchslim.quantizing.qat as qat

from retinanet.dataloader import CocoDataset, Normalizer, Augmenter, Resizer
from retinanet import model


trainset = CocoDataset("/data2/xiongyizhe/coco2014/coco", set_name='train2014',
                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

testset = CocoDataset("/data2/xiongyizhe/coco2014/coco", set_name='val2014',
                            transform=transforms.Compose([Normalizer(), Resizer()]))

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
    print("img:", data["img"].size(), "scale:", data["scale"],"annot", data["annot"].size())
    # x,y=data
    # print(x.size(),y.size())
    # flag+=1
    # if flag==2:
    #     break


# solver = qat.QATSolver(retinanet, config)
# solver.run()