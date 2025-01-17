import torch
import torch.nn as nn

import copy
import torchslim
import torchslim.slim_solver as slim_solver


import torchpruner as pruner
import torchpruner.model_tools as model_tools

from torchslim.modules.rep_modules import ACNet_CR, RepModule, Efficient_ACNet_CR

acnet_class_maping = {"acnet_cr": ACNet_CR, "efficient_acnet_cr": Efficient_ACNet_CR}


def acnet_convertor_generator(acnet_type):
    acnet_class = acnet_class_maping[acnet_type]

    def acnet_convert_function(name, origin_module):
        if origin_module.kernel_size[0] == 1 and origin_module.kernel_size[1] == 1:
            return origin_module
        return acnet_class(origin_module)

    return acnet_convert_function


def deploy_convert_function(name, origin_module):
    return origin_module.convert()


# acnet convertor
def convert_to_acnet(model, acnet_type):
    model = model_tools.replace_object_by_class(
        model, nn.Conv2d, acnet_convertor_generator(acnet_type)
    )
    return model


def deploy_convert(model):
    return model_tools.replace_object_by_class(
        model, RepModule, deploy_convert_function
    )


def optimizer_generator(params, config):
    return torch.optim.SGD(params, lr=0.0)


def scheduler_generator(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epoch"])


def init_hook(self):
    self.model = convert_to_acnet(self.model, self.config["acnet_type"])
    print(self.model)


class ACNetSolver(slim_solver.CommonSlimSolver):
    __config_setting__ = [
        ("task_name", str, "defualt", False, "The task name"),
        ("acnet_type", str, "acnet_cr", False, "The type of the acnet block"),
        (
            "save_deploy_format",
            bool,
            True,
            False,
            "convert the ACNet to conv when saving the model",
        ),
        ("lr", float, 0.1, False, "The learning rate of the optimizer"),
        ("epoch", int, 360, False, "The total epoch to train the model"),
        ("batch_size", int, 128, False, "The batch size per step"),
        ("test_batch_size", int, 128, False, "The evaluation batch size per step"),
        ("momentum", float, 0.9, False, "The momentum for the optimizer"),
        ("weight_decay", float, 1e-4, False, "The wegith decay for the parameters"),
        ("save_keyword", str, "acc", False, "The keyword for save"),
        ("save_dir", str, "checkpoints", False, "The model save dir"),
        ("num_workers", int, 0, False, "The number of workers to read data"),
        ("devices", list, None, False, "The device to be used in training"),
        ("log_interval", int, 200, False, "The interval to report the log"),
        # generate the optimizer
        (
            "optimizer_generator",
            "function",
            optimizer_generator,
            False,
            "The optimizer generator (params,config)->optimizer",
        ),
        # generate the scheduler
        (
            "scheduler_generator",
            "function",
            scheduler_generator,
            True,
            "the scheduler generator for the task (optmizer,config)->scheduler",
        ),
        # predict the result
        (
            "predict_function",
            "function",
            None,
            False,
            "get the prediction of the data (model,batch_data)->predict",
        ),
        # calculate the loss for one iteration
        (
            "calculate_loss_function",
            "function",
            None,
            False,
            "(predict,batch_data)->loss",
        ),
        # get the evaluate result for one iteration
        (
            "evaluate_function",
            "function",
            None,
            True,
            "(predict,batch_data)->evaluate_dict",
        ),
        # get the dataset
        (
            "dataset_generator",
            "function",
            None,
            True,
            "()->dataset_train,dataset_validation",
        ),
    ]

    def __init__(self, model, config):
        super(ACNetSolver, self).__init__(model, config)
        self.regist_init_hook(init_hook)

    def generate_params_setting(self):
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        base_lr = self.config["lr"]
        weight_decay = self.config["weight_decay"]
        momentum = self.config["momentum"]
        params = []
        for key, value in model.named_parameters():
            apply_weight_decay = weight_decay
            apply_momentum = momentum
            apply_lr = base_lr
            # if 'scale' in key:
            #     apply_weight_decay=0
            if "bias" in key:
                apply_lr = apply_lr * 2
            params += [
                {
                    "params": [value],
                    "lr": apply_lr,
                    "weight_decay": apply_weight_decay,
                    "momentum": apply_momentum,
                }
            ]
        return params

    def save_model(self):
        model = copy.deepcopy(self.model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.cpu()
        if self.config["save_deploy_format"]:
            model = deploy_convert(model)
        torch.save(
            {
                self.config["save_keyword"]: self.variable_dict["save_target"],
                "net": model,
            },
            self.variable_dict["save_path"],
        )
