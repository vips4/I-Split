#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import torch

from torchvision import transforms
from matplotlib import pyplot as plt

from models.vgg_models import VGGBottleneck, vgg16_bottleneck
from models.resnet_models import ResNetBottleneck, resnet50_bottleneck
from models.bottlenecks.undercomplete_autoencoder import AutoEncoderUnderComplete

# Define preprocessing for training and inference.
TRAIN_SIZE = (224, 224)


def get_transform(is_train=False):
    resize = [transforms.Resize(TRAIN_SIZE)]
    augment_transforms = [
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomApply(
            torch.nn.ModuleList(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
                    ),
                ]
            ),
            p=0.4,
        )]
    output_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    if is_train:
        res = transforms.Compose(resize + augment_transforms + output_transforms)
    else:
        res = transforms.Compose(resize + output_transforms)
    return res


def preload_transform():
    resize = transforms.Resize(TRAIN_SIZE)
    return resize


def get_size(tensor: torch.Tensor):
    # dtype = eval(tensor.type())
    float_precision: int
    if isinstance(tensor, torch.FloatTensor):
        float_precision = 32
    elif isinstance(tensor, torch.DoubleTensor):
        float_precision = 64
    else:
        raise RuntimeError()

    bytes_size = tensor.numel()*float_precision/8

    # print(bytes_size, "bytes")
    # print(bytes_size / 1024, "Kb")
    # print(bytes_size / 1024 / 1024, "Mb")
    return bytes_size


def get_network_intermediate_sizes(
        MODEL_TYPE=None,
        show=False,
        exlcude_batchnorm=True,
        exclude_relu=True,
        include_input=False,
        inject_bottleneck=False,
        bottleneck_index=None
    ):

    print("Loading network [...]")
    if MODEL_TYPE == "VGG" or MODEL_TYPE == "vgg16":
        vgg: VGGBottleneck = vgg16_bottleneck(False)
        if inject_bottleneck:
            vgg16_structure = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
            vgg16_structure.insert(bottleneck_index + 1, "B_conv")
            vgg.inject_bottleneck(vgg16_structure,
                                  AutoEncoderUnderComplete,
                                  expansions=[2],
                                  kernel_size=3)
        ticks = vgg.get_layer_names(include_input=include_input)
        # skip = vgg.get_excluded_layers()
        model = vgg
    elif MODEL_TYPE == "ResNet" or MODEL_TYPE == "resnet50":
        resnet: ResNetBottleneck = resnet50_bottleneck()
        if inject_bottleneck:
            resnet.inject_bottleneck(bottleneck_index+1,
                                     AutoEncoderUnderComplete,
                                     expansions=[2],
                                     kernel_size=3)
        ticks = resnet.get_layer_names(exlcude_batchnorm=exlcude_batchnorm,
                                       exclude_relu=exclude_relu,
                                       include_input=include_input)
        # skip = resnet.get_excluded_layers()
        model = resnet
    else:
        raise NotImplementedError()
    print("Network loaded!")

    xin_shape = (1, 3, 224, 224)
    xin = torch.rand(xin_shape)

    sizes = model.get_sizes(xin,
                            exlcude_batchnorm=exlcude_batchnorm,
                            exclude_relu=exclude_relu,
                            include_input=include_input)

    values = [get_size(torch.rand(s)) for s in sizes]

    if inject_bottleneck:
        ticks = list(range(len(values)))

    fname = f"{MODEL_TYPE}_sizes"
    if inject_bottleneck:
        fname += f"_bottleneck_idx={bottleneck_index}"

    if show:
        plt.figure(figsize=(10, 8))
        plt.title(f"Size of intermediate data in {MODEL_TYPE}")
        plt.plot(values)
        plt.xticks(range(len(values)), ticks, rotation=90)
        plt.savefig(fname + ".jpg")

    return values, ticks
