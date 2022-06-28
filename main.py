#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import os
import torch
import random
import numpy as np
import torchvision
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from utils import get_network_intermediate_sizes

from models.vgg_models import *
from models.vgg_models import VGG
from models.resnet_models import *
from models.resnet_models import ResNet
from datasets.imagenet import ImageNet
import torchvision.transforms as transforms
from filters_extraction import get_filters


def norm_sum_of_values(s_map):
    return round(abs(s_map.sum() / s_map.size), 30)


def compute_saliency(model: VGG,
                     experiments_dir: str,
                     file_savepath: str,
                     save_interval: int = 10000,
                     device: torch.device = torch.device("cpu")):
    pickle_out = file_savepath
    exp_dir = experiments_dir
    os.makedirs(exp_dir, exist_ok=True)

    model = model.to(device)

    # Test with ImageNet.
    # val_loader = DataLoader(
    #     ImageNet("data/imagenet", "val"),
    #     batch_size=1,
    #     shuffle=False
    # )

    # Test with CIFAR10.
    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              shuffle=False)

    # saliency_maps = <(idx, gt_class): map>.
    saliency_maps = {}
    last_label = None

    for i, (img, gt_label) in enumerate(tqdm(test_loader)):
        gt_label_val = int(gt_label.detach().cpu().numpy().item())
        if last_label != gt_label_val:
            last_label = gt_label_val
            print(f"Running on class: {gt_label_val}")

        img = img.to(device)

        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        res, grads, fwd_layer_list = model.forward_with_grads(img)

        # Get maximum val.
        top_score = res[0, torch.argmax(res)]

        # Compute grads.
        top_score.backward()

        layers_saliency = []
        for grad_idx, grad in enumerate(grads):
            layer_for_features: List[torch.nn.Module] = \
                fwd_layer_list[:grad_idx+1]
            saliency = get_filters(img, layer_for_features, grad)
            layers_saliency.append(saliency)

        saliency_maps[(i, gt_label_val)] = layers_saliency

        if i % save_interval == 0:
            with open(pickle_out, "wb") as fp:
                pkl.dump(saliency_maps, fp)

    with open(pickle_out, "wb") as fp:
        pkl.dump(saliency_maps, fp)

    return saliency_maps


def get_saliency_mean(
    data: Dict[Tuple[int, int], List[torch.Tensor]],
    mean_maps_save_file: Optional[str],
    class_idx: Optional[int] = None
):
    # Data is expected to be a dict of:
    # <iteration_idx, class_idx> : List[class_activation_maps(i.e. tensors)]

    if len(data) == 0:
        raise RuntimeError()

    for idx_and_class, maps_list in data.items():
        # Get the N. of layers.
        num_maps = len(maps_list)
        break

    maps_per_layer = [None] * num_maps
    for i, (idx_and_class, maps_list) in enumerate(data.items()):
        iter_idx, c_idx = idx_and_class
        if class_idx is not None:
            if c_idx != class_idx:
                continue
        for j, layermap in enumerate(maps_list):
            if maps_per_layer[j] is None:
                maps_per_layer[j] = []
            maps_per_layer[j].append(layermap)

    mean_map_per_layer = []
    for layermap_list in maps_per_layer:
        maps = np.asarray([m.detach().cpu().numpy() \
            if isinstance(m, torch.Tensor) else np.asarray(m) \
                for m in layermap_list])
        mean_map = np.mean(maps, axis=0)
        mean_map_per_layer.append(mean_map)

    if mean_maps_save_file is not None:
        # with open(pickle_out, "wb") as fp:
        #     pkl.dump(mean_map_per_layer, fp)
        np.save(mean_maps_save_file, mean_map_per_layer)

    return mean_map_per_layer


def maps_analisys(net_type,
                  layers_list_to_skip,
                  layers_tick_names,
                  maps_per_class: Dict[int, np.ndarray],
                  mean_maps_per_layer,
                  exp_dir: str):
    os.makedirs(exp_dir, exist_ok=True)
    skip = layers_list_to_skip

    def split(x): return [e for (i, e) in enumerate(x) if i not in skip]

    mean_maps_per_layer = split(mean_maps_per_layer)
    if len(layers_tick_names) != len(mean_maps_per_layer)+1:
        input_name = layers_tick_names[0]
        layers_tick_names = layers_tick_names[1:]
        layers_tick_names = split(layers_tick_names)
        layers_tick_names.insert(0, input_name)

    assert len(layers_tick_names) == len(mean_maps_per_layer) + 1, \
        "Wrong alignment with ticknames!"

    # Computing I-Split curve.
    maps = mean_maps_per_layer

    vals = [0]
    for k, mean_map in enumerate(maps):
        val = norm_sum_of_values(mean_map)
        vals.append(val)
    vals[0] = vals[1]

    x = np.arange(len(vals))

    fig, ax = plt.subplots(figsize = (12, 8), sharex=True, sharey=True)
    plt.xticks(x, layers_tick_names)
    ax.tick_params(axis='both', labelsize=22)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel("Layers",  fontsize=30, labelpad=10)
    ax.set_ylabel("CUI (log10)", color='orange', fontsize=30, labelpad=10)

    log_vals = np.log10(vals)
    maxima = (np.diff(np.sign(np.diff(log_vals))) < 0).nonzero()[0] + 1

    ax.plot(log_vals, linewidth=5, color='orange', label='I-Split', zorder=1)
    ax.scatter(np.asarray(x)[maxima],
               np.asarray(log_vals)[maxima],
               marker="*",
               s=550,
               edgecolors='yellow',
               c='red',
               linewidth=1,
               zorder=2)

    # Legend.
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, bbox_to_anchor=(0.36, 0.98), fontsize=26)

    # Vertical lines at the points.
    for xc in x:
        plt.axvline(x=xc, color='gray', linewidth=0.3, ymax=0.98)

    fig.tight_layout()
    plt.savefig(f"{exp_dir}/i_split_curve.png", dpi=400)
    plt.close()

    # Computing Layers Output size (MB) curve.
    fig, ax = plt.subplots(figsize = (12, 8), sharex=True, sharey=True)
    if net_type == "ResNet":
        sizes, ticks = get_network_intermediate_sizes(net_type,
                                                      exlcude_batchnorm=True,
                                                      exclude_relu=True,
                                                      include_input=True)
    elif net_type == "VGG":
        sizes, ticks = get_network_intermediate_sizes(net_type,
                                                      exlcude_batchnorm=True,
                                                      exclude_relu=True,
                                                      include_input=True)

    plt.xticks(x, ticks)
    ax.tick_params(axis='both', labelsize=22)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel("Layers",  fontsize=30, labelpad=10)
    ax.set_ylabel("Output size (MB)", color='k', fontsize=30, labelpad=10)

    ax.plot(sizes,
            linestyle='dotted',
            linewidth=5,
            color='k',
            label="CDE",
            zorder=1)

    # Legend.
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, bbox_to_anchor=(0.96, 0.96), fontsize=26)

    # Vertical lines at the points.
    for xc in x:
        plt.axvline(x=xc, color='gray', linewidth=0.3, ymax=0.98)

    fig.tight_layout()
    plt.savefig(f"{exp_dir}/output_size_curve.png", dpi=400)
    plt.close()


def main(TModel: str):
    device = torch.device("cuda:0") \
        if torch.cuda.is_available() else torch.device("cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    expt_dirname = f"expts/{TModel}_res"

    isplit_by_class = f"expts/{TModel}_res/isplit_by_class.pkl"
    isplit_mean_by_class = f"expts/{TModel}_res/isplit_mean_by_class.npy"
    isplit_mean_all_classes = f"expts/{TModel}_res/isplit_mean_all_classes.npy"

    # Choice of supported models (VGG16 and ResNet50).
    if TModel == "VGG":
        model = vgg16(pretrained=True)
        # ReLU layers of VGG16.
        skip = [1, 2, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
        layer_ticknames = model.get_layer_names(include_input=True)
    elif TModel == "ResNet":
        model = resnet50(pretrained=True)
        # ReLU and BatchNorm layers of ResNet-50.
        skip = model.get_excluded_layers()
        layer_ticknames = model.get_layer_names(exclude_relu=False,
                                                exlcude_batchnorm=False,
                                                include_input=True)
    else:
        raise NotImplementedError()

    if not os.path.isfile(isplit_mean_all_classes) or \
       not os.path.isfile(isplit_mean_by_class):
        if not os.path.isfile(isplit_by_class):
            compute_saliency(model,
                             expt_dirname,
                             isplit_by_class,
                             device=device)

        with open(isplit_by_class, "rb") as fp:
            data = pkl.load(fp)

        mean_map_per_layer = get_saliency_mean(data, isplit_mean_all_classes)
        to_save = {}

        class_collected = set()
        for iter_classidx, _ in data.items():
            _, class_idx = iter_classidx
            if class_idx not in class_collected:
                class_collected.add(class_idx)
                mean_map_class_i = get_saliency_mean(data, None, class_idx)

                to_save[class_idx] = mean_map_class_i

        np.save(isplit_mean_by_class, to_save)
        maps_per_class = to_save
    else:
        maps_per_class: Dict[int, np.ndarray] = \
            np.load(isplit_mean_by_class, allow_pickle=True).item()
        mean_map_per_layer = np.load(isplit_mean_all_classes, allow_pickle=True)

    maps_analisys(TModel,
                  skip,
                  layer_ticknames,
                  maps_per_class,
                  mean_map_per_layer,
                  expt_dirname)


if __name__ == "__main__":
    main("ResNet")
    main("VGG")
