#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List


def get_filters(input_img: torch.Tensor,
                fwd_layer_list: List[torch.nn.Module],
                gradients: torch.Tensor,
                normalize: bool = False,
                show: bool = False) -> None:
    # Pool the gradients across the channels.
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    if len(input_img.shape) == 3:
        input_img = input_img.unsqueeze(0)

    # Get the activations of the last convolutional layer.
    _input = input_img
    for l in fwd_layer_list:
        _input = l(_input)

    # activations = fwd_layer_list(input_img).detach()
    activations = _input.detach()

    # Weight the channels by corresponding gradients.
    chs = activations.shape[1]

    for i in range(chs):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Average the channels of the activations.
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()

    # ReLU on top of the heatmap (Eq. 2 of https://arxiv.org/pdf/1610.02391.pdf).
    heatmap = np.maximum(heatmap, 0)

    heatmap = torch.from_numpy(heatmap)
    if normalize:
        heatmap /= torch.max(heatmap) if torch.max(heatmap) != 0 else 1

    # Draw the heatmap.
    if show:
        plt.figure()
        plt.imshow(heatmap.squeeze())
        plt.savefig("test.jpg")
        plt.pause(0.5)

    return heatmap
