#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        in_shape: torch.Size,
        filters=64,
        kernel_size=5,
        expansions=None,
        init_weights: bool = True,
    ):
        super().__init__()
        in_ch = in_shape[0]
        self.in_channel = in_ch
        self.filters = filters
        self.kernel = kernel_size

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(
                in_ch,
                filters,
                kernel_size,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),

            nn.Conv2d(
                filters,
                filters*expansions[0],
                kernel_size,
                stride=2,
                padding=1
            ),
           
            nn.BatchNorm2d(filters * expansions[0]),
            nn.ReLU(True),
        )

        self.out_channels = filters * sum([1 for m in self.encoder_cnn if isinstance(m, nn.Conv2d)])

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self, modules=None):
        if not modules:
            modules = self.modules()

        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        is_debug = False

        if is_debug:
            print(f"Encoder Forward")
            print(f"\t(START)")
            print(f"\tForward input: {x.shape}")
            # Debug
            for i, m in enumerate(self.encoder_cnn):
                try:
                    x = m(x)
                except ValueError:
                    continue

                print(f"\tForward step {i}: {x.shape}")
            print("\t(END)")

            # print("Encoder output shape: ", x.shape)
            # print(x.shape)
        else:
            x = self.encoder_cnn(x)
        return x
