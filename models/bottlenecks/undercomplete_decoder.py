#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


from torch import nn


class Decoder(nn.Module):
    def __init__(
        self,
        out_ch=64,
        filters=64,
        kernel_size=5,
        expansions=None,

        fix_first=False,
        fix_second=False,

        fix_first_pad=False,
        fix_second_pad=False,

        init_weights: bool = True,
    ):
        super().__init__()
        self.out_channels = out_ch
        self.filters = filters
        self.kernel = kernel_size

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                # filters * 2, 
                filters * expansions[0], 
                filters, 
                kernel_size, 
                stride=2, 

                padding=1 if not fix_first_pad else 0,
                output_padding=1 if not fix_first else 0
            ),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                filters,
                out_ch,
                kernel_size,
                stride=2,
                
                padding=1 if not fix_second_pad else 0,
                output_padding=1 if not fix_second else 0
            ),
        )

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

            print(f"Decoder Forward")
            print(f"\t(START)")
            print(f"\tForward input: {x.shape}")

            # Debug
            for i, m in enumerate(self.decoder_conv):
                x = m(x)
                print(f"\tForward step {i}: {x.shape}")

            print(f"\t(END)")
            out = x
        else:
            out = self.decoder_conv(x)

        # print("Decoder output shape: ", x.shape)
        return out
