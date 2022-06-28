#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import torch
from torch import nn

from typing import Optional, List

from models.bottlenecks.base import CompressionBottleneck
from models.bottlenecks.undercomplete_encoder import Encoder
from models.bottlenecks.undercomplete_decoder import Decoder


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


class AutoEncoderUnderComplete(nn.Module, CompressionBottleneck):
    def __init__(self,
                 input_shape: torch.Size,
                 expansions: Optional[List[int]] = None,
                 # output_shape: torch.Size,
                 # n_filters=64,
                 compression_ratio=8,
                 kernel_size=3,
                 init_weights=True):
        super().__init__()
        assert isinstance(input_shape, (tuple, list, torch.Size, int)), "Input shape must be a tuple or a list of dimensions"

        if not isinstance(input_shape, int):
            assert len(input_shape) == 3, "Input shape must be in the shape: [C,H,W]"
        else:
            input_shape = [input_shape]

        if kernel_size % 2 == 0:
            w = "[AutoEncoder] WARNING! Even kernel size will give dimensions mismatch with small feature maps. Change the decoder code or use an odd kernel size"
            import warnings
            warnings.warn(w)

        if expansions is None:
            expansions = [2]

        self.input_channels = input_shape[0]
        self.out_channels = input_shape[0]

        self._init_weights = init_weights

        self.kernel_size = kernel_size
        self.n_filters = int(round(self.input_channels / compression_ratio))

        self.expansions = expansions

        self.encoder: Encoder = Encoder(
            in_shape=input_shape,
            filters=self.n_filters,
            kernel_size=kernel_size,
            expansions=expansions,
            init_weights=init_weights,

        )
        # self.decoder: Decoder = Decoder(in_ch=input_channels, filters=n_filters, init_weights=init_weights)
        # self.decoder: Decoder = Decoder(out_ch=out_channels, filters=n_filters, init_weights=init_weights)
        self.decoder: Decoder = None

        fake_xin = torch.randn(input_shape).type(torch.FloatTensor).unsqueeze(0)
        self.create_decoder(fake_xin)

    def freeze(self):
        for name, module in self.named_parameters():
            if hasattr(module, "weight"):
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                module.bias.requires_grad = False

    def unfreeze(self):
        for n, p in self.named_parameters():
            p.requires_grad = True

    def create_decoder(self, fake_xin) -> bool:

        def _fix(fix_first, fix_second, fix_first_pad, fix_second_pad):

            self.decoder = Decoder(
                out_ch=self.out_channels,
                filters=self.n_filters,
                init_weights=self._init_weights,
                kernel_size=self.kernel_size,
                expansions=self.expansions,

                fix_first=fix_first,
                fix_second=fix_second,
                fix_first_pad=fix_first_pad,
                fix_second_pad=fix_second_pad
            )
            encoded = self.encoder(fake_xin)
            decoded = self.decoder(encoded)

            s1 = fake_xin.shape
            s2 = decoded.shape

            return s1 == s2

        # normal run
        attempt = _fix(False, False, False, False)
        if attempt:
            return True

        # fixes

        # Output padding
        attempt = _fix(True, False, False, False)
        if attempt:
            return True
        # Output padding + padding
        attempt = _fix(True, False, True, False)
        if attempt:
            return True

        # Output padding
        attempt = _fix(False, True, False, False)
        if attempt:
            return True
        # Output padding + padding
        attempt = _fix(False, True, False, True)
        if attempt:
            return True

        attempt = _fix(True, True, False, False)
        if attempt:
            return True
        attempt = _fix(True, True, True, True)
        if attempt:
            return True

        # a mali estremi...
        attempt = _fix(False, False, True, False)
        if attempt:
            return True
        attempt = _fix(False, False, False, False)
        if attempt:
            return True
        attempt = _fix(False, False, True, True)
        if attempt:
            return True

        import warnings
        warnings.warn("[Bottleneck AutoEncoder] Decoder creation failed! Decoder output will have a different shape")
        return False

    def forward(self, x):
        encoded = self.encoder(x)

        # byte_size = get_size(encoded)
        if False:
            # TODO: getsize breaks something if tensors are on gpu, investigate!
            insize = get_size(x)
            outsize = get_size(self.encoder(x))

            print("Compression info (Input->Encoded):")
            print(f"Input size\t{insize/1000:.2f} Kb  --  {insize/1000/1000:.4f} Mb")
            print(f"Output size\t{outsize/1000:.2f} Kb  --  {outsize/1000/1000:.4f} Mb")
            print(f"Compression ratio: {outsize/insize:.4f}")
            print(f"Compression: {100 - outsize*100/insize:.2f}%")
            print(f"Information preserved: {outsize*100/insize:.2f}%")
            print()

        decoded = self.decoder(encoded)
        return decoded

    def get_split_size(self, x):
        return self.encoder(x).shape
        
