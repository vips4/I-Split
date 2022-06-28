#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple, Type, Any, Callable, Union, List, Optional

from models.bottlenecks.base import CompressionBottleneck
from models.bottlenecks.undercomplete_autoencoder import AutoEncoderUnderComplete

# ResNet models.
__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def unfreeze(self):
        for n, p in self.named_parameters():
            if hasattr(p, "unfreeze"):
                p.unfreeze()
            else:
                p.requires_grad = True

    def freeze(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                if module.bias is not None:
                    module.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.first_block = [self.conv1, self.bn1, self.relu, self.maxpool]

        self.block = block
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def set_num_class(self, num_classes: int = 1, copy_weights: bool = True) -> None:
        new_fc = nn.Linear(512 * self.block.expansion, num_classes)

        if copy_weights:
            if hasattr(self.fc, "weight"):

                if self.fc.weight.shape == new_fc.weight.shape and copy_weights:
                    new_fc.weight = self.fc.weight
                else:
                    nn.init.normal_(new_fc.weight, 0, 0.01)
                    nn.init.constant_(new_fc.bias, 0)
        else:
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.constant_(new_fc.bias, 0)

        self.fc = new_fc

        self.num_classes = num_classes
        return self

    def get_excluded_layers(self) -> List[int]:
        # res = [1, 2]  # bachnorm, relu
        res = []

        curr = 0
        tmp = []
        for i, v in enumerate(self.first_block):
            if isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                tmp.append(i+curr)
        res += tmp

        curr = len(self.first_block)
        tmp = []
        for i, v in enumerate(self.layer1):
            if isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                tmp.append(i+curr)

        res += tmp

        curr = len(self.layer1)
        tmp = []
        for i, v in enumerate(self.layer2):
            if isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                tmp.append(i+curr)

        res += tmp

        curr = len(self.layer2)
        tmp = []
        for i, v in enumerate(self.layer3):
            if isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                tmp.append(i+curr)

        res += tmp

        curr = len(self.layer3)
        tmp = []
        for i, v in enumerate(self.layer4):
            if isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                tmp.append(i+curr)

        res += tmp

        return res

    def get_layer_names(self, exlcude_batchnorm: bool = False, exclude_relu: bool = False, include_input: bool = False) -> List[int]:
        res = ["input"] if include_input else []

        for v in self.first_block:
            if exlcude_batchnorm and isinstance(v, torch.nn.BatchNorm2d):
                continue
            if exclude_relu and isinstance(v, torch.nn.ReLU):
                continue
            res.append(f"{type(v).__name__.lower()}")

        tmp = 1
        for i, v in enumerate(self.layer1):
            if exlcude_batchnorm and isinstance(v, torch.nn.BatchNorm2d):
                continue
            if exclude_relu and isinstance(v, torch.nn.ReLU):
                continue
            res.append(f"layer1_{type(v).__name__.lower()}_{tmp}")
            tmp += 1

        tmp = 1
        for i, v in enumerate(self.layer2):
            if exlcude_batchnorm and isinstance(v, torch.nn.BatchNorm2d):
                continue
            if exclude_relu and isinstance(v, torch.nn.ReLU):
                continue

            res.append(f"layer2_{type(v).__name__.lower()}_{tmp}")
            tmp += 1

        tmp = 1
        for i, v in enumerate(self.layer3):
            if exlcude_batchnorm and isinstance(v, torch.nn.BatchNorm2d):
                continue
            if exclude_relu and isinstance(v, torch.nn.ReLU):
                continue

            res.append(f"layer3_{type(v).__name__.lower()}_{tmp}")
            tmp += 1

        tmp = 1
        for i, v in enumerate(self.layer4):
            if exlcude_batchnorm and isinstance(v, torch.nn.BatchNorm2d):
                continue
            if exclude_relu and isinstance(v, torch.nn.ReLU):
                continue

            res.append(f"layer4_{type(v).__name__.lower()}_{tmp}")
            tmp += 1

        # res += "avg_pool"

        return res

    def get_sizes(self, x, exlcude_batchnorm=True, exclude_relu=True, include_input=False, include_bottlenecks_deep=True):
        sizes = [] if not include_input else [x.shape]

        x = self.conv1(x)
        sizes.append(x.shape)

        if not exlcude_batchnorm:
            x = self.bn1(x)
            sizes.append(x.shape)

        if not exclude_relu:
            x = self.relu(x)
            sizes.append(x.shape)

        x = self.maxpool(x)
        sizes.append(x.shape)

        layer_sequentials = [self.layer1, self.layer2, self.layer3, self.layer4]

        _input = x
        for layer in layer_sequentials:
            for _, f in enumerate(layer):
                features: torch.Tensor = f(_input)
                if exclude_relu and isinstance(f, torch.nn.ReLU):
                    continue
                if exlcude_batchnorm and isinstance(f, torch.nn.BatchNorm2d):
                    continue
                
                if include_bottlenecks_deep and isinstance(f, CompressionBottleneck):
                    sizes.append(f.get_split_size(_input))
                else:
                    sizes.append(features.shape)
                _input = features

        return sizes

    def forward_with_grads(self, x) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.nn.Module]]:
        grads: List[torch.Tensor] = []  # this variable will be filled only AFTER backward()

        fwd_layers: List[torch.nn.Module] = []
        # fwd_layers: List[torch.nn.Module] = [
        #     self.conv1,
        #     self.bn1,
        #     self.relu,
        #     self.maxpool
        # ]

        def hook(grad):
            grads.insert(0, grad)
            # grads.append(grad)

        # x = self.conv1(x)
        # x.register_hook(hook)

        # x = self.bn1(x)
        # x.register_hook(hook)

        # x = self.relu(x)
        # x.register_hook(hook)

        # x = self.maxpool(x)
        # x.register_hook(hook)

        for v in self.first_block:
            fwd_layers.append(v)
            x = v(x)
            x.register_hook(hook)

        layer_sequentials = [self.layer1, self.layer2, self.layer3, self.layer4]

        _input = x
        for layer in layer_sequentials:
            for f_index, f in enumerate(layer):
                fwd_layers.append(f)
                features: torch.Tensor = f(_input)
                features.register_hook(hook)
                _input = features

        x = self.avgpool(features)
        # fwd_layers.append(self.avgpool)
        # x.register_hook(hook)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, grads, fwd_layers

    def forward(self, x: Tensor) -> Tensor:

        # See note [TorchScript super()]
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        for layer in self.first_block:
            x = layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNetBottleneck(ResNet):
    def __init__(
            self, block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups,
                         width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer)
        self._bottleneck_idx = None
        self._bottleneck_block = None
        self._bottleneck_inserted = False

    def forward(self, x: torch.Tensor, idx: Optional[int] = None):
        if idx is not None:
            return self.indexed_forward(x, idx)
        else:
            return super().forward(x)

    def indexed_forward(self, x: Tensor, idx: int):

        layers = [self.first_block, self.layer1, self.layer2, self.layer3, self.layer4]

        assert self._bottleneck_inserted, "Not implemented without a bottleneck"

        # TOFIX: accrocchio per farlo andare, per ora...
        idx -= 1
        block_n = self._bottleneck_block
        if idx < 0:
            assert self._bottleneck_block-1 >= 0, "NOT IMPLEMENTED BOTTLENECK 0"
            block_n = self._bottleneck_block - 1
            idx = len(layers[block_n])-1

        # IDX index is not important, just run up to bottleneck
        # TODO: per il futuro, cambiare l'indexed o cambiare struttura delle due reti. uniformare i layer in modo lineare

        # if idx != self._bottleneck_idx:
        #     max_n = len(layers[self._bottleneck_block])
        #     assert idx < max_n, "Index out of bounds. expected to be the bottleneck or in the same block. otherwise is not implemented yet"

        f1 = None

        # counter = 0
        _input = x
        for j, seq in enumerate(layers):
            for i, v in enumerate(seq):
                _input = v(_input)
                if j == block_n and i == idx:
                    f1 = _input
                # if not isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                #     counter += 1

        assert f1 is not None, "Unable to find the bottleneck"

        out_f = _input
        out_f = self.avgpool(out_f)
        out_f = torch.flatten(out_f, 1)
        out_f = self.fc(out_f)

        return out_f, f1

    def inject_bottleneck(
            self, insert_index, bottleneck_type: Type[CompressionBottleneck],
            skip_batchnorm: bool = True, skip_relu: bool = True, **kwargs):

        fake_xin = torch.rand((1, 3, 224, 224))

        found = False
        counter = 0
        insertion = 0
        block_n = 0
        layer_sequentials = [self.first_block, self.layer1, self.layer2, self.layer3, self.layer4]

        _input = fake_xin
        for j, sequential in enumerate(layer_sequentials):
            for i, v in enumerate(sequential):
                if skip_batchnorm and isinstance(v, torch.nn.BatchNorm2d):
                    continue
                if skip_relu and isinstance(v, torch.nn.ReLU):
                    continue
                if counter == insert_index:
                    found = True
                    insertion = i
                    block_n = j
                    break
                counter += 1
                _input = v(_input)
            if found:
                break

        if not found:
            raise NotImplementedError()

        # ho l'indice di inserimento

        # print(f"Block N = {block_n}; idx = {insertion}")

        # sequentials = layer_sequentials[:block_n+1]

        # _input = fake_xin
        # for j, seq in enumerate(sequentials):
        #     tmp_counter = 0
        #     for i, v in enumerate(seq):
        #         _input = v(_input)
        #         if isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
        #             continue
        #         if tmp_counter == insertion and j == block_n:
        #             break
        #         tmp_counter += 1

        shape = _input.shape[1:]

        ae: AutoEncoderUnderComplete = bottleneck_type(shape, kernel_size=3)

        tmp = list(layer_sequentials[block_n])
        # idx = insertion + 1  # insert after the requested idx
        idx = insertion  # insert exatcly where it was asked to
        tmp.insert(idx, ae)

        self._bottleneck_idx_abs = idx + sum([len(s) for s in layer_sequentials[:block_n]])
        self._bottleneck_idx = idx
        self._bottleneck_idx_in_block = idx
        self._bottleneck_block = block_n

        if block_n == 0:
            # this means the saved models will have a sequential. in the original model is a list
            self.first_block = nn.Sequential(*tmp)
        elif block_n == 1:
            self.layer1 = nn.Sequential(*tmp)
        elif block_n == 2:
            self.layer2 = nn.Sequential(*tmp)
        elif block_n == 3:
            self.layer3 = nn.Sequential(*tmp)
        elif block_n == 4:
            self.layer4 = nn.Sequential(*tmp)
        else:
            raise NotImplementedError()

        self._bottleneck_inserted = True
        if block_n != 0:
            self.first_block = torch.nn.Sequential(*self.first_block)  # anyway transform it to a sequential after insertion of bottleneck

    def freeze(self, exclude_bottleneck=True):
        """
        Freeze all the weights
        """
        idx = self._bottleneck_idx
        layer_sequentials = [self.first_block, self.layer1, self.layer2, self.layer3, self.layer4]

        for j, sequential in enumerate(layer_sequentials):
            for i, module in enumerate(sequential):
                if j == self._bottleneck_block and i == idx and exclude_bottleneck:
                    continue
                if hasattr(module, "weight"):
                    module.weight.requires_grad = False
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        module.bias.requires_grad = False
                if hasattr(module, "freeze") and not isinstance(module, CompressionBottleneck):
                    module.freeze()
                if isinstance(module, CompressionBottleneck) and not exclude_bottleneck:
                    module.freeze()

        for module in [self.fc]:
            if hasattr(module, "weight"):
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                module.bias.requires_grad = False

    def unfreeze(self):
        for n, p in self.named_parameters():
            p.requires_grad = True


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def _resnet_bottleneck(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNetBottleneck(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_bottleneck(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_bottleneck("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def __test_bottleneck__():
    from models.bottlenecks.undercomplete_autoencoder import AutoEncoderUnderComplete
    xin = torch.rand((1, 3, 224, 224))
    for i in range(2, 18):
        print(f"Test insertion idx={i}")
        m: ResNetBottleneck = resnet50_bottleneck()
        m.inject_bottleneck(i, AutoEncoderUnderComplete)

        print(m)

        m(xin)

        m.forward_with_grads(xin)

        print("Successful inserted bottleneck!")


if __name__ == "__main__":
    __test_bottleneck__()
