#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import torch.utils.model_zoo as model_zoo

from PIL.Image import Image
from typing import List, Optional, Tuple, Type

from models.bottlenecks.base import CompressionBottleneck


# VGG models.
__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
    "vgg19_bn_bottleneck_conv",
]


model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(self, features: torch.nn.Sequential, num_classes: int = 1000, init_weights: bool = True):
        super(VGG, self).__init__()

        self._init_weights = init_weights
        self.num_classes = num_classes
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    # method for the activation exctraction
    def get_activations(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.features[:idx](x)

    def get_saliency_from_layer(self, img: Image, layer_idx: int, method: str = "gradients"):
        methods = ["gradients", "gradcam"]
        if method not in methods:
            raise RuntimeError(f"Saliency method not supported: {method}")

        size = 224
        preprocess = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x[None]),
        ])

        deprocess = T.Compose([
            # Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
            #   => Y/(1/σ) follows Distribution(0,σ)
            #   => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
            T.Lambda(lambda x: x[0]),
            T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            T.ToPILImage(),
        ])

        if method == "gradients":
            print("Ignoring idx")
            # raw gradients extraction
            # preprocess the image
            X = preprocess(img)

            # we would run the model in evaluation mode
            model.eval()

            # we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
            X.requires_grad_()

            scores = self.forward(X)

            # Get the index corresponding to the maximum score and the maximum score itself.
            score_max_index = scores.argmax()
            score_max = scores[0, score_max_index]

            score_max.backward()

            """
            Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
            R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
            across all colour channels.
            """
            saliency, _ = torch.max(X.grad.data.abs(), dim=1)

            from matplotlib import pyplot as plt
            plt.imshow(saliency[0], cmap=plt.cm.hot)
            plt.axis("off")
            plt.show()
        elif method == "gradcam":
            pass
        else:
            raise NotImplementedError

    def get_activations_image(self, img, max_batch_size=20, show=False):

        # https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

        self.eval()

        # batching to increase the error
        if max_batch_size > 1:
            device = img.device
            if len(img.size()) == 4:
                tmp = img.squeeze(0)
            else:
                tmp = img
            tmp = tmp.detach().cpu().numpy()
            tmp = np.repeat(tmp[None, ...], max_batch_size, 0)
            img_in = torch.from_numpy(tmp).to(device)
        else:
            img_in = img

        pred = self(img_in)

        if max_batch_size > 1:
            pred = pred[0, :].unsqueeze(0)  # 1xN

        # get the gradient of the output with respect to the parameters of the model
        if pred.numel() > 1:
            import warnings
            warnings.warn("Using argmax")
            pred = pred[0, pred.argmax(dim=1)]
        pred.backward()

        # pull the gradients out of the model
        gradients = self.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach()

        # weight the channels by corresponding gradients
        chs = activations.shape[1]
        for i in range(chs):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        heatmap = torch.from_numpy(heatmap)
        # normalize the heatmap
        heatmap /= torch.max(heatmap) if torch.max(heatmap) != 0 else 1

        if show:
            # draw the heatmap
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(heatmap.squeeze())
            plt.pause(0.5)

        return heatmap

    def set_num_class(self, num_classes: int = 1, copy_weights: bool = True) -> None:
        inshape = 512 * 7 * 7  # (with input 224x224)
        # inshape = 512 * 8 * 8  # (with input 256x256)
        # inshape = 1024 * 2 * 8 * 8  # (with input 512x512)
        new_c = nn.Sequential(
            nn.Linear(inshape, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        for m in self.classifier:
            if not hasattr(m, "weight"):
                continue

            for k in new_c:
                if not hasattr(k, "weight"):
                    continue
                if m.weight.shape == k.weight.shape and copy_weights:
                    k.weight = m.weight
                else:
                    self._initialize_weights([k])

        self.classifier = new_c

        # if self._init_weights:
        # self._initialize_weights(self.classifier)
        self.num_classes = num_classes
        return self

    def _validate_net(self):
        shape = (3, 224, 224)
        xin = torch.randn(shape).unsqueeze(0)
        try:
            self.forward(xin)
        except Exception as e:
            raise e

    def forward(self, x):
        features = self.features(x)

        features = self.avgpool(features)
        linear_features = features.view(features.size(0), -1)

        res = self.classifier(linear_features)

        return res

    def forward_with_grads(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.nn.Module]]:
        grads = []  # this variable will be filled only AFTER backward()

        def hook(grad):
            grads.insert(0, grad)
            # grads.append(grad)

        _input = x
        for f_index, f in enumerate(self.features):
            features: torch.Tensor = f(_input)
            features.register_hook(hook)
            _input = features

        features = self.avgpool(features)
        linear_features = features.view(features.size(0), -1)

        res = self.classifier(linear_features)

        fwd_layers = [l for l in self.features]

        return res, grads, fwd_layers

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

    def get_excluded_layers(self) -> List[int]:
        res = []
        for i, v in enumerate(self.features):
            if isinstance(v, (torch.nn.ReLU, torch.nn.BatchNorm2d)):
                res.append(i)
        return res

    def get_layer_names(self, include_input=True) -> List[int]:
        layer_ticknames = [
            "input_1",
            "block1_conv1",
            "block1_conv2",
            "block1_pool",
            "block2_conv1",
            "block2_conv2",
            "block2_pool",
            "block3_conv1",
            "block3_conv2",
            "block3_conv3",
            "block3_pool",
            "block4_conv1",
            "block4_conv2",
            "block4_conv3",
            "block4_pool",
            "block5_conv1",
            "block5_conv2",
            "block5_conv3",
            "block5_pool",
            # "flatten"
            # "fc1"
            # "fc2"
            # "predictions"
        ]
        return layer_ticknames[1:] if not include_input else layer_ticknames

    def get_sizes(self, x, exlcude_batchnorm: bool = True, exclude_relu: bool = True, include_input=True, include_bottlenecks_deep=True):
        sizes = [x.shape] if include_input else []

        _input = x
        for i, v in enumerate(self.features):
            _input = v(_input)
            if exlcude_batchnorm and isinstance(v, torch.nn.BatchNorm2d):
                continue
            if exclude_relu and isinstance(v, torch.nn.ReLU):
                continue
            
            if include_bottlenecks_deep and isinstance(v, CompressionBottleneck):
                sizes.append(v.get_split_size(_input))
            else:
                sizes.append(_input.shape)

        return sizes


class VGGBottleneck(VGG):
    def __init__(self, features: torch.nn.Sequential, num_classes: int = 1000, init_weights: bool = True):
        super().__init__(features, num_classes=num_classes, init_weights=init_weights)

        self._init_weights = init_weights
        self.num_classes = num_classes
        self.features = features

        # placeholder for bottleneck
        self._bottleneck_inserted = False
        self._bottleneck_idx = lambda x: [
            idx for idx, _ in enumerate(x) if isinstance(_, CompressionBottleneck)
        ]

        # placeholder for the gradients
        self.gradients = None
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def freeze(self, exclude_bottleneck=True):
        """
        Freeze all the weights
        """
        idxs = self._bottleneck_idx(self.features)
        for i, module in enumerate(self.features):
            if i in idxs and exclude_bottleneck:
                continue
            if hasattr(module, "weight"):
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                module.bias.requires_grad = False
            if isinstance(module, CompressionBottleneck) and not exclude_bottleneck:
                module.freeze()

        for module in self.classifier:
            if hasattr(module, "weight"):
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                module.bias.requires_grad = False

    def fine_tuning_freeze(self):
        """
        Freeze all the weights
        """
        idxs = self._bottleneck_idx(self.features)
        for i, module in enumerate(self.features):
        
            if hasattr(module, "weight"):
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                module.bias.requires_grad = False
            if isinstance(module, CompressionBottleneck):
                module.freeze()

        for module in self.classifier:
            if hasattr(module, "weight"):
                if module.weight.shape == (self.num_classes, 4096):
                    continue
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                module.bias.requires_grad = False

    def unfreeze(self):
        for n, p in self.named_parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        if self._bottleneck_inserted:
            idxs: List[int] = self._bottleneck_idx(self.features)
            if len(idxs) != 1:  # both less and greater than
                raise NotImplementedError
            idx = idxs[0]
            if idx > len(self.features) - 1:
                # get activation of last layer of bottleneck... even if it"s not right probably
                print("aaaaaaaaaaaaaaaaaa not implementedddd fffffffffff")
                raise NotImplementedError
            else:
                up_to_bottleneck = self.features[0: idx + 1]
                remaining = self.features[idx + 1:]

                f1 = up_to_bottleneck(x)
                return f1
        else:
            return self.features[:36](x)

    def get_bottleneck_idx(self):
        return self._bottleneck_idx(self.features)

    def get_activations_image(self, img, max_batch_size=20, show=False):

        # https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

        self.eval()

        # batching to increase the error
        if max_batch_size > 1:
            device = img.device
            if len(img.size()) == 4:
                tmp = img.squeeze(0)
            else:
                tmp = img
            tmp = tmp.detach().cpu().numpy()
            tmp = np.repeat(tmp[None, ...], max_batch_size, 0)
            img_in = torch.from_numpy(tmp).to(device)
        else:
            img_in = img

        pred = self(img_in)

        if max_batch_size > 1:
            pred = pred[0, :].unsqueeze(0)  # 1xN

        # get the gradient of the output with respect to the parameters of the model
        if pred.numel() > 1:
            import warnings
            warnings.warn("Using argmax")
            pred = pred[0, pred.argmax(dim=1)]
        pred.backward()

        # pull the gradients out of the model
        gradients = self.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach()

        # weight the channels by corresponding gradients
        chs = activations.shape[1]
        for i in range(chs):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        heatmap = torch.from_numpy(heatmap)
        # normalize the heatmap
        heatmap /= torch.max(heatmap) if torch.max(heatmap) != 0 else 1

        if show:
            # draw the heatmap
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(heatmap.squeeze())
            plt.pause(0.5)

        return heatmap

    def set_num_class(self, num_classes: int = 1, copy_weights: bool = True) -> None:
        inshape = 512 * 7 * 7  # (with input 224x224)
        # inshape = 512 * 8 * 8  # (with input 256x256)
        # inshape = 1024 * 2 * 8 * 8  # (with input 512x512)
        new_c = nn.Sequential(
            nn.Linear(inshape, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        for m in self.classifier:
            if not hasattr(m, "weight"):
                continue

            for k in new_c:
                if not hasattr(k, "weight"):
                    continue
                if m.weight.shape == k.weight.shape and copy_weights:
                    k.weight = m.weight
                else:
                    self._initialize_weights([k])

        self.classifier = new_c

        # if self._init_weights:
        # self._initialize_weights(self.classifier)
        self.num_classes = num_classes
        return self

    def _gen_cfg(self):
        res = []
        for i, m in enumerate(self.features):
            if isinstance(m, nn.modules.BatchNorm2d) or isinstance(m, nn.modules.ReLU):
                continue
            if isinstance(m, nn.modules.MaxPool2d):
                res.append("M")
            elif isinstance(m, nn.modules.Conv2d):
                res.append(m.out_channels)
            elif isinstance(m, CompressionBottleneck):
                res.append("B_conv")

        return res

    def inject_bottleneck(self, configuration, bottleneck_type: Type[CompressionBottleneck], **kwargs):

        if sum([1 for v in configuration if v == "B_conv"]) == 1 and configuration[-1] == "B_conv":
            fake_xin = torch.rand((1, 3, 224, 224))
            feats = self.features(fake_xin).squeeze(0).shape
            new_f = list(self.features)
            new_f.append(
                bottleneck_type(
                    input_shape=feats,
                    # n_filters=filters // 2,
                    # n_filters=self.features[-3].out_channels,
                    n_filters=512,
                    **kwargs
                )
            )
        else:
            new_f = []
            c_idx = 0
            bn_in_ch: Optional[int] = None
            last_conv_layer: Optional[nn.Conv2d] = None
            for i, m in enumerate(self.features):
                if isinstance(m, nn.modules.BatchNorm2d) or isinstance(m, nn.modules.ReLU):
                    new_f.append(m)
                    continue

                expected = configuration[c_idx]

                # print(expected, m._get_name(), m)

                expected_type = None

                if expected == "B_conv":
                    # print("SKIPPING / INSERTING Bottleneck")

                    prev_idx = c_idx - 1
                    if prev_idx < 0:
                        # if previous is negative, use the next
                        # prev = "__NEXT__"  # to trigger the "try next" mode
                        prev = "__NEXT__"
                        bn_in_ch = 3  # IS THE FIRST LAYER
                    else:
                        prev = configuration[c_idx - 1]

                    if isinstance(prev, int):
                        filters = prev
                    elif prev == "M":
                        # Get number
                        # ----------------------
                        # Let F = 2 (2x2 window)
                        # Stride, S = 2
                        # Depth, D = 5 (depth from the previous layer)
                        # [(I - F) / S] + 1 x D
                        # i.e.
                        # [(30 – 2) / 2] + 1 x D
                        # Hin = last_conv_layer.out_channels
                        # Padding = 0
                        # Dilation = 1
                        # KernelSize = 2
                        # Stride = 2
                        # filters = floor(
                        #     ((Hin + 2 * Padding - Dilation * (KernelSize - 1)) / Stride) + 1
                        # )

                        filters = last_conv_layer.out_channels

                        # fake_xin = torch.rand(1, 3, 224, 224)

                        # for jjj in range(len(self.features)):
                        #     tmp = self.features[:jjj+1](fake_xin)
                        #     s = tmp.shape
                        #     print(s)
                        #     if jjj > i:
                        #         break

                        # feats = self.features[:i+1]().squeeze(0).shape
                        # assert feats[1] == feats[2]
                        # filters = feats[1]  #

                    elif prev == "__NEXT__":
                        try:
                            next_p = configuration[c_idx + 1]
                            if isinstance(next_p, int):
                                filters = next_p
                            else:
                                raise RuntimeError(
                                    "Don't know how to append after a non-conv layer"
                                )
                        except IndexError:
                            raise RuntimeError("Index out of bounds for configuration")
                    else:
                        raise NotImplementedError(
                            f"This is not supported (previous = '{prev}')"
                        )

                    # Currently downsampling to half. Other parameters are default
                    feats = self.features[:i](torch.rand(1, 3, 224, 224)).squeeze(0).shape
                    if bn_in_ch is None:
                        bneck = bottleneck_type(
                            input_shape=feats,
                            # n_filters=filters // 2,
                            # n_filters=filters,
                            **kwargs
                        )
                    else:
                        bneck = bottleneck_type(
                            # input_shape=bn_in_ch,
                            input_shape=feats,
                            # n_filters=filters,
                            **kwargs
                        )
                        bn_in_ch = None

                    new_f.append(bneck)  # adding bneck
                    new_f.append(m)  # and current module

                    c_idx += 2  # bneck + current

                elif expected == "M":
                    expected_type = nn.modules.MaxPool2d
                    new_f.append(m)
                    c_idx += 1
                elif isinstance(expected, int):
                    expected_type = nn.modules.Conv2d
                    new_f.append(m)
                    last_conv_layer = m
                    # if batch_norm:
                    #     clock += 2
                    # else:
                    #     clock += 1
                    c_idx += 1
                else:
                    raise NotImplementedError()

                if expected_type is not None:
                    e = isinstance(m, expected_type)
                    # print(e)
                    if not e:
                        raise RuntimeError(
                            "Wrong bottleneck insertion. This is probably due to code error"
                        )
                    if isinstance(expected, int):  # Conv2d layers
                        if m.out_channels != expected:
                            raise RuntimeError(
                                "Incompatible channel dimensions. This is probably due to code error"
                            )

        self._bottleneck_inserted = True
        new_features = nn.Sequential(*new_f)
        self.features = new_features

        validate = self._gen_cfg()
        try:
            for j, v in enumerate(configuration):
                # print(validate[j], v, " are equal? ", validate[j] == v)
                assert validate[j] == v, "Invalid value"
        except Exception as e:
            # handle
            raise e

        self._validate_net()

    def _validate_net(self):
        shape = (3, 224, 224)
        xin = torch.randn(shape).unsqueeze(0)
        # try:
        self.forward(xin)
        # except Exception as e:
        #     if self._bottleneck_inserted:
        #         idx = self._bottleneck_idx(self.features)[0]
        #         ae = self.features[idx]
        #         if not isinstance(ae, CompressionBottleneck):
        #             raise e
        #         fmap = self.features[0:idx](xin)
        #         ae.try_fix(fmap)
        #     else:
        #         raise e

    def forward(self, x, idx=None):
        if idx is not None:
            return self.indexed_forward(x, idx)

        if self._bottleneck_inserted:
            idxs: List[int] = self._bottleneck_idx(self.features)
            if len(idxs) != 1:  # both less and greater than
                raise NotImplementedError
            idx = idxs[0]
            if idx > len(self.features) - 1:
                # get activation of last layer of bottleneck... even if it"s not right probably
                print("aaaaaaaaaaaaaaaaaa not implementedddd fffffffffff")
                raise NotImplementedError
            else:
                up_to_bottleneck = self.features[0: idx + 1]
                remaining = self.features[idx + 1:]

                f1 = up_to_bottleneck(x)
                if f1.requires_grad:
                    try:
                        f1.register_hook(self.activations_hook)
                    except:
                        import warnings
                        warnings.warn("Unable to register gradient hook!")
                features = remaining(f1)
        else:
            try:
                # N.B. where to "grep" the gradients highly depends on how the net is created.
                # You need to get to the last conv2d
                idx = 0
                for j in reversed(range(len(self.features))):
                    if isinstance(self.features[j], torch.nn.Conv2d):
                        idx = j+1  # activation layer after conv2d (if bn, +1)
                        break
                if isinstance(self.features[idx], torch.nn.BatchNorm2d):
                    idx += 1
                if idx == len(self.features)-1:
                    raise NotImplementedError
                f1 = self.features[:idx+1](x)

                if f1.requires_grad:
                    # register the hook
                    f1.register_hook(self.activations_hook)

                # apply the remaining pooling
                features = self.features[idx+1:](f1)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                # maybe bottleneck has not been inserted for real.
                # raise RuntimeError("Did you set debug(model) ? You should not.")
                features = self.features(x)

        features = self.avgpool(features)
        linear_features = features.view(features.size(0), -1)
        # print(features.shape, linear_features.shape)
        res = self.classifier(linear_features)
        return res

    def indexed_forward(self, x, idx):
        layers_1 = self.features[:idx]
        layers_2 = self.features[idx:]

        if len(layers_1) == 0:
            raise NotImplementedError()

        if len(layers_2) == 0:
            f1 = layers_1(x)
            features = self.avgpool(f1)
        else:
            f1 = layers_1(x)
            features = layers_2(f1)
            features = self.avgpool(features)

        linear_features = features.view(features.size(0), -1)
        # print(features.shape, linear_features.shape)
        res = self.classifier(linear_features)
        return res, f1

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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# fmt: off
cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],

    "S1_v1": [64, 64, "M", 128, 128, "B_conv", "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    "S1_v2": [64, 64, "M", 128, 128, "M", 256, 256, "B_conv", 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    "S1_v3": [64, 64, "M", 128, 128, "M", "B_conv", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    "S1_v4": ["B_conv", 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
# fmt: on


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["A"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg11"]))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["A"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg11_bn"]))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["B"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg13"]))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["B"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg13_bn"]))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["D"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg16"]))
    return model


def vgg16_bottleneck(pretrained: bool, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGGBottleneck(make_layers(cfg["D"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg16"]))

    # if isinstance(bottleneck_version, str):
    #     bn = cfg[bottleneck_version]
    # else:
    #     bn = bottleneck_version

    # model.inject_bottleneck(bn, bottleneck_class)
    return model


def vgg16_bn(pretrained=False, bottleneck_config=None, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["D"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg16_bn"]))
    if isinstance(bottleneck_config, str):
        bn = cfg[bottleneck_config]
    else:
        bn = bottleneck_config

    model.inject_bottleneck(bn, batch_norm=False)
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["E"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg19"]))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["E"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg19_bn"]))
    return model


def vgg19_bn_bottleneck_conv(pretrained=False, bottleneck_version="S1_v3", **kwargs):
    """VGG 19-layer model (configuration "S1") with batch normalization and a
    Bottlneck which performs neural compression.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGGBottleneck(make_layers(cfg["E"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg19_bn"]))

    if isinstance(bottleneck_version, str):
        bn = cfg[bottleneck_version]
    else:
        bn = bottleneck_version

    model.inject_bottleneck(bn, batch_norm=True)
    return model


# if __name__ == "__main__":
#     import torch
#     import copy
#     from tqdm import tqdm
#     from src.evaluation.evaluation import activation_maps

#     dev, _ = torch_import()
#     dev = torch.device("cpu")
#     models = []

#     # models.append(vgg19_bn(pretrained=True))
#     # models.append(vgg19(pretrained=True))

#     print("Creating VGG configurations")
#     for i in tqdm(range(len(cfg["E"]))):
#         # if i < 11: continue
#         vgg_struct = copy.deepcopy(cfg["E"])
#         vgg_struct.insert(i, "B_conv")
#         models.append(vgg19_bn_bottleneck_conv(pretrained=True, bottleneck_version=vgg_struct))
#         # break

#     # models.append(vgg19_bn_bottleneck_conv(pretrained=True, bottleneck_version="S1_v1"))
#     # models.append(vgg19_bn_bottleneck_conv(pretrained=True, bottleneck_version="S1_v2"))
#     # models.append(vgg19_bn_bottleneck_conv(pretrained=True, bottleneck_version="S1_v3"))
#     # models.append(vgg19_bn_bottleneck_conv(pretrained=True, bottleneck_version="S1_v4"))
#     # models.append(vgg19_bn_bottleneck_conv(pretrained=True, bottleneck_version="E"))
#     for k, model in enumerate(tqdm(models)):
#         # fake(dev, model, enable_debug_shapes=False)
#         activation_maps(dev, model, k)


def __test_bottleneck__():
    from src.models.bottlenecks.undercomplete_autoencoder import AutoEncoderUnderComplete
    xin = torch.rand((1, 3, 224, 224))
    m: VGGBottleneck = vgg16_bottleneck(False)

    import copy
    for i in range(15, len(m.features)):
        print(f"Test insertion idx={i}")
        m: VGGBottleneck = vgg16_bottleneck(False)
        vgg_struct = copy.deepcopy(cfg["D"])
        vgg_struct.insert(i, "B_conv")
        m.inject_bottleneck(vgg_struct, AutoEncoderUnderComplete)

        print(m)

        m(xin)

        m.forward_with_grads(xin)

        print("Successful inserted bottleneck!")


if __name__ == "__main__":
    __test_bottleneck__()
