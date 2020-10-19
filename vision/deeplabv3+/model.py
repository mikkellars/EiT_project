"""
"""


import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


def deeplabv3_resnet(bb_type:str='resnet50', n_classes:int=1, pretrained:bool=True,
                     progress:bool=True, aux_loss=None, **kwargs):
    """DeepLabV3+ with ResNet50.

    Args:
        bb_type (str, optional): Name of backbone architecture. Defaults to 'resnet50'.
        n_classes (int, optional): Number of classes. Defaults to 1.
        pretrained (bool, optional): Use pretrained weights. Defaults to True.
        progress (bool, optional): Show progress of downloading weights. Defaults to True.
        aux_loss ([type], optional): [description]. Defaults to None.

    Returns:
        nn.Module: DeepLabV3+ with Resnet backbone model.
    """

    if pretrained: aux_loss = True

    if bb_type == 'resnet50':
        backbone = torchvision.models.resnet50(pretrained, progress)
        inplanes = 2048
        in_aux = 1024
    elif bb_type == 'resnet18':
        backbone = torchvision.models.resnet18(pretrained, progress)
        inplanes = 512
        in_aux = 256
    elif bb_type == 'resnet34':
        backbone = torchvision.models.resnet34(pretrained, progress)
        inplanes = 512
        in_aux = 256
    elif bb_type == 'resnet101': 
        backbone = torchvision.models.resnet101(pretrained, progress)
        inplanes = 2048
        in_aux = 1024
    else: raise ValueError()

    return_layers = {'layer4': 'out'}
    if aux_loss: return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers)

    aux_classifier = None
    if aux_loss: aux_classifier = FCNHead(in_aux, n_classes)

    classifier = DeepLabHead(inplanes, n_classes)
    
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model


class DeepLabV3(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None):

        super(DeepLabV3, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        size = x.shape[-2:]

        features = self.backbone(x)

        ret = OrderedDict()

        x = features['out']
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        ret['out'] = x

        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            ret['aux'] = x

        return ret['out']


class FCNHead(nn.Sequential):
    """FCN Head.

    Args:
        in_channels (int): In channels.
        channels (int): Channels.
    """

    def __init__(self, in_channels: int, channels: int):

        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class DeepLabHead(nn.Sequential):
    """Deep Lab Head.

    Args:
        in_channels (int): In channels.
        n_classes (int): Number of classes.
    """

    def __init__(self, in_channels: int, n_classes: int):

        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, 1),
            nn.Tanh()
        )


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, dilation):

        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]

        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int):

        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class ASPP(nn.Module):

    def __init__(self, in_channels: int, atrous_rate):

        super(ASPP, self).__init__()

        out_channels = 256
        
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        rate1, rate2, rate3 = tuple(atrous_rate)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        ret = [conv(x) for conv in self.convs]
        ret = torch.cat(ret, dim=1)
        ret = self.project(ret)
        return ret


class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    def show(img):
        np_img = img.detach().numpy()
        plt.imshow(np.transpose(np_img, (1,2,0)), interpolation='nearest')
        plt.axis('off')
        plt.show()
    
    model = deeplabv3_resnet('resnet18')
    model.eval()

    times = list()

    inputs = [torch.rand(3, 256, 256) for _ in range(5)]
    show(make_grid(inputs, padding=10))

    for inp in inputs:
        inp = inp.unsqueeze(0)

        start_time = time.time()
        pred = model(inp)
        times.append((time.time() - start_time))

    print(f'Done! It took {np.mean(times):.4f} seconds to predict each image.')
