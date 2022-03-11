# -*- coding: utf-8 -*-
"""
    define network module
"""
import torch
from torch import nn
from torchvision.models.squeezenet import Fire


class IModule(nn.Module):
    """ I_Module for VoNet """
    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super(IModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)


class FModule(nn.Module):
    """ F_Module for VoNet """
    def __init__(self, in_dim: int, out_dim: int):
        super(FModule, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        return x

class VoNet(nn.Module):
    """ カラー対応版VoNet """
    def __init__(self):
        super(VoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            IModule(in_dim=64, h_dim=96, out_dim=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(256, 256),
            FModule(256, 384),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(384, 384),
            FModule(384, 512),
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.25),
            #nn.Conv2d(1024, 1536, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Softmax(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class VoNet_origin(nn.Module):
    """ カラー対応版VoNet """
    def __init__(self):
        super(VoNet_origin, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            IModule(in_dim=64, h_dim=96, out_dim=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(256, 256),
            FModule(256, 384),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(384, 384),
            FModule(384, 512),
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            #nn.Softmax()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
        #x = torch.flatten(x, 1)
        #x = self.LinearDetection(x)
        #return x

class VoNetSingleCh(nn.Module):
    """ グレースケール対応版VoNet """
    def __init__(self):
        super(VoNetSingleCh, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            IModule(in_dim=64, h_dim=96, out_dim=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(256, 256),
            FModule(256, 384),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(384, 384),
            FModule(384, 512),
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class SqueezeNetSingleCh(nn.Module):
    """ グレースケール画像対応版SqueezeNet """
    def __init__(self, num_classes=6):
        super(SqueezeNetSingleCh, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
