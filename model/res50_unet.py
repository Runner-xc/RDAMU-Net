import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from model.utils.modules import DoubleConv

class RES50_UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(RES50_UNet, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder1 = resnet.layer1  # 256
        self.encoder2 = resnet.layer2  # 512
        self.encoder3 = resnet.layer3  # 1024
        self.encoder4 = resnet.layer4  # 2048

        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = DoubleConv(2048, 512)
        self.decoder2 = DoubleConv(1024, 256)
        self.decoder3 = DoubleConv(512, 64)
        self.final_conv = nn.Conv2d(64, num_classes, 1, 1, 0)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        c = self.bottleneck(x5)

        d1 = self.up(c)
        d1 = torch.cat((d1, x4), dim=1)
        d1 = self.decoder1(d1)

        d2 = self.up(d1)
        d2 = torch.cat((d2, x3), dim=1)
        d2 = self.decoder2(d2)

        d3 = self.up(d2)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.decoder3(d3)
        out = self.final_conv(d3)
        return out
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
