import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, in_chennel):
        super(Model, self).__init__()
        self.in_chennel = in_chennel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chennel, out_channels=4*in_chennel, stride=2,  kernel_size=3),
            nn.BatchNorm2d(4*in_chennel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*in_chennel, out_channels=8*in_chennel, stride=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=8*in_chennel, out_channels=16*in_chennel, stride=1, kernel_size=3),
            nn.BatchNorm2d(16*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=16*in_chennel, out_channels=8*in_chennel, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chennel, out_channels=4*in_chennel, stride=2,  kernel_size=3),
            nn.BatchNorm2d(4*in_chennel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4*in_chennel, out_channels=8*in_chennel, stride=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=8*in_chennel, out_channels=8*in_chennel, stride=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=8*in_chennel, out_channels=8*in_chennel, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1*in_chennel, out_channels=4*in_chennel, stride=2,  kernel_size=3),
            nn.BatchNorm2d(4*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=4*in_chennel, out_channels=8*in_chennel, stride=1, kernel_size=3),
            nn.Conv2d(in_channels=8*in_chennel, out_channels=8*in_chennel, stride=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Conv2d(in_channels=8*in_chennel, out_channels=8*in_chennel, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True),
        )
        self.layer_cat = nn.Sequential(
            nn.Conv2d(in_channels=24*in_chennel, out_channels=16*in_chennel, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(16*in_chennel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16*in_chennel, out_channels=8*in_chennel, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(8*in_chennel),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Sequential(
            nn.Linear(in_features=in_chennel*5*5*8, out_features=20*in_chennel),
            nn.Linear(in_features=20 * in_chennel, out_features=10)
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(x)
        layer = torch.cat((layer1, layer2, layer3), 1)
        layer = self.layer_cat(layer)
        layer = self.layer(layer.reshape(layer.size(0), -1))
        return layer
