import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, in_chennel=1):
        super(Model, self).__init__()
        self.in_chennel = in_chennel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chennel, out_channels=32*in_chennel, stride=1,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32*in_chennel, out_channels=32*in_chennel, stride=1,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32*in_chennel, out_channels=32*in_chennel, stride=1,  kernel_size=3, padding=1),
            nn.BatchNorm2d(32*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2,stride=2, padding=1),

            nn.Conv2d(in_channels=32*in_chennel, out_channels=64*in_chennel, stride=1,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64*in_chennel, out_channels=32*in_chennel, stride=1,  kernel_size=3, padding=1),
            nn.BatchNorm2d(32*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(in_channels=32*in_chennel, out_channels=64*in_chennel, stride=1,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64*in_chennel, out_channels=32*in_chennel, stride=1,  kernel_size=3, padding=1),
            nn.BatchNorm2d(32*in_chennel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.layer = nn.Sequential(
            nn.Linear(in_features=in_chennel*32*5*5, out_features=20*in_chennel),
            nn.Linear(in_features=20 * in_chennel, out_features=10)
        )

    def forward(self, x):
        layer = self.layer1(x)
        print(layer.shape)
        layer = self.layer(layer.reshape(layer.size(0), -1))
        return layer
