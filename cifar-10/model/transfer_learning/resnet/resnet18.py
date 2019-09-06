import torch.nn as nn
from torchvision.models.resnet import resnet18


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()

        self.net = resnet18(pretrained=True)

        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.net(x)
