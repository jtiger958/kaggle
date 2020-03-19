import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from tqdm import tqdm
from utils import AverageMeter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = resnet50(pretrained=True)
        num_features = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.Linear(4096, 1),
            nn.Hardtanh(min_val=0, max_val=4)
        )

    def forward(self, x):
        return self.net(x)

    def train_model(self, data_loader, criterion, optimizer):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train()
        self.to(device)
        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        loss_avg.reset()
        acc_avg.reset()

        for label, image in tqdm(data_loader):
            image: torch.Tensor = image.to(device)
            label: torch.Tensor = label.to(device).float().unsqueeze(dim=1)

            pred: torch.Tensor = self.forward(image)

            loss: torch.Tensor = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())
            pred = (pred + 0.5).long()
            num_correct = (pred == label).sum().item()
            acc_avg.update(num_correct/image.shape[0])

        print(f'Average [Accuracy: {acc_avg.avg:.8f}][Loss: {loss_avg.avg:.8f}]')
        return loss_avg.avg

    def predict_image(self, image: torch.Tensor):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.eval()
        self.to(device)

        label: torch.Tensor = self.net(image)
        label = nn.functional.softmax(label, dim=1)
        return label
