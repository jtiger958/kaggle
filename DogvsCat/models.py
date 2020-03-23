import torch
import torch.nn as nn
from torchvision.models.vgg import vgg11
from tqdm import tqdm
from utils import AverageMeter


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.net = vgg11(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
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

            pred = pred >= 0.5
            loss_avg.update(loss.item())
            num_correct = (pred == label).sum().item()
            acc_avg.update(num_correct/image.shape[0])

        return loss_avg.avg, acc_avg.avg

    def validate_model(self, data_loader, criterion):
        with torch.no_grad():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.eval()
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

                pred = pred >= 0.5
                loss_avg.update(loss.item())
                num_correct = (pred == label).sum().item()
                acc_avg.update(num_correct / image.shape[0])

            return loss_avg.avg, acc_avg.avg

    def predict_image(self, image: torch.Tensor):
        with torch.no_grad():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.eval()
            self.to(device)

            label: torch.Tensor = self.net(image)
            # label = nn.functional.softmax(label, dim=1)
            return label
