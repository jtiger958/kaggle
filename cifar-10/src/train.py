from model.transfer_learning.resnet.resnet18 import Resnet18
import torch
import os
from glob import glob
import torch.nn as nn
from visdom import Visdom

from utils.utils import AverageMeter, LambdaLR


class Trainer:
    def __init__(self, config, train_loader, num_class=10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.num_class = num_class
        self.learning_rate = config.lr
        self.train_loader = train_loader
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.decay_epoch = config.decay_epoch
        self.loss_dir = config.loss_dir
        self.visdom = Visdom()

        self.build_model()

    def build_model(self):
        self.net = Resnet18(self.num_class)
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        model = glob(os.path.join(self.checkpoint_dir, f"checkpoint-{self.epoch-1}.pth"))

        if not model:
            print("[!] No checkpoint in ", str(self.checkpoint_dir))
            return

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print("[*] Load Model from %s: " % str(self.checkpoint_dir), str(model[-1]))

    def read_loss_info(self):
        train_accuracy_path = glob(os.path.join(self.loss_dir, "train_accuracy.txt"))
        train_loss_path = glob(os.path.join(self.loss_dir, "train_loss.txt"))
        learning_rate_path = glob(os.path.join(self.loss_dir, "learning_rate_info.txt"))
        epoch_info_path = glob(os.path.join(self.loss_dir, "epoch_info.txt"))

        if not train_loss_path:
            return [], [], [], []

        accuracy_file = open(train_accuracy_path[0], 'r')
        loss_file = open(train_loss_path[0], 'r')
        learning_rate_file = open(learning_rate_path[0], 'r')
        epoch_file = open(epoch_info_path[0], 'r')

        accuracy = accuracy_file.readline().split(' ')[:-1]
        loss = loss_file.readline().split(' ')[:-1]
        learning_rate = learning_rate_file.readline().split(' ')[:-1]
        epoch = epoch_file.readline().split(' ')[:-1]

        accuracy = [float(accuracy_item) for accuracy_item in accuracy]
        loss = [float(loss_item) for loss_item in loss]
        learning_rate = [float(learning_rate_item) for learning_rate_item in learning_rate]
        epoch = [int(epoch_item) for epoch_item in epoch]

        accuracy_file.close()
        loss_file.close()
        learning_rate_file.close()
        epoch_file.close()

        return accuracy, loss, learning_rate, epoch

    def save_loss_info(self, accuracy, loss, lr, epoch):
        train_accuracy_path = os.path.join(self.loss_dir, "train_accuracy.txt")
        train_loss_path = os.path.join(self.loss_dir, "train_loss.txt")
        learning_rate_path = os.path.join(self.loss_dir, "learning_rate_info.txt")
        epoch_info_path = os.path.join(self.loss_dir, "epoch_info.txt")

        if not os.path.exists(self.loss_dir):
            os.makedirs(self.loss_dir)

        accuracy_file = open(train_accuracy_path, 'w')
        loss_file = open(train_loss_path, 'w')
        learning_rate_file = open(learning_rate_path, 'w')
        epoch_info_file = open(epoch_info_path, 'w')

        for accuracy_item in accuracy:
            accuracy_file.write(f"{accuracy_item} ")

        for loss_item in loss:
            loss_file.write(f"{loss_item} ")

        for lr_item in lr:
            learning_rate_file.write(f"{lr_item} ")

        for epoch_item in epoch:
            epoch_info_file.write(f"{epoch_item} ")

        accuracy_file.close()
        loss_file.close()
        learning_rate_file.close()
        epoch_info_file.close()

    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         LambdaLR(self.num_epoch, self.epoch,
                                                                  self.decay_epoch).step)

        total_step = len(self.train_loader)
        losses = AverageMeter()
        accuracy = AverageMeter()
        accuracy_set, loss_set, lr_set, epoch_set = self.read_loss_info()

        loss_window = self.visdom.line(Y=[1])
        lr_window = self.visdom.line(Y=[1])
        accuracy_window = self.visdom.line(Y=[1])

        for epoch in range(self.epoch, self.num_epoch):
            losses.reset()
            for step, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                predicted = (predicted == labels).sum().item()

                losses.update(loss.item(), self.batch_size)
                accuracy.update(predicted/self.batch_size, self.batch_size)

                if step % 10 == 0:
                    print(f'Epoch [{epoch}/{self.num_epoch}], Step [{step}/{total_step}], Loss: {losses.avg:.4f}, '
                          f'Accuracy: {accuracy.avg:.4f}')

            accuracy_set += [accuracy.avg]
            loss_set += [losses.avg]
            lr_set += [optimizer.param_groups[0]['lr']]
            epoch_set += [epoch]
            loss_window = self.visdom.line(Y=loss_set, X=epoch_set, win=loss_window, update='replace')
            lr_window = self.visdom.line(Y=lr_set, X=epoch_set, win=lr_window, update='replace')
            accuracy_window = self.visdom.line(Y=accuracy_set, X=epoch_set, win=accuracy_window, update='replace')

            self.save_loss_info(accuracy_set,loss_set, lr_set, epoch_set)
            torch.save(self.net.state_dict(), '%s/checkpoint-%d.pth' % (self.checkpoint_dir, epoch))
            lr_scheduler.step()
