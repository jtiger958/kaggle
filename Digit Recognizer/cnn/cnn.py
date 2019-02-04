from data_loader import load_data, get_loader
from model import Model
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch
from glob import glob
import os

num, image = load_data()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = 0.001
        self.epoch = 30
        self.batch_size = 32
        self.optimizer = optim.Adam
        self.model_path = './checkpoint'
        self.train_loader, self.test_loader, self.answer_loader = get_loader(batch_size=self.batch_size)
        self.build_model()

    def build_model(self):
        self.net = Model()
        self.net.to(self.device)
        os.makedirs(self.model_path, exist_ok=True)
        if self.model_path != '':
            self.load_model()

    def load_model(self):
        print("[*] Load models from {}...".format(self.model_path))

        paths = glob(os.path.join(self.model_path, 'CNN*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.model_path))
            return

        filename = paths[-1]
        self.net.load_state_dict(torch.load(filename, map_location=self.device))
        print("[*] Model loaded: {}".format(filename))

    def train(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-7)
        criterion = CrossEntropyLoss()

        for epoch in range(self.epoch):
            for step, (image, num) in enumerate(self.train_loader):

                target = num.to(self.device)
                image = image.to(self.device).float()

                pred = self.net(image)
                loss = criterion(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('[%d][%d] loss:%.6f' % (epoch, step, loss))
            torch.save(self.net.state_dict(), '%s/CNN_epoch_%d.pth' % (self.model_path, epoch))

    def test(self):
        correct = 0
        total = 0
        for step, (image, num) in enumerate(self.test_loader):
            target = num.to(self.device)
            image = image.to(self.device).float()

            pred = self.net(image)

            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))

    def answer(self):
        f = open("answer.csv", 'w')
        f.write('ImageId,Label\n')
        index = 1
        for step, image in enumerate(self.answer_loader):
            image = image.to(self.device).float()
            pred = self.net(image)
            _, predicted = torch.max(pred.data, 1)
            answer = predicted.cpu().numpy()

            for num in answer:
                data = '%d,%d\n' % (index, num)
                f.write(data)
                index = index + 1
        f.close()




trainer = Trainer()
trainer.train()
trainer.test()
trainer.answer()