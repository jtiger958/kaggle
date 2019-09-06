from model.transfer_learning.resnet.resnet18 import Resnet18
import torch
import os
from glob import glob


class Tester:
    def __init__(self, config, answer_loader, num_class=10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.num_class = num_class
        self.learning_rate = config.lr
        self.answer_loader = answer_loader
        self.epoch = config.epoch
        self.category = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.build_model()

    def build_model(self):
        self.net = Resnet18(self.num_class)
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        if not os.listdir(self.checkpoint_dir):
            raise Exception("[!] No checkpoint in ", str(self.checkpoint_dir))

        model = glob(os.path.join(self.checkpoint_dir, "checkpoint-*.pth"))

        file_index = [int(index.split('-')[-1].split('.')[0]) for index in model]
        file_index.sort()
        path = model[0].split('-')[0] + '-' + f"{file_index[-1]}.pth"

        self.net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[*] Load Model from {path}")

    def test(self):

        answer_file = open('answer.csv', 'w')
        answer_file.write('id,label\n')

        for step, (images, image_file_index) in enumerate(self.answer_loader):
            images = images.to(self.device)
            image_file_index = image_file_index
            preds = self.net(images)

            _, predicted = torch.max(preds.data, 1)

            for index, label in enumerate(predicted):
                answer_file.write(f'{image_file_index[index]},{self.category[label]}\n')

            if step % 100 == 0:
                print('make answer... {}/{}'.format(step, len(self.answer_loader)))

