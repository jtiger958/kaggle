import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms
from torch.utils.data.dataset import random_split

def load_data(path='train.csv'):
    data = pd.read_csv(path).values
    return data[:, 0], data[:, 1:]


class _Dataset(Dataset):
    def __init__(self, path, image_size=28):
        self.path = path
        self.image_size = image_size
        self.data = pd.read_csv(self.path)

    def __getitem__(self, index):
        self.image = self.data.iloc[index, 1:].as_matrix().reshape(self.image_size, self.image_size)
        self.num = np.array(self.data.iloc[index, 0])
        return np.expand_dims(self.image, axis=0), self.num

    def __len__(self):
        return len(self.data)


class __Dataset(Dataset):
    def __init__(self, path, image_size=28):
        self.path = path
        self.image_size = image_size
        self.data = pd.read_csv(self.path)

    def __getitem__(self, index):
        self.image = self.data.iloc[index, :].as_matrix().reshape(self.image_size, self.image_size)
        return np.expand_dims(self.image, axis=0)

    def __len__(self):
        return len(self.data)


def get_loader(data_folder='train.csv', batch_size=4, image_size=28, shuffle=False, num_workers=0):
    train_dataset = _Dataset(data_folder, image_size)
    answer_dataset = __Dataset('test.csv', image_size)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    answer_loader = DataLoader(dataset=answer_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, answer_loader


def get_mnist_loader(data_dir='dataset', batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])
    return DataLoader(dataset=MNIST(root=data_dir, train=True, transform=transform, download=True), batch_size=batch_size,
                      shuffle=True, num_workers=num_workers)