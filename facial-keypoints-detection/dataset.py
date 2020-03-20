import torch.utils.data
from PIL import Image
import os
from glob import glob
from torchvision import transforms
import pandas as pd
import csv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size):
        self.image_size = image_size
        self.label_df = pd.read_csv(os.path.join(f'{data_dir}', 'label.csv'))
        self.num_row, self.num_col = self.label_df.shape
        self.data_dir = data_dir

    def __getitem__(self, item):
        image: Image.Image = Image.open(os.path.join(f'{self.data_dir}', f'{item}.png')).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        return self.label_df.iloc[item], transform(image)

    def __len__(self):
        return self.num_row

    def get_num_col(self):
        return self.num_col

    def get_num_row(self):
        return self.num_col


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size):
        self.files = glob(os.path.join(f'{data_dir}', '*.jpeg'))
        self.image_size = image_size

    def __getitem__(self, item):
        image: Image.Image = Image.open(self.files[item]).convert('RGB')
        index = self.files[item].split('\\')[-1].split('.')[0]
        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        return index, transform(image)

    def __len__(self):
        return len(self.files)


def get_loader(data_dir, image_size, batch_size):
    dataset = Dataset(data_dir, image_size)
    return torch.utils.data.DataLoader(dataset, batch_size, True)


def get_test_loader(data_dir, image_size, batch_size):
    dataset = TestDataset(data_dir, image_size)
    return torch.utils.data.DataLoader(dataset, batch_size)
