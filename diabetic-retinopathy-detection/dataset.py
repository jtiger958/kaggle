import torch.utils.data
from PIL import Image
import os
from glob import glob
from torchvision import transforms
import pandas as pd
import csv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size):
        self.files = glob(os.path.join(f'{data_dir}', '*.jpeg'))
        self.image_size = image_size
        self.files.sort()
        self.label_df = pd.read_csv(os.path.join(f'{data_dir}', 'trainLabels.csv'))

        with open(os.path.join(f'dataset', 'train', 'trainLabels.csv')) as infile:
            reader = csv.reader(infile)
            self.label_dict = {rows[0]: rows[1] for rows in reader}

    def __getitem__(self, item):
        image: Image.Image = Image.open(self.files[item]).convert('RGB')
        image_id: str = self.files[item].split('\\')[-1].split('.')[-2] == 'cat'

        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(4),
            transforms.ToTensor(),
        ])

        return self.label_dict[image_id], transform(image)

    def __len__(self):
        return len(self.files)


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
