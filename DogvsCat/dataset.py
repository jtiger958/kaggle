import torch.utils.data
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import random_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size):
        self.files = glob(os.path.join(f'{data_dir}', '*.jpg'))
        self.image_size = image_size
        self.files.sort()

    def __getitem__(self, item):
        image: Image.Image = Image.open(self.files[item]).convert('RGB')
        label = 0 if self.files[item].split('\\')[-1].split('.')[-3] == 'cat' else 1

        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation((-2, 2)),
            transforms.ToTensor(),
            # transforms.RandomErasing(),
        ])

        return label, transform(image)

    def __len__(self):
        return len(self.files)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size):
        self.files = glob(os.path.join(f'{data_dir}', '*.jpg'))
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

    train_length = int(0.9 * len(dataset))
    validation_length = len(dataset) - train_length

    train_dataset, validation_dataset = random_split(dataset, (train_length, validation_length))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size, False)
    return train_loader, validation_loader


def get_test_loader(data_dir, image_size, batch_size):
    dataset = TestDataset(data_dir, image_size)
    return torch.utils.data.DataLoader(dataset, batch_size)
