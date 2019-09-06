from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms


class TrainDatasets(Dataset):
    def __init__(self, data_dir, train_label_path, transform=None, image_size=224):
        self.transform = transform
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_label = {}
        self.label_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                            'horse': 7, 'ship': 8, 'truck': 9}

        with open(train_label_path) as f:
            for line in f.readlines():
                if line.startswith('id'):
                    continue
                line = line[:-1]
                file_index, label = line.split(',')
                self.image_label[file_index] = label

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        images = self.image_path[item]
        image_file_index = images.split('.')[0].split('/')[-1]
        label = self.label_index[self.image_label[image_file_index]]
        images = Image.open(images).convert('RGB')

        if self.transform is not None:
            images = self.transform(images)
        else:
            images = transforms.ToTensor()(images)

        return images, label

    def __len__(self):
        return len(self.image_path)


class TestDatasets(Dataset):
    def __init__(self, data_dir, transform=None, image_size=224):
        self.transform = transform
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        images = self.image_path[item]
        image_file_index = images.split('.')[0].split('/')[-1]
        images = Image.open(images).convert('RGB')

        if self.transform is not None:
            images = self.transform(images)
        else:
            images = transforms.ToTensor()(images)

        return images, image_file_index

    def __len__(self):
        return len(self.image_path)
