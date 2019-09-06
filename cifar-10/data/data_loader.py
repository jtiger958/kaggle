import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from data.dataset import TrainDatasets, TestDatasets


def get_cifar_10_loader(config):
    transform_image = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = torchvision.datasets.cifar.CIFAR10(root='./data/dataset/',
                                                       train=True,
                                                       transform=transform_image,
                                                       download=True)

    test_dataset = torchvision.datasets.cifar.CIFAR10(root='./data/dataset/',
                                                      train=False,
                                                      transform=transform_image)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             shuffle=False)

    return train_loader, test_loader


def get_loader(config):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform  = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_loader = DataLoader(dataset=TrainDatasets(config.train_dir, train_label_path=config.train_label_path,
                                                    transform=train_transform),
                              batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)

    test_loader = DataLoader(dataset=TestDatasets(config.test_dir, transform=test_transform),
                             batch_size=config.batch_size,num_workers=config.num_workers, shuffle=True)
    return train_loader, test_loader
