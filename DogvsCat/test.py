import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from models import Model
from dataset import get_test_loader

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--data_dir', type=str, default='dataset/test', help='directory of dataset')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=24, help='start number of epochs to train for')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")

args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = Model(args.num_classes)

data_loader = get_test_loader(args.data_dir, args.image_size, args.batch_size)

net.load_state_dict(torch.load(os.path.join(f'{args.checkpoint_dir}', f'{args.epoch}.pth'), map_location=device))
file = open('output.csv', 'w')
file.write('id,label\n')

for indexs, images in tqdm(data_loader):
    with torch.no_grad():
        images: torch.Tensor = images.to(device)
        indexs: torch.Tensor = indexs
        preds = net.predict_image(images)

        for i in range(len(indexs)):
            file.write(f'{indexs[i]},{preds[i].item()}\n')