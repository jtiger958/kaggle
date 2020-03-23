import torch
import torch.nn as nn
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

from models import Model
from dataset import get_loader

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=448, help='the height / width of the input image to network')
parser.add_argument('--data_dir', type=str, default='dataset/train', help='directory of dataset')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=0, help='start number of epochs to train for')
parser.add_argument('--num_epoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--decay_epoch', type=int, default=30, help='decay epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--log_path', default='logs', help="directory to save train log")

args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = Model()

if os.path.exists(os.path.join(f'{args.checkpoint_dir}', f'{args.epoch-1}.pth')):
    net.load_state_dict(torch.load(os.path.join(f'{args.checkpoint_dir}', f'{args.epoch-1}.pth'), map_location=device))
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_epoch)
data_loader = get_loader(args.data_dir, args.image_size, args.batch_size)
scheduler.step(args.epoch)

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
writer = SummaryWriter(args.log_path)

for iter in range(args.epoch, args.num_epoch):
    print(iter)
    train_loss, train_acc = net.train_model(data_loader, criterion, optimizer)
    validation_loss, validation_acc = net.validate_model(data_loader, criterion)
    writer.add_scalars('Loss', {'train': train_loss, 'val': validation_loss}, iter)
    writer.add_scalars('Accuracy', {'train': train_acc, 'val': validation_acc}, iter)
    torch.save(net.state_dict(), os.path.join(f'{args.checkpoint_dir}', f'{iter}.pth'))
    scheduler.step()