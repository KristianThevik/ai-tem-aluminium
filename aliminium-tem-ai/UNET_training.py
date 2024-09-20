# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:51:01 2024

@author: espen
"""

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import adam
from torchvision import transforms as tf
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
import cv2
import pandas as pd
import time
import os

number_of_epochs = 51
learning_rate = 5e-4
depth = 3
filters = 6

batch_size = 10

transform = tf.Compose([
    tf.Resize(size = (512, 512), interpolation=Image.NEAREST)
])

# base_path = r"/cluster/work/espenjgr/cross_data/cross_data_1024_v5_old"
# out_path = r"/cluster/work/espenjgr/u_net_models/cross2"
base_path = r"C:\Users\krist\OneDrive\Dokumenter\masterProsjekt\training\training_data"
out_path = r"C:\Users\krist\OneDrive\Dokumenter\masterProsjekt"


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

f = open(out_path+filename+'log.txt', "w")
f.write(f'Device is {device}\n')
f.write('#'*20+'\n')
f.close()

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        
        https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/d6c0210143daa133bbdeddaffc8993b1e17b5174/util/unet.py
        
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
    
def dice(predb, yb):
    pb = predb.argmax(dim=1)
    yb = yb.to(device)
    
    cls_ = predb.shape[1]
    
    mean_dice = 0
    dice_list = torch.zeros(cls_)
    
    for i in range(cls_):
        p = pb == i
        y = yb == i
    
        volume_sum = torch.sum(y, dim=(2,1)) + torch.sum(p, dim=(2,1))

        volume_intersect = torch.logical_and(p, y)
        volume_intersect = torch.sum(volume_intersect, dim=(2,1))
        
        # if volume_sum
        dice = (2*volume_intersect / volume_sum).mean()
        
        mean_dice += dice
        dice_list[i] = dice
 
    return mean_dice/cls_, (dice_list)

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None, mask_transform = None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.images = sorted([i for i in os.listdir(image_dir) if 'mask' not in i and i.endswith('.jpg')])
        self.masks = sorted([i for i in os.listdir(image_dir) if ('mask' in i and i.endswith('.png'))])


    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.image_dir, self.masks[idx])
        
        image = Image.open(img_name).convert('L')
        # print(image)
        raw = np.stack([np.array(image),], axis=2)
        raw = raw.transpose((2,0,1))
        image =  2*(raw / np.max(raw)) - 1 # normalize: -1 to 1
        image = torch.tensor(image, dtype = torch.float32)
        mask = Image.open(mask_name).convert('L')
        
        raw = np.stack([np.array(mask)>0,], axis=2)
        
        raw = raw.transpose((2,0,1))
        mask = torch.tensor(raw, dtype = torch.torch.int64)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask[0]




data_train = CustomDataset(base_path+r"/train", transform = transform)
data_valid = CustomDataset(base_path+r"/valid", transform = transform)

train_data = DataLoader(data_train, batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data_valid, batch_size=batch_size, shuffle=True)
model = UNet(in_channels = 1, n_classes = 2, depth = depth, wf = filters, padding = True)
model = model.to(device)
metric = dice
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start = time.time()

train_loss, valid_loss = [], []
train_met, valid_met = [], []

save = True

for epoch in range(number_of_epochs):
    f = open(out_path+filename+'log.txt', "a")
    f.write('-' * 100+'\n')
    f.write('Epoch {}/{}\n'.format(epoch, number_of_epochs - 1))
    for phase in ['Train', 'Valid']:
        if phase == 'Train':
            model.train(True)
            datal = train_data
            
            
        else:
            model.train(False)
            datal = valid_data

        running_loss = 0.0
        running_met = 0.0
        running_met_arr = torch.zeros(2) # number of classes

        step = 0

        # iterate over data
        for x, y in datal:
            x = x.to(device)
            y = y.to(device)
            
            step += 1

            if phase == 'Train':
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(x)
                    loss = loss_function(outputs, y)

            with torch.no_grad():
                met, met_list = metric(outputs, y)

            running_met  += met * datal.batch_size
            running_loss += loss * datal.batch_size
            running_met_arr += met_list * datal.batch_size
                
                
            # if step % (datal.batch_size*10) == 0:
            #     print('Current step: {}  \nLoss: {:.4f}  Dice score: {:.4f}'.format(step, loss, met))
                
        epoch_loss = running_loss / len(datal.dataset)
        epoch_met = running_met / len(datal.dataset)
        epoch_met_arr = running_met_arr / len(datal.dataset)

        f.write('{} Loss: {:.4f}; Dice: {:0.4f} \n'.format(phase, epoch_loss, epoch_met))
        print('Dice #1: %.4f, Dice #2: %.4f'%(epoch_met_arr[0], epoch_met_arr[1]))

        train_loss.append(epoch_loss) if phase=='Train' else valid_loss.append(epoch_loss)
        train_met.append(epoch_met) if phase=='Train' else valid_met.append(epoch_met)
    
    patience = 5
    if (len(valid_loss) > patience) and (len(valid_loss) - torch.argmin(torch.tensor(valid_loss)) >= patience) and save:
        save = False
        f.write("Early save at epoch: {} and patience: {}\n".format(epoch, patience))
        f.close()
        PATH = out_path+filename+'patience.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
            'loss': 'CrossEntropy',
            'epoch': number_of_epochs,
            'batch_size': datal.batch_size, 
            'lr': learning_rate,
            'train_loss':train_loss, 
            'valid_loss': valid_loss,
            'valid_met': valid_met,
        }, PATH)
        # break
    f.close()

time_elapsed = time.time() - start
f = open(out_path+filename+'log.txt', "a")
f.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60)) 
f.close()

PATH = out_path+filename+'model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optim': optimizer.state_dict(),
    'loss': 'CrossEntropy',
    'epoch': number_of_epochs,
    'batch_size': datal.batch_size, 
    'lr': learning_rate,
    'train_loss':train_loss, 
    'valid_loss': valid_loss,
    'valid_met': valid_met,
}, PATH)