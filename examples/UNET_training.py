# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:51:01 2024

@author: espen
"""
from testMaster.UNET.CustomDataset import CustomDataset
from testMaster.UNET.model_utils import UNet
from testMaster.UNET.training_utils import train_model, dice
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
from pathlib import Path
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


number_of_epochs = 51
learning_rate = 5e-4
depth = 3
filters = 6
patience = 5

batch_size = 10

transform = tf.Compose([
    tf.Resize(size = (512, 512), interpolation=Image.NEAREST)
])

this_dir = Path(__file__).resolve().parent
base_path = this_dir.parent / "data"
Path(base_path).mkdir(parents=True, exist_ok=True)
out_path = this_dir.parent / "data" / "models"
Path(out_path).mkdir(parents=True, exist_ok=True)



f = open(os.path.join(out_path, "Unet_log.txt"), "w")
f.write(f'Device is {device}\n')
f.write('#'*20+'\n')
f.close()
    

data_train = CustomDataset(base_path+r"/train", transform = transform)
data_valid = CustomDataset(base_path+r"/valid", transform = transform)

train_data = DataLoader(data_train, batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data_valid, batch_size=batch_size, shuffle=True)
model = UNet(in_channels = 1, n_classes = 2, depth = depth, wf = filters, padding = True)
model = model.to(device)
metric = dice
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, optimizer, loss_function, metric, train_data, valid_data, number_of_epochs, patience, out_path, learning_rate, device)