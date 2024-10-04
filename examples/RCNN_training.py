# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:23:48 2024

@author: espen
"""
import torch
from typing import Tuple, List, Dict, OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.roi_heads import maskrcnn_loss
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.models.detection.rpn import concat_box_prediction_layers
from data_loader import CustomDataset
from model_utils import get_model, initialize_optimizer
from training_utils import train_model
 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Values
# =============================================================================

model_name = r"/v13"
#base_path =  r"/cluster/work/espenjgr/rotated_labels/new_needle2"
base_path = r"C:\Users\krist\OneDrive\Dokumenter\masterProsjekt\training\training_data"
#out_path = r"/cluster/work/espenjgr/cross_models"
out_path = r"C:\Users\krist\OneDrive\Dokumenter\masterProsjekt\training\training_data\out_path"


batch_size = 1
lr=1e-4
transform = None

f = open(out_path+r"\log.txt", "w")
f.write(f'Device \t\t: {device}\n')
f.write(f'Batch_size \t: {batch_size}\n')
f.write(f'Learning_rate \t: {lr}\n')
f.write(f'Data_path \t: {base_path}\n')
f.write('-'*60+'\n')
f.close()

# Load datasets

data_train = CustomDataset(base_path+"/train", transform = transform)   
data_valid = CustomDataset(base_path+"/valid", transform = transform) 

# Load model and optimizer

print('Starting model loading')

model = get_model(device)
optimizer = initialize_optimizer(model, lr)
model.train(True)

print('Model loaded')

# Start training
train_model(model, optimizer, data_train, data_valid, device, out_path, batch_size, lr)




