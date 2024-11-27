# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:23:48 2024

@author: espen
"""
import torch
from testMaster.RCNN.CustomDataset import CustomDataset
from testMaster.RCNN.model_utils import get_model, initialize_optimizer
from testMaster.RCNN.training_utils import train_model, plot_training_results
from pathlib import Path
import os
 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Values
# =============================================================================

this_dir = Path(__file__).resolve().parent
base_path = this_dir.parent / "data"
Path(base_path).mkdir(parents=True, exist_ok=True)
out_path = this_dir.parent / "data" / "models"
Path(out_path).mkdir(parents=True, exist_ok=True)


batch_size = 1
lr=1e-4
transform = None

f = open(os.path.join(out_path, "RCNN_log.txt"), "w")
f.write(f'Device \t\t: {device}\n')
f.write(f'Batch_size \t: {batch_size}\n')
f.write(f'Learning_rate \t: {lr}\n')
f.write(f'Data_path \t: {base_path}\n')
f.write('-'*60+'\n')
f.close()

# Load datasets

data_train = CustomDataset(base_path / "train_cross", transform = transform)   
data_valid = CustomDataset(base_path / "valid_cross", transform = transform) 

# Load model and optimizer

print('Starting model loading')

model = get_model(device)
optimizer = initialize_optimizer(model, lr)
model.train(True)

print('Model loaded')

# Start training
train_loss, valid_loss, valid_IoU, train_mask_loss, valid_mask_loss = train_model(model, optimizer, data_train, data_valid, device, out_path, batch_size, lr)

plot_training_results(train_loss, valid_loss, valid_IoU, train_mask_loss, valid_mask_loss)
