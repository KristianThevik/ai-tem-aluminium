# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:23:48 2024

@author: espen
"""

import torch 
import numpy as np
import cv2
import pandas as pd
import os
import torchvision as tv
import torch.nn as nn
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Tuple, List, Dict, OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.roi_heads import maskrcnn_loss
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.models.detection.rpn import concat_box_prediction_layers
import json
from scipy import ndimage as nd
 

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

class CustomDataset():
    """
    Load dataset
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.file = open(self.image_dir+"/_annotations.coco.json",'r')
        self.data = json.load(self.file)
        self.img  = pd.DataFrame(self.data['images'])
        self.an   = pd.DataFrame(self.data['annotations'])
        self.iterator = iter(self.img.id)
        # self.resize = resize
    def __len__(self):
        return len(self.img)

    def reset_iter(self):
        self.iterator = iter(self.img.id)
    def get_data(self, batch_size):

        img_list = []
        data_list = []
        
        for idx in range(batch_size):
            self.id = next(self.iterator,-1)
            if not self.id+1:
                if idx>0:
                    self.reset_iter()
                    return img_list, data_list
                else:
                    self.reset_iter()
                    if batch_size == 1:
                        self.id = next(self.iterator,-1)
                    else:
                        continue
                
            img_name = os.path.join(self.image_dir, self.img[self.img.id == self.id].file_name.iloc[0])
            # image = np.load(img_name)
            image = np.array(Image.open(img_name).convert('L'))
            size = np.shape(image)[-1]

            image = np.expand_dims(image, axis=0)
            image = image/np.max(image)
            image = torch.tensor(image, dtype = torch.float32)
            data = {}
            
            
    
            
            num_objs = len(self.an[self.an.image_id == self.id])
            mask     = np.zeros([num_objs,size,size], dtype = np.uint8)
            boxes    = torch.zeros([num_objs,4], dtype=torch.float32)

            for num, annotation in enumerate(self.an[self.an.image_id == self.id].iloc):
                x,y,w,h = annotation.bbox
                box = np.array([x,y,x+w,y+h]).astype(np.int32)
                boxes[num] = torch.tensor(box)
                segments = np.array(annotation.segmentation).astype(np.int32)
                segments = segments.reshape((np.shape(segments)[-1]//2, 2))
                for index, value in enumerate(segments):
                    segments[index] -= box[:2]
                segments = segments.astype(np.int32)
                
                mask[num][box[1]:box[3]+1,box[0]:box[2]+1] = (cv2.fillPoly(mask[num][box[1]:box[3]+1,box[0]:box[2]+1], [segments] , color = (255,255,255)))>0
            mask = torch.as_tensor(mask)
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
    
    
            data["boxes"] =  boxes
            data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
            data["masks"] = mask
    
            
    
    
            img_list.append(image)
            data_list.append(data)
            
        return img_list, data_list


def calculate_iou_loss(pred_masks, target_masks, pred_score):
    """
    Calculate intersection over union (IoU) between predicted and target masks.
    Parameters:
        pred_masks (torch.Tensor): predicted masks
        target_masks (torch.Tensor): target masks
    Returns:
        float: value of IoU
    """
    pred = pred_masks.detach().cpu().numpy()
    target = np.any(target_masks.detach().cpu().numpy(),axis = 0)
    size = len([scr for scr in pred_score if scr>0.9])
    new_pred = np.zeros([size,np.shape(pred)[-1],np.shape(pred)[-1]])
    for index in range(len(pred_masks)):
        if pred_score[index] > 0.9:
            new_pred[index] = pred[index]
    pred = new_pred
    intersection = np.sum(np.logical_and(pred,target))
    union = np.sum(pred) + np.sum(target)
    iou = (2*intersection) / (union)
    
    
    return iou   


data_train = CustomDataset(base_path+"/train", transform = transform)   
data_valid = CustomDataset(base_path+"/valid", transform = transform) 

print('Starting model loading')  


model=tv.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT', min_size=1024, max_size=2048) #CHANGE

# =============================================================================
# If ResNet18, use model below
# =============================================================================

# backbone = tv.models.resnet18(weights = 'DEFAULT')
# backbone = tv.models.detection.backbone_utils._resnet_fpn_extractor(backbone, 1)
# model = tv.models.detection.MaskRCNN(backbone, num_classes=2, min_size=1024, max_size=2048)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.detections_per_img = 300
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)         
model.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr = lr)
model.train(True)
print('Model loaded')  







train_loss , valid_loss, valid_IoU  = [] , [], []
train_m_loss , valid_m_loss         = [] , []
for i in range(101): #Epochs
    f = open(out_path+r"\log.txt", "a")
    for phase in ['train','valid']:
        running_loss = 0
        running_m_loss = 0
        total_iou_mask = 0
        
        if phase == 'train':
            model.train(True)
            datal = data_train
        else:
            model.train(False)
            datal = data_valid
        
        for cycle in range(int(np.ceil(len(datal)/batch_size))):
            
            images, targets= datal.get_data(batch_size)
            images = list(image.to(device) for image in images)
            targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
            
            if phase == 'train':
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                # losses = loss_dict['loss_mask']
                running_loss+= loss_dict['loss_mask'] * batch_size
                losses.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    model.eval()
                    outputs = model(images,targets)
                    model.train(True)
                    loss_valid = model(images, targets)
                    model.train(False)
                    losses = sum(loss for loss in loss_valid.values())
                    running_loss+= losses * batch_size
                    running_m_loss += loss_valid['loss_mask'] * batch_size
                    for j in range(len(outputs)):
                        preds_masks = outputs[j]['masks'] > 0.5  # Convert mask probabilities to binary masks
                        preds_score = outputs[j]['scores']
                        gt_masks = targets[j]['masks']
                        iou_mask = calculate_iou_loss(preds_masks, gt_masks, preds_score)
                        total_iou_mask += iou_mask
        epoch_loss = running_loss/len(datal)
        epoch_m_loss = running_m_loss/len(datal)
        train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
        train_m_loss.append(epoch_m_loss) if phase == 'train' else valid_m_loss.append(epoch_m_loss)
    avg_iou_mask = total_iou_mask/ len(datal)
    valid_IoU.append(avg_iou_mask)
    f.write('Epoch: {} ; Train_loss: {} ; Valid_loss: {} ; Valid_dice: {}\n'.format(i, train_loss[-1],valid_loss[-1],valid_IoU[-1]))
    f.close()
    if i%10==0:
        torch.save({
        'model_state_dict': model.state_dict(),
        'optim': optimizer.state_dict(),
        'loss': 'CrossEntropy',
        'epoch': i,
        'bs': batch_size, 
        'lr': lr,
        'valid_loss': valid_loss,
        'valid_dice': valid_IoU,
        'train_loss': train_loss,
        'tmask_loss': train_m_loss,
        'vmask_loss': valid_m_loss,
        }, out_path+model_name+r"/normal"+str(i)+".pth")
    torch.save({
    'model_state_dict': model.state_dict(),
    'optim': optimizer.state_dict(),
    'loss': 'CrossEntropy',
    'epoch': i,
    'bs': batch_size, 
    'lr': lr,
    'valid_loss': valid_loss,
    'valid_dice': valid_IoU,
    'train_loss': train_loss,
    'tmask_loss': train_m_loss,
    'vmask_loss': valid_m_loss,
    }, out_path+model_name+r"/temp_save.pth")
