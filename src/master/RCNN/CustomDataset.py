import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd


class CustomDataset():
    """
    Load dataset
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.file = open(os.path.join(self.image_dir, "_annotations.coco.json"),'r')
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
