import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np

import matplotlib.pyplot as plt

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
    
def get_mask():
    # Load the COCO annotation file
    coco = COCO(r'C:\Users\krist\Documents\masterRepo\data\train_cross\_annotations.coco.json')

    # Load an image and its annotations
    image_id = 131  # Use the correct image ID
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)

    # Create an empty mask
    image_info = coco.loadImgs(image_id)[0]
    mask = np.zeros((image_info['height'], image_info['width']))

    # Iterate through all annotations for this image
    for ann in annotations:
        mask = np.maximum(mask, coco.annToMask(ann))

    # Display the mask using matplotlib
    plt.imshow(mask, cmap='gray')
    plt.axis('off')  # To remove axis ticks
    plt.show()

# Call the function to display the mask
get_mask()