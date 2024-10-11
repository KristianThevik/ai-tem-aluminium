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
        # There are no mask files in dataset, create them with mask function
        generate_and_save_masks(image_dir)
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
    
def generate_and_save_masks(image_dir):
    """
    This function generates and saves mask files for each image based on the COCO annotations.
    The masks will be saved in the same directory as the images, with '_mask.png' suffixes.
    
    """
    
    # Load the COCO annotations
    coco = COCO(os.path.join(image_dir, '_annotations.coco.json'))

    # Get all image IDs
    image_ids = coco.getImgIds()

    # Iterate over each image
    for image_id in image_ids:
        # Load image info to get file name and dimensions
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        width, height = image_info['width'], image_info['height']
        
        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Get all annotation IDs for this image
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)

        # Iterate through annotations and update the mask
        for ann in annotations:
            ann_mask = coco.annToMask(ann)
            mask = np.maximum(mask, ann_mask)

        # Create mask file name
        mask_file_name = file_name.replace('.jpg', '_mask.png')
        mask_file_path = os.path.join(image_dir, mask_file_name)

        # Save the mask as a PNG file
        Image.fromarray(mask * 255).save(mask_file_path)