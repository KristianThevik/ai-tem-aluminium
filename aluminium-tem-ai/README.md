# Automatic Precipitate Statistics Extraction

This repository contains code used for automatic extraction of precipitate statistics
using Mask R-CNN and U-Net models. The prediction are implemented in Python using Jupyter Notebooks. 

## File Structure

- 'RCNN.ipynb': Jupyter Notebook for Mask R-CNN predictions.
- 'UNET.ipynb': Jupyter Notebook for U-Net predictions.
- 'training': Folder containing Python script for Mask R-CNN and U-Net training, in addition to the training data in  COCO-segmentation format 
- 'models': Folder containing files ending in '*.pth', that are the trained ML models.
- '_dm3_lib.py': Python script for  loading of (*.dm3) files. Created by P. Rayna [[1](https://github.com/piraynal/pyDM3reader/blob/main/dm3_lib/_dm3_lib.py)] 
- 'u_net_pytorch.py': Python script for U-Net architecture. Created by E. Stevens and L. P. G. Antiga [[2](https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/util/unet.py)]

## Requirements

To install the required dependencies, run: 

```bash
pip install -r requirements.txt
```

The file 'requirements.txt' contains all Python packages present in the conda environment

## Usage

1. Ensure that the trained model files (*.pth) are placed in the models, while dm3_lib.py, and u_net_pytorch.py are in the same directory as .ipynb files
2. Run 'RCNN.ipynb' for Mask R-CNN predictions or 'UNET.ipynb' for U-Net predictions.
3. Follow instructions in Notebooks.