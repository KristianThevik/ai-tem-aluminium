import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from skimage.segmentation import clear_border
from testMaster.u_net_pytorch import UNet
from  PIL import Image
from itertools import product
import pandas as pd
from skimage import measure, color
import pandas as pd

"""
    Method for converting Digital Micrograph files to arrays should be implemented
    since datasets could be dm3 files.
"""

class Evaluator:
    """ 
        This is the general evaluator class, were each specfic model evaluator are subclasses
        dataset_dir: path to directory containing the dataset images
    """
    def __init__(self, dataset_dir, model: Path, cross = True, device = 'cpu'):
        self.dataset_dir = dataset_dir
        self.image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.path = model
        self.cross = cross
        self.device = device
        self.nm_per_px = 0.069661 # This has to be dynamically set or given as function paramter

    def predict(self, img):
        raise NotImplementedError("Subclasses must implement predict method.")
    

    def statistics(self):

        """
           This function calls the predict function specific to each class
           This function uses the predict function of the model evaluator class that is a an instance
           of Evaluator class
        """
        image_ids = []
        len_cross_vals = []
        confidence_scores = []

        for id, image in enumerate(self.image_paths):
            prediction, scores = self.predict(image)
            for i in range(len(prediction)):
                image_ids.append(id)
                len_cross_vals.append(prediction[i])
                confidence_scores.append(scores[i])

        if self.cross:
            values = 'Cross section [nm^2]'
        else:
            values = 'Length [nm]'

        stats_dict = {'Image ID' : image_ids, values : len_cross_vals, 'Confidence score' : confidence_scores}

        df = pd.DataFrame(stats_dict)
        df.to_csv(os.path.join(os.path.dirname(self.dataset_dir), 'statistics.csv'))
        




    
        

    
    
class RCNNEvaluator(Evaluator):
    def __init__(self, model, cross, device, other, stuff, specific, f_or, r_cnn):
        super().__init__(model, cross, device) # Inherits what is common for all model evaluator from Evalutaor class
        self.size = 1024
        self.treshold = 0.9 if cross else 0.4
        self.erode_it = 0 if cross else 4

        # Load the model
        self.checkpoint = torch.load(self.path, map_location=torch.device(self.device))
        self.model = maskrcnn_resnet50_fpn(weights='DEFAULT', min_size=1024, max_size=2048, box_detections_per_img=500)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print('Mask-RCNN Model Loaded')


    def to_tensor(self, file) -> torch.Tensor:
        """
        Opens the image, resizes it (if applicable), and converts it to a pytorch tensor.
        """
        image = np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        image = cv2.resize(image, dsize=(self.size, self.size))
        image = np.expand_dims(image, axis=0)  # Adding channel dimension
        image = image / np.max(image)  # Normalize to range [0, 1]
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def predict(self, img):
        """
        Analyze mask and box predictions and return statistics like area or length.
        """

        self.area    = []
        self.lengths = []
        self.confidence_scores = []
        print('Mask threshold set to {0:.2f}'.format(self.threshold))
        print('Calibration used: {0:.4f} nm/px'.format(self.nm_per_px))
        im = self.to_tensor(self, img).unsqueeze(0).to(self.device)
        self.nm_per_px *=2 #Original calibration for 2048x2048, but images are resized to 1024x1024
        with torch.no_grad(): #Predict
            pred = self.model(im)
            
        if self.cross:
            for i in range(len(pred[0]['masks'])):
                msk = pred[0]['masks'][i,0].detach().cpu().numpy()
                scr = pred[0]['scores'][i].detach().cpu().numpy()
                mask    = msk>self.threshold
                kernel  = np.ones((2, 2), np.uint8) 
                mask_er = cv2.erode(mask.astype(np.float32), kernel, iterations = self.erode_it)  
                msk    = mask_er>0
                clear = clear_border(msk)
                area1 = np.sum(clear)
                area2 = np.sum(msk)
                if scr>0.9 and area1 == area2:
                   self.area.append(area1)
                   self.confidence_scores.append(scr)
            return self.area*self.nm_per_px**2, self.confidence_scores
        else:
            for i in range(len(pred[0]['masks'])):
                scr = pred[0]['scores'][i].detach().cpu().numpy()
                msk = pred[0]['masks'][i,0].detach().cpu().numpy()
                mask    = msk>self.threshold
                kernel  = np.ones((2, 2), np.uint8) 
                mask_er = cv2.erode(mask.astype(np.float32), kernel, iterations = self.erode_it)  
                msk     = clear_border(mask_er>0,buffer_size=10)
                if scr>0.9 and np.any(msk):
                    rect = cv2.minAreaRect(np.argwhere((msk>self.threshold)))
                    (center), (width,height), angle = rect
                    length = np.max([width,height])
                    self.lengths.append(length)
                    self.confidence_scores.append(scr)
            return self.lengths*self.nm_per_px, self.confidence_scores
        


class UNETEvaluator(Evaluator):
    def __init__(self, model, cross, device, other, stuff, specific, f_or, r_cnn):
        super().__init__(model, cross, device) # Inherits what is common for all model evaluator from Evalutaor class
        self.size      = 1024
        self.tile_size = 512
        self.checkpoint = torch.load(model, map_location=torch.device(device))
        self.model = UNet(in_channels = 1, n_classes = 2, depth = 3, wf = 6, padding = True)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(' Unet Model Loaded')    
    
    def tile_img(self, arr, d) -> list:
        """
        Tile the image into equal parts of size (d x d) pixels, for example 1024x1024 image 
        into 4 images of 512x512

        """
        img = Image.fromarray(arr)
        w, h = img.size
        grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
        img_list = []
        for i, j in grid:
            box = (j, i, j + d, i + d)
            img_list.append(img.crop(box))
        return img_list

    def to_tensor(self, file) -> list:
        """
        Opens the image in grayscale, resizes it, and converts it to a pytorch tensor.
        For now keep these functions separate for each model-evaluator. Double check if UNET expects input
        files in range [-1,1], and RCNN [0,1]. If not, this function could be put in the Evaluator class
        """
        image = np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        image = cv2.resize(image, dsize=(self.size, self.size))
        tensors = []
        images = self.tile_img(np.array(image), self.tile_size)

        for i in images:
            im = np.expand_dims(i, axis=0)
            im = 2 * (im / np.max(im)) - 1 # Normalizing to values [-1,1]
            im = torch.tensor(im, dtype=torch.float32)
            tensors.append(im)
        return np.array(image), tensors

    def predict(self, img):
        """
        Prediction function
        
        """
        self.prediction = []
        self.area = []
        self.lengths = []

        # Process the single image tiled into multiple images, see tile_image function
        true_img, imgs = self.to_tensor(self, img)
        new_im = Image.new('RGB', (self.size, self.size))
        self.nm_per_px *= 2  # Original calibration for 2048x2048, but images are resized to 1024x1024

        for index, im in enumerate(imgs):
            im = im.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(im)
                output = torch.argmax(pred, dim=1)  # Get the index of the channel with the highest probability
                output = output.squeeze(0).cpu().numpy()

            y_offset = int(self.tile_size * (index > 1))
            x_offset = int(self.tile_size * (index % 2))
            out = Image.fromarray(output.astype('uint8') * 255).convert('RGB')
            new_im.paste(out, box=(x_offset, y_offset))

        self.prediction.append(np.array(new_im))

        if self.cross:
            self.area += self.watershed(self.prediction[-1], plot=False)
        else:
            self.calc_length(self.prediction[-1])

        if self.cross:
            return self.area*self.nm_per_px**2, true_img , self.prediction
        else:
            return self.lengths*self.nm_per_px, true_img , self.prediction