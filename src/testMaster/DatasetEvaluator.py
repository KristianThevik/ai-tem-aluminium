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
        
        mean_value = np.mean(len_cross_vals)
        std_value = np.std(len_cross_vals)
        std_mean = std_value / np.sqrt(len(len_cross_vals))  # Standard error of the mean
        number_density = len(len_cross_vals) / (len(image_ids)*(self.size*self.nm_per_px)**2)

        if self.cross:
            values = 'Cross section [nm^2]'
            unit = 'nm^2'
        else:
            values = 'Length [nm]'
            unit = 'nm'

        stats_dict = {'Image ID' : image_ids, values : len_cross_vals, 'Confidence score' : confidence_scores}

        df = pd.DataFrame(stats_dict)
        df.to_csv(os.path.join(os.path.dirname(self.dataset_dir), 'statistics.csv'))

        print('Average: {0:.2f} {5}, STDev: {1:.2f} {5}, Number counted: {2:d}, STDev of mean: {3:.2f}, Number density: {4:.7f}nm^-2'.format(mean_value, std_value, len(len_cross_vals), std_mean , number_density, unit))

        
        hist_vals = np.array(len_cross_vals)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(hist_vals, bins=20, alpha=0.5, edgecolor="black")

        ax.set_xlabel("Cross Section [nm^2]" if self.cross else "Length [nm]")  # Adjust based on your measurement
        ax.set_xlim(0, max(hist_vals))
        ax.set_ylabel("Number Frequency")
        ax.set_title("Distribution of Cross-Sectional Areas" if self.cross else "Distribution of Lengths")
       
        plt.show()
    
class RCNNEvaluator(Evaluator):
    def __init__(self, dataset_dir, model, cross, device):
        super().__init__(dataset_dir, model, cross, device) # Inherits what is common for all model evaluator from Evalutaor class
        self.size = 1024
        self.threshold = 0.9 if cross else 0.4
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
        im = self.to_tensor(img).unsqueeze(0).to(self.device)
        temp_nm_per_px = self.nm_per_px *2 #Original calibration for 2048x2048, but images are resized to 1024x1024
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
            return np.array(self.area)*temp_nm_per_px**2, self.confidence_scores
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
            return np.array(self.lengths)*temp_nm_per_px, self.confidence_scores
        


class UNETEvaluator(Evaluator):
    def __init__(self, dataset_dir, model, cross, device):
        super().__init__(dataset_dir, model, cross, device) # Inherits what is common for all model evaluator from Evalutaor class
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
        true_img, imgs = self.to_tensor(img)
        new_im = Image.new('RGB', (self.size, self.size))
        temp_nm_per_px = self.nm_per_px * 2  # Original calibration for 2048x2048, but images are resized to 1024x1024

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
            self.area += self.watershed(self.prediction[-1], temp_nm_per_px, plot=False)
        else:
            self.calc_length(self.prediction[-1], temp_nm_per_px)

        if self.cross:
            return np.array(self.area)*temp_nm_per_px**2, np.ones(len(self.area))  # Dummy array, should implement confidence score
        else:
            return np.array(self.lengths)*temp_nm_per_px, np.ones(len(self.lengths)) # Dummy array, should implement confidence score
        
    def watershed(self, img, nm_px_ratio, plot = False):
        
        """
        Performs the watershed algorithm on the prediction img
    
        img  : PIL.Image (semantic segmentation prediction map)
        plot : bool (True if you want to see the watershed processing steps)
        
        Documentation: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
        
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = clear_border(gray)
        ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = cv2.dilate(bin_img, kernel, iterations=20) 
        dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5) 
        
       
        #foreground area 
        ret, sure_fg = cv2.threshold(dist, 0.15 * dist.max(), 255, cv2.THRESH_BINARY) 
        sure_fg = sure_fg.astype(np.uint8)   
          
        # unknown area 
        unknown = cv2.subtract(sure_bg, sure_fg) 
        ret, markers = cv2.connectedComponents(sure_fg) 
          
        # Add one to all labels so that background is not 0, but 1 
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers) 
        
        if plot:
            fig, axes = plt.subplots(2,2)
            axes[0,0].imshow(gray) 
            axes[0, 0].set_title('Img') 
            axes[0,1].imshow(dist) 
            axes[0, 1].set_title('Distance Transform') 
              
            axes[1,0].imshow(sure_fg) 
            axes[1, 0].set_title('Sure Foreground') 
            axes[1,1].imshow(markers) 
            axes[1, 1].set_title('Markers') 
    
        img2 = color.label2rgb(markers,bg_label = 1,bg_color=(0, 0, 0))
        props = measure.regionprops_table(markers, intensity_image=gray, 
                                      properties=['label',
                                                  'area', 'equivalent_diameter',
                                                  'mean_intensity', 'solidity'])
        
        df = pd.DataFrame(props)
        area = list(df[(df.mean_intensity > 100) & (df.area > 1.5/nm_px_ratio**2)].area)
        return area

    def calc_length(self, img, nm_px_ratio):
        """
        Estimates the length of precipitates
        """
        grey = img[:,:,0]
        contours, hierarchy = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        l      = []
        for contour in contours:
            (center), (width,height), angle = cv2.minAreaRect(contour)
            length = np.max([width,height])
            l.append([length, angle+(angle<0)*90])
        for index, (length, angle) in enumerate(l):
            median_angle = np.median([angle for (length, angle) in l if length*nm_px_ratio > 5]) #Find angles of all detections longer than 5nm
            error        = 5.0 #degrees
            if  (median_angle - error<angle<median_angle + error) and length*nm_px_ratio>3: #If precipitate in correct direction (within error) and longer than 3nm, accept detection
                self.lengths.append(length) 
    