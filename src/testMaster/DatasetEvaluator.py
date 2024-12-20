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
from skimage.measure import label
import pandas as pd
import testMaster._dm3_lib as dm
from scipy.stats import gaussian_kde



class Evaluator:
    """ 
        This is the general evaluator class, were each specfic model evaluator are subclasses
        dataset_dir: path to directory containing the dataset images
    """
    def __init__(self, dataset_dir, model: Path, nm_per_px, cross = True, device = 'cpu'):
        self.dataset_dir = dataset_dir
        self.image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.dm3')]
        self.path = model
        self.cross = cross
        self.device = device
        self.nm_per_px = nm_per_px 

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
            if image.endswith('.dm3'):
                image, self.nm_per_px[id] = self.DM_2_array(dm.DM3(image))
            #print(self.nm_per_px[id])

            prediction, scores, gray_img, overlay_img = self.predict(image, self.nm_per_px[id])
            self.check_image(gray_img, overlay_img, id)

            for i in range(len(prediction)):
                image_ids.append(id)
                len_cross_vals.append(prediction[i])
                confidence_scores.append(scores[i])
        
        mean_value = np.mean(len_cross_vals)
        std_value = np.std(len_cross_vals)
        std_mean = std_value / np.sqrt(len(len_cross_vals))  # Standard error of the mean
        number_density = len(len_cross_vals) / np.sum(((self.size*2*np.array(self.nm_per_px))**2)) # Assuming all images in a dataset has same size
        # Multiply self.size with 2 above since images are originally 2048x2048 and the original nm_per_px is used
        if self.cross:
            values = 'Cross section [nm^2]'
            unit = 'nm^2'
        else:
            values = 'Length [nm]'
            unit = 'nm'

        # Filling lists with nan values so that the values do not repaeat over all rows
        stats_dict = {
            'Image ID' : image_ids, 
            values : np.round(len_cross_vals, 1), 
            'Confidence score' : np.round(confidence_scores, 3),
            f'Average [{unit}]': [np.round(mean_value, 2)] + [np.nan] * (len(len_cross_vals) - 1),
            f'Standard Deviation [{unit}]': [np.round(std_value, 2)] + [np.nan] * (len(len_cross_vals) - 1),
            'Number Counted': [len(len_cross_vals)] + [np.nan] * (len(len_cross_vals) - 1),
            f'Standard Error of Mean [{unit}]': [np.round(std_mean, 2)] + [np.nan] * (len(len_cross_vals) - 1),
            'Number Density [nm^-2]': [np.round(number_density, 7)] + [np.nan] * (len(len_cross_vals) - 1)  
        }

        df = pd.DataFrame(stats_dict)
        df.to_csv(os.path.join(os.path.dirname(self.dataset_dir), 'statistics.csv'))

        print('Average: {0:.2f} {5}, STDev: {1:.2f} {5}, Number counted: {2:d}, STDev of mean: {3:.2f} {5}, Number density: {4:.7f}nm^-2'.format(mean_value, std_value, len(len_cross_vals), std_mean , number_density, unit))

        file_path = "cross.csv" if self.cross else "lengths.csv"
        data = pd.read_csv(file_path, header=None, engine='python', names=['integer', 'decimal'], sep=',')

        # Fill missing decimal values with 0
        data['decimal'] = data['decimal'].fillna(0)

        # Combine the integer and decimal parts
        data_combined = data['integer'] + data['decimal'] / (10 ** data['decimal'].astype(str).str.len())

        # Convert to a list or save back to a new file
        number_list = data_combined.tolist()
        print(len(number_list), number_list)
        print('Mean: ', np.mean(np.array(number_list)), 'STD: ', np.std(np.array(number_list)))
        hist_numbers = np.array(number_list)
        kde_n = gaussian_kde(hist_numbers)
        x_vals_n = np.linspace(0, max(hist_numbers), 1000)  # Generate x-axis values
        pdf_n = kde_n(x_vals_n)  # Evaluate the KDE at the x-axis values

        x_lim = 120

        if self.cross:
            x_lim = 30
        else:
            x_lim = 120


        hist_vals = np.array(len_cross_vals)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(hist_vals, bins=20, alpha=0.5, edgecolor="black")

        ax.set_xlabel("Cross Section [nm^2]" if self.cross else "Length [nm]")  # Adjust based on your measurement
        ax.set_xlim(0, max(hist_vals))
        ax.set_ylabel("Number Frequency")
        ax.set_title("Distribution of Cross-Sectional Areas" if self.cross else "Distribution of Lengths")

        kde = gaussian_kde(hist_vals)
        x_vals = np.linspace(0, max(hist_vals), 1000)  # Generate x-axis values
        pdf = kde(x_vals)  # Evaluate the KDE at the x-axis values

        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, pdf, color="blue", label="YOLO")
        plt.plot(x_vals, pdf_n, color="red", label="Manual")
        plt.xlabel(" Precipitate Cross Section [nm^2]" if self.cross else "Precipitate Length [nm]", fontsize=22)
        plt.ylabel("Normalized Distribution" + (" [1/nm^2]" if self.cross else " [1/nm]"), fontsize=22)
        plt.title("Distribution of " + ("cross sections" if self.cross else "lengths"), fontsize=26)
        plt.legend(fontsize=16)

        plt.grid(True)
        plt.xlim(0, x_lim)  # Limits the x-axis from 0 to 120

        plt.show()
       
        

    def check_image(self, img, mask, n):

        pred_path = os.path.join(os.path.dirname(self.dataset_dir), 'image_predictions')
        os.makedirs(pred_path, exist_ok=True) 

        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img, cmap = 'gray')
        axes[0].set_title('Original Grayscale Image',  fontsize=10)
        axes[1].imshow(mask)
        axes[1].set_title('Predicted Precipitates',  fontsize=10)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        fig.subplots_adjust(wspace=0.2)
        fig.savefig(os.path.join(pred_path, f'Image{n}.png'), dpi = 300)
        plt.close(fig)
        
    def DM_2_array(self, img) -> np.array:
        """
        Convert Digital Micrograph file to numpy array

        img: An instance of the DM3 class from _dm3_lib.py

        returns a numpy array of the grayscale image
        """
        nm_per_px = img.pxsize[0]
        cons = img.contrastlimits
        im   = img.imagedata
        im[im>cons[1]] = cons[1]
        im[im<cons[0]] = cons[0]
        im =  ((im-cons[0])/(cons[1]-cons[0]))*255  #0 to 1
        return im.astype(np.uint8), nm_per_px


    
class RCNNEvaluator(Evaluator):
    def __init__(self, dataset_dir, model, nm_per_px, cross, device):
        super().__init__(dataset_dir, model, nm_per_px, cross, device) # Inherits what is common for all model evaluator from Evalutaor class
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
        if isinstance(file, np.ndarray):  # Check if it's already processed by DM_2_array
            image = file
        else:
            image = np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

        image = cv2.resize(image, dsize=(self.size, self.size))
        image = np.expand_dims(image, axis=0)  # Adding channel dimension
        image = image / np.max(image)  # Normalize to range [0, 1]
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def predict(self, img, nm_px_ratio):
        """
        Analyze mask and box predictions and return statistics like area or length.
        """

        self.area    = []
        self.lengths = []
        self.confidence_scores = []
        print('Mask threshold set to {0:.2f}'.format(self.threshold))
        print('Calibration used: {0:.4f} nm/px'.format(nm_px_ratio))
        im = self.to_tensor(img).unsqueeze(0).to(self.device)
        temp_nm_per_px = nm_px_ratio *2 #Original calibration for 2048x2048, but images are resized to 1024x1024
        im_np = im[0].detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC format
        gray = cv2.cvtColor(im_np,cv2.COLOR_BGR2RGB)
        overlay = gray.copy()

        with torch.no_grad():
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
                   # Add red overlay to the accepcted areas
                   overlay[msk] = [1, 0, 0]

            return np.array(self.area)*temp_nm_per_px**2, self.confidence_scores, gray, overlay
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
                    overlay[msk] = [1, 0, 0]
            return np.array(self.lengths)*temp_nm_per_px, self.confidence_scores, gray, overlay
        


class UNETEvaluator(Evaluator):
    def __init__(self, dataset_dir, model, nm_per_px, cross, device):
        super().__init__(dataset_dir, model, nm_per_px, cross, device) # Inherits what is common for all model evaluator from Evalutaor class
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
        if isinstance(file, np.ndarray):  # Check if it's already processed by DM_2_array
            image = file
        else:
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

    def predict(self, img, nm_px_ratio):
        """
        Prediction function
        
        """
        self.prediction = []
        self.area = []
        self.lengths = []
        self.confidence_scores = []

        # Process the single image tiled into multiple images, see tile_image function
        true_img, imgs = self.to_tensor(img)  
        overlay = cv2.cvtColor(true_img, cv2.COLOR_GRAY2RGB)  
        new_im = Image.new('RGB', (self.size, self.size))
        full_prob_map = np.zeros((self.size, self.size))
        temp_nm_per_px = nm_px_ratio * 2  # Original calibration for 2048x2048, but images are resized to 1024x1024

        for index, im in enumerate(imgs):
            im = im.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(im)
                output = torch.argmax(pred, dim=1)  # Get the index of the channel with the highest probability
                output = output.squeeze(0).cpu().numpy()
                probability_map = torch.sigmoid(pred) # Get probabilites of each pixel belonging to the assigned class, Using sigmoid since we have binary sgementation
                probability_map = probability_map.squeeze(0).cpu().numpy()  # Convert to numpy array for easier processing
                object_prob_map = probability_map[1]  # index 1 is the second channel, the percipitate class

            y_offset = int(self.tile_size * (index > 1))
            x_offset = int(self.tile_size * (index % 2))
            out = Image.fromarray(output.astype('uint8') * 255).convert('RGB')
            new_im.paste(out, box=(x_offset, y_offset))
            full_prob_map[y_offset:y_offset + self.tile_size, x_offset:x_offset + self.tile_size] = object_prob_map

        self.prediction.append(np.array(new_im))


        if self.cross:
            area, markers, labels = self.watershed(self.prediction[-1], temp_nm_per_px, plot=False)
            self.area += area

            # Markers is a 2d array with sahpe equal to input image, (w,h).
            # Background pixels has a value of 1. "Unknown" pixels has a value of 0
            # Pixels of the first region/object has a value 2, and for the next region/object the values
            # of the pixels are incremented by 1. 
            
            for obj_label in labels:
                object_mask = (markers == obj_label)
                overlay[object_mask] = [255, 0, 0]  
                object_probabilities = full_prob_map[object_mask]
                confidence_score = np.mean(object_probabilities[object_probabilities > 0.5])
                self.confidence_scores.append(confidence_score)  
            
            return np.array(self.area) * temp_nm_per_px**2, self.confidence_scores, true_img, overlay  
        else:
            overlay_img = self.calc_length(self.prediction[-1], temp_nm_per_px, full_prob_map, overlay)
            return np.array(self.lengths) * temp_nm_per_px, self.confidence_scores, true_img, overlay_img 

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
        filtered_labels = list(df[(df.mean_intensity > 100) & (df.area > 1.5/nm_px_ratio**2)].label)
        return area, markers, filtered_labels

    def calc_length(self, img, nm_px_ratio, full_prob_map, overlay):
        """
        Estimates the length of precipitates
        """
        grey = img[:,:,0]
        contours, hierarchy = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        l      = []
        conf_score = []
        for contour in contours:
            (center), (width,height), angle = cv2.minAreaRect(contour)
            length = np.max([width,height])
            l.append([length, angle+(angle<0)*90])
            mask = np.zeros_like(grey, dtype=np.uint8) # Mask of the contour 
            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED) # Filling the area inside contour and contour 
            # with values of 1. Rest of pixels in the image are 0. Drawing only 1 contour(1 precipitate)
            # Extract the probability values for the pixels within the contour
            object_probabilities = full_prob_map[mask > 0]
            conf_score.append(np.mean(object_probabilities[object_probabilities > 0.5]))

        for index, (length, angle) in enumerate(l):
            median_angle = np.median([angle for (length, angle) in l if length*nm_px_ratio > 5]) #Find angles of all detections longer than 5nm
            error        = 5.0 #degrees
            if  (median_angle - error<angle<median_angle + error) and length*nm_px_ratio>3: #If precipitate in correct direction (within error) and longer than 3nm, accept detection
                mask = np.zeros_like(grey, dtype=np.uint8)
                cv2.drawContours(mask, [contours[index]], -1, 1, thickness=cv2.FILLED)
                overlay[mask > 0] = [255, 0, 0]
                self.lengths.append(length) 
                self.confidence_scores.append(conf_score[index])

        return overlay

class YOLOEvaluator(Evaluator):
    def __init__(self, dataset_dir, model, nm_per_px, cross, device):
        super().__init__(dataset_dir, model, nm_per_px, cross, device) # Inherits what is common for all model evaluator from Evalutaor class
        self.size      = 1024
        self.model = model
        self.threshold = 0.25

        print('YOLOv11 Model Loaded')

        
    def predict(self, img, nm_px_ratio):
        """
        Performs YOLOv11 prediction on a single image
        """
        self.area = []
        self.lengths = []
        print('Prediction confidence threshold set to {0:.2f}'.format(self.threshold))

        
        if isinstance(img, np.ndarray):  # convert to RGB if grayscale(.dm3 case), if image not .dm3 file, img variable is a string
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
           

        # .predict() accepts both image files, str of image path, numpy array.
        result = self.model.predict(img, conf=self.threshold)[0]
        self.confidence_scores = result.boxes.conf
        orig_img = result.orig_img
        orig_img = cv2.resize(orig_img, dsize=(self.size, self.size))
        print(np.shape(orig_img))
        overlay = orig_img.copy()
        
        if self.cross:
            masks = result.masks.data
            
            for i, mask in enumerate(masks):

                binary_mask = mask > 0  # Convert to binary (1 where mask is present, else 0)
                #print(np.shape(binary_mask))
                pixel_count = binary_mask.sum().item()  
                area = pixel_count * ((nm_px_ratio*2) ** 2) # Muliplying nm_px_ratio with 2 since images are resized
                self.area.append(area)
                overlay[binary_mask > 0] = [255, 0, 0]
                

            return self.area, self.confidence_scores, orig_img, overlay
        
        else:
            masks = result.masks.data

            for i, mask in enumerate(masks):
                binary_mask = mask > 0  
                binary_mask_np = binary_mask.cpu().numpy()

                # Apply erosion to make the mask thinner
                kernel = np.ones((2, 2), np.uint8)
                binary_mask_eroded = cv2.erode(binary_mask_np.astype(np.float32), kernel, iterations=4)

                # Remove regions touching the border
                binary_mask = clear_border(binary_mask_eroded > 0, buffer_size=10)

                # Clear border and erode heavily influence the results

                points = np.argwhere(binary_mask)[:, ::-1]  # Convert from (row, column) to (x, y)

                if len(points) >= 5:  # Ensure sufficient points for minAreaRect
                    rect = cv2.minAreaRect(points.astype(np.float32))
                    (center), (width, height), angle = rect
                    length = np.max([width, height]) * nm_px_ratio * 2
                    self.lengths.append(length)

                    
                    overlay[binary_mask > 0] = [255, 0, 0]

            return self.lengths, self.confidence_scores, orig_img, overlay
        