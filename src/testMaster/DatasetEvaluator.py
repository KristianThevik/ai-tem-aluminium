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


class DatasetEvaluator:
    def __init__(self, image_dir):
        """
        image_dir: The path to the folder containing images to evaluate.
        """
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.results = []  # Store results for all images

    def predict(self, evaluator):
        """
        Loops through each image in the dataset folder, evaluates the mask and generates plots.
        """
        for idx, img_path in enumerate(self.image_paths):
            print(f"Evaluating image {idx+1}/{len(self.image_paths)}: {img_path}")
            result = evaluator.evaluate(img_path)
            self.results.append(result)
            self._generate_plot(idx, result, evaluator)

    def _generate_plot(self, img_idx, result, evaluator):
        """
        Generate plots for each image based on the evaluator's results.
        """
        if evaluator.cross:
            # Plot histogram of areas (cross-section evaluation)
            areas = np.array(result["area"]) * evaluator.nm_per_px ** 2
            plt.figure()
            plt.hist(areas, bins=20, color='b', alpha=0.7)
            plt.title(f"Image {img_idx+1}: Cross-section Histogram")
            plt.xlabel('Area (nm²)')
            plt.ylabel('Count')
            plt.show()
        else:
            # Plot lengths (length evaluation)
            lengths = np.array(result["lengths"]) * evaluator.nm_per_px
            plt.figure()
            plt.hist(lengths, bins=20, color='g', alpha=0.7)
            plt.title(f"Image {img_idx+1}: Length Histogram")
            plt.xlabel('Length (nm)')
            plt.ylabel('Count')
            plt.show()

class Evaluator:
    def __init__(self, path_model: Path):
        self.path = path_model

    def evaluate(self, img):
        raise NotImplementedError("Subclasses must implement evaluate method.")

class YOLOEvaluator(Evaluator):
    def evaluate(self, img):
        # Implementation of YOLO evaluation logic
        pass

class RCNNEvaluator(Evaluator):
    def __init__(self, path_model: Path, cross=True, device='cpu'):
        super().__init__(path_model)
        self.cross = cross  # If True, evaluating cross-section; otherwise length
        self.size = 1024
        self.device = device
        self.threshold = 0.9 if cross else 0.5
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

        # Example pixel size calibration
        self.nm_per_px = 0.069661  # Example, this can be set dynamically later

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

    def evaluate(self, img_file):
        """
        Evaluate a single image.
        """
        image = self.to_tensor(img_file).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(image)
        
        # Analyze prediction results
        return self.statistics(prediction)

    def statistics(self, prediction):
        """
        Analyze mask and box predictions and return statistics like area or length.
        """
        if self.cross:
            area = []
            for i in range(len(prediction[0]['masks'])):
                mask = prediction[0]['masks'][i, 0].detach().cpu().numpy()
                score = prediction[0]['scores'][i].detach().cpu().numpy()
                if score > self.threshold:
                    mask = mask > self.threshold
                    mask_eroded = cv2.erode(mask.astype(np.float32), np.ones((2, 2), np.uint8), iterations=self.erode_it)
                    clear_mask = clear_border(mask_eroded > 0)
                    area1 = np.sum(clear_mask)
                    area2 = np.sum(mask_eroded > 0)
                    if area1 == area2:
                        area.append(area1)
            return {"area": area}
        else:
            lengths = []
            for i in range(len(prediction[0]['masks'])):
                score = prediction[0]['scores'][i].detach().cpu().numpy()
                mask = prediction[0]['masks'][i, 0].detach().cpu().numpy()
                if score > self.threshold:
                    mask = mask > self.threshold
                    mask_eroded = cv2.erode(mask.astype(np.float32), np.ones((2, 2), np.uint8), iterations=self.erode_it)
                    clear_mask = clear_border(mask_eroded > 0)
                    rect = cv2.minAreaRect(np.argwhere(clear_mask > 0))
                    lengths.append(max(rect[1]))
            return {"lengths": lengths}

class UNETEvaluator(Evaluator):
    def __init__(self, path_model: Path, cross=True, device='cpu'):
        super().__init__(path_model)
        self.cross = cross
        self.device = device
        self.size = 1024
        self.tile_size = 512

        # Load the UNet model
        self.checkpoint = torch.load(self.path, map_location=torch.device(self.device))
        self.model = UNet(in_channels=1, n_classes=2, depth=3, wf=6, padding=True)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print('UNet Model Loaded')

    def tile_img(self, arr, d) -> list:
        """
        Tile the image into equal parts of size (d x d) pixels
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
        """
        try:
            image = np.array(Image.open(file).convert('L'))
        except Exception:
            raise ValueError("Something went wrong when loading the image.")
        
        image = cv2.resize(image, dsize=(self.size, self.size))
        tensors = []
        images = self.tile_img(np.array(image), self.tile_size)

        for i in images:
            im = np.expand_dims(i, axis=0)
            im = 2 * (im / np.max(im)) - 1
            im = torch.tensor(im, dtype=torch.float32)
            tensors.append(im)
        return np.array(image), tensors

    def watershed(self, img, plot=False):
        """
        Performs the watershed algorithm on the prediction img.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = clear_border(gray)
        ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = cv2.dilate(bin_img, kernel, iterations=20)
        dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

        # Foreground area
        ret, sure_fg = cv2.threshold(dist, 0.15 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # Unknown area
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that background is not 0, but 1
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)

        img2 = color.label2rgb(markers, bg_label=1, bg_color=(0, 0, 0))
        props = measure.regionprops_table(markers, intensity_image=gray,
                                          properties=['label', 'area', 'equivalent_diameter',
                                                      'mean_intensity', 'solidity'])
        df = pd.DataFrame(props)
        area = list(df[(df.mean_intensity > 100) & (df.area > 1.5 / self.nm_per_px ** 2)].area)
        return area

    def calc_length(self, img):
        """
        Estimates the length of precipitates.
        """
        grey = img[:, :, 0]
        contours, hierarchy = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lengths = []
        for contour in contours:
            (center), (width, height), angle = cv2.minAreaRect(contour)
            length = np.max([width, height])
            lengths.append([length, angle + (angle < 0) * 90])
        return lengths

    def evaluate(self, img_file):
        """
        Evaluate a single image.
        """
        true_img, imgs = self.to_tensor(img_file)
        new_im = Image.new('RGB', (self.size, self.size))
        self.nm_per_px = 0.069661

        for index, im in enumerate(imgs):
            im = im.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(im)
                output = torch.argmax(pred, dim=1)
                output = output.squeeze(0).cpu().numpy()

            y_offset = int(self.tile_size * (index > 1))
            x_offset = int(self.tile_size * ((index) % 2))
            out = Image.fromarray(output.astype('uint8') * 255).convert('RGB')
            new_im.paste(out, box=(x_offset, y_offset))

        if self.cross:
            return {"area": self.watershed(np.array(new_im))}
        else:
            return {"lengths": self.calc_length(np.array(new_im))}
