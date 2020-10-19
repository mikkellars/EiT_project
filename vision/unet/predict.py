"""
"""

import os
import sys
sys.path.append(os.getcwd())

import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from vision.unet.model import *


class PredictUNet:

    def __init__(self, model_file: str):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_model = torch.load(model_file)
        self.model = UNet(3, 1)
        self.model.load_state_dict(self.save_model["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.resize = transforms.Resize(size=(512, 512))

        print(f'Loaded {model_file}.')
        print(f'The model is trained for {self.save_model["epoch"]} epochs with a loss of {self.save_model["loss"]}.')


    def predict(self, img):

        img = self.resize(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))

        img = img.to(self.device, dtype=torch.float32)
        img = img.view(1, img.size(0), img.size(1), img.size(2))

        with torch.set_grad_enabled(False):
            pred_time = time.time()
            pred = self.model(img)
            pred_time = time.time() - pred_time

        pred = pred.data.cpu().detach().numpy()[0]
        pred[pred > 0] = 255
        pred[pred < 0] = 0

        return pred[0]

    def show(self, img, pred):

        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(pred)
        plt.title('Mask')
        plt.axis('off')
        plt.tight_layout(True)
        plt.show()


class PredictDetectNet:

    def __init__(self, model_file: str):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_model = torch.load(model_file)
        self.model = UNet(15, 1)
        self.model.load_state_dict(self.save_model["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.resize = transforms.Resize(size=(512, 512))
        self.ddepth = cv2.CV_16S

        print(f'Loaded {model_file}.')
        print(f'The model is trained for {self.save_model["epoch"]} epochs with a loss of {self.save_model["loss"]}.')

    def laplacian_filter(self, img, reps: int):
        ret = img.copy()
        for i in range(reps):
            ret = cv2.Laplacian(ret, self.ddepth, ksize=3)
            ret = cv2.convertScaleAbs(ret)
        return ret

    def predict(self, img):

        img = self.resize(img)

        img = np.array(img)

        # Sobel x and y
        sobel_x = cv2.Sobel(img, self.ddepth, 1, 0)
        sobel_x = cv2.convertScaleAbs(sobel_x)

        sobel_y = cv2.Sobel(img, self.ddepth, 0, 1)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        # Laplacian
        laplace_4 = self.laplacian_filter(img, 2)

        laplace_8 = self.laplacian_filter(img, 3)

        # Transpose
        img = np.transpose(img, (2, 0, 1))
        sobel_x = np.transpose(sobel_x, (2, 0, 1))
        sobel_y = np.transpose(sobel_y, (2, 0, 1))
        laplace_4 = np.transpose(laplace_4, (2, 0, 1))
        laplace_8 = np.transpose(laplace_8, (2, 0, 1))

        # To tensor
        img = torch.from_numpy(img.astype(np.float32))
        sobel_x = torch.from_numpy(sobel_x.astype(np.float32))
        sobel_y = torch.from_numpy(sobel_y.astype(np.float32))
        laplace_4 = torch.from_numpy(laplace_4.astype(np.float32))
        laplace_8 = torch.from_numpy(laplace_8.astype(np.float32))

        img = torch.cat([img, sobel_x, sobel_y, laplace_4, laplace_8], dim=0)

        img = img.to(self.device, dtype=torch.float32)
        img = img.view(1, img.size(0), img.size(1), img.size(2))

        with torch.set_grad_enabled(False):
            pred_time = time.time()
            pred = self.model(img)
            pred_time = time.time() - pred_time

        pred = pred.data.cpu().detach().numpy()[0]
        pred[pred > 0] = 255
        pred[pred < 0] = 0

        return pred[0]

    def show(self, img, pred):

        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(pred)
        plt.title('Mask')
        plt.axis('off')
        plt.tight_layout(True)
        plt.show()


class PredictTexels:

    def __init__(self, model_file: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_model = torch.load(model_file)
        self.model = UNet(3, 1)
        self.model.load_state_dict(self.save_model["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f'Loaded {model_file}.')
        print(f'The model is trained for {self.save_model["epoch"]} epochs with a loss of {self.save_model["loss"]}.')

    def predict(self, img):
        assert img.shape == (64, 64, 3), f'Image shape should be (X, X, 3), got {img.shape}'
 
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32))

        img = img.to(self.device, dtype=torch.float32)
        img = img.view(1, img.size(0), img.size(1), img.size(2))

        with torch.set_grad_enabled(False):
            pred_time = time.time()
            pred = self.model(img)
            pred_time = time.time() - pred_time

        pred = pred.data.cpu().detach().numpy()[0]
        pred[pred > 0] = 255
        pred[pred < 0] = 0

        return pred[0]

    def show(self, img, pred):
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(pred)
        plt.title('Mask')
        plt.axis('off')
        plt.tight_layout(True)
        plt.show()


if __name__ == '__main__':
    start_time = time.time()

    predictor = PredictTexels(model_file='vision/unet/models/texel_checkpoint.pt')

    img = cv2.imread('vision/data/lorenz-fence-inspection-examples/Eskild_fig_3_16.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width = img.shape[1]
    height = img.shape[0]
    n_rows = height // 64
    n_cols = width // 64

    for i in range(0, n_rows):
        for j in range(0, n_cols):
            roi = img[i * height // n_rows : i * height // n_rows + height // n_rows,
                      j * width // n_cols : j * width // n_cols + width // n_cols]
            
            roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)
            
            pred = predictor.predict(roi)
            predictor.show(roi, pred)
    
    end_time = time.time() - start_time

    print(f'Done! It took {end_time:.4f} seconds.')
