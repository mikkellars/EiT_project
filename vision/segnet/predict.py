"""
"""

import os
import sys
sys.path.append(os.getcwd())

import time
import torch
import numpy as np
from torchvision import transforms
from vision.segnet.model import *


class PredictSegNet:

    def __init__(self, model_file: str):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_model = torch.load(model_file)
        self.model = Segnet(3, 1)
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
        pred[pred > 0.05] = 255
        pred[pred <= 0.5] = 0

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

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from PIL import Image

    path = 'vision/data/lorenz-fence-inspection-examples'
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith('.jpg') or f.endswith('.png')
    ]

    predictor = PredictSegNet(model_file='vision/segnet/models/segnet_model.pt')

    loop = tqdm(files)
    for f in loop:
        img = Image.open('vision/data/lorenz-fence-inspection-examples/crop_fence.jpg').convert('RGB')
        pred = predictor.predict(img)
        predictor.show(np.array(img), pred)

    print('Done!')
