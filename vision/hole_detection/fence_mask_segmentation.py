"""
"""


import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms


sys.path.append(os.getcwd())


class HoleDetection:
    def __init__(self, model: str = 'unet', weights_path: str = '', n_chs: int = 3, n_cls: int = 1):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_chs = n_chs
        self.n_cls = n_cls

        # self.tfs = transforms.Compose([
        #     transforms.Resize(512),
        #     transforms.ToTensor(),
        # ])
        self.resize = transforms.Resize(size=(512, 512))

        saved_model = torch.load(weights_path)
        print(f'Loadded {model} model trained for {saved_model["epoch"]} epochs with loss {saved_model["loss"]}')

        if model == 'unet':
            from vision.unet.model import UNet
            self.model = UNet(15, n_cls)

            self.model.load_state_dict(saved_model["model_state_dict"])

            for param in self.model.parameters():
                param.requires_grad = True

            self.model.to(self.device)
        else:
            raise NotImplementedError(f'Unsupported model, got {model}') 
    
    def predict(self, x):
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x.astype(np.float32))

        x = x.to(self.device, dtype=torch.float32)
        x = x.view(1, x.size(0), x.size(1), x.size(2))

        with torch.set_grad_enabled(False):
            x = self.model(x)
        
        x = x.data.cpu().detach().numpy()[0]

        x[x <= 0.5] = 0
        x[x > 0.5] = 255

        return np.transpose(x, (1, 2, 0))[:, :, 0]



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model_path = 'vision/unet/models/detectnet_model.pt'
    hole_detection = HoleDetection(model='unet', weights_path=model_path)

    img_dir = 'vision/data/lorenz-fence-inspection-examples'
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

    for f in files:
        if f.endswith('.jpg') is False:
            continue

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

        pred = hole_detection.predict(img)

        plt.figure(figsize=(6,3))

        plt.subplot(1,2,1)
        plt.title('Input')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.title('Output')
        plt.imshow(pred)
        plt.axis('off')

        plt.tight_layout(True)
        plt.show()