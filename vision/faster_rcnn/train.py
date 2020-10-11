"""
Trainer of the Faster R-CNN.
"""


import os
import sys
sys.path.append(os.getcwd())


import matplotlib
from tqdm import tqdm
from torch.utils import data
from vision.faster_rcnn.utils import *


