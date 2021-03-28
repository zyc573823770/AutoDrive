import os
import torch
import cv2
import json
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs
from PIL import Image, ImageGrab

num = 1
for name in os.listdir('images'):
    path = os.path.join('images', name)
    img = cv2.imread(path)
    img = img[32:-1,:-1,:]
    print(img.shape)
    cv2.imwrite(os.path.join('images', "{}.jpg".format(num)), img)
    num+=1