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

while(True):
    image =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[0:2]

    # gps_track = np.array([[[168,84,243]]]).repeat(600, axis=0).repeat(800, axis=1)
    height, width, _ = image.shape
    select_r = np.array(image[:,:,2]==168, dtype=np.uint8)
    select_g = np.array(image[:,:,1]==84, dtype=np.uint8)
    select_b = np.array(image[:,:,0]==243, dtype=np.uint8)
    oup_img = select_r*select_g*select_b
    # print(np.sum(oup_img))
    cv2.imshow('win', oup_img*255)
    cv2.imshow('win2', image)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break