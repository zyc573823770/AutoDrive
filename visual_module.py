import argparse
import importlib
import json
import os
import pprint
import sys
import time
from copy import deepcopy
import win32gui
import math
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageGrab
from torch import nn

from config import system_configs
from nnet.py_factory import NetworkFactory
from utils import normalize_
import keys

torch.backends.cudnn.benchmark = False
GREEN = [0, 255, 0]
WHITE = [255, 255, 255]
RED = [0, 0, 255]

def PostProcess(outputs, target_sizes):
    out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2
    prob = F.softmax(out_logits, -1)
    _, labels = prob.max(-1)
    labels[labels != 1] = 0
    results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)

    return results

def get_lane_model():
    input_size = [360, 640]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
    with open('config\\LSTR.json', "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = 'LSTR'
    system_configs.update_config(configs["system"])
    nnet = NetworkFactory()

    with open('cache\\nnet\\LSTR\\LSTR_500000.pkl', "rb") as f:
        params = torch.load(f)
        model_dict = nnet.model.state_dict()
        if len(params) != len(model_dict):
            pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
        else:
            pretrained_dict = params
        model_dict.update(pretrained_dict)

        nnet.model.load_state_dict(model_dict)
    nnet.cuda()
    nnet.eval_mode()
    return nnet, input_size, mean, std

def lane_detection(ori_image, mean, std, input_size, nnet, point=True):
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[0:2]
    images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
    masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
    orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
    pad_image     = image.copy()
    pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
    resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
    resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
    masks[0][0]   = resized_mask.squeeze()
    resized_image = resized_image / 255.
    normalize_(resized_image, mean, std)
    resized_image = resized_image.transpose(2, 0, 1)
    images[0]     = resized_image
    images        = torch.from_numpy(images).cuda(non_blocking=True)
    masks         = torch.from_numpy(masks).cuda(non_blocking=True)
    torch.cuda.synchronize(0)  # 0 is the GPU id
    outputs, _      = nnet.test([images, masks])
    torch.cuda.synchronize(0)  # 0 is the GPU id
    results = PostProcess(outputs, orig_target_sizes)

    pred = results[0].cpu().numpy()
    img  = pad_image
    img_h, img_w, _ = img.shape
    pred = pred[pred[:, 0].astype(int) == 1]
    # overlay = np.zeros_like(img, np.uint8)
    overlay_rgb = img.copy()
    point_xy = []
    for i, lane in enumerate(pred):
        lane = lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions

        # generate points from the polynomial
        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                            lane[5]) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]
        point_xy.append(points)
        if point:
            for xxx, yyy in points:
                # cv2.circle(overlay, (xxx, yyy), 1, color=WHITE, thickness=1)
                cv2.circle(overlay_rgb, (xxx, yyy), 1, color=GREEN, thickness=1)
        else:
            for current_point, next_point in zip(points[:-1], points[1:]):
                # overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=WHITE, thickness=1)
                overlay_rgb = cv2.line(overlay_rgb, tuple(current_point), tuple(next_point), color=GREEN, thickness=1)

    return overlay_rgb, point_xy

def move_window(hwnd, x, y, n_width, n_height, b_repaint):
    win32gui.MoveWindow(hwnd, x - 7, y, n_width, n_height, b_repaint)

def transform_point(pointxy, M):
    result = []
    for lanes in pointxy:
        ps = lanes.shape[0]
        pad = np.ones((ps,1), dtype=lanes.dtype)
        pad = np.concatenate([lanes, pad], axis=1)
        res = np.matmul(pad, M.T)
        result.append(np.floor(res[:,:2]/res[:,-1][:, np.newaxis]).astype(np.int32))
    return result

if __name__ == "__main__":
    nnet, input_size, mean, std = get_lane_model()
    # 图片查看的校正
    # hwnd = win32gui.FindWindow(None, "图片查看")
    # move_window(hwnd, 10, 10, 650, 650, True)
    # bbox = (40,119,840,719)

    while(True):
        last_time = time.time()
        image =  np.array(ImageGrab.grab(bbox=bbox))

        mix, points_xy = lane_detection(image, mean, std, input_size, nnet, True)

        # original pts
        pts_o = np.float32([[0, 200], [0, 600], [800, 600], [800, 200]]) # 这四个点为原始图片上数独的位置
        pts_d = np.float32([[0, 0], [336, 400], [444, 400], [800, 0]]) # 这是变换之后的图上四个点的位置

        # get transform matrix
        M = cv2.getPerspectiveTransform(pts_o, pts_d)
        # apply transformation
        black_bg = np.zeros((400, 800, 3), dtype=np.float32)
        # black_bg = cv2.warpPerspective(black_bg, M, (800, 400))
        # print(black_bg.shape)
        new_points = transform_point(points_xy, M)
        ploynomial = []
        left_lane = (1000,None)
        right_lane = (1000,None)
        for lanes in new_points:
            lanes = lanes[(lanes[:,0]>=0)&(lanes[:,1]>=0)]
            ploynomial.append(np.polyfit(lanes[:,1]/400, lanes[:,0]/800, deg=3))
            for xxx,yyy in lanes:
                cv2.circle(black_bg, (xxx, yyy), 1, color=WHITE, thickness=3)
            a,b,c,d = ploynomial[-1]

            abcd = a+b+c+d
            if abcd<0.5 and (0.5-abcd)<left_lane[0]:
                left_lane = (0.5-abcd, ploynomial[-1])
            if 0.5<abcd and (abcd-0.5)<right_lane[0]:
                right_lane = (abcd-0.5, ploynomial[-1])
        
        if left_lane[0]!=1000 and right_lane[0]!=1000:

            for xx in range(400):
                x = xx/400
                a,b,c,d = left_lane[1]
                y1 = np.floor((a*x**3+b*x**2+c*x+d)*800).astype(np.int32)
                # cv2.circle(black_bg, (y1, xx), 1, color=GREEN, thickness=1)
                a,b,c,d = right_lane[1]
                y2 = np.floor((a*x**3+b*x**2+c*x+d)*800).astype(np.int32)
                # cv2.circle(black_bg, (y2, xx), 1, color=GREEN, thickness=1)
                cv2.circle(black_bg, (np.floor((y1+y2)/2).astype(np.int32), xx), 1, color=RED, thickness=1)
            
        cv2.imshow('proce', black_bg)
        black_bg = black_bg[:,200:-200,:]
        black_bg = cv2.resize(black_bg, (90, 90))
        # mix = cv2.cvtColor(mix, cv2.COLOR_BGR2BGRA)
        # process = cv2.cvtColor(process, cv2.COLOR_BGR2BGRA)
        w = 0.6
        mix[-130:-40,-130:-40,:] = w*mix[-130:-40,-130:-40,:] + (1-w)*black_bg
        # mix[40:130,-130:-40,:] = np.clip(mix[40:130,-130:-40,:] + process, None, 255)

        # cv2.imshow('win', lane)
        cv2.imshow('ori', mix)
        print(time.time()-last_time)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
