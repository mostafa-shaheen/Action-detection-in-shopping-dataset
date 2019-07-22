from __future__ import print_function, division
import cv2
import os
import torch
import numpy as np
import pickle
import scipy.io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datasetLoading import VideoDataset

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

idx_to_action = {        0:'None'              ,
                         1:'Reach To Shelf'    ,
                         2:'Retract From Shelf',
                         3:'Hand In Shelf'     ,
                         4:'Inspect Product'   ,
                         5:'Inspect Shelf'      }

videos_dir = '/home/mostafa/git_workspace/Action-detection-in-shopping-dataset/test/video'
label_dir = '/home/mostafa/git_workspace/Action-detection-in-shopping-dataset/test/labels'
chunk_step = 6
channels = 3
width = 224
hight = 224

data_transform =    transforms.Compose([
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
print('start')
dataset = VideoDataset(videos_dir, label_dir,chunk_step,channels,width,hight, transform=data_transform)
print('done')
dataloader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=0)


for frame, label in dataloader:
    frame = torch.squeeze(frame, 0)
    frame = frame.permute(1, 2, 0)
    frame = frame.numpy()
    cv2.imshow('Stream', frame)
    print(idx_to_action[int(label)])
    ch = 0xFF & cv2.waitKey(100)
    if ch == 27:
        break
