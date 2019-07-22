from __future__ import print_function, division
import cv2
import os
import torch
import numpy as np
import pickle
import scipy.io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils




class VideoDataset(Dataset):
   
    def __init__(self,videos_dir, label_dir,chunk_step,channels,width,hight, transform=None):
    

        self.videos_dir = videos_dir
        self.label_dir  = label_dir
        self.chunk_step = chunk_step
        self.channels = channels
        self.width  = width
        self.hight  = hight
        self.transform  = transform
        video_file_list = os.listdir(self.videos_dir)
        label_file_list = os.listdir(self.label_dir)
        self.all_frames  = torch.FloatTensor(1,channels,width,hight)
        self.all_labels  = torch.FloatTensor(1)
        
        
        videos2labels={}

        for video in video_file_list:
            for labels in label_file_list:
                if video[:-9] == labels[:-10]:
                    videos2labels[video] = labels
                    break
                    
        for video in video_file_list:
            labels_file = videos2labels[video]
            new_frames, new_labels = self.readVideoLabels(videos_dir+'/'+video, label_dir+'/'+labels_file)
            self.all_frames = torch.cat((self.all_frames,new_frames),0)
            self.all_labels = torch.cat((self.all_labels,new_labels),0)

	self.all_frames = self.all_frames[2:]
        self.all_labels = self.all_labels[2:]

    def __len__(self):
        return len(self.all_frames)

    def readVideoLabels(self, videoFile, labelsFile):
        # Open the video file
        cap = cv2.VideoCapture(videoFile)
        out_frames = torch.FloatTensor(1,self.channels,self.width,self.hight)
        out_labels  = torch.FloatTensor(1)
        mat = scipy.io.loadmat(labelsFile)
        labels = mat['tlabs']
        frames2actions = {}
        frames2labels  = {}

        idx_to_action = {0:'None'              ,
                         1:'Reach To Shelf'    ,
                         2:'Retract From Shelf',
                         3:'Hand In Shelf'     ,
                         4:'Inspect Product'   ,
                         5:'Inspect Shelf'      }

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame in range(1,count+1):
            frames2actions[frame]='None'
            frames2labels[frame] = 0

        for i,action in enumerate(labels):
            for n in range(len(labels[i][0])):
                start = labels[i][0][n][0]
                end = labels[i][0][n][1]
                for key in range(start,end+1):
                    frames2actions[key]=idx_to_action[i]
                    frames2labels[key] = i+1
                    
        frame_idx=0
        for i in range(0,count):
            ret, frame = cap.read()

            if not ret:
                break
                
            frame_idx +=1

            if(frame_idx % self.chunk_step == 0):
                frame = cv2.resize(frame, (self.width, self.hight))
		frame = transforms.functional.to_pil_image(frame)

                if self.transform is not None:
                    frame = self.transform(frame)

                frame = torch.unsqueeze(frame,0)
                frame_label =  torch.Tensor([frames2labels[frame_idx]])
                out_frames = torch.cat((out_frames,frame),0)
                out_labels = torch.cat((out_labels,frame_label),0)
        return out_frames, out_labels

    def __getitem__(self, idx):
            
        return self.all_frames[idx], self.all_labels[idx]

