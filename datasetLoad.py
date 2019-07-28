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
    """Dataset Class for Loading Video"""

    def __init__(self,videos_dir, label_dir,chunk_step,channels,width,hight, transform=None):
    

        self.videos_dir = videos_dir
        self.label_dir  = label_dir
        self.chunk_step = chunk_step
        self.channels = channels
        self.width  = width
        self.hight  = hight
        self.transform  = transform
        self.video_file_list = os.listdir(self.videos_dir)
        self.label_file_list = os.listdir(self.label_dir)
        self.video_file_list.sort()
        self.all_labels = torch.LongTensor(1)
        self.video2endframe = {}

        self.generate_labels()


        
        
    def generate_labels(self):
      
      videos2labels={}
      for video in self.video_file_list:
          for labels in self.label_file_list:
              if video[:-9] == labels[:-10]:
                  videos2labels[video] = labels
                  break
                  
      total_count = 0
      for video in self.video_file_list:
          cap = cv2.VideoCapture(self.videos_dir+'/'+video)
          vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          out_frames = ((vid_frames-6)//self.chunk_step)+1
          total_count += out_frames
          self.video2endframe[video]= total_count

          labels_file = videos2labels[video]
          mat = scipy.io.loadmat(self.label_dir+'/'+labels_file)
          labels = mat['tlabs']
          frames2labels  = {}



          for f in range(1,vid_frames+1):
              frames2labels[f] = 0


          for i,action in enumerate(labels):
              for n in range(len(labels[i][0])):
                  start = labels[i][0][n][0]
                  end = labels[i][0][n][1]
                  for key in range(start,end+1):
                      frames2labels[key] = i+1

          for k in range (7,vid_frames+1,self.chunk_step):
              self.all_labels = torch.cat((self.all_labels,torch.LongTensor([frames2labels[k]])),0)

      self.all_labels = self.all_labels[1:]

    def __len__(self):
        return len(self.all_labels[1:])
    
    
    def get_frame(self,videoFile, count_from_offset):
        cap = cv2.VideoCapture(self.videos_dir+'/'+videoFile)
        cap.set(cv2.CAP_PROP_POS_FRAMES,(count_from_offset*self.chunk_step)-6)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (self.width, self.hight))
      
        frame = transforms.functional.to_pil_image(frame)

        if self.transform is not None:
            frame = self.transform(frame)            
        return frame
      
    

    def __getitem__(self, idx):
      starting_frame=0
      for video in self.video_file_list:
          if idx<self.video2endframe[video]:
              count_from_offset = idx-starting_frame
              frame = self.get_frame(video,count_from_offset)
              label = self.all_labels[idx]
              break
          starting_frame = self.video2endframe[video] 
            
      return frame, label

