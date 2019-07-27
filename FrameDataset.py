
from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np
import pickle
import scipy.io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class FrameDataset(Dataset):
    def __init__(self,Frames_Dir,ChunkSize=6,transform=None):
        self.transform=transform
        self.frameslist = os.listdir(Frames_Dir)
        self.frameslist = sorted(self.frameslist,key=self.sortt)
        self.frameslist = self.frameslist[ChunkSize//2::ChunkSize]        
    def sortt(self,x):
      return int(x[0:-6])
                                     
    def __len__(self):
        return len(self.frameslist)

    def __getitem__(self, idx):
      frame = cv2.imread(Frames_Dir+'/'+self.frameslist[idx])
      frame = cv2.resize(frame, (224, 224))
      frame = transforms.functional.to_pil_image(frame)
      label = int(self.frameslist[idx][-5])
      label = torch.LongTensor([label])
      if self.transform:
        frame = self.transform(frame)
      return frame,label
                                     
                                  
