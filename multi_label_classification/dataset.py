import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from sklearn.preprocessing import MultiLabelBinarizer
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

class ScalpDataset(D.Dataset):
    def __init__(self, path, data, label, transform=None):
        self.path = path
        self.data = data
        self.label = label
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = Image.open(self.path + self.data[idx])
        label = self.label[idx] 
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Proces_dataset():
    def __init__(self,path,json_name):
        self.path = path
        self.json_name = json_name
        with open(os.path.join(self.path ,'annotations', self.json_name),'r') as f:
            coco = json.load(f)
        self.coco = coco
    def ohe_label(self,namelist):
        cls_list = []
        images_info = self.coco['images']
        namelist = list(map(lambda x: x['file_name'],self.coco['images']))
        for i in namelist:
            idx = namelist.index(i)
            cls = images_info[idx]['class']
            cls_list.append(cls)
        ohe = MultiLabelBinarizer()
        ohe_cls = ohe.fit_transform(cls_list)
        return ohe_cls
        
    def train_val_test(self,dota_path,mode):
        train_img = sorted(list(map(lambda x: x.replace('.txt','.png'),os.listdir(f'{dota_path}/clean_train'))))
        val_img = sorted(list(map(lambda x: x.replace('.txt','.png'),os.listdir(f'{dota_path}/clean_val'))))
        test_img = sorted(list(map(lambda x: x.replace('.txt','.png'),os.listdir(f'{dota_path}/clean_test'))))
        if mode == 'train':
            train_label = self.ohe_label(train_img)
            return train_img,train_label
        if mode == 'val':
            val_label = self.ohe_label(val_img)    
            return val_img,val_label
        if mode == 'test':
            test_label = self.ohe_label(test_img)
            return test_img,test_label
