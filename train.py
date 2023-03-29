from dataset import ScalpDataset,Proces_dataset
from model import StyleModel

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

import argparse


import timm
import random
import torch.backends.cudnn as cudnn
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
device = 'cuda:0'
def train(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    processing = Proces_dataset(opt.path, 'instances_default.json')
    train_img,train_label = processing.train_val_test('/data/dmc/3_dota','train')
    val_img,val_label = processing.train_val_test('/data/dmc/3_dota','val')
    transformer = T.Compose([
        T.ToTensor(),
#         T.Normalize((49.14544531, 52.61869218, 52.28344427), (75.67587604, 80.40975236, 81.53936774)),
        ])
    train_dataset = ScalpDataset(opt.path+'images/', train_img,train_label,transform = transformer)
    val_dataset = ScalpDataset(opt.path+'images/', val_img,val_label,transform = transformer)

    
    train_dataloader = D.DataLoader(train_dataset, batch_size = opt.batch, shuffle = True, drop_last=False)
    val_dataloader = D.DataLoader(val_dataset, batch_size = opt.batch, shuffle = True, drop_last=False)

    pretrained = timm.create_model('convnext_base',pretrained=True)
    model = StyleModel(pretrained)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=0.1,eta_min=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    total_step = len(train_dataloader)
    best_val_acc = 0
    EPOCH = 30
    for epoch in range(EPOCH):
        train_acc_list = []
        running_loss = 0
        
        model.train()
        
        for i, (images, labels) in tqdm(enumerate(train_dataloader),position=0, leave=True):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()

            probs= model(images)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            probs  = probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            preds = probs > 0.5
            batch_acc = (labels == preds).mean()
            train_acc_list.append(batch_acc)
        
        train_acc = np.mean(train_acc_list)
        print(f'Epoch [{epoch+1}/{EPOCH}], Step [{i+1}/{total_step}], Loss: {running_loss/total_step}, Acc {train_acc}')

        model.eval()
        valid_acc_list = []
        with torch.no_grad():

            for images, labels in val_dataloader:
                images = images.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                probs = model(images)
                # valid_loss = criterion(probs, labels)

                probs  = probs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                preds = probs > 0.5
                batch_acc = (labels == preds).mean()
                valid_acc_list.append(batch_acc)
                
            val_acc = np.mean(valid_acc_list)
            print(f'Validation acc: {val_acc}')
        lr_scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{opt.save_dir}best_cls_{epoch}.pth')
        torch.save(model.state_dict(), f'{opt.save_dir}cls_{epoch}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='/data/dmc/2_coco/')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--save_dir',type=str,default = '/data/dmc/cls_ckpt/')
    opt = parser.parse_args()
    train(opt)