import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import timm 
from timm.models.efficientnet import *
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from tqdm import tqdm

from models import *
from dataset import *

epochs = 10
img_sizes=(480, 270)
train_path = '/kaggle/input/gta-v-data/training_png_dataset'
scheduler = 'cosine'
optimizer = 'adam'
init_lr = 3e-4
batch_size = 64
num_classes = 9
mixed_precision = True
mask = False
if mask:
    raise ValueError('Mask is not available. Set it to `False`')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

train_df = pd.read_csv(f'{train_path}/training_labels.csv')
print('\n')
print('training DF:')
print(train_df.head(4))


train_dataset = SDC_V0_DataRetriever(df=train_df, mode='train',
                                     albumentations=True, image_sizes=(480,270))

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset),
                         num_workers=4, drop_last=False)

print('len(training_data): {}'.format(len(train_df)))

# get model:
model = SDC_V0_Model(num_classes=num_classes, pretrained=True)
model.to(device)

# criterion
cls_criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

# LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-1)
# scaler
scaler = torch.cuda.amp.GradScaler()

# Train Loop
# TODO: add validation!
for epoch in range(epochs):
    if epoch > 0:
        past_train_loss = cls_loss

    CHECKPOINT = 'effnetv2_m_480_270_v1-{}.pth'.format(epoch)
    print('-'*37+f' EPOCH: {epoch} '+'-'*37)
    
    #------------
    scheduler.step()
    model.train()
    
    loop = tqdm(train_loader)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device).float().long()
        labels = torch.argmax(labels, 1)

        #labels = torch.tensor(labels, dtype=torch.long, device=device)

        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            cls_outputs = model(images)
            #print(cls_outputs)
            cls_loss = cls_criterion(cls_outputs, labels)
        
        # scale stuff
        scaler.scale(cls_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_description('Epoch {}/{} | LR: {} | loss: {}'.format(epoch, epochs-1, optimizer.param_groups[0]['lr'], cls_loss))
    
    print('End of epoch: {}, training loss: {}'.format(epoch, cls_loss))

    if epoch > 0:
        if cls_loss <= past_train_loss:
            print('Loss improved from {} to {}\nSaving model: {}'.format(past_train_loss, cls_loss, CHECKPOINT))
            torch.save(model.state_dict(), CHECKPOINT)
    else:
        print('Saving the first epoch`s model')
        torch.save(model.state_dict(), CHECKPOINT)

# TODO: Add validation step.
            