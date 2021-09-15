import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from common import *
from dataset import *
from models import *

epochs = 30

df = pd.read_csv(f'{absolute_path}/training_labels.csv') #'/kaggle/input/gta-v-data/training_png_dataset'
train_dataset = SDC_V0_DataRetriever(df=df, mode='train', albumentations=True, image_sizes=(480,270))

train_loader = DataLoader(train_dataset, batch_size=16, sampler=RandomSampler(train_dataset),
                         num_workers=4, drop_last=False)



print('Train size: {} '.format(len(df)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = SDC_V0_Model(num_classes=9, pretrained=True)
model.to(device)

#------
cls_criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-1)

scaler = torch.cuda.amp.GradScaler()

train_losses = []
for epoch in range(epochs):
    if epoch > 0:
        past_train_loss = cls_loss

    CHECKPOINT = 'effnetv2_m_480_270_v0-{}.pth'.format(epoch)
    
    print('Epoch: {}'.format(epoch))
    scheduler.step()
    model.train()

    loop = tqdm(train_loader)
    for images,labels in loop:
        #print('img: {}'.format(images.shape))
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.float()

        optimizer.zero_grad()
        #mixed precision
        with torch.cuda.amp.autocast():
            cls_outputs = model(images)
            cls_loss = cls_criterion(cls_outputs, labels)
            train_losses.append(cls_loss.item())
        
        scaler.scale(cls_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description('Epoch {}/{} | LR: {} | cls_loss: {}'.format(epoch, epochs-1, optimizer.param_groups[0]['lr'], cls_loss))

    print('train loss: {} '.format(cls_loss))
        
    if epoch > 0:
        if cls_loss > past_train_loss:
            print('Train loss didn`t improve... :(')
    else:
        print('Loss improved! Saving model: {}'.format(CHECKPOINT))
        torch.save(model.state_dict(), CHECKPOINT)



