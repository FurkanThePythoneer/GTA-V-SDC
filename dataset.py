import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
import cv2

from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD # for normalizing image



# Dataset Generator:
class SDC_V0_DataRetriever(torch.utils.data.Dataset):
    def __init__(self, df, mode, albumentations=True, image_sizes=(480,270)):
        super(SDC_V0_DataRetriever,self).__init__()
        self.df = df
        self.image_sizes = image_sizes
        self.mode = mode
        self.albumentations = albumentations

        if self.mode == 'train':
            self.transform = A.Compose([
            A.Resize(self.image_sizes[0], self.image_sizes[1]),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
            ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = cv2.imread('{}/train/{}'.format(train_path, self.df.id.values[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.albumentations:
            img = self.transform(image=img)['image']
        else:
            raise NotImplementedError('You have to use augmentations sorry.')
        #img = img/255.0
        label = self.df.loc[index, 'choice']
        label = eval(label)
        '''
        if [1,0,0,0,0,0,0,0,0] == label:
            label = 1
        elif [0,1,0,0,0,0,0,0,0] == label:
            label = 2
        elif [0,0,1,0,0,0,0,0,0] == label:
            label = 3      
        elif [0,0,0,1,0,0,0,0,0] == label:
            label = 4
        elif [0,0,0,0,1,0,0,0,0] == label:
            label = 5
        elif [0,0,0,0,0,1,0,0,0] == label:
            label = 6
        elif [0,0,0,0,0,0,1,0,0] == label:
            label = 7
        elif [0,0,0,0,0,0,0,1,0] == label:
            label = 8
        elif [0,0,0,0,0,0,0,0,1] == label:
            label = 9
        else:
            raise ValueError('!')
        '''
        label = torch.Tensor(label)

        return img, label