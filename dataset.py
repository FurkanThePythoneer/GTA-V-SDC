import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from common import absolute_path #/kaggle/input/gta-v-data/training_png_dataset


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
				ToTensorV2()
				])


	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		img = cv2.imread('{}/train/{}'.format(absolute_path, self.df.id.values[index]))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		if self.albumentations:
			img = self.transform(image=img)['image']
		else:
			raise NotImplementedError('You have to use augmentations sorry.')
		img = img/255.0
		label = torch.Tensor(self.df.choice.values[index])

		return img, label



