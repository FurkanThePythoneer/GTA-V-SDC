import tensorflow as tf
import cv2
import numpy as np
from models import * # Using effnet model (version: 1)
from madgrad import MadGrad # A different optimizer method that beats adam and sgd

import math
import pandas as pd

from utils.dataset import *
from utils.tpu_utils import *

import wandb
from wandb.keras import WandbCallback

def log_in_wandb(key='95f7dfe314870c15547061ef7e42f3cb18f3cf31'):
	wandb.login(key=key)
#from kaggle_datasets import KaggleDatasets # GCS path. for TPU

def make_train_test_split(df, test_size=0.22):
    '''
    Create train, test
    split for dataframes..
    '''
    splits = []
    train_num = len(df) - math.ceil(len(df)*test_size)
    for i in range(len(df)):
        if i < train_num:
            splits.append('train')
        else:
            splits.append('val')

    df['split'] = splits
    # ---------------
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    train_df = df.loc[df['split'] == 'train']
    valid_df = df.loc[df['split'] == 'val']

    del df
    
    return train_df, valid_df

def get_strategy():
	strategy = auto_select_accelerator(); return strategy

def train(epochs, test_size, init_lr, min_lr, strategy, dataframe, device='GPU', n_labels = 9, batch_size=32):
	log_in_wandb()
	print('Logged in to wandb.ai')

	# Initialize wandb with your project name
	run = wandb.init(project='self_driving_car_gta-v',
	                 config={  # and include hyperparameters and metadata
	                     "learning_rate": 0.001,
	                     "epochs": 10,
	                     "batch_size": 16,
	                     "loss_function": "categorical_crossentropy",
	                     "architecture": "CNN",
	                     "dataset": "SDC-GTA-V-V0"
	                     })
	config = wandb.config
	tf.keras.backend.clear_session()

	train_df, val_df = make_train_test_split(df=dataframe, test_size=test_size)

	print('Epochs: ', epochs)
	print('test_size:', test_size)
	print('init_lr: ',init_lr)
	print('min_lr:  ', min_lr)
	print('Batch size: ',batch_size)
	print('device: ',device)
	print('Training DF:\n{}'.format(df.head(3)))
	print('Val df: {}'.format(val_df))

	
	train_paths  = train_df['id'].to_list()
	train_labels = train_df['choice'].to_list()
	
	valid_paths  = val_df['id'].to_list()
	valid_labels = val_df['choice'].to_list()
	
	actual_labels  = [] # training labels
	actual_labels2 = [] # validation labels
	
	for choice in train_labels:
		#print(type(choice))
		if choice   == '[1, 0, 0, 0, 0, 0, 0, 0, 0]':
			actual_labels.append(0)
		elif choice == '[0, 1, 0, 0, 0, 0, 0, 0, 0]':
			actual_labels.append(1)
		elif choice == '[0, 0, 1, 0, 0, 0, 0, 0, 0]':
			actual_labels.append(2)
		elif choice == '[0, 0, 0, 1, 0, 0, 0, 0, 0]':
			actual_labels.append(3)						
		elif choice == '[0, 0, 0, 0, 1, 0, 0, 0, 0]':
			actual_labels.append(4)
		elif choice == '[0, 0, 0, 0, 0, 1, 0, 0, 0]':
			actual_labels.append(5)
		elif choice == '[0, 0, 0, 0, 0, 0, 1, 0, 0]':
			actual_labels.append(6)
		elif choice == '[0, 0, 0, 0, 0, 0, 0, 1, 0]':
			actual_labels.append(7)						
		elif choice == '[0, 0, 0, 0, 0, 0, 0, 0, 1]':
			actual_labels.append(8)
		else:
			raise ValueError('No choice?!')
	

	for choice in valid_labels:
		#print(choice)
		if choice   == '[1, 0, 0, 0, 0, 0, 0, 0, 0]':
			actual_labels2.append(0)
		elif choice == '[0, 1, 0, 0, 0, 0, 0, 0, 0]':
			actual_labels2.append(1)
		elif choice == '[0, 0, 1, 0, 0, 0, 0, 0, 0]':
			actual_labels2.append(2)
		elif choice == '[0, 0, 0, 1, 0, 0, 0, 0, 0]':
			actual_labels2.append(3)						
		elif choice == '[0, 0, 0, 0, 1, 0, 0, 0, 0]':
			actual_labels2.append(4)
		elif choice == '[0, 0, 0, 0, 0, 1, 0, 0, 0]':
			actual_labels2.append(5)
		elif choice == '[0, 0, 0, 0, 0, 0, 1, 0, 0]':
			actual_labels2.append(6)
		elif choice == '[0, 0, 0, 0, 0, 0, 0, 1, 0]':
			actual_labels2.append(7)						
		elif choice == '[0, 0, 0, 0, 0, 0, 0, 0, 1]':
			actual_labels2.append(8)
		else:
			raise ValueError('No choice?!')

	actual_labels = tf.keras.utils.to_categorical(actual_labels, 9)
	actual_labels2 = tf.keras.utils.to_categorical(actual_labels2, 9)



	decoder = build_decoder(with_labels=True, target_size=(480,270), ext='png')
	test_decoder = build_decoder(with_labels=False, target_size=(480,270),ext='png')

	train_dataset = build_dataset(
		train_paths, actual_labels, bsize=batch_size, decode_fn=decoder
	)

	valid_dataset = build_dataset(
		valid_paths, actual_labels2, bsize=batch_size, decode_fn=decoder,
		repeat=False, shuffle=False, augment=False
	)

	with strategy.scope(): # get the model
		model = effnetv2_b2_model(input_shape=(480,270,3), weights='imagenet',
								 include_top=False, num_labels=n_labels)

		model.compile(optimizer=MadGrad(learning_rate=init_lr),
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])

		#print(model.summary())

	#steps_per_epoch = train_paths.shape[0] // batch_size
	steps_per_epoch = 69912 // batch_size	
	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		f'/kaggle/working/effnetv2-b0_v1-480.h5', save_best_only=True, monitor='val_loss', mode='min')

	lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
		monitor="val_loss", patience=3, min_lr=min_lr, mode='min')

	history = model.fit(
		train_dataset,
		epochs=epochs,
		verbose=1,
		callbacks=[checkpoint, lr_reducer, WandbCallback()],
		steps_per_epoch=steps_per_epoch,
		validation_data=valid_dataset)
	
	run.finish()

	return history

	
if __name__ == '__main__':
	#df = pd.read_csv('/kaggle/input/gta-v-data/training_png_dataset/training_labels.csv')
	df = pd.read_csv('utils/v3_train.csv')

	strategy = get_strategy()
	history  = train(epochs=10, test_size=0.20, init_lr=1e-3, min_lr=1e-6,
					strategy=strategy, dataframe=df, device='GPU', batch_size=16)

	hist_df = pd.DataFrame(history.history)
	hist_df.to_csv(f'/kaggle/working/history.csv')

