import tensorflow as tf
import cv2
import numpy as np
from models import xception_model # Using Xception model (version: 1)
from madgrad import MadGrad # A different optimizer method that beats adam and sgd

import pandas as pd

from utils.dataset import *
from utils.tpu_utils import *


#from kaggle_datasets import KaggleDatasets # GCS path. for TPU

def get_strategy():
	strategy = auto_select_accelerator(); return strategy

def train(epochs, test_size, init_lr, min_lr, strategy, dataframe, device='GPU', n_labels = 9, batch_size=64):
	train_df, val_df = make_train_test_split(df=dataframe, test_size=test_size)
	
	print('\n')
	print('Epochs: ', epochs)
	print('test_size:', test_size)
	print('init_lr: ',init_lr)
	print('min_lr:  ', min_lr)
	print('Batch size: ',batch_size)
	print('device: ',device)
	print('Training DF:\n{}'.format(df.head(3)))
	print('Val df: {}'.format(val_df))
	print('\n')
	
	train_paths  = train_df['id'].to_list()
	train_labels = train_df['choice'].to_list()
	
	valid_paths  = val_df['id'].to_list()
	valid_labels = val_df['choice'].to_list()

	decoder = build_decoder(with_labels=True, target_size=(480,270), ext='png')
	test_decoder = build_decoder(with_labels=False, target_size=(480,270),ext='png')

    train_dataset = build_dataset(
        train_paths, train_labels, bsize=batch_size, decode_fn=decoder
    )

    valid_dataset = build_dataset(
        valid_paths, valid_labels, bsize=batch_size, decode_fn=decoder,
        repeat=False, shuffle=False, augment=False
    )

    with strategy.scope():
    	model = xception_model(input_shape=(480,270,3), weights='imagenet',
    							 include_top=False, num_labels=n_labels)

    	model.compile(optimizer=MadGrad(learning_rate=init_lr),
    				  loss='categorical_crossentropy',
    				  metrics=['accuracy'])

    	print(model.summary())

    steps_per_epoch = train_paths.shape[0] // batch_size

	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		f'/kaggle/working/xception_v1-480.h5', save_best_only=True, monitor='val_loss', mode='min')

	lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, min_lr=min_lr, mode='min')

	history = model.fit(
		train_dataset,
		epochs=epochs,
		verbose=1,
		callbacks=[checkpoint, lr_reducer],
		steps_per_epoch=steps_per_epoch,
		validation_data=valid_dataset)
    

	return history

	
if __name__ == '__main__':
	#df = pd.read_csv('/kaggle/input/gta-v-data/training_png_dataset/training_labels.csv')
	df = pd.read_csv('utils/v3_train.csv')

	strategy = get_strategy()
	history  = train(epochs=10, test_size=0.20, init_lr=1e-3, min_lr=1e-6,
					strategy=strategy, dataframe=df, device='GPU')

	hist_df = pd.DataFrame(history.history)
	hist_df.to_csv(f'/kaggle/working/history.csv')

