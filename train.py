import os
import sys
import time
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
Take the pre-formatted data and train the model on it.
"""

# def read():
def read_Nshuffle():
	"""Splits and shuffles training data into train, test, and validation sets."""
	rent = "/Volumes/HomeXx/compuir/hands_ml/FreiHAND_pub_v2"
	with np.load(os.path.join(rent, "preprocessed.npz")) as data:
		xtrain, ytrain = data['xtrain'], data['ytrain']
		xtest, ytest = data['xtest'], data['ytest']
		xval, yval = data['xval'], data['yval']

	train_data = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
	test_data = tf.data.Dataset.from_tensor_slices((xtest, ytest))
	val_data = tf.data.Dataset.from_tensor_slices((xval, yval))

	shuffled_train = train_data.shuffle(buffer_size=10000)
	shuffled_test = test_data.shuffle(buffer_size=10000)
	shuffled_val = val_data.shuffle(buffer_size=10000)

	return shuffled_train, shuffled_test, shuffled_val

def process_imgs(t_img, label): # pass images read directly to the model
	img = tf.io.read_file(t_img) # read the image's contents
	img = tf.io.decode_jpeg(img, channels=3) # decode it
	img = tf.image.resize(img, [224, 224]) # reshape for consistency
	img = img / 255.0 # normalize from 0 - 1
	return img, label

def batch_Nnormal(shuffled_train, shuffled_test, shuffled_val):
	batch_sz = 50
	_train_data = shuffled_train.map(process_imgs)
	train_data_ = _train_data.batch(batch_sz)

	test_data_ = shuffled_test.map(process_imgs)
	_test_data = test_data_.batch(batch_sz)

	val_data_ = shuffled_val.map(process_imgs)
	_val_data = val_data_.batch(batch_sz)

	return train_data_, _test_data, _val_data

def design():
	model = tf.keras.Sequential([
		# tf.keras.layers.Input(shape=(224, 224, 1)),
		tf.keras.layers.Input(shape=(224, 224, 3)),

		tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
		tf.keras.layers.MaxPooling2D((2, 2)),

		tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
		tf.keras.layers.MaxPooling2D((2, 2)),

		tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
		tf.keras.layers.MaxPooling2D((2, 2)),

		tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
		tf.keras.layers.MaxPooling2D((2, 2)),

		tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
		tf.keras.layers.MaxPooling2D((2, 2)),

		tf.keras.layers.Flatten(),

		# tf.keras.layers.Dense(1024, activation='relu'),
		# tf.keras.layers.Dropout(0.7),
		tf.keras.layers.Dense(1024, activation='linear'),
		tf.keras.layers.Dropout(0.2),

		tf.keras.layers.Dense(42, activation='linear')
	])

	return model

def main():
	"""Trains a tf.keras.Sequential model on the given images & coordinate pairs."""
	try:
		train, test, val = read_Nshuffle()
		train_data, test_data, val_data = batch_Nnormal(train, test, val)

		model = design()

		model.compile(
			optimizer=tf.keras.optimizers.AdamW(),
			# optimizer='adam',
			# loss='mse',
			loss=tf.keras.losses.Huber(delta=10.0),
			metrics=['mae']
		)

		n_ep = 7
		history = model.fit(
			train_data,
			epochs=n_ep, 
			validation_data=(val_data),
			verbose=1
		)

		loss, mae = model.evaluate(test_data)

		print(f"test loss: {loss:.4f}; test mae: {mae:.4f}")

		# hist(history) # visualize
		model.save("handModel_iv226k.keras")
		record(history.history, model.optimizer.get_config()) # track
	
	except KeyboardInterrupt:
		print("boss kill't it; shutter down")
		sys.exit(0)

def hist(history):
	acc = history.history['mae']
	val_acc = history.history['val_mae']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(acc) + 1)

	plt.figure(figsize=(12, 5))

	plt.subplot(1, 2, 1)
	plt.plot(epochs, acc, 'b', label='Training mae')
	plt.plot(epochs, val_acc, 'r', label='Validation mae')
	plt.title('Training and Validation MAE')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Average Error')
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(epochs, loss, 'b', label='Training mse')
	plt.plot(epochs, val_loss, 'r', label='Validation mse')
	plt.title('Training and Validation MSE')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.legend()

	plt.show()

def record(history, config):
	record = "/Volumes/HomeXx/compuir/hands_ml/recorded_stats.md"

	jshistory = json.dumps(history, indent=4, default=str)
	jsconfig = json.dumps(config, indent=4, default=str)

	with open(record, 'a') as f:
		f.write(f"\n{jsconfig}\n{jshistory}\n")

if __name__=="__main__":
	main()