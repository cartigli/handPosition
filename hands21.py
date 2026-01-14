import os
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
test loss: 0.3974; test avg. pixel err: 4.7683 (pixel_error: 10.6791)
"""

def targs():
	"""Finds the target's coordinate values."""
	targets = pd.read_json("/Volumes/HomeXx/compuir/hands/FreiHAND_pub_v2/training_xyz.json")
	y_ = np.array(targets.values.tolist())

	return y_, targets

def vals():
	"""Finds the images' extraneous data."""
	angles = pd.read_json("/Volumes/HomeXx/compuir/hands/FreiHAND_pub_v2/training_K.json")
	angles = np.array(angles.values.tolist())
	return angles

def fimgs():
	"""Finds the images available."""
	images_dir = "/Volumes/HomeXx/compuir/hands/FreiHAND_pub_v2/training/rgb"

	imgns = 0
	imgs = []

	for obj in os.scandir(images_dir):
		if obj.is_file():
			root, ext = os.path.splitext(obj.path)
			if ext in (".png", ".jpeg", ".jpg"):
				imgs.append(obj.path)
				imgns += 1

	print(f"{imgns} images")
	print(f"{int(imgns/3)} should be ~32560")

	imgs.sort()
	t_imgs = imgs[:32560]

	return t_imgs

def apply(xyz, K):
	"""Applies focal point, perspective K to coordinate pairs x, y, and z."""
	xyzT = xyz.T # (21x3) -> (3x21)
	xy = np.matmul(K, xyzT).T # (3x3)*(3x21) = (3x21)

	cpairs = xy[:,:2] / xy[:,2:] # x/z, y/z
	return cpairs.flatten()

def collect():
	"""Assembles the training data and targets."""
	y_, targets = targs() # target coordinate pairs
	# y_ = y_.reshape(-1, 21 * 3) # flatten # not if apply() is used

	t_imgs = fimgs() # training image files
	K = vals() # focal point & camera weights

	assert len(t_imgs) == len(y_), f"data misaligned {len(t_imgs)}:{len(y_)}" # sanity
	return y_, targets, t_imgs, K

def process_imgs(t_img, label): # pass images read directly to the model
	img = tf.io.read_file(t_img) # read the image's contents
	img = tf.io.decode_jpeg(img) #. channels=3)
	img = tf.image.resize(img, [224, 224]) # reshape for consistency
	img /= 255. # normalize from 0 - 1

	return img, label

def format():
	"""Splits and shuffles training data into train, test, and validation sets."""
	y_, targets, t_imgs, K = collect()

	cpairs = np.array([apply(y_[i], K[i]) for i in range(len(y_))])
	cpairs /= 224.

	xTrain, xtest, yTrain, ytest = train_test_split(t_imgs, cpairs, test_size=0.2, random_state=17)
	xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5, random_state=17)

	train_data = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))
	test_data = tf.data.Dataset.from_tensor_slices((xtest, ytest))
	val_data = tf.data.Dataset.from_tensor_slices((xval, yval))

	shuffled_train_data = train_data.shuffle(7)
	shuffled_test_data = test_data.shuffle(7)
	shuffled_val_data = val_data.shuffle(7)

	batch_sz = 100
	_train_data = shuffled_train_data.map(process_imgs)
	train_data_ = _train_data.batch(batch_sz)

	test_data_ = shuffled_test_data.map(process_imgs)
	_test_data = test_data_.batch(batch_sz)

	val_data_ = shuffled_val_data.map(process_imgs)
	_val_data = val_data_.batch(batch_sz)

	return train_data_, _test_data, _val_data

def design():
	model = tf.keras.Sequential([
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

		tf.keras.layers.Dense(1024, activation='relu'),
		tf.keras.layers.Dropout(0.7), # killa

		tf.keras.layers.Dense(512, activation='linear'),
		tf.keras.layers.Dropout(0.2),

		tf.keras.layers.Dense(42)
	])

	return model

def pixel_err(y_true, y_pred):
	return tf.reduce_mean(tf.abs(y_true - y_pred))*224.0

def main():
	"""Trains a tf.keras model on the given images & coordinate pairs."""
	train_data, test_data, val_data = format()
	# train_data, test_data, val_data = format(t_imgs, cpairs)
	# train_data, test_data, val_inputs, val_targets = format(t_imgs, y_)

	model = design()

	model.compile(
		optimizer='adam',
		loss='mse', 
		metrics=['mae', pixel_err]
	)

	n_ep = 5
	history = model.fit(
		train_data,
		epochs=n_ep, 
		validation_data=(val_data),
		verbose=1
	)

	loss, mae, px_err = model.evaluate(test_data)
	print(f"test loss: {(loss*100):.4f}; test avg. pixel err: {(mae*100):.4f} (pixel_error: {px_err:.4f})")

	hist(history) # visualize
	record(history, model.optimizer.get_config()) # track

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
	file = "/Volumes/HomeXx/compuir/hands_ml/recorded_stats.md"

	print(f"\n{history.history}\n{config}")

	with open(file, 'a') as f:
		f.write(f"\n{history.history}\n{config}")

def overlay(img_path, xyz, K):
	"""Overlays calculated values onto file.png"""
	coord_pairs = apply(xyz, K)

	skeleton = [
		(0, 1), (1, 2), (2, 3), (3, 4), # thumb
		(0, 5), (5, 6), (6, 7), (7, 8), # index
		(0, 9), (9, 10), (10, 11), (11, 12), # middle
		(0, 13), (13, 14), (14, 15), (15, 16), # ring
		(0, 17), (17, 18), (18, 19), (19, 20) # pinky 
	]

	img = mpimg.imread(img_path)
	plt.figure()
	plt.imshow(img)

	for start, finish in skeleton:
		x_vals = [coord_pairs[start, 0], coord_pairs[finish, 0]]
		y_vals = [coord_pairs[start, 1], coord_pairs[finish, 1]]
		plt.plot(x_vals, y_vals, c="red", linewidth=2)
	
	plt.scatter(coord_pairs[:, 0], coord_pairs[:, 1], c="blue", s=20)
	plt.axis("off")
	plt.show()

def check_dims(flist):
	"""Finds the dimensions of the images' file paths passed."""
	sizes = {}

	for obj in flist:
		if os.path.exists(obj):
			with Image.open(obj) as i:
				sizes[obj] = i.size

	uni = set()
	for szs in sizes.values():
		uni.add(szs)

	print(len(uni)) # 1
	print(uni) # (224x224)

def pra(): # test / example
	"""Takes a random file and maps its projected points to its paired image."""
	idx = random.randint(1, 32560)

	y_, targets = targs()
	angles = vals()
	imgs = fimgs()

	assert len(imgs) == len(targets), f"data misaligned {len(imgs)}:{len(targets)}"

	K = angles[idx] # 3x3
	xyz = targets[idx] # 21x3 points

	img_path = imgs[idx]

	if img_path:
		xy = overlay(img_path, xyz, K)
	else:
		print('error in logic flow')

if __name__=="__main__":
	main()