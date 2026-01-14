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
	imgs = []

	for obj in os.scandir(images_dir):
		if obj.is_file():
			root, ext = os.path.splitext(obj.path)
			if ext in (".png", ".jpeg", ".jpg"):
				imgs.append(obj.path)
				# imgns += 1

	imgs.sort()
	t_imgs = imgs[:32560]
	return t_imgs

def apply(xyz, K):
	"""Applies focal point, perspective K to coordinate pairs x, y, and z."""
	xyz_T = xyz.T # (21x3) -> (3x21)
	xyz_K = np.matmul(K, xyz_T).T # (3x3)*(3x21) = (3x21)

	xy = xyz_K[:,:2] / xyz_K[:,2:] # x/z, y/z
	return xy.flatten()

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

	img = tf.io.decode_jpeg(img, channels=3)
	img = tf.image.resize(img, [224, 224]) # reshape for consistency

	img /= 255. # normalize RGB
	return img, label

def format():
	"""Splits and shuffles training data into train, test, and validation sets."""
	y_, targets, t_imgs, K = collect()

	xy = np.array([apply(y_[i], K[i]) for i in range(len(y_))])
	xy /= 224.

	xTrain, xtest, yTrain, ytest = train_test_split(t_imgs, xy, test_size=0.2, random_state=17)
	xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5, random_state=17)

	train_data = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))
	test_data = tf.data.Dataset.from_tensor_slices((xtest, ytest))
	val_data = tf.data.Dataset.from_tensor_slices((xval, yval))

	shuffled_val_data = val_data.shuffle(7)

	val_data_ = shuffled_val_data.map(process_imgs)
	_val_data = val_data_.batch(100)

	for image, label in val_data_.take(1):
		print("Image's shape:", image.shape)
		print("Top-left-ish pixel's RGB:", image[10, 10, :].numpy())

	return train_data_, _test_data, _val_data

def main():
	"""Trains a tf.keras model on the given images & coordinate pairs."""
	train_data, test_data, val_data = format()


if __name__=="__main__":
	main()