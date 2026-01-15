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
Writes formatted and split training, testing, and validation data to "./data".
"""

dataSource = "/Volumes/HomeXx/compuir/hands_ml/FreiHAND_pub_v2"

def targs():
	"""Finds the target's coordinate values."""
	# targets = pd.read_json("/Volumes/HomeXx/compuir/hands_ml/data_noScale/FreiHAND_pub_v2/training_xyz.json")
	target = "training_xyz.json"
	source = os.path.join(dataSource, target)
	targets = pd.read_json(source)
	y_ = np.array(targets.values.tolist())

	return y_

def vals():
	"""Finds the images' extraneous data."""
	# angles = pd.read_json("/Volumes/HomeXx/compuir/hands_ml/data_noScale/FreiHAND_pub_v2/training_K.json")
	target = "training_K.json"
	source = os.path.join(dataSource, target)
	angles = pd.read_json(source)
	angles = np.array(angles.values.tolist())
	return angles

def fimgs():
	"""Finds the images available."""
	# images_dir = "/Volumes/HomeXx/compuir/hands_ml/data_noScale/FreiHAND_pub_v2/training/rgb"
	target = "training/rgb"
	source = os.path.join(dataSource, target)

	imgns = 0
	imgs = []

	for obj in os.scandir(source):
		if obj.is_file():
			root, ext = os.path.splitext(obj.path)
			# if ext in (".png", ".jpeg", ".jpg"):
			if ext in (".jpg"):
				imgs.append(obj.path)
				imgns += 1

	print(f"{imgns} images")

	imgs.sort()
	t_imgs = imgs[:32560]

	return t_imgs

def apply(xyz, K):
	"""Applies focal point, perspective K to coordinate pairs x, y, and z."""
	xyzT = xyz.T # (21x3) -> (3x21)
	xy = np.matmul(K, xyzT).T # (3x3)*(3x21) = (3x21)

	cpairs = xy[:,:2] / xy[:,2:] # x/z, y/z

	return cpairs.flatten()

def apply2(xyz, K):
	"""Applies focal point, perspective K to coordinate pairs x, y, and z."""
	pairs = []
	for i in range(len(xyz)):
		xyzT = xyz[i].T # (21x3) -> (3x21)
		xy = np.matmul(K[i], xyzT).T # (3x3)*(3x21) = (3x21)

		cpairs = xy[:,:2] / xy[:,2:] # x/z, y/z

		pairs.append(cpairs.flatten())
	
	return np.array(pairs) / 224.0

def collect():
	"""Assembles the training data and targets."""
	y_ = targs() # target coordinate pairs

	t_imgs = fimgs() # training image files
	K = vals() # focal point & camera weights

	assert len(t_imgs) == len(y_), f"data misaligned {len(t_imgs)}:{len(y_)}" # sanity
	return y_, t_imgs, K

def format():
	"""Writes training, testing, and validation datasets to the disk."""
	y_, t_imgs, K = collect()

	# cpairs = np.array([apply(y_[i], K[i]) for i in range(len(y_))])
	targets = apply2(y_, K)

	xtrain, xtest, ytrain, ytest = train_test_split(t_imgs, targets, test_size=0.3, random_state=42)
	xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5, random_state=42)

	# dump = "/Volumes/HomeXx/compuir/hands_ml/FreiHAND_pub_v2/preprocessed.npz"
	# np.savez_compressed(dump,
	np.savez_compressed(os.path.join(dataSource, "preprocessed.npz"),
		xtrain=xtrain, ytrain=ytrain, 
		xtest=xtest, ytest=ytest,
		xval=xval, yval=yval
	)

def find_missing():
	target = "training/rgb"
	source = os.path.join(dataSource, target)

	for i in range(32560):
		if i < 10:
			f = "0000000" + f"{i}.jpg"
		elif i < 100:
			f = "000000" + f"{i}.jpg"
		elif i < 1000:
			f = "00000" + f"{i}.jpg"
		elif i < 10000:
			f = "0000" + f"{i}.jpg"
		elif i < 100000:
			f = "000" + f"{i}.jpg"
		
		ff = os.path.join(source, f)
		
		if not os.path.exists(ff):
			print(f"{f} is shot")

def main():
	"""Writes training, testing, and validation datasets to the disk."""
	format()
	# find_missing()

if __name__=="__main__":
	main()