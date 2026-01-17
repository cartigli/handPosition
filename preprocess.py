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

dataSource = "/Volumes/HomeXx/compuir/hands_ml/data/FreiHAND_pub_v2"

def targs():
	"""Finds the target's coordinate values."""
	target = "training_xyz.json"
	source = os.path.join(dataSource, target)
	y_s = pd.read_json(source)
	targets = np.array(y_s.values.tolist())

	return targets

def vals():
	"""Finds the images' extraneous data."""
	target = "training_K.json"
	source = os.path.join(dataSource, target)
	angles = pd.read_json(source)
	angles = np.array(angles.values.tolist())
	return angles

def fimgs():
	"""Finds the images available."""
	source = os.path.join(dataSource, "training/rgb")
	imgs = []

	for obj in os.scandir(source):
		if obj.is_file():
			root, ext = os.path.splitext(obj.path)
			if ext in (".jpg"):
				imgs.append(obj.path)
	imgs.sort()

	return imgs[32560:65120]
	# return imgs[:32560]

def apply2(xyz, K):
	"""Applies focal point, perspective K to coordinate pairs x, y, and z."""
	pairs = []
	for i in range(len(xyz)):
		xyzT = xyz[i].T # (21x3) -> (3x21)
		xy = np.matmul(K[i], xyzT).T # (3x3)*(3x21) = (3x21)

		cpairs = xy[:,:2] / xy[:,2:] # x/z, y/z

		# pairs.append(cpairs.flatten()) # reshape for as inputs
		pairs.append(cpairs)
		# ^comment this out for the custom weight matrix

	return np.array(pairs) / 224.0 # normalize 0 - 1

def collect():
	"""Assembles the training data and targets."""
	y_ = targs() # target coordinate pairs

	t_imgs = fimgs() # training image files
	K = vals() # focal point & camera weights

	# sanity
	assert len(t_imgs) == len(y_), f"data misaligned {len(t_imgs)}:{len(y_)}"

	return y_, t_imgs, K

def format():
	"""Writes training, testing, and validation datasets to the disk."""
	y_, t_imgs, K = collect()

	targets = apply2(y_, K)

	xtrain, xtest, ytrain, ytest = train_test_split(t_imgs, targets, test_size=0.1, random_state=42)
	xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5, random_state=42)

	np.savez_compressed(os.path.join(os.path.dirname(dataSource), "preprocessed0.npz"),
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