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
Writes formatted and split training, testing, and validation data."
"""

SAMPLE_SIZE = 40

dataTarget = "preprocessed9.npz" # json dump with training splits
dataSource = "/Volumes/HomeXx/compuir/hands_ml/data/FreiHAND_pub_v2"

def targs():
	"""Finds the target's coordinate values."""
	source = os.path.join(dataSource, "training_xyz.json")

	with open(source, 'r') as f:
		targets = json.load(f)

	return np.array(targets)

def vals():
	"""Finds the images' camera's metadata."""
	source = os.path.join(dataSource, "training_K.json")

	with open(source, 'r') as f:
		angles = json.load(f)

	return np.array(angles)

def fimgs(beg, end):
	"""Finds the images available."""
	source = os.path.join(dataSource, "training/rgb")
	imgs = []

	for obj in os.scandir(source):
		if obj.is_file():
			root, ext = os.path.splitext(obj.path)
			if ext in (".jpg", ".png"):
				imgs.append(obj.path)
	imgs.sort()

	return imgs[beg:end]

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

	t_imgs = fimgs(beg=32560, end=65120) # training image files
	K = vals() # focal point & camera weights

	# sanity
	assert len(t_imgs) == len(y_), f"data misaligned {len(t_imgs)}:{len(y_)}"

	return y_, t_imgs, K

def collectMini():
	"""Assembles the training data and targets."""
	y_ = targs()[:SAMPLE_SIZE]

	beg = 32560 
	end = beg + SAMPLE_SIZE
	t_imgs = fimgs(beg, end)
	K = vals()[:SAMPLE_SIZE]

	# sanity
	assert len(t_imgs) == len(y_), f"data misaligned {len(t_imgs)}:{len(y_)}"

	return y_, t_imgs, K

def format():
	"""Writes training, testing, and validation datasets to the disk."""
	xyz, t_imgs, K = collect()
	targets = apply2(xyz, K)

	xtrain, xtest, ytrain, ytest = train_test_split(
		t_imgs,
		targets,
		test_size=0.1, # 90% train
		random_state=42
	)

	xtest, xval, ytest, yval = train_test_split(
		xtest,
		ytest,
		test_size=0.5, # 10% val/test
		random_state=42
	)

	np.savez_compressed(
		os.path.join(
			os.path.dirname(dataSource), 
			dataTarget
		),
		xtrain=xtrain, ytrain=ytrain,
		xtest=xtest, ytest=ytest,
		xval=xval, yval=yval
	)


def formatMini():
	"""Writes training, testing, and validation datasets to the disk."""
	xyz, t_imgs, K = collectMini()
	targets = apply2(xyz, K)

	xtrain, xtest, ytrain, ytest = train_test_split(
		t_imgs,
		targets,
		test_size=0.2, # 90% train
		random_state=42
	)

	xtest, xval, ytest, yval = train_test_split(
		xtest,
		ytest,
		test_size=0.5, # 10% val/test
		random_state=42
	)

	np.savez_compressed(
		os.path.join(
			os.path.dirname(dataSource), 
			dataTarget
		),
		xtrain=xtrain, ytrain=ytrain,
		xtest=xtest, ytest=ytest,
		xval=xval, yval=yval
	)

def main():
	"""Writes training, testing, and validation datasets to the disk."""
	formatMini()
	# format()

if __name__=="__main__":
	main()