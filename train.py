import os
import sys
import json
import shutil
import argparse
import subprocess
import contextlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply, Dropout


"""
Take the pre-processed data and train the model on it.
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rent = "/Volumes/HomeXx/compuir/hands_ml"

class Stream:
	"""Manual control over stdout & stderr for writing to multiple places."""

	def __init__(self, *streams):
		"""*streams: accept any number of args in a packed tuple."""
		self.streams = streams

	def write(self, data):
		for stream in self.streams:
			stream.write(data)
			stream.flush()

	def flush(self):
		for stream in self.streams:
			stream.flush()

def postProcess():
	"""Splits and shuffles training data into train, test, and validation sets."""
	with np.load(os.path.join(rent, "data", "preprocessed1.npz")) as data:
		xtrain, ytrain = data['xtrain'], data['ytrain']
		xtest, ytest = data['xtest'], data['ytest']
		xval, yval = data['xval'], data['yval']

	training = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
	testing = tf.data.Dataset.from_tensor_slices((xtest, ytest))
	validating = tf.data.Dataset.from_tensor_slices((xval, yval))

	return training, testing, validating

def process_imgs(t_img, label):
	"""Reads image files and passes them straight to the model."""
	img = tf.io.read_file(t_img) # read the image's contents
	img = tf.io.decode_jpeg(img, channels=3) # RGB (=3)

	# img = tf.image.rgb_to_grayscale(img) # GRAYSCALE
	img = tf.image.resize(img, [224, 224]) # reshape for consistency
	img = img / 255.0 # normalize 0 - 1

	return img, label

def shuffle_Nmap(training, testing, validating):
	"""Applies splits to the processing for images (above) and batches results."""
	batch = 32
	io_tune = tf.data.AUTOTUNE

	mapped_train_data = training.map(process_imgs, num_parallel_calls=io_tune)
	# shuffled_train_data = mapped_train_data.shuffle(buffer_size=10000)
	shuffled_train_data = mapped_train_data.cache().shuffle(buffer_size=10000)
	train_targets = shuffled_train_data.batch(batch).prefetch(io_tune)

	mapped_test_data = testing.map(process_imgs, num_parallel_calls=io_tune)
	# test_targets = mapped_test_data.batch(batch).prefetch(io_tune)
	test_targets = mapped_test_data.cache().batch(batch).prefetch(io_tune)

	mapped_val_data = validating.map(process_imgs, num_parallel_calls=io_tune)
	# shuffled_val_data = mapped_val_data.shuffle(buffer_size=10000)
	shuffled_val_data = mapped_val_data.cache().shuffle(buffer_size=10000)
	val_targets = shuffled_val_data.batch(batch).prefetch(io_tune)

	return train_targets, test_targets, val_targets

def setup():
	"""Collects and returns training, test, and validation data for the model."""
	training, testing, validating = postProcess()
	return shuffle_Nmap(training, testing, validating)

def design():
	"""Define the model to train."""
	inputs = layers.Input(shape=(224, 224, 3))

	x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
	x = layers.ReLU()(x)
	x = cbam_block(x)
	x = layers.Dropout(0.1)(x)
	x = layers.MaxPooling2D((2, 2))(x)

	x = layers.Conv2D(64, (3, 3), padding='same')(x)
	x = layers.ReLU()(x)
	x = cbam_block(x)
	x = layers.Dropout(0.1)(x)
	x = layers.MaxPooling2D((2, 2))(x)

	x = layers.Conv2D(128, (3, 3), padding='same')(x)
	x = layers.ReLU()(x)
	x = cbam_block(x)
	# x = layers.Dropout(0.1)(x)
	x = layers.MaxPooling2D((2, 2))(x)

	x = layers.Conv2D(256, (3, 3), padding='same')(x)
	x = layers.ReLU()(x)
	x = cbam_block(x)
	# x = layers.Dropout(0.1)(x)
	x = layers.MaxPooling2D((2, 2))(x)

	x = layers.Conv2D(512, (3, 3), padding='same')(x)
	x = layers.ReLU()(x)
	x = cbam_block(x)
	# x = layers.Dropout(0.1)(x)
	x = layers.MaxPooling2D((2, 2))(x)

	x = layers.Conv2D(1024, (3, 3), padding='same')(x)
	x = layers.ReLU()(x)
	x = cbam_block(x)
	# x = layers.Dropout(0.1)(x)
	x = layers.MaxPooling2D((2, 2))(x)

	x = layers.Flatten()(x)

	x = layers.Dense(42)(x)
	outputs = layers.Reshape((21, 2))(x)

	return keras.Model(inputs, outputs)

def _compile(model):
	"""Compile the model's optimizers, loss, & metrics."""

	model.compile(
		optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0008, global_clipnorm=1.0),
		# loss='mae',
		loss='mse',
		# loss=anatomical_loss,
		metrics=[pxl_mae]
	)

	return model

def _train(model, train_targets, test_targets, val_targets):
	"""Fit the model to the training and validation data."""
	dump = os.path.join(rent, "safetyStop_model.keras")

	# callbacks
	checkpoint = ModelCheckpoint(dump, monitor='val_loss', save_best_only=True)
	earlystop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

	history = model.fit(
		train_targets,
		epochs=14,
		validation_data=(val_targets),
		# callbacks=[checkpoint, earlystop],
		verbose=1
	)
	loss, mae = model.evaluate(test_targets)

	bureaucrat(model, history)
	return loss, mae

def main():
	"""Reads data, compiles model, trains, validates and tests on data splits."""
	try:
		train_targets, test_targets, val_targets = setup()

		model = _compile(design())

		loss, mae = _train(model, train_targets, test_targets, val_targets)

		print(f"test loss: {loss:.4f}; test pxl_mae: {mae:.4f}")

	except KeyboardInterrupt:
		print("boss kill't it; shutter down")
		sys.exit(0)

def cbam_block(x, ratio=16, kernel_size=7):
	"""Applies channel and spatial attention [in that order]."""
	x = channel_attention(x, ratio)
	x = spatial_attention(x, kernel_size)
	return x

def channel_attention(x, ratio=16):
	"""'Which channel features are valued/needed most from this input?'"""
	channels = x.shape[-1] # (h*w*channels)[-1] = channels
	squeezed = GlobalAveragePooling2D()(x) # squeeze out one value per channel

	# splits inputs channels by the ratio (16) and activate w.ReLU
	excited = Dense(channels // ratio, activation='relu')(squeezed)
	dropped = Dropout(0.1)(excited) # drop a few for overfitting

	# activate the weights with sigmoid (0,1) for each weights' weight
	ratios = Dense(channels, activation='sigmoid')(dropped)
	reshaped = Reshape((1, 1, channels))(ratios)

	return Multiply()([x, reshaped]) # per-channel weighted attention

def spatial_attention(x, kernel_size=7):
	"""'Which spatial locations matter most?'"""
	# compress channels over avg & max pooling.
	avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
	max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

	# concatenate the pools
	concat = Concatenate()([avg_pool, max_pool])

	# convolve the sum over sigmoid (0,1) and return result * inputs
	attention = Conv2D(1, kernel_size, activation='sigmoid', padding='same')(concat)
	return Multiply()([x, attention])

def pxl_mae(y_true, y_pred):
	"""Re-calculate error by un-normalizing pixel coordinates."""
	adjusted = tf.reduce_mean(tf.abs(y_true - y_pred))*224.0
	return adjusted

def strux_mae(y_true, y_pred):
	"""Applies the custom (arbitrary) weights to the model's predictions."""
	bump_weights = tf.constant([
		2.0, # wrist 
		2.0, 1.0, 1.0, 1.5, # thumb
		2.0, 1.0, 1.0, 1.5, # index
		1.2, 1.0, 1.0, 1.2, # middle
		1.2, 1.0, 1.0, 1.2, # ring
		1.2, 1.0, 1.0, 1.2, # pinky
	], dtype=tf.float32)

	sqr_err = tf.square(y_true - y_pred) # (batch, 21, 2) | coordinate prediction error (per coord)

	# EUCLIDEAN loss
	per_key_sum = tf.reduce_sum(sqr_err, axis=-1) # get each point's err individually
	weighted_err = tf.sqrt(per_key_sum + 1e-7) * bump_weights # and apply them to the weights

	# MAE loss
	# mae_key_err = tf.reduce_mean(sqr_err, axis=-1) # (batch, 21) | avg. accuracy
	# weighted_err = mae_key_err * bump_weights # (batch, 21) @ (21,) | apply custom weights

	# adjusted = tf.reduce_mean(weighted_err) # mean custom weighted errors
	# return adjusted
	return tf.reduce_mean(weighted_err)

def anatomical_loss(y_true, y_pred):
	"""Adjusts loss based on deviation from skeletal length consistency."""
	# model accuracy
	mae = tf.reduce_mean(tf.abs(y_true - y_pred))

	anatomy = [ # bone connectivity [start joint, end joint]
		[0, 1], [1, 2], [2, 3], [3, 4], # thumb
		[0, 5], [5, 6], [6, 7], [7, 8], # index
		[0, 9], [9, 10], [10, 11], [11, 12], # middle
		[0, 13], [13, 14], [14, 15], [15, 16], # ring
		[0, 17], [17, 18], [18, 19], [19, 20] # pinky
	]

	# actual starts/ends of each joint
	true_start = tf.gather(y_true, [bone[0] for bone in anatomy], axis=1)
	true_end = tf.gather(y_true, [bone[1] for bone in anatomy], axis=1)

	# predicted starts/ends of each joint
	pred_start = tf.gather(y_pred, [bone[0] for bone in anatomy], axis=1)
	pred_end = tf.gather(y_pred, [bone[1] for bone in anatomy], axis=1)

	# get bone vectors (dx, dy)
	true_vectors = true_start - true_end
	pred_vectors = pred_start - pred_end

	# get bone euclidean norms (+ epsilon for NaN values)
	true_lengths = tf.norm(true_vectors, axis=-1) + 1e-6
	pred_lengths = tf.norm(pred_vectors, axis=-1) + 1e-6

	# get the error in predictions
	sym_err_ratio = tf.math.log(pred_lengths / (true_lengths + 1e-7))
	bone_loss = tf.reduce_mean(tf.abs(sym_err_ratio))

	# impact/consideration of bone_position loss + mae
	return mae + (bone_loss*1.0)

def bureaucrat(model, history):
	"""Records training results, model config, and model itself."""
	i = 85

	base = "handModel"
	dirname = base + f"_i{i}"
	modelname = base + ".keras"

	lo = os.path.join(rent, dirname)

	while True:
		if not os.path.exists(lo):
			os.makedirs(lo)
			break
		i += 1
		dirname = base + f"_i{i}"
		lo = os.path.join(rent, dirname)

	source = os.path.join(rent, "safetyStop_model.keras")
	backup = os.path.join(lo, "safetyStop_model.keras")

	if os.path.exists(source):
		shutil.copy2(source, backup)

	final = os.path.join(lo, modelname)
	model.save(final)

	plot = os.path.join(lo, "training.png")
	vis(history, plot)
	record(history.history, model.optimizer.get_config(), lo)

	subprocess.run(["/Volumes/HomeXx/compuir/ml/bin/python", "/Volumes/HomeXx/compuir/iStuff/iHistory.py", "-m", "Training finished."])

def vis(history, lo):
	"""Plot the model's training loss and val_loss + metrics from training."""
	mae = history.history['pxl_mae']
	val_mae = history.history['val_pxl_mae']

	epochs = range(1, len(mae) + 1)
	plt.figure(figsize=(8, 6))

	plt.plot(epochs, mae, 'b', label='Training M.A.E.')
	plt.plot(epochs, val_mae, 'r', label='Validation M.A.E.')

	plt.title('Training and Validation M.A.E.')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Average Error')
	plt.legend()

	plt.xlim(left=0)
	plt.ylim(bottom=0)

	plt.savefig(lo)
	plt.close()

def record(history, config, base):
	"""Write the training history and model config to a records file."""
	config_dmp = os.path.join(base, "modelConfig.json")
	history_dmp = os.path.join(base, "modelHistory.json")

	jsconfig = json.dumps(config, indent=4, default=str)
	jshistory = json.dumps(history, indent=4, default=str)

	with open(config_dmp, 'a') as f:
		f.write(jsconfig)
	with open(history_dmp, 'a') as f:
		f.write(jshistory)

if __name__=="__main__":
	psr = argparse.ArgumentParser()
	psr.add_argument("-i", "--index", action="store_true", help="sends stdout & stderr to index.html as well as console")

	args = psr.parse_args()
	manageOutput = False

	if args.index:
		manageOutput = True
	
	if not manageOutput:
		main()
		sys.exit(0)

	with open("/Volumes/HomeXx/compuir/neat/cartigliaClub/index.html", "w") as trainingOut:
		strm = Stream(sys.stdout, sys.stderr, trainingOut)

		with contextlib.redirect_stdout(strm), contextlib.redirect_stderr(strm):
			main()
