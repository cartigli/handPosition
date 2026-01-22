import os
import time
import random

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


i = 89
machine = f"/Volumes/HomeXx/compuir/hands_ml/handModel_i{i}/handModel.keras"
model = tf.keras.models.load_model(machine, compile=False)

def data():
    rand = random.randint(0, 500)

    with np.load("data/preprocessed.npz") as data:
        test_path = data['xtest'][rand]
        true_coords = data['ytest'][rand]
    
    return test_path, true_coords

def format(test_path, true_coords):
    img = Image.open(test_path).convert("RGB")

    img = img.resize((224, 224))
    img_arr = np.array(img)

    # normalize RGB
    img_arr = img_arr / 255.0

    img_arr = np.expand_dims(img_arr, axis=0)
    predictions = model.predict(img_arr)

    img_display = np.squeeze(img_arr)
    coords = predictions[0]*224

    return coords, img_display

def plott(coords, img_display):
    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)

    # for i in range(0, len(coords), 2): # flat predictions
        # x, y = coords[i], coords[i+1]

    for i, (x, y) in enumerate(coords): # coupled predictions
        plt.plot(x, y, 'ro', markersize=8)
        plt.annotate(f'{i//2}', (x, y), color='white', fontsize=10)

    plt.title("Predicted Hand Points")
    plt.show()

def video():
    parent = "/Volumes/HomeXx/compuir/hands_ml/outputs"
    vid = os.path.join(parent, "inputs", "hands_video2.MOV")
    i = 10
    out = os.path.join(parent, f"hands_pdx{i}.mp4")

    if os.path.exists(out):
        while True:
            if not os.path.exists(out):
                break
            i += 1
            out = os.path.join(parent, f"hands_pdx{i}.mp4")

    frcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, frcc, 30, (224, 224))

    video = cv2.VideoCapture(vid)

    while True:
        success, frame = video.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAYSCALE) # GRAYSCALE

        img = cv2.resize(frame, (224, 224))
        img_norm = img / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        predictions = model.predict(img_batch, verbose=0)
        coords = predictions[0]*224.0

        # for i in range(0, len(coords), 2): # flat predictions
        #     x, y = coords[i], coords[i+1]

        for x, y in coords: # coupled predictions
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        cimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # RGB
        # cimg = cv2.cvtColor(img, cv2.COLOR_GRAYSCALE2BGR) # GRAYSCALE

        writer.write(cimg)

    writer.release()
    video.release()

if __name__=="__main__":
    video()
