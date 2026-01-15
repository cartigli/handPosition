import os
import time
import random

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

machine = "/Volumes/HomeXx/compuir/hands_ml/handModel_iv226k.keras"
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

    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]
        plt.plot(x, y, 'ro', markersize=8)
        plt.annotate(f'{i//2}', (x, y), color='white', fontsize=10)

    plt.title("Predicted Hand Points")
    plt.show()

def video():
    parent = "/Volumes/HomeXx/compuir/hands_ml/outputs"
    vid = os.path.join(parent, "hands_video3.MOV")
    out = os.path.join(parent, "hands_outvid14.mp4")

    frcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, frcc, 30, (224, 224))

    video = cv2.VideoCapture(vid)

    while True:
        success, frame = video.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = cv2.resize(frame, (224, 224))
        img_norm = img / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        predictions = model.predict(img_batch)
        coords = predictions[0]*224.0*224.0

        for i in range(0, len(coords), 2):
            x, y = round(coords[i]), round(coords[i+1])
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        
        cimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Hand Tracking", img)
        writer.write(cimg)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    writer.release()
    video.release()
    # cv2.destroyAllWindows()


def main():
    # test_paths, true_coords = data()
    # coords, img_display = format(test_paths, true_coords)
    # plott(coords, img_display)

    video()

if __name__=="__main__":
    main()
