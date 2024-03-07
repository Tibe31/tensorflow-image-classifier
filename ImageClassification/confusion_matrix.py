# import the necessary packages
from tensorflow.keras.models import Model, save_model, load_model
# from keras.models import save_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import imutils
import cv2
from utilities.utils import get_key
import time
import os
import random
import msvcrt
import matplotlib.pyplot as plt
from utilities.utils_images import plot_activation, convert_image_to_max
import pandas as pd
import imghdr
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")


args = vars(ap.parse_args())
mylist = args["model"].split('/')
name_model = mylist[1]
name_model_split = name_model.split('_')
INPUT_SHAPE = (200,200,3)
print (INPUT_SHAPE)
PATH = 'test'


args = vars(ap.parse_args())

dizionario_label = {
    '0': 0,
    '1': 1
}

# load the trained convolutional neural network
print("[INFO] loading network...")


model = load_model(args["model"])
print (model.summary())


print("[INFO] model loaded...")

count_images = 0
ko_images = 0

scores_negative = []
scores_positive = []
ground_truth_labels = []
predicted_probabilities = []



for directory in (os.listdir(PATH)):
    os.mkdir('out/' + directory)
    print ('inizio nuova classe')

    for img in os.listdir(PATH+'/'+directory):
        count_images = count_images + 1
        type = imghdr.what(PATH + '/' + directory+'/'+img)
        image_read = cv2.imread(PATH + '/' + directory+'/'+img)
        # print (img,directory)
        #image = image[20:700, 0:image.shape[1]]
        image = cv2.resize(image_read, (int(INPUT_SHAPE[1]),int(INPUT_SHAPE[0])), cv2.INTER_AREA)
        show_image = image.copy()
        image = image.astype("float") / 255.0
        # image = img_to_array(image)
        if (INPUT_SHAPE[2]=='1'):
            last_axis = -1
            image = np.expand_dims(image, last_axis)

        image = np.expand_dims(image, axis=0)
        start = time.process_time()
        predizioni = model.predict(image)[0]
        print (predizioni)
        
        gt_label = int(directory)
        print (gt_label)
        predicted_probabilities.append(predizioni[0])
        ground_truth_labels.append(gt_label)
        
        # if (gt_label == 0):
            # scores_negative.append(predizioni)
        cv2.imwrite('out/' + directory + '/' + str(predizioni[0]) + '_' + img, image_read)
        # else:
            # cv2.imwrite('out/' + '1' + '/' + str(predizioni[0]) + '_' + img, image_read)
            # if (predizioni[0] < 0.5):
        scores_positive.append(predizioni)

def find_best_threshold(labels, scores):
    thresholds = np.linspace(0, 1, num=100)  # Genera 100 valori di soglia tra 0 e 1
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        predicted_labels = (scores >= threshold).astype(int)
        f1 = f1_score(labels, predicted_labels)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

best_threshold, f1 = find_best_threshold(ground_truth_labels, predicted_probabilities)
print (ground_truth_labels)
print(predicted_probabilities)
print("Soglia migliore:", best_threshold)
print("F1 score:", f1)

