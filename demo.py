# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, save_model, load_model
# from keras.models import save_model
import tensorflow as tf
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

model_path = 'modelli/'
model_name = '8_35_110_3_2022-01-24preprocess-function'

name_model_split = model_name.split('_')

INPUT_SHAPE = ("%s"%name_model_split[1],"%s"%name_model_split[2],"%s"%name_model_split[3])
PATH = 'immagini'

# load the trained convolutional neural network
print("[INFO] loading network...")


model = load_model(model_path +'/' + model_name)

print("[INFO] model loaded...")

while True:
    for img in os.listdir(PATH):
        if (img):
            print ('Immagine ricevuta, inizio inferenza!')
            print ('Cancello esito precedente...')
            os.remove("esiti/esito.txt")

            type = imghdr.what(PATH +'/'+img)
            if (type != 'png'):
                continue
            if (INPUT_SHAPE[2]=='1'):
                image = cv2.imread(PATH +'/'+img,cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(PATH +'/'+img)
                image = cv2.resize(image, (int(INPUT_SHAPE[1]),int(INPUT_SHAPE[0])), cv2.INTER_AREA)
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                if (INPUT_SHAPE[2]=='1'):
                    last_axis = -1
                    image = np.expand_dims(image, last_axis)

                image = np.expand_dims(image, axis=0)
                predizioni = model.predict(image)[0]

                if (predizioni > 0.5):
                    index_predicted = 1
                else:
                    index_predicted = 0

                print ('Predizione effettuata, scrivo esito nel file...')
                f = open("esiti/esito.txt", "a")
                f.write(str(index_predicted))
                f.close()

                if (index_predicted==0):
                    Esito='BUONO'
                else:
                    Esito='SCARTO'
                print ('Esito: ' + Esito)

                os.remove(PATH +'/'+img)
                print ('Immagine rimossa!')
    time.sleep(5)
