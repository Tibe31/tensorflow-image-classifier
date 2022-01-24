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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")


args = vars(ap.parse_args())
mylist = args["model"].split('/')
name_model = mylist[1]
name_model_split = name_model.split('_')
INPUT_SHAPE = ("%s"%name_model_split[1],"%s"%name_model_split[2],"%s"%name_model_split[3])
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

y_test = []
predic = []
pred_time_list = []

for directory in (os.listdir(PATH)):
    for img in os.listdir(PATH+'/'+directory):
        count_images = count_images + 1
        type = imghdr.what(PATH + '/' + directory+'/'+img)
        if (type != 'png'):
            continue
        if (INPUT_SHAPE[2]=='1'):
            image = cv2.imread(PATH + '/' + directory+'/'+img,cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(PATH + '/' + directory+'/'+img)
        # print (img,directory)
        #image = image[20:700, 0:image.shape[1]]
        image = cv2.resize(image, (int(INPUT_SHAPE[1]),int(INPUT_SHAPE[0])), cv2.INTER_AREA)
        show_image = image.copy()
        image = image.astype("float") / 255.0
        # image = img_to_array(image)
        if (INPUT_SHAPE[2]=='1'):
            last_axis = -1
            image = np.expand_dims(image, last_axis)

        image = np.expand_dims(image, axis=0)
        start = time.process_time()
        predizioni = model.predict(image)[0]
        end = time.process_time()
        pred_time_list.append(end-start)
        if (predizioni > 0.5):
            index_predicted = 1
        else:
            index_predicted = 0
        classe_predicted = ([k for k,v in dizionario_label.items() if v == index_predicted])
        classe_predicted = int(classe_predicted[0])

        gt_label = int(directory)

        y_test.append(int(directory))
        predic.append(classe_predicted)

        if (classe_predicted!=int(directory)):
            #conto gli errori
            ko_images = ko_images + 1
            show_image = cv2.resize(show_image, (700, 700), cv2.INTER_AREA)
            print (img,directory)
            print (np.max(predizioni))
            org = (15, 50)

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2
            show_image = cv2.putText(show_image, 'P:%s'%classe_predicted + ' GT: %s'%gt_label, org,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
            cv2.imshow("Display 1", show_image)
            cv2.waitKey(0)



from string import ascii_uppercase
from pandas import DataFrame
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # for data visualization


columns = ['%s' %(i) for i in ['0','1','2','3','4','6'][0:len(np.unique(y_test))]]

confm = confusion_matrix(y_test, predic)
confm = confm.astype('float') / confm.sum(axis=1)[:, np.newaxis]
df_cm = DataFrame(confm, index=columns, columns=columns)

plt.title("Confusion Matrix", fontsize =20)
ax = sn.heatmap(df_cm, cmap='PiYG', annot=True)
ax.set_xlabel("Predicted", fontsize = 20)
ax.set_ylabel("Ground Truth", fontsize = 20)
plt.show()


#create dataframe
#%KO
KO=(ko_images/count_images)*100

print (KO)
print (count_images)
npa = np.asarray(pred_time_list, dtype=np.float32)
print ('Tempo mediano',np.median(npa))
