#!/usr/bin/env python
# coding: utf-8

# In[1]:

import csv
import tensorflow as tf
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from imutils import paths
from tensorflow.keras.models import Sequential,load_model, Model
from tensorflow.keras.layers import BatchNormalization,Dropout,MaxPooling2D,GlobalAveragePooling2D,Conv2D, GaussianNoise,Dense,Flatten
from datetime import date
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from sklearn.metrics import recall_score
from tensorflow.keras.regularizers import l2
from PIL import ImageFilter
# import mxnet as mx
from numpy import random
from PIL import Image, ImageEnhance




today = date.today()
print("Today's date:", today)

train_dir='immagini/train'
val_dir='immagini/val'

dizionario_label = {
    'KO': 0,
    'OK': 1
}

INPUT_SHAPE = (200,200,3)
BATCH_SIZE = [16]
LR = 1e-3
EPOCHS = 1000
DECAY_AFTER_EPOCHS = 5
Dropouts = [0.4]

for b in BATCH_SIZE:

    for d in Dropouts:


        def random_contrast_and_blur(image):
            contrast_factor = tf.random.uniform(shape=[], minval=0.8, maxval=1.2)  # Genera un fattore di contrasto casuale tra 0.5 e 1.5
            adjusted_image = tf.image.adjust_contrast(image, contrast_factor)
            return adjusted_image
            
           
        checkpoint_filepath = 'modelli/%s'%b+'_%s'%INPUT_SHAPE[0]+'_%s'%INPUT_SHAPE[1]+'_%s'%INPUT_SHAPE[2]+'_%s'%today+'_2'


        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
            
            
        class WriteValMetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self, file_path):
                self.file_path = file_path

            def on_epoch_end(self, epoch, logs=None):
                val_loss = logs.get('val_loss')
                val_accuracy = logs.get('val_binary_accuracy')

                with open(self.file_path, 'a') as file:
                    file.write(f'Dropout {d} - Epoch {epoch + 1}: Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}\n')


        file_path = 'val_metrics_%d'%b + '_%f'%d +'.txt'
        write_val_metrics_callback = WriteValMetricsCallback(file_path)

        vgg_conv = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape = INPUT_SHAPE)
        model = Sequential()
        model.add(vgg_conv)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(d))
        model.add(GaussianNoise(5))
        model.add(Dense(1,activation="sigmoid"))




        print (model.summary())

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255.,
            rotation_range=2,
            zoom_range = 0.02,
            width_shift_range=0.02,
            height_shift_range=0.02,
            brightness_range = [0.9,1.1],
            fill_mode='nearest'
            )
            


        # Note that the validation data should not be augmented!
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255.,
            #brightness_range = [0.8,1.2],
            #zoom_range = 0.01,
            #width_shift_range=0.01,
            #height_shift_range=0.01
        )

        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                batch_size=b,
                class_mode='binary',
                shuffle=True)
        validation_generator = test_datagen.flow_from_directory(
                val_dir,
                target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                batch_size=b,
                class_mode='binary',
                shuffle = True)
                
                
        # x= train_generator.next()
        # for i in range(0,BATCH_SIZE-1):
            # image = x[0][i]
            # plt.imshow(image)
            # plt.show()


        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            LR,
            decay_rate=0.99,
            decay_steps=int((train_generator.samples/b))*DECAY_AFTER_EPOCHS,
            staircase=False)

        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                      loss = tf.keras.losses.BinaryCrossentropy(name='val_accuracy'),
                      metrics =[tf.keras.metrics.BinaryAccuracy(threshold = 0.5),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])



        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[model_checkpoint_callback,write_val_metrics_callback]
        )


