#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from imutils import paths
from tensorflow.keras.models import Sequential,load_model, Model
from tensorflow.keras.layers import BatchNormalization,Dropout,GlobalAveragePooling2D,Conv2D, Dense,Flatten
from datetime import date
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications.vgg16 import VGG16
import mxnet as mx
from numpy import random
from PIL import Image, ImageEnhance




def ChangeContrast(img):
    example_image_copy = img.copy()
    aug = mx.image.BrightnessJitterAug(brightness=0.15)
    aug_image = aug(example_image_copy)
    return tf.image.random_contrast(
    aug_image, 0.8, 1.4, seed=None)


today = date.today()
print("Today's date:", today)

train_dir='C:/Users/Matteo/Desktop/ML/gefran/immagini/train'
val_dir='C:/Users/Matteo/Desktop/ML/gefran/immagini/val'

dizionario_label = {
    'KO': 0,
    'OK': 1
}

INPUT_SHAPE = (35,110,3)
BATCH_SIZE = [8]
EPOCHS = 400

checkpoint_filepath = 'C:/Users/Matteo/Desktop/ML/gefran/modelli/%s'%b+'_%s'%INPUT_SHAPE[0]+'_%s'%INPUT_SHAPE[1]+'_%s'%INPUT_SHAPE[2]+'_%s'%today+'preprocess-function'


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,#0.01
    decay_rate=0.99,
    decay_steps=10000,
    staircase=True)


    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu', input_shape=INPUT_SHAPE),#data_format="channels_first"),
    #     tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
    #     tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu',padding='same'),
    #     tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(6, activation='relu'),
    #     tf.keras.layers.Dropout(0.1),
    #     tf.keras.layers.Dense(1, activation='sigmoid')])
    # Res = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None,input_shape=INPUT_SHAPE, pooling=None)
    #
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape = INPUT_SHAPE)

for layer in vgg_conv.layers[:-8]:
    layer.trainable = False
model = Sequential()
model.add(vgg_conv)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(6,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))


model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              loss = tf.keras.losses.BinaryCrossentropy(name='val_loss'),
              metrics =['accuracy'])

              # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255.,
    vertical_flip = True,
    height_shift_range=0.03,
    width_shift_range=0.03,
    zoom_range=0.02,
    preprocessing_function=ChangeContrast
)


# Note that the validation data should not be augmented!
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255.,
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
        class_mode='binary')


x= train_generator.next()
for i in range(0,4):
    image = x[0][i]
    plt.imshow(image)
    plt.show()


history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[model_checkpoint_callback]
)




acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'r', label='loss')
plt.plot(epochs, val_loss, 'y', label='val_loss')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
