import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Dropout, MaxPooling2D, GlobalAveragePooling2D, Conv2D, GaussianNoise, Dense, Flatten
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    GaussianNoise,
    GlobalAveragePooling2D,
    Input
)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, GaussianNoise
from tensorflow.keras.models import Sequential



class ImageClassificationModel():
    def __init__(self, cfg, drop_rate):
        self.input_shape = cfg['input_shape']
        self.drop_rate = cfg['dropout']
        self.gaussian_noise = cfg['gaussian_noise']

    def build_model(self):
        base_conv = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model = Sequential()
        model.add(base_conv)
        model.add(GlobalAveragePooling2D())
        model.add(GaussianNoise(self.gaussian_noise))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(1, activation='sigmoid'))

        return model