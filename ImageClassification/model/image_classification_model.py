import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Dropout, MaxPooling2D, GlobalAveragePooling2D, Conv2D, GaussianNoise, Dense, Flatten


class ImageClassificationModel():
    def __init__(self, cfg, drop_rate):
        self.input_shape = cfg.input_shape
        self.drop_rate = drop_rate
        self.gaussian_noise = cfg.gaussian_noise

    def build_model(self):
        base_conv = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model = Sequential()
        model.add(base_conv)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(self.drop_rate))
        model.add(GaussianNoise(self.gaussian_noise))
        model.add(Dense(1, activation="sigmoid"))
        return model
        