import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cfg
from augmentation_cfg import train_augmentation_parameters, val_augmentation_parameters
from model.image_classification_model import ImageClassificationModel
from utils.utils import show_augmentations
from utils.callbacks import WriteValMetricsCallback

for batch in cfg.batch_sizes:

    for drop_rate in cfg.dropouts:

        checkpoint_filepath = cfg.checkpoint_filepath_main + '/%s'%batch+'_%s'%cfg.input_shape[0]+'_%s'%cfg.input_shape[1]+'_%s'%cfg.input_shape[2]+'_%s'

        model = ImageClassificationModel(cfg, drop_rate).build_model()

        print (model.summary())

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        write_val_metrics_callback = WriteValMetricsCallback(cfg.path_metrics + 'val_metrics_%d'%batch + '_%f'%drop_rate +'.txt')


        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**train_augmentation_parameters)

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**val_augmentation_parameters)

        train_generator = train_datagen.flow_from_directory(
                cfg.train_dir,
                target_size=(cfg.input_shape[0], cfg.input_shape[1]),
                batch_size=batch,
                class_mode='binary',
                shuffle=True)
        validation_generator = test_datagen.flow_from_directory(
                cfg.val_dir,
                target_size=(cfg.input_shape[0], cfg.input_shape[1]),
                batch_size=batch,
                class_mode='binary',
                shuffle = True)
                
        if (cfg.show_augmentations == 'true'):
            show_augmentations(batch, train_generator)


        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.lr,
            decay_rate=0.99,
            decay_steps=int((train_generator.samples/batch))*cfg.decay_after_n_epochs,
            staircase=False)

        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                      loss = tf.keras.losses.BinaryCrossentropy(name='val_accuracy'),
                      metrics =[tf.keras.metrics.BinaryAccuracy(threshold = 0.5),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])



        history = model.fit(
            train_generator,
            epochs=cfg.epochs,
            validation_data=validation_generator,
            callbacks=[model_checkpoint_callback,write_val_metrics_callback]
        )
