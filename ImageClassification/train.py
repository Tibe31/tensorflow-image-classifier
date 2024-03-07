import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cfg
from augmentation_cfg import train_augmentation_parameters, val_augmentation_parameters
from model.image_classification_model import ImageClassificationModel


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


        class WriteValMetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self, file_path):
                self.file_path = file_path

            def on_epoch_end(self, epoch, logs=None):
                val_loss = logs.get('val_loss')
                val_accuracy = logs.get('val_binary_accuracy')

                with open(self.file_path, 'a') as file:
                    file.write(f'Dropout {drop_rate} - Epoch {epoch + 1}: Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}\n')


        file_path = 'val_metrics_%d'%batch + '_%f'%drop_rate +'.txt'
        write_val_metrics_callback = WriteValMetricsCallback(file_path)


        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**train_augmentation_parameters)


        # Note that the validation data should not be augmented!
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255.,
        )

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


        # x= train_generator.next()
        # for i in range(0,BATCH_SIZE-1):
            # image = x[0][i]
            # plt.imshow(image)
            # plt.show()


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
