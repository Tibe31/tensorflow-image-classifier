import tensorflow as tf
import datetime
import os
import numpy as np

from utils.config_loader import Config
from model.image_classification_model import ImageClassificationModel
from utils.utils import show_augmentations
from utils.callbacks import ModelCheckpointCallback
from utils.data_splitter import perform_auto_split  # ⬅️ nuovo import

# === CONFIG ===
config = Config()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === Augmentations ===
train_augmentation_parameters = config['augmentation']['train']
val_augmentation_parameters = config['augmentation']['val']

# === Parametri ===
drop_rate = config['dropout']
batch = config['batch_size']

# === Costruzione modello ===
if config['use_pretrained_model']:
    print(f"Carico modello preaddestrato da: {config['pretrained_model_path']}")
    model = tf.keras.models.load_model(config['pretrained_model_path'])
    for layer in model.layers:
        layer.trainable = True
else:
    print("Creo un nuovo modello da zero")
    model = ImageClassificationModel(config.cfg, drop_rate).build_model()
    print(model.summary())

# === Callback naming ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{batch}_{config['input_shape'][0]}_{config['input_shape'][1]}_{config['input_shape'][2]}_{timestamp}"

callback_instance = ModelCheckpointCallback(
    os.path.join(config['checkpoint_filepath'], model_name),
    monitor=config['monitor_metric'],
    mode=config['mode'],
    save_best_only=True
)
model_checkpoint_callback = callback_instance.get_callback()

# === AUTO SPLIT ===
train_dir, val_dir, test_dir = perform_auto_split(config)

# === Data Generators ===
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**train_augmentation_parameters)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**val_augmentation_parameters)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=tuple(config['input_shape'][:2]),
    batch_size=batch,
    class_mode='binary',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=tuple(config['input_shape'][:2]),
    batch_size=batch,
    class_mode='binary',
    shuffle=False
)

if config['show_augmentations']:
    show_augmentations(batch, train_generator)

# === Compilazione modello ===
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    config['lr'],
    decay_rate=0.99,
    decay_steps=2000,
    staircase=False
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.BinaryCrossentropy(name='loss'),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(threshold=0.5),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives(),
    ]
)

# === Addestramento ===
history = model.fit(
    train_generator,
    epochs=config['epochs'],
    validation_data=validation_generator,
    callbacks=[model_checkpoint_callback]
)
