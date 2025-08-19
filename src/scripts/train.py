import tensorflow as tf
import datetime
import os
import numpy as np
import subprocess
import time
from src.utils.config_loader import Config
from src.model.image_classification_model import ImageClassificationModel
from src.utils.utils import show_augmentations
from src.utils.callbacks import ModelCheckpointCallback
from src.utils.data_splitter import perform_auto_split

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


monitor_metric = config['monitor_metric']
if config['classes'] == 2:
    monitor_metric = 'val_binary_accuracy'
else:
    monitor_metric = 'val_accuracy'

callback_instance = ModelCheckpointCallback(
    os.path.join(config['checkpoint_filepath'], model_name),
    monitor=monitor_metric,
    mode=config['mode'],
    save_best_only=True
)
model_checkpoint_callback = callback_instance.get_callback()

# === TensorBoard Callback ===
log_dir = os.path.join("logs", "fit", model_name)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Launch TensorBoard in a separate process
print(f"Launching TensorBoard. Log directory: {log_dir}")
tb_process = subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006"], shell=True)
time.sleep(5) # Give TensorBoard a moment to start
print("TensorBoard should be accessible at http://localhost:6006/")
print("To stop TensorBoard, find its process and terminate it (e.g., using Task Manager on Windows).")

try:

    # === AUTO SPLIT ===
    # train_dir, val_dir, test_dir = perform_auto_split(config)

    # === DATA PATHS ===
    # The data split is now handled exclusively by app.py.
    # train.py reads the resulting paths from the configuration.
    train_dir = config['train_dir']
    val_dir = config['val_dir']
    test_dir = config['test_path']

    # === Data Generators ===
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**train_augmentation_parameters)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**val_augmentation_parameters)

    # === Class mode e configurazione dinamica ===
    num_classes = config['classes']

    if num_classes == 2:
        final_class_mode = 'binary'
        final_loss = tf.keras.losses.BinaryCrossentropy(name='loss')
        final_metrics = [
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
        ]
    else:
        final_class_mode = 'categorical'  # oppure 'sparse' se non usi one-hot encoding
        final_loss = tf.keras.losses.CategoricalCrossentropy(name='loss')
        final_metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]


    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=tuple(config['input_shape'][:2]),
        batch_size=batch,
        class_mode=final_class_mode,
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=tuple(config['input_shape'][:2]),
        batch_size=batch,
        class_mode=final_class_mode,
        shuffle=False
    )

    if config['show_augmentations']:
        augmentation_image_path = show_augmentations(train_dir, batch, config['input_shape'], train_augmentation_parameters, config['classes'])
        print(f"AUGMENTATION_IMAGE_PATH:{augmentation_image_path}")

    # === Compilazione modello ===
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config['lr'],
        decay_rate=0.99,
        decay_steps=2000,
        staircase=False
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=final_loss,
        metrics=final_metrics 
    )

    # === Addestramento ===
    history = model.fit(
        train_generator,
        epochs=config['epochs'],
        validation_data=validation_generator,
        callbacks=[model_checkpoint_callback, tensorboard_callback]
    )

finally:
    print("Terminating TensorBoard process...")
    tb_process.terminate()
    tb_process.wait() # Wait for the process to actually terminate
    print("TensorBoard process terminated.")
