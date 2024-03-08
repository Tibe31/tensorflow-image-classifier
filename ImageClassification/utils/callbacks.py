import tensorflow as tf

class WriteValMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        self.file_path = file_path

    def on_epoch_end(self, epoch, drop_rate, logs=None):
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_binary_accuracy')

        with open(self.file_path, 'a') as file:
            file.write(f'Dropout {drop_rate} - Epoch {epoch + 1}: Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}\n')
