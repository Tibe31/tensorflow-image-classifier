import tensorflow as tf
import numpy as np
from datetime import datetime

class ModelCheckpointCallback:
    def __init__(self, checkpoint_filepath, monitor='val_binary_accuracy', mode='max', save_best_only=True):
        self.checkpoint_filepath = checkpoint_filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only

    def get_callback(self):
        class CustomModelCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, filepath, monitor, mode, save_best_only):
                super().__init__()
                self.filepath = filepath
                self.monitor = monitor
                self.mode = mode
                self.save_best_only = save_best_only
                self.best_acc = -np.Inf
                self.best_loss = np.Inf

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current_acc = logs.get('val_binary_accuracy')
                current_loss = logs.get('val_loss')

                if current_acc is None or current_loss is None:
                    return

                save = False

                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    self.best_loss = current_loss
                    save = True
                elif current_loss < self.best_loss and current_acc >= self.best_acc:
                    self.best_loss = current_loss
                    save = True

                if save:
                    filepath = self.filepath.format(epoch=epoch, **logs)
                    self.model.save(filepath)

                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("performance_monitor.txt", "a") as f:
                        f.write(f"{current_time} - Epoch {epoch + 1}: model saved. val_loss: {current_loss}, val_binary_accuracy: {current_acc}\n")
                    print(f"{current_time} - Epoch {epoch + 1}: model saved. val_loss: {current_loss}, val_binary_accuracy: {current_acc}")

        return CustomModelCheckpoint(
            filepath=self.checkpoint_filepath,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=self.save_best_only
        )
