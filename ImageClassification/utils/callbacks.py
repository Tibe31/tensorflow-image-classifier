import tensorflow as tf
import numpy as np


class ModelCheckpointCallback:
    def __init__(self, checkpoint_filepath, monitor, mode, save_best_only = True):
        self.checkpoint_filepath = checkpoint_filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only


def get_callback(self):
            # Define the custom ModelCheckpoint with added functionality to log to a file
            class CustomModelCheckpoint(tf.keras.callbacks.Callback):
                def __init__(self, filepath, monitor, mode, save_best_only):
                    super(CustomModelCheckpoint, self).__init__()
                    self.filepath = filepath
                    self.monitor = monitor
                    self.mode = mode
                    self.save_best_only = save_best_only
                    self.best = -np.Inf if mode == 'max' else np.Inf

                def on_epoch_end(self, epoch, logs=None):
                    current = logs.get(self.monitor)
                    if current is None:
                        return

                    if (self.mode == 'max' and current > self.best) or (self.mode == 'min' and current < self.best):
                        self.best = current
                        if self.save_best_only:
                            self.model.save(self.filepath.format(epoch=epoch, **logs))
                            # Log the model saving action to performance_monitor.txt
                            with open("performance_monitor.txt", "a") as f:
                                f.write(f"Epoch {epoch + 1}: {self.monitor} improved to {current}\n")
                            print(f"Epoch {epoch + 1}: {self.monitor} improved to {current}, model saved.")

            return CustomModelCheckpoint(
                filepath=self.checkpoint_filepath,
                monitor=self.monitor,
                mode=self.mode,
                save_best_only=self.save_best_only
            )
