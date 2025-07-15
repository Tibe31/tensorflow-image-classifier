import tensorflow as tf
import numpy as np
from datetime import datetime

class ModelCheckpointCallback:
    def __init__(self, checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True):
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
                
                # Inizializza i valori migliori in base al mode
                if self.mode == 'max':
                    self.best_metric = -np.Inf
                else:  # mode == 'min'
                    self.best_metric = np.Inf
                
                self.best_loss = np.Inf

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current_metric = logs.get(self.monitor)
                current_loss = logs.get('val_loss')

                if current_metric is None or current_loss is None:
                    print(f"Warning: {self.monitor} or val_loss not found in logs")
                    return

                save = False

                # Logica per decidere se salvare basata sul mode
                if self.mode == 'max':
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.best_loss = current_loss
                        save = True
                    elif current_metric == self.best_metric and current_loss < self.best_loss:
                        self.best_loss = current_loss
                        save = True
                else:  # mode == 'min'
                    if current_metric < self.best_metric:
                        self.best_metric = current_metric
                        self.best_loss = current_loss
                        save = True
                    elif current_metric == self.best_metric and current_loss < self.best_loss:
                        self.best_loss = current_loss
                        save = True

                if save:
                    filepath = self.filepath.format(epoch=epoch, **logs)
                    self.model.save(filepath)

                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("performance_monitor.txt", "a") as f:
                        f.write(f"{current_time} - Epoch {epoch + 1}: model saved. val_loss: {current_loss:.4f}, {self.monitor}: {current_metric:.4f}\n")
                    print(f"{current_time} - Epoch {epoch + 1}: model saved. val_loss: {current_loss:.4f}, {self.monitor}: {current_metric:.4f}")

        return CustomModelCheckpoint(
            filepath=self.checkpoint_filepath,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=self.save_best_only
        )