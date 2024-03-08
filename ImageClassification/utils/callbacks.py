import tensorflow as tf


class ModelCheckpointCallback:
    def __init__(self, checkpoint_filepath, monitor, mode, save_best_only = True):
        self.checkpoint_filepath = checkpoint_filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only


    def get_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=False,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=self.save_best_only
        )