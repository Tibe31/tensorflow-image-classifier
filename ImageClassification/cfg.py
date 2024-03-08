batch_size = 32
learning_rate = 0.001
label_dictionary = {
    "KO": 0,
    "OK": 1
}
epochs = 1000
train_dir='immagini/train'
val_dir='immagini/val'
input_shape = (200,200,3)
batch_sizes = [16]
lr = 1e-3
decay_after_n_epochs = 5
dropouts = [0.4]
checkpoint_filepath_main = 'modelli'
gaussian_noise = 5
file_path = 'val_metrics_%d'%batch + '_%f'%drop_rate +'.txt'
