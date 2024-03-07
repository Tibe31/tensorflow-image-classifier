train_augmentation_parameters = {
    "rescale": 1./255.,
    "rotation_range": 2,
    "zoom_range": 0.02,
    "width_shift_range": 0.02,
    "height_shift_range": 0.02,
    "brightness_range": [0.9, 1.1],
    "fill_mode": "nearest"
}

val_augmentation_parameters = {
    "rescale": 1./255.
}
