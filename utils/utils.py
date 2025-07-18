import matplotlib.pyplot as plt
import tensorflow as tf
import os

#show all the images in the training batch
def show_augmentations(train_dir, batch_size, input_shape, augmentation_parameters, num_classes, output_path="temp_augmentations.png"):
  class_mode = 'binary' if num_classes == 2 else 'categorical'
  # Create a temporary ImageDataGenerator for visualization
  temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**augmentation_parameters)
  
  temp_generator = temp_datagen.flow_from_directory(
      train_dir,
      target_size=tuple(input_shape[:2]),
      batch_size=batch_size,
      class_mode=class_mode,
      shuffle=True
  )

  x = temp_generator.next()
  fig, axes = plt.subplots(int(batch_size**0.5), int(batch_size**0.5), figsize=(10, 10))
  axes = axes.flatten()
  for i in range(0, min(batch_size, len(x[0]))):
    axes[i].imshow(x[0][i])
    axes[i].axis('off')
  plt.tight_layout()
  plt.savefig(output_path) # Save the figure instead of showing it
  plt.close(fig) # Close the figure to free memory
  return output_path
