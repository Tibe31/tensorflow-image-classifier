# Image Classification with TensorFlow

This repository contains a configurable image classification training pipeline using TensorFlow and Keras. It supports training from scratch or fine-tuning a pre-trained model, with configurable data augmentations, callbacks, and training parameters via a YAML file.

## Project Structure

- `train.py`: Main script to train the model.
- `config.yaml`: Centralized configuration file for training parameters.
- `model/`: Contains the model architecture definition.
- `utils/`: Contains utility functions, configuration loader, augmentation visualizations, and custom callbacks.

## Setup Instructions

To configure a virtual environment and train the classifier:

1. Open a command prompt;
2. Create a virtual environment with Anaconda:
   ```bash
   conda create -n "name_env" python=3.9
   ```
3. Activate the environment:
   ```bash
   conda activate "name_env"
   ```
4. Install pip (if not already installed):
   ```bash
   conda install pip
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Once the environment is set up and the configuration file is properly edited, run the training script:

```bash
python train.py
```

The model will be trained on the dataset defined in `config.yaml`, and the best model will be saved based on the monitored metric.
