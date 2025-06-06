# Image Classification with TensorFlow

This repository contains a configurable image classification training pipeline using TensorFlow and Keras. It supports training from scratch or fine-tuning a pre-trained model, with configurable data augmentations, callbacks, and training parameters via a YAML file.

## Project Structure

- `train.py`: Main script to train the model.
- `config.yaml`: Centralized configuration file for training parameters.
- `model/`: Contains the model architecture definition.
- `utils/`: Contains utility functions, configuration loader, augmentation visualizations, and custom callbacks.
- `confusion_matrix.py`: Script to evaluate model performance on a test set with custom F1-score and threshold optimization.

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

## Model Evaluation

The script confusion_matrix.py evaluates a trained binary classification model by analyzing its predictions on a test dataset. It computes a custom F1-score and can automatically find the best classification threshold. It also optionally saves test images into structured folders for inspection.

Features
Custom F1-score metric implemented as a TensorFlow/Keras metric.

Two image-saving modes:

standard: saves images by their true class.

folder_split: saves images into folders based on predicted class.

Automatic threshold optimization for F1-score.

Compatible with models trained and saved by train.py.

   ```bash
   python confusion_matrix.py -m <path_to_model> [-s <threshold>] [--mode standard|folder_split]
   ``
Parameters
-m, --model: Path to the saved model (required).

-s, --threshold: Classification threshold (default: 0.5).

--mode: Output mode:

standard: saves images under folders matching ground-truth labels.

folder_split: saves images into out/0 and out/1 based on predicted class.
