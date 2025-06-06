# Image Classification with TensorFlow

This repository provides a configurable image classification training pipeline using TensorFlow and Keras. It supports training from scratch or fine-tuning a pre-trained model, with customizable data augmentation, callbacks, and training parameters defined in a YAML configuration file.

---

## Project Structure

- `train.py` — Main script to train the model.
- `config.yaml` — Centralized configuration file for training parameters.
- `model/` — Contains model architecture definitions.
- `utils/` — Utility functions, configuration loader, data augmentation previews, and custom callbacks.
- `confusion_matrix.py` — Script for model evaluation with custom F1-score metric and threshold optimization.

---

## Setup Instructions

1. Open a terminal or command prompt.
2. Create a virtual environment using Anaconda:
   ```bash
   conda create -n name_env python=3.9
   ```
3. Activate the environment:
   ```bash
   conda activate name_env
   ```
4. Ensure `pip` is installed:
   ```bash
   conda install pip
   ```
5. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training

After setting up the environment and configuring `config.yaml`, run the training script:

```bash
python train.py
```

The model will be trained on the dataset specified in the configuration file, and the best model will be saved according to the selected evaluation metric.

---

## Model Evaluation

The `confusion_matrix.py` script evaluates a trained binary classification model using a test set. It calculates a custom F1-score and can determine the optimal classification threshold. Optionally, it saves test images into structured folders for easier inspection.

### Key Features

- Custom F1-score metric (TensorFlow/Keras-compatible)
- Automatic threshold optimization
- Two image-saving modes:
  - `standard`: saves images by true class
  - `folder_split`: saves images into `out/0` and `out/1` based on predictions
- Compatible with models trained using `train.py`

### Usage

```bash
python confusion_matrix.py -m <path_to_model> [-s <threshold>] [--mode standard|folder_split]
```

### Parameters

- `-m`, `--model`: Path to the saved model (required)
- `-s`, `--threshold`: Classification threshold (default: 0.5)
- `--mode`: Output mode: `standard` or `folder_split`
