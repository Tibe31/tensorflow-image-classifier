import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Metric
from sklearn.metrics import f1_score
from src.utils.config_loader import Config

# ---------------------------
# Custom F1Score metric
# ---------------------------
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='val_f1_score', threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * ((precision * recall) / (precision + recall + 1e-7))

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

    def get_config(self):
        config = super(F1Score, self).get_config()
        config.update({"threshold": self.threshold})
        return config


# ---------------------------
# Threshold optimization
# ---------------------------
def find_best_threshold(labels, scores):
    thresholds = np.linspace(0, 1, num=100)
    best_f1 = 0
    best_threshold = 0
    for threshold in thresholds:
        predicted = (np.array(scores) >= threshold).astype(int)
        f1 = f1_score(labels, predicted)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


# ---------------------------
# Main
# ---------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model folder")
ap.add_argument("-s", "--threshold", type=float, default=0.5, help="soglia per classificazione")
ap.add_argument("--mode", choices=["standard", "folder_split"], default="standard",
                help="modalità di salvataggio immagini: standard oppure folder_split per scarti/buoni")
args = vars(ap.parse_args())

config = Config()

# 🧠 Estrai input_shape dal nome file
model_path = args["model"]
filename = os.path.basename(model_path)
try:
    shape_part = filename.split('_')[:4]  # es: 8_512_512_3
    input_shape = tuple(map(int, shape_part[1:]))  # (512, 512, 3)
except Exception as e:
    print(f"[ERRORE] Impossibile estrarre input_shape da: {filename}, uso default (512, 512, 3)")
    input_shape = (512, 512, 3)

DATA_PATH = config['test_path']
threshold = args["threshold"]
mode = args["mode"]

# ✅ Carica modello con oggetti custom
print("[INFO] Caricamento modello...")
model = load_model(model_path, compile=False)
model.compile(
    optimizer="sgd",  # anche placeholder
    loss="binary_crossentropy",
    metrics=[F1Score(name="val_f1_score")])
print("[INFO] Modello caricato.")
print(model.summary())

count_images = 0
ground_truth_labels = []
predicted_probabilities = []

if mode == "folder_split":
    # Creo le cartelle "0" e "1" dentro out
    output_dirs = {
        0: os.path.join("out", "0"),
        1: os.path.join("out", "1")
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

for directory in os.listdir(DATA_PATH):
    dir_path = os.path.join(DATA_PATH, directory)

    # In modalità standard creo sotto-cartella out/<classe>
    if mode == "standard":
        output_dir = os.path.join('out', directory)
        os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Analizzo classe: {directory}")

    for img_name in os.listdir(dir_path):
        count_images += 1
        image_path = os.path.join(dir_path, img_name)
        image_bgr = cv2.imread(image_path)
        image_resized = cv2.resize(image_bgr, (input_shape[1], input_shape[0]))
        image_normalized = image_resized.astype("float32") / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

        # Predizione
        prediction = model.predict(image_input)[0][0]
        predicted_probabilities.append(prediction)
        ground_truth_labels.append(int(directory))

        output_filename = f"{prediction:.4f}_{img_name}"

        if mode == "standard":
            output_path = os.path.join(output_dir, output_filename)
        else:  # folder_split
            label_folder = 1 if prediction >= threshold else 0
            output_path = os.path.join(output_dirs[label_folder], output_filename)

        cv2.imwrite(output_path, image_bgr)

# ---------------------------
# Valutazione finale
# ---------------------------
best_threshold, best_f1 = find_best_threshold(ground_truth_labels, predicted_probabilities)
print(f"[INFO] Totale immagini analizzate: {count_images}")
print(f"[INFO] Miglior soglia trovata: {best_threshold:.3f}")
print(f"[INFO] F1-score ottimale: {best_f1:.3f}")
