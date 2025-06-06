import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
import cfg  # configurazione con percorso dati

# ---------------------------
# Custom F1 Score Metric
# ---------------------------
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='val_f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
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
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

# ---------------------------
# Trova la soglia ottimale
# ---------------------------
def find_best_threshold(labels, scores):
    thresholds = np.linspace(0, 1, num=100)
    best_f1, best_threshold = 0, 0
    for t in thresholds:
        preds = (np.array(scores) >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Percorso al modello salvato")
    parser.add_argument("-s", "--threshold", type=float, default=0.5, help="Soglia iniziale di classificazione")
    parser.add_argument("--mode", choices=["standard", "folder_split"], default="standard",
                        help="Modalità di salvataggio immagini: standard o folder_split")
    args = parser.parse_args()

    model_path = args.model
    threshold = args.threshold
    mode = args.mode
    data_path = cfg.test_path

    # Estrai input_shape dal nome del file modello
    try:
        shape_part = os.path.basename(model_path).split('_')[:4]
        input_shape = tuple(map(int, shape_part[1:]))
    except Exception:
        print(f"[ERRORE] Impossibile estrarre input_shape da nome file. Uso default (512, 512, 3)")
        input_shape = (512, 512, 3)

    # Carica modello
    print("[INFO] Caricamento modello...")
    model = load_model(model_path, compile=False)
    model.compile(
        optimizer="sgd",  # Placeholder
        loss="binary_crossentropy",
        metrics=[F1Score(name="val_f1_score")]
    )
    print("[INFO] Modello caricato.")
    print(model.summary())

    count, y_true, y_scores = 0, [], []

    if mode == "folder_split":
        output_dirs = {0: os.path.join("out", "0"), 1: os.path.join("out", "1")}
        for path in output_dirs.values():
            os.makedirs(path, exist_ok=True)

    for class_dir in os.listdir(data_path):
        class_path = os.path.join(data_path, class_dir)
        output_dir = os.path.join("out", class_dir)
        if mode == "standard":
            os.makedirs(output_dir, exist_ok=True)

        print(f"[INFO] Elaborazione classe: {class_dir}")
        for img_file in os.listdir(class_path):
            count += 1
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (input_shape[1], input_shape[0]))
            img_normalized = img_resized.astype("float32") / 255.0
            input_tensor = np.expand_dims(img_normalized, axis=0)

            score = model.predict(input_tensor, verbose=0)[0][0]
            y_scores.append(score)
            y_true.append(int(class_dir))

            output_name = f"{score:.4f}_{img_file}"
            if mode == "standard":
                cv2.imwrite(os.path.join(output_dir, output_name), img)
            else:
                label_dir = output_dirs[1 if score >= threshold else 0]
                cv2.imwrite(os.path.join(label_dir, output_name), img)

    # Risultati finali
    best_thresh, best_f1 = find_best_threshold(y_true, y_scores)
    print(f"[INFO] Immagini elaborate: {count}")
    print(f"[INFO] Soglia ottimale trovata: {best_thresh:.3f}")
    print(f"[INFO] F1-score massimo: {best_f1:.3f}")

if __name__ == "__main__":
    main()
