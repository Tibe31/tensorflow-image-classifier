import subprocess
import shutil
import time
import os

# Lista di configurazioni (YAML)
trainings = [
    {
        "config_yaml": "configs/config_st1.yaml",
        "augmentation_yaml": "configs/augmentation_st1.yaml",
        "name": "ST1"
    },
    {
        "config_yaml": "configs/config_st3.yaml",
        "augmentation_yaml": "configs/augmentation_st3.yaml",
        "name": "ST3"
    }
]

for t in trainings:
    print(f"\nAvvio training per {t['name']}...")

    # Copia i file nella posizione attesa da Config
    shutil.copy(t["config_yaml"], "config.yaml")
    shutil.copy(t["augmentation_yaml"], "augmentation.yaml")

    # Avvia train.py
    result = subprocess.run(["python", "train.py"])

    if result.returncode != 0:
        print(f"Training {t['name']} fallito. Interruzione.")
        break

    print(f"Training {t['name']} completato.")
    time.sleep(2)

print("\nTutti i training completati.")
