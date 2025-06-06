import subprocess
import shutil
import time

# Lista di configurazioni
trainings = [
    {
        "cfg": "cfg_st1_best.py",
        "aug": "augmentation_cfg_1.py",
        "name": "ST1"
    },
    {
        "cfg": "cfg_st3_best.py",
        "aug": "augmentation_cfg_3.py",
        "name": "ST3"
    }
]

for t in trainings:
    print(f"\n🧠 Avvio training per {t['name']}...")

    # Copia file config nel modulo usato da train_weights
    shutil.copy(t["cfg"], "cfg.py")
    shutil.copy(t["aug"], "augmentation_cfg.py")

    # Avvia train_weights.py
    result = subprocess.run(["python", "train.py"])

    if result.returncode != 0:
        print(f"❌ Training {t['name']} fallito.")
        break

    print(f"✅ Training {t['name']} completato.")
    time.sleep(2)

print("\n🏁 Tutti i training completati.")
