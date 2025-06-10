import shutil
import random
from pathlib import Path

def perform_auto_split(config):
    if not config['auto_split']['enabled']:
        return config['train_dir'], config['val_dir']

    source_dir = Path(config['auto_split']['source_dir'])
    output_dir = Path(config['auto_split']['output_dir'])
    val_split = config['auto_split']['val_split']

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    class_counts = {}

    # Conta immagini per ogni classe
    for class_dir in class_dirs:
        class_counts[class_dir.name] = len(list(class_dir.glob("*")))

    # Trova numero minimo di immagini tra le classi
    min_images = min(class_counts.values())
    num_val_per_class = int(min_images * val_split)

    for class_dir in class_dirs:
        images = list(class_dir.glob("*"))
        random.shuffle(images)

        val_imgs = images[:num_val_per_class]
        train_imgs = images[num_val_per_class:]

        for subset_dir, subset_imgs in [(train_dir, train_imgs), (val_dir, val_imgs)]:
            class_subdir = subset_dir / class_dir.name
            class_subdir.mkdir(parents=True, exist_ok=True)
            for img_path in subset_imgs:
                shutil.copy(img_path, class_subdir)

    print(f"Split equilibrato completato: {num_val_per_class} val immagini per classe.")
    return str(train_dir), str(val_dir)
