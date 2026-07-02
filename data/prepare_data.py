import os
from datasets import load_dataset
from PIL import Image

IMG_SIZE = 64
DATA_DIR = "data/images"
os.makedirs(f"{DATA_DIR}/cat", exist_ok=True)
os.makedirs(f"{DATA_DIR}/dog", exist_ok=True)

ds = load_dataset("microsoft/cats_vs_dogs", split="train")

sample_per_class = 150
counts = {"cat": 0, "dog": 0}

for i, example in enumerate(ds):
    label = "cat" if example["labels"] == 0 else "dog"
    if counts[label] >= sample_per_class:
        continue
    img = example["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    fname = f"{label}_{counts[label]:03d}.jpg"
    img.save(f"{DATA_DIR}/{label}/{fname}")
    counts[label] += 1
    if all(v >= sample_per_class for v in counts.values()):
        break

print(f"Saved {counts['cat']} cats, {counts['dog']} dogs -> {DATA_DIR}/")
