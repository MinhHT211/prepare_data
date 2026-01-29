import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

# ================= CONFIG =================
DATASET_ROOT = "../locv3"
OUT_DIR = "../processed"

IMG_SIZE = (32, 32)
NORMALIZE_255 = True
STANDARDIZE = True

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42

MIN_SAMPLES_PER_CLASS = 7   # lọc class ít mẫu

# =========================================

os.makedirs(OUT_DIR, exist_ok=True)

print("DATASET_ROOT =", os.path.abspath(DATASET_ROOT))
print("Exists:", os.path.exists(DATASET_ROOT))

# ============ LOAD DATASET ============
label_map = {}
label_counter = 0

X = []
y = []
meta = []

for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img_path = os.path.join(root, file)

            # label = tên folder cuối
            label_name = os.path.basename(root)

            if label_name not in label_map:
                label_map[label_name] = label_counter
                label_counter += 1

            label_id = label_map[label_name]

            try:
                img = Image.open(img_path).convert('L')
            except:
                continue

            img = img.resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32)
            arr = arr.flatten()  # 32x32 -> 1024

            X.append(arr)
            y.append(label_id)

            meta.append({
                "path": img_path,
                "label_name": label_name,
                "label_id": label_id
            })

X = np.array(X)
y = np.array(y)

print("\n=== RAW DATASET ===")
print("Total samples:", X.shape[0])
print("Total classes:", len(label_map))

# ============ COUNT PER LABEL ============
counts = Counter(y)
id2label = {v: k for k, v in label_map.items()}

print("\n=== SAMPLE COUNT PER LABEL ===")
for label_id, count in sorted(counts.items(), key=lambda x: x[0]):
    print(f"Label {label_id:3d} | {id2label[label_id]:25s} | Samples: {count}")

# ============ FILTER RARE CLASSES ============
valid_classes = {cls for cls, c in counts.items() if c >= MIN_SAMPLES_PER_CLASS}
mask = np.array([label in valid_classes for label in y])

X = X[mask]
y = y[mask]

print("\n=== AFTER FILTERING RARE CLASSES ===")
print("Samples:", X.shape[0])
print("Classes:", len(set(y)))

removed = {cls: c for cls, c in counts.items() if c < MIN_SAMPLES_PER_CLASS}
print("Removed rare classes:", len(removed))

# ============ NORMALIZATION ============
if NORMALIZE_255:
    X = X / 255.0

# ============ STANDARDIZATION (OPTIONAL) ============
if STANDARDIZE:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    np.save(os.path.join(OUT_DIR, "mean.npy"), mean)
    np.save(os.path.join(OUT_DIR, "std.npy"), std)

# ============ STRATIFIED SPLIT ============
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=(1-TRAIN_RATIO),
    random_state=RANDOM_SEED,
    stratify=y
)

val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=(1-val_ratio_adjusted),
    random_state=RANDOM_SEED,
    stratify=y_temp
)

print("\n=== SPLIT DATASET ===")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# ============ SAVE DATASET ============
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)

np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)

np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

# save label map
with open(os.path.join(OUT_DIR, "label_map.txt"), "w", encoding="utf-8") as f:
    for k, v in label_map.items():
        f.write(f"{v}: {k}\n")

# save metadata
pd.DataFrame(meta).to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)

print("\n=== DONE ===")
print("Processed dataset saved to:", os.path.abspath(OUT_DIR))
