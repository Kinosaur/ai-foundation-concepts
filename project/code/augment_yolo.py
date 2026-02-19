import os
import cv2
import glob
import random
from pathlib import Path

import albumentations as A

# ====== SETTINGS ======
SRC = Path(r"/Users/kaungkhantlin/Developer/2_2025/AI_Concepts/project/receipts/YOLODataset")                 # your original dataset folder
DST = Path("YOLODataset_aug")             # output augmented dataset folder
COPIES_PER_IMAGE = 5                      # how many augmented versions per train image

random.seed(42)

# Albumentations pipeline (safe, common augmentations)
# Albumentations pipeline (optimized for Document/Text Detection)
transform = A.Compose(
    [
        # 1. GEOMETRY (Conservative)
        # minimal rotation to simulate slightly crooked scans
        A.SafeRotate(limit=3, p=0.5), 
        
        # random perspective simulates taking a photo at an angle
        A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.3), 

        # 2. PIXEL LEVEL (Aggressive is okay here)
        # Simulates different scanner lights or phone camera flashes
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Simulates different paper ages (yellowing) or printer ink levels
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        
        # 3. NOISE (Simulate bad cameras/printing)
        # ISO noise is better than blur for text because edges remain sharp
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        
        # Only use very weak blur if necessary
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=["class_labels"])
)

def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            bboxes.append([x, y, w, h])
            class_labels.append(cls)
    return bboxes, class_labels

def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, (x, y, w, h) in zip(class_labels, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def copy_tree():
    # Copy val set as-is (no augmentation)
    for split in ["val"]:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path in glob.glob(str(SRC / "images" / split / "*.*")):
            img_path = Path(img_path)
            label_path = SRC / "labels" / split / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            os.makedirs(DST / "images" / split, exist_ok=True)
            os.makedirs(DST / "labels" / split, exist_ok=True)
            cv2.imwrite(str(DST / "images" / split / img_path.name), cv2.imread(str(img_path)))
            with open(label_path, "r", encoding="utf-8") as f:
                (DST / "labels" / split / label_path.name).write_text(f.read(), encoding="utf-8")

    # Copy dataset.yaml
    (DST / "dataset.yaml").write_text((SRC / "dataset.yaml").read_text(encoding="utf-8"), encoding="utf-8")

def augment_train():
    (DST / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "train").mkdir(parents=True, exist_ok=True)

    train_imgs = glob.glob(str(SRC / "images" / "train" / "*.*"))
    for img_path in train_imgs:
        img_path = Path(img_path)
        label_path = SRC / "labels" / "train" / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        bboxes, class_labels = read_yolo_labels(label_path)

        # 1) Copy original
        cv2.imwrite(str(DST / "images" / "train" / img_path.name), image)
        (DST / "labels" / "train" / label_path.name).write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")

        # 2) Make augmented copies
        for k in range(COPIES_PER_IMAGE):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_cls = augmented["class_labels"]

            # Skip if augmentation removed all boxes (can happen rarely)
            if len(aug_boxes) == 0:
                continue

            new_name = f"{img_path.stem}_aug{k}{img_path.suffix}"
            new_lbl = f"{img_path.stem}_aug{k}.txt"

            cv2.imwrite(str(DST / "images" / "train" / new_name), aug_img)
            write_yolo_labels(DST / "labels" / "train" / new_lbl, aug_boxes, aug_cls)

if __name__ == "__main__":
    copy_tree()
    augment_train()
    print("✅ Augmented dataset created at:", DST)
