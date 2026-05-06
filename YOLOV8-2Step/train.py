from ultralytics import YOLO
import os
import shutil

# ── Build classification dataset structure ──────────────────
BASE = r"C:\Users\User\Desktop\intelligent system\Intelligent-Systems-Assignment-1\dataset\fall_dataset\crops"
CLASSIFY_DIR = r"C:\Users\User\Desktop\intelligent system\Intelligent-Systems-Assignment-1\dataset\classify_dataset"

for split in ['train', 'val']:
    for cls in ['person', 'non_person']:
        os.makedirs(os.path.join(CLASSIFY_DIR, split, cls), exist_ok=True)

# Split crops into train and val (80/20)
for cls in ['person', 'non_person']:
    src_dir = os.path.join(BASE, cls)
    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for f in train_files:
        shutil.copy(os.path.join(src_dir, f),
                    os.path.join(CLASSIFY_DIR, 'train', cls, f))
    for f in val_files:
        shutil.copy(os.path.join(src_dir, f),
                    os.path.join(CLASSIFY_DIR, 'val', cls, f))

print("Dataset structure created!")

# ── Train YOLOv8 Classification Model ──────────────────────
model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=CLASSIFY_DIR,
    epochs=50,
    imgsz=224,
    batch=8,
    name="person_nonperson_classifier",
    patience=20,
    device='cpu',

    # augmentation settings
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    degrees=10.0,
    translate=0.1,
    scale=0.5
)

print("Training complete!")
print("Best model: runs/classify/person_nonperson_classifier/weights/best.pt")