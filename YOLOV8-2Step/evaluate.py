import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Load model ──────────────────────────────────────────────
model = YOLO("runs/classify/person_nonperson_classifier/weights/best.pt")

CLASSIFY_DIR = r"C:\Users\User\Desktop\intelligent system\Intelligent-Systems-Assignment-1\dataset\classify_dataset"
VAL_DIR = os.path.join(CLASSIFY_DIR, "val")

# ── Collect ground truth and predictions ────────────────────
class_names = sorted(os.listdir(VAL_DIR))   # ['non_person', 'person']
print(f"Classes found: {class_names}")

y_true = []
y_pred = []
y_conf = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(VAL_DIR, class_name)
    image_files = [f for f in os.listdir(class_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Running inference on {len(image_files)} '{class_name}' images...")

    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        results = model(img_path, verbose=False)

        # Top-1 predicted class index and confidence
        probs = results[0].probs
        pred_idx = int(probs.top1)
        pred_conf = float(probs.top1conf)

        y_true.append(class_idx)
        y_pred.append(pred_idx)
        y_conf.append(pred_conf)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ── Print full report ────────────────────────────────────────
print("\n" + "="*55)
print("         FULL CLASSIFICATION REPORT")
print("="*55)
print(classification_report(y_true, y_pred,
                             target_names=class_names,
                             digits=4))

print(f"Overall Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro Precision  : {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Macro Recall     : {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"Macro F1-Score   : {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"Weighted F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

# ── Confusion matrix ─────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(f"{'':>15}", end="")
for cn in class_names:
    print(f"{cn:>15}", end="")
print()
for i, cn in enumerate(class_names):
    print(f"{cn:>15}", end="")
    for val in cm[i]:
        print(f"{val:>15}", end="")
    print()

# ── Save confusion matrix plot ────────────────────────────────
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Person vs Non-Person")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix.png")