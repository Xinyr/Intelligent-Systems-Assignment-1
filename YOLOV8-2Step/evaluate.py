import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

model = YOLO("runs/classify/person_nonperson_classifier5/weights/best.pt")

CLASSIFY_DIR = r"C:\Users\User\Desktop\intelligent system\Intelligent-Systems-Assignment-1\dataset\classify_dataset"

def evaluate_split(split_name):
    split_dir = os.path.join(CLASSIFY_DIR, split_name)
    
    if not os.path.exists(split_dir):
        print(f"\n{split_name} folder not found, skipping...")
        return

    class_names = sorted(os.listdir(split_dir))
    y_true = []
    y_pred = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        print(f"  Running inference on {len(image_files)} '{class_name}' images...")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path, verbose=False)
            probs = results[0].probs
            pred_idx = int(probs.top1)
            y_true.append(class_idx)
            y_pred.append(pred_idx)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n{'='*55}")
    print(f"   CLASSIFICATION REPORT — {split_name.upper()} SET")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                 target_names=class_names,
                                 digits=4))
    print(f"Overall Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro Precision  : {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Macro Recall     : {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Macro F1-Score   : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"{'':>15}", end="")
    for cn in class_names:
        print(f"{cn:>15}", end="")
    print()
    for i, cn in enumerate(class_names):
        print(f"{cn:>15}", end="")
        for val in cm[i]:
            print(f"{val:>15}", end="")
        print()

    # Save confusion matrix plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix – {split_name.upper()} Set")
    plt.tight_layout()
    filename = f"confusion_matrix_{split_name}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {filename}")


# ── Run for all 3 splits ─────────────────────────────────────
for split in ['train', 'val', 'test']:
    print(f"\nEvaluating {split} set...")
    evaluate_split(split)

print("\nDone! All 3 sets evaluated.")