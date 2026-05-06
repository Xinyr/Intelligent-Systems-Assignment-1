import os
import cv2
import random

IMAGE_DIR = r"C:\Users\User\Desktop\intelligent system\Intelligent-Systems-Assignment-1\dataset\fall_dataset\images\train"
LABEL_DIR = r"C:\Users\User\Desktop\intelligent system\Intelligent-Systems-Assignment-1\dataset\fall_dataset\labels\train"
OUTPUT_DIR = r"C:\Users\User\Desktop\intelligent system\Intelligent-Systems-Assignment-1\dataset\fall_dataset"

person_img_dir = os.path.join(OUTPUT_DIR, "crops", "person")
nonperson_img_dir = os.path.join(OUTPUT_DIR, "crops", "non_person")
os.makedirs(person_img_dir, exist_ok=True)
os.makedirs(nonperson_img_dir, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR,
                  os.path.splitext(img_file)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    person_boxes = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 5:
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            person_boxes.append((x1, y1, x2, y2))

            # Save person crop
            px1 = max(0, x1-5)
            py1 = max(0, y1-5)
            px2 = min(w, x2+5)
            py2 = min(h, y2+5)
            crop = img[py1:py2, px1:px2]
            if crop.size > 0:
                base = os.path.splitext(img_file)[0]
                cv2.imwrite(os.path.join(person_img_dir,
                            f"{base}_p{i}.jpg"), crop)

    # Save non-person crops from background
    attempts = 0
    saved = 0
    while attempts < 100 and saved < 3:
        patch_w = random.randint(60, 150)
        patch_h = random.randint(60, 150)
        rx1 = random.randint(0, max(0, w - patch_w))
        ry1 = random.randint(0, max(0, h - patch_h))
        rx2 = rx1 + patch_w
        ry2 = ry1 + patch_h

        # Check no overlap with person boxes
        overlap = False
        for (bx1, by1, bx2, by2) in person_boxes:
            if not (rx2 < bx1 or rx1 > bx2 or ry2 < by1 or ry1 > by2):
                overlap = True
                break

        if not overlap:
            nonperson_crop = img[ry1:ry2, rx1:rx2]
            if nonperson_crop.size > 0:
                base = os.path.splitext(img_file)[0]
                cv2.imwrite(os.path.join(nonperson_img_dir,
                            f"{base}_np{saved}.jpg"), nonperson_crop)
                saved += 1
        attempts += 1

print(f"Done!")
print(f"Person crops saved to: {person_img_dir}")
print(f"Non-person crops saved to: {nonperson_img_dir}")