import cv2
import sys
import os
from ultralytics import YOLO

# ── Load models ──────────────────────────────────────────────
detect_model = YOLO("yolov8n.pt")
classify_model = YOLO("runs/classify/person_nonperson_classifier5/weights/best.pt")

def test_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image.")
        return

    h, w = img.shape[:2]
    print(f"\nImage: {image_path} ({w}x{h})")
    print("-" * 40)

    # ── Step 1: Detect regions using YOLOv8 ─────────────────
    detect_results = detect_model(image_path, verbose=False, conf=0.05, classes=[0])
    boxes = detect_results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No person detected by detector.")
        print("Running classifier on whole image...\n")

        # Run classifier on whole image
        classify_results = classify_model(img, verbose=False)
        probs = classify_results[0].probs
        pred_idx = int(probs.top1)
        non_person_prob = float(probs.data[0])
        person_prob = float(probs.data[1])

        class_names = ["non_person", "person"]
        label = class_names[pred_idx]

        print(f"Result:")
        print(f"  Predicted  : {label}")
        print(f"  Person     : {person_prob * 100:.2f}%")
        print(f"  Non-Person : {non_person_prob * 100:.2f}%")

        # Draw result on image
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        text = f"{label} {max(person_prob, non_person_prob) * 100:.1f}%"
        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        output_path = "test_result.jpg"
        cv2.imwrite(output_path, img)
        print(f"\nResult saved to: {output_path}")

        cv2.imshow("Two-Step Detection Result", img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    person_count = 0
    non_person_count = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Crop the detected region
        pad = 5
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = img[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        # ── Step 2: Classify crop as person or non_person ───
        classify_results = classify_model(crop, verbose=False)
        probs = classify_results[0].probs
        pred_idx = int(probs.top1)
        pred_conf = float(probs.top1conf)

        non_person_prob = float(probs.data[0])
        person_prob = float(probs.data[1])

        class_names = ["non_person", "person"]
        label = class_names[pred_idx]

        print(f"Region {i+1}:")
        print(f"  Predicted  : {label}")
        print(f"  Person     : {person_prob * 100:.2f}%")
        print(f"  Non-Person : {non_person_prob * 100:.2f}%")
        print(f"  Detect conf: {conf:.2f}")

        if label == "person":
            color = (0, 255, 0)
            person_count += 1
        else:
            color = (0, 0, 255)
            non_person_count += 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {pred_conf * 100:.1f}%"
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    total = person_count + non_person_count
    person_pct = (person_count / total * 100) if total > 0 else 0
    non_person_pct = (non_person_count / total * 100) if total > 0 else 0

    print("-" * 40)
    print(f"Total regions analysed    : {total}")
    print(f"Total persons detected    : {person_count} ({person_pct:.1f}%)")
    print(f"Total non-persons detected: {non_person_count} ({non_person_pct:.1f}%)")
    print("-" * 40)
    print(f"FINAL RESULT: {'PERSON DETECTED' if person_count > non_person_count else 'NO PERSON DETECTED'}")

    output_path = "test_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"\nResult saved to: {output_path}")

    cv2.imshow("Two-Step Detection Result", img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_path = input("Enter image path: ").strip().strip('"')
        test_image(image_path)
    else:
        test_image(sys.argv[1])