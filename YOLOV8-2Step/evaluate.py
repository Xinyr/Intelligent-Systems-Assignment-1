from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/person_detector/weights/best.pt")

# Evaluate on test set
metrics = model.val(data="data.yaml", split="test")

print("\n===== TEST SET RESULTS =====")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall:    {metrics.box.r.mean():.4f}")