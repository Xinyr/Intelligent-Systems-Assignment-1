from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=416,
    batch=8,
    name="person_detector",
    patience=10,
    device='cpu'
)

print("Training complete!")
print("Best model saved at: runs/detect/person_detector/weights/best.pt")