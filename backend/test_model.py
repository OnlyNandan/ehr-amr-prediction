from ultralytics import YOLO
import os

# Load model and print class names
MODEL_PATH = "models/bccd_yolov8_best.pt"
model = YOLO(MODEL_PATH)

print("Model class names:")
print(model.names)
print("\nModel info:")
print(f"Number of classes: {len(model.names)}")
for idx, name in model.names.items():
    print(f"  Class {idx}: {name}")
