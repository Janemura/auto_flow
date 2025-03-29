import torch
import cv2
import numpy as np
from pathlib import Path

# Load the pre-trained YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def detect_cars_yolo(image_path):
    # Read image
    image = cv2.imread(image_path)
    
    # Convert to RGB (YOLO expects RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform detection
    results = model(image_rgb)
    
    # Get detected objects
    detections = results.pandas().xyxy[0]
    
    # Filter only cars/trucks
    vehicle_classes = ["car", "truck", "bus", "motorbike"]
    cars_detected = detections[detections["name"].isin(vehicle_classes)]
    
    # Draw bounding boxes
    for _, row in cars_detected.iterrows():
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save processed image
    processed_path = image_path.replace(".jpg", "_processed.jpg")
    cv2.imwrite(processed_path, image)
    
    return len(cars_detected)
