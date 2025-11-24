from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, List
import cv2
import numpy as np
from io import BytesIO
import base64
from ultralytics import YOLO
import os

router = APIRouter(
    prefix="/api/image-analysis",
    tags=["Image Analysis"]
)

# Load YOLO model (do this once at module load time)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bccd_yolov8_best.pt")
model = YOLO(MODEL_PATH)

# Class mapping (BCCD dataset actual classes)
CLASS_NAMES = {
    0: "RBC",
    1: "WBC", 
    2: "Platelets"
}

def detect_blood_cells(image_array: np.ndarray) -> Dict:
    """
    Real YOLO detection using BCCD-trained model
    """
    # Run inference with lower confidence threshold for better detection
    results = model(image_array, conf=0.15, iou=0.45)
    
    # Parse results
    detections = []
    counts = {"WBC": 0, "RBC": 0, "Platelets": 0}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Map class ID to name
            cell_type = CLASS_NAMES.get(cls, "Unknown")
            
            # Count cells
            if cell_type in counts:
                counts[cell_type] += 1
            
            # Store detection
            detections.append({
                "type": cell_type,
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "confidence": conf
            })
    
    return {
        "detections": detections,
        "counts": counts
    }

def draw_detections(image_array: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on the image"""
    annotated_image = image_array.copy()
    
    colors = {
        "WBC": (255, 0, 0),        # Blue
        "RBC": (0, 0, 255),        # Red
        "Platelets": (0, 255, 0)   # Green
    }
    
    for detection in detections:
        cell_type = detection["type"]
        bbox = detection["bbox"]
        color = colors.get(cell_type, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Add label
        label = f"{cell_type} {detection['confidence']:.2f}"
        cv2.putText(annotated_image, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return annotated_image

@router.post("/analyze-blood-smear")
async def analyze_blood_smear(file: UploadFile = File(...)) -> Dict:
    """
    Analyze a blood smear image and return cell counts + annotated image
    """
    try:
        # Read image from upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect cells
        results = detect_blood_cells(image)
        
        # Draw detections
        annotated_image = draw_detections(image, results["detections"])
        
        # Convert annotated image to base64 for frontend display
        _, buffer = cv2.imencode('.png', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "counts": results["counts"],
            "annotated_image": f"data:image/png;base64,{annotated_base64}",
            "total_detections": len(results["detections"])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "image-analysis"}
