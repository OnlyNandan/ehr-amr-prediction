from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, List
import cv2
import numpy as np
from io import BytesIO
import base64

router = APIRouter(
    prefix="/api/image-analysis",
    tags=["Image Analysis"]
)

# Mock cell detection for now - in production, you'd use a trained YOLO model
def detect_blood_cells(image_array: np.ndarray) -> Dict:
    """
    Mock detection function. In production, replace with:
    from ultralytics import YOLO
    model = YOLO('path/to/trained/model.pt')
    results = model(image_array)
    """
    # Simulate detection results
    height, width = image_array.shape[:2]
    
    # Mock detections with random positions
    np.random.seed(42)  # For reproducibility
    
    # Simulate WBC, RBC, and Platelet counts
    wbc_count = np.random.randint(15, 25)
    rbc_count = np.random.randint(80, 120)
    platelet_count = np.random.randint(40, 60)
    
    # Generate mock bounding boxes
    detections = []
    
    # Add WBC detections (larger, less numerous)
    for i in range(wbc_count):
        x = np.random.randint(0, width - 30)
        y = np.random.randint(0, height - 30)
        detections.append({
            "type": "WBC",
            "bbox": [int(x), int(y), int(x + 25), int(y + 25)],
            "confidence": float(np.random.uniform(0.85, 0.98))
        })
    
    # Add RBC detections (smaller, more numerous)
    for i in range(rbc_count):
        x = np.random.randint(0, width - 15)
        y = np.random.randint(0, height - 15)
        detections.append({
            "type": "RBC",
            "bbox": [int(x), int(y), int(x + 12), int(y + 12)],
            "confidence": float(np.random.uniform(0.80, 0.95))
        })
    
    # Add Platelet detections (smallest)
    for i in range(platelet_count):
        x = np.random.randint(0, width - 8)
        y = np.random.randint(0, height - 8)
        detections.append({
            "type": "Platelet",
            "bbox": [int(x), int(y), int(x + 6), int(y + 6)],
            "confidence": float(np.random.uniform(0.75, 0.92))
        })
    
    return {
        "detections": detections,
        "counts": {
            "WBC": wbc_count,
            "RBC": rbc_count,
            "Platelet": platelet_count
        }
    }

def draw_detections(image_array: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on the image"""
    annotated_image = image_array.copy()
    
    colors = {
        "WBC": (255, 0, 0),      # Blue
        "RBC": (0, 0, 255),      # Red
        "Platelet": (0, 255, 0)  # Green
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
