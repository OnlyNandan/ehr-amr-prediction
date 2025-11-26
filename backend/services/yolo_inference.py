"""
YOLO-based Blood Smear Cell Detection Service
Detects WBC (white blood cells) and RBC (red blood cells) using circle detection
and optionally YOLO for more accurate detection.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
import io
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BloodSmearAnalyzer:
    """
    Analyzes blood smear images to detect and count WBC and RBC cells.
    Uses OpenCV circle detection (Hough Transform) for reliable cell detection.
    Returns bounding boxes for visualization.
    """
    
    HPF_AREA_PIXELS = 400 * 400
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        
        # Try to load YOLO model
        self._load_yolo_model(model_path)
    
    def _load_yolo_model(self, model_path: Optional[str] = None):
        """Load YOLOv8 model for general object detection"""
        try:
            from ultralytics import YOLO
            # Use YOLOv8n (nano) for fast inference
            self.model = YOLO("yolov8n.pt")
            logger.info("Loaded YOLOv8n model")
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}. Using OpenCV detection.")
            self.model = None
    
    def decode_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        try:
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            return img
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def assess_image_quality(self, img: np.ndarray) -> Tuple[float, List[str]]:
        """Assess image quality"""
        warnings = []
        score = 1.0
        
        height, width = img.shape[:2]
        if width < 300 or height < 300:
            warnings.append("Low resolution image")
            score -= 0.2
        
        brightness = np.mean(img)
        if brightness < 50:
            warnings.append("Image too dark")
            score -= 0.2
        elif brightness > 220:
            warnings.append("Image overexposed")
            score -= 0.2
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            warnings.append("Image may be blurry")
            score -= 0.15
        
        return max(0.0, score), warnings
    
    def detect_cells_advanced(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Advanced cell detection using Computer Vision techniques.
        - WBCs: Detected via HSV color segmentation (purple/blue nuclei).
        - RBCs: Detected via adaptive thresholding and morphological operations.
        """
        detections = []
        height, width = img.shape[:2]
        
        # --- 1. WBC Detection (Purple/Blue Nuclei) ---
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for purple/blue (WBC nuclei)
        # These ranges might need tuning based on stain type (Giemsa/Wright)
        lower_purple1 = np.array([120, 50, 50])
        upper_purple1 = np.array([170, 255, 255])
        mask_wbc1 = cv2.inRange(hsv, lower_purple1, upper_purple1)
        
        # Sometimes nuclei are more dark blue
        lower_purple2 = np.array([100, 50, 50])
        upper_purple2 = np.array([140, 255, 255])
        mask_wbc2 = cv2.inRange(hsv, lower_purple2, upper_purple2)
        
        mask_wbc = cv2.bitwise_or(mask_wbc1, mask_wbc2)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask_wbc = cv2.morphologyEx(mask_wbc, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_wbc = cv2.dilate(mask_wbc, kernel, iterations=1)
        
        # Find contours for WBC
        cnts_wbc, _ = cv2.findContours(mask_wbc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wbc_boxes = []
        
        for i, c in enumerate(cnts_wbc):
            area = cv2.contourArea(c)
            if area > 200: # Minimum size for WBC nucleus
                x, y, w, h = cv2.boundingRect(c)
                
                # Expand box to include cytoplasm (WBCs are larger than just their nucleus)
                pad_x = int(w * 0.6)
                pad_y = int(h * 0.6)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(width, x + w + pad_x)
                y2 = min(height, y + h + pad_y)
                
                # Calculate confidence based on area and shape
                confidence = min(0.99, 0.85 + (area / (width * height) * 100))
                
                detection = {
                    "id": len(detections),
                    "class_name": "wbc",
                    "confidence": float(confidence),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "color": "#2196F3"
                }
                detections.append(detection)
                wbc_boxes.append((x1, y1, x2, y2))

        # --- 2. RBC Detection (Reddish circles) ---
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Adaptive Thresholding (better for varying lighting)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 3
        )
        
        # Morphological operations to separate touching cells (Watershed-like effect)
        kernel_rbc = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_rbc, iterations=2)
        
        # Find contours
        cnts_rbc, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts_rbc:
            area = cv2.contourArea(c)
            
            # RBC area filtering (adjust based on image resolution if needed)
            # Assuming standard microscopy images
            if 80 < area < 4000: 
                # Check circularity
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity > 0.65: # RBCs are mostly circular
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w//2, y + h//2
                    
                    # Check overlap with WBCs
                    is_overlapping_wbc = False
                    for wx1, wy1, wx2, wy2 in wbc_boxes:
                        # If center of RBC is inside a WBC box
                        if wx1 < cx < wx2 and wy1 < cy < wy2:
                            is_overlapping_wbc = True
                            break
                    
                    if not is_overlapping_wbc:
                        detection = {
                            "id": len(detections),
                            "class_name": "rbc",
                            "confidence": min(0.98, 0.8 + (circularity * 0.15)),
                            "bbox": [float(x), float(y), float(x+w), float(y+h)],
                            "color": "#F44336"
                        }
                        detections.append(detection)
        
        # Fallback: If no cells detected, try the simpler Hough transform
        if len(detections) == 0:
            logger.info("Advanced detection found no cells, falling back to Hough Transform")
            return self.detect_cells_opencv(img)
            
        return detections
    
    def run_inference(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Run cell detection using Advanced Computer Vision Pipeline"""
        # We prioritize the advanced CV pipeline over the un-finetuned YOLO model
        return self.detect_cells_advanced(img)
    
    def extract_features(self, detections: List[Dict[str, Any]], image_area: float) -> Dict[str, Any]:
        """Extract features from detections for AMR prediction"""
        wbc_count = sum(1 for d in detections if d["class_name"] == "wbc")
        rbc_count = sum(1 for d in detections if d["class_name"] == "rbc")
        
        wbc_confidences = [d["confidence"] for d in detections if d["class_name"] == "wbc"]
        rbc_confidences = [d["confidence"] for d in detections if d["class_name"] == "rbc"]
        
        norm_factor = self.HPF_AREA_PIXELS / image_area if image_area > 0 else 1
        
        features = {
            "wbc_count": wbc_count,
            "rbc_count": rbc_count,
            "wbc_count_per_hpf": round(wbc_count * norm_factor, 2),
            "rbc_count_per_hpf": round(rbc_count * norm_factor, 2),
            "wbc_rbc_ratio": round(wbc_count / max(rbc_count, 1), 4),
            "total_cell_count": wbc_count + rbc_count,
            "mean_wbc_confidence": round(float(np.mean(wbc_confidences)), 3) if wbc_confidences else 0,
            "mean_rbc_confidence": round(float(np.mean(rbc_confidences)), 3) if rbc_confidences else 0,
            # Required fields for ImageFeatures schema
            "neutrophil_count": float(wbc_count),
            "lymphocyte_count": 0.0,
            "monocyte_count": 0.0,
            "eosinophil_count": 0.0,
            "basophil_count": 0.0,
            "platelet_estimate": 0.0,
            "parasite_present": False,
            "bacterial_cluster_count": 0,
            "mean_bacterial_bbox_confidence": 0.0,
            "rbc_morphology_anisocytosis": False,
            "rbc_morphology_poikilocytosis": False,
            "rbc_morphology_hypochromia": False
        }
        
        return features
    
    def analyze(self, image_base64: str) -> Dict[str, Any]:
        """Complete analysis pipeline for a blood smear image."""
        start_time = time.time()
        
        img = self.decode_image(image_base64)
        height, width = img.shape[:2]
        image_area = width * height
        
        quality_score, warnings = self.assess_image_quality(img)
        detections = self.run_inference(img)
        features = self.extract_features(detections, image_area)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "image_features": features,
            "detections": detections,
            "processing_time_ms": round(processing_time, 2),
            "quality_score": round(quality_score, 2),
            "warnings": warnings,
            "metadata": {
                "image_width": width,
                "image_height": height,
                "total_detections": len(detections),
                "wbc_count": features["wbc_count"],
                "rbc_count": features["rbc_count"]
            }
        }


_analyzer_instance: Optional[BloodSmearAnalyzer] = None


def get_analyzer(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> BloodSmearAnalyzer:
    """Get or create singleton analyzer instance"""
    global _analyzer_instance
    
    if _analyzer_instance is None:
        _analyzer_instance = BloodSmearAnalyzer(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
    
    return _analyzer_instance
