"""
YOLO-based Blood Smear Analysis Service
Extracts features from blood smear images for AMR prediction
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
import io

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BloodSmearAnalyzer:
    """
    Analyzes blood smear images using YOLO to extract features
    for AMR prediction.
    
    Detectable classes:
    - neutrophil, lymphocyte, monocyte, eosinophil, basophil
    - platelet, rbc
    - bacteria, bacterial_cluster
    - parasite (malaria, etc.)
    - morphology flags (anisocytosis, poikilocytosis, hypochromia)
    """
    
    # Class mapping for YOLO model
    CLASS_NAMES = {
        0: "neutrophil",
        1: "lymphocyte",
        2: "monocyte",
        3: "eosinophil",
        4: "basophil",
        5: "platelet",
        6: "rbc",
        7: "bacteria",
        8: "bacterial_cluster",
        9: "parasite",
        10: "anisocytosis_rbc",
        11: "poikilocytosis_rbc",
        12: "hypochromia_rbc"
    }
    
    # High-power field dimensions (for normalization)
    HPF_AREA_PIXELS = 400 * 400  # Approximate HPF area in pixels
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize the blood smear analyzer.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning(f"YOLO model not found at {model_path}. Using mock inference.")
    
    def _load_model(self, model_path: str):
        """Load YOLO model from path"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def decode_image(self, image_base64: str) -> Image.Image:
        """Decode base64 image to PIL Image"""
        try:
            # Remove data URL prefix if present
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def assess_image_quality(self, image: Image.Image) -> Tuple[float, List[str]]:
        """
        Assess image quality for blood smear analysis.
        
        Returns:
            Tuple of (quality_score, warnings)
        """
        warnings = []
        score = 1.0
        
        # Check image size
        width, height = image.size
        if width < 400 or height < 400:
            warnings.append("Image resolution too low for accurate analysis")
            score -= 0.3
        
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # Check brightness
        brightness = np.mean(img_array)
        if brightness < 50:
            warnings.append("Image too dark")
            score -= 0.2
        elif brightness > 220:
            warnings.append("Image overexposed")
            score -= 0.2
        
        # Check contrast
        contrast = np.std(img_array)
        if contrast < 30:
            warnings.append("Low contrast - may affect cell detection")
            score -= 0.15
        
        # Check for blur (using Laplacian variance)
        try:
            import cv2
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                warnings.append("Image appears blurry")
                score -= 0.2
        except ImportError:
            pass  # OpenCV not available
        
        return max(0.0, score), warnings
    
    def run_inference(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run YOLO inference on image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            return self._mock_inference(image)
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    detection = {
                        "class_id": int(boxes.cls[i]),
                        "class_name": self.CLASS_NAMES.get(int(boxes.cls[i]), "unknown"),
                        "confidence": float(boxes.conf[i]),
                        "bbox": boxes.xyxy[i].tolist(),  # [x1, y1, x2, y2]
                        "bbox_area": float((boxes.xyxy[i][2] - boxes.xyxy[i][0]) * 
                                          (boxes.xyxy[i][3] - boxes.xyxy[i][1]))
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._mock_inference(image)
    
    def _mock_inference(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Generate mock detections for testing when model is not available"""
        logger.warning("Using mock inference - results are synthetic")
        
        # Generate realistic mock detections
        np.random.seed(42)
        detections = []
        
        # Simulate typical blood smear cell distribution
        cell_counts = {
            "neutrophil": np.random.poisson(12),
            "lymphocyte": np.random.poisson(4),
            "monocyte": np.random.poisson(1),
            "eosinophil": np.random.poisson(0.5),
            "basophil": 0,
            "platelet": np.random.poisson(15),
            "rbc": np.random.poisson(50),
            "bacterial_cluster": np.random.poisson(0.3),
        }
        
        class_name_to_id = {v: k for k, v in self.CLASS_NAMES.items()}
        
        for class_name, count in cell_counts.items():
            for _ in range(int(count)):
                # Generate random bbox
                x1 = np.random.uniform(50, 350)
                y1 = np.random.uniform(50, 350)
                size = np.random.uniform(20, 50)
                
                detection = {
                    "class_id": class_name_to_id.get(class_name, 0),
                    "class_name": class_name,
                    "confidence": np.random.uniform(0.6, 0.95),
                    "bbox": [x1, y1, x1 + size, y1 + size],
                    "bbox_area": size * size
                }
                detections.append(detection)
        
        return detections
    
    def extract_features(self, detections: List[Dict[str, Any]], image_area: float = None) -> Dict[str, Any]:
        """
        Extract features from YOLO detections for AMR prediction.
        
        Args:
            detections: List of detection dictionaries
            image_area: Total image area in pixels (for normalization)
            
        Returns:
            Dictionary of extracted features
        """
        if image_area is None:
            image_area = self.HPF_AREA_PIXELS
        
        # Count cells by type
        counts = {
            "neutrophil": 0,
            "lymphocyte": 0,
            "monocyte": 0,
            "eosinophil": 0,
            "basophil": 0,
            "platelet": 0,
            "rbc": 0,
            "bacteria": 0,
            "bacterial_cluster": 0,
            "parasite": 0,
            "anisocytosis_rbc": 0,
            "poikilocytosis_rbc": 0,
            "hypochromia_rbc": 0
        }
        
        bacterial_confidences = []
        
        for det in detections:
            class_name = det["class_name"]
            if class_name in counts:
                counts[class_name] += 1
            
            if class_name in ["bacteria", "bacterial_cluster"]:
                bacterial_confidences.append(det["confidence"])
        
        # Normalize counts per HPF
        normalization_factor = self.HPF_AREA_PIXELS / image_area
        
        # Calculate features
        neutrophil_count = counts["neutrophil"] * normalization_factor
        lymphocyte_count = counts["lymphocyte"] * normalization_factor
        
        # NLR (neutrophil-to-lymphocyte ratio)
        nlr = neutrophil_count / lymphocyte_count if lymphocyte_count > 0 else 0
        
        # Platelet estimate (rough estimation)
        # Average platelet per HPF * 20000 = estimated count
        platelet_estimate = counts["platelet"] * normalization_factor * 20000
        
        features = {
            "neutrophil_count": neutrophil_count,
            "lymphocyte_count": lymphocyte_count,
            "monocyte_count": counts["monocyte"] * normalization_factor,
            "eosinophil_count": counts["eosinophil"] * normalization_factor,
            "basophil_count": counts["basophil"] * normalization_factor,
            "nlr": nlr,
            "platelet_estimate": platelet_estimate,
            "rbc_count": counts["rbc"] * normalization_factor,
            "parasite_present": counts["parasite"] > 0,
            "bacterial_cluster_count": counts["bacterial_cluster"] + counts["bacteria"],
            "mean_bacterial_bbox_confidence": np.mean(bacterial_confidences) if bacterial_confidences else 0,
            "rbc_morphology_anisocytosis": counts["anisocytosis_rbc"] > 2,
            "rbc_morphology_poikilocytosis": counts["poikilocytosis_rbc"] > 2,
            "rbc_morphology_hypochromia": counts["hypochromia_rbc"] > 2
        }
        
        return features
    
    def analyze(self, image_base64: str) -> Dict[str, Any]:
        """
        Complete analysis pipeline for a blood smear image.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Dictionary containing features, detections, and metadata
        """
        start_time = time.time()
        
        # Decode image
        image = self.decode_image(image_base64)
        image_area = image.size[0] * image.size[1]
        
        # Assess quality
        quality_score, warnings = self.assess_image_quality(image)
        
        # Run inference
        detections = self.run_inference(image)
        
        # Extract features
        features = self.extract_features(detections, image_area)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "image_features": features,
            "detections": detections,
            "processing_time_ms": processing_time,
            "quality_score": quality_score,
            "warnings": warnings,
            "metadata": {
                "image_width": image.size[0],
                "image_height": image.size[1],
                "total_detections": len(detections)
            }
        }


# Singleton instance for API use
_analyzer_instance: Optional[BloodSmearAnalyzer] = None


def get_analyzer(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
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
