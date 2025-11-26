"""
Flask API Routes for AMR Prediction System
REST endpoints for image analysis and AMR prediction
"""
import logging
from datetime import datetime
from functools import wraps
import time

from flask import Blueprint, request, jsonify, current_app, g
from pydantic import ValidationError

from models.schemas import (
    AMRPredictionRequest,
    AMRPredictionResponse,
    ImagePredictionRequest,
    ImageAnalysisResponse,
    ImageFeatures,
    ErrorResponse,
    AuditLogEntry
)
from services.yolo_inference import get_analyzer
from services.amr_predictor import get_predictor
from config import get_config

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint("api", __name__)


def validate_request(schema_class):
    """Decorator to validate request body against Pydantic schema"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                json_data = request.get_json()
                if json_data is None:
                    return jsonify(ErrorResponse(
                        error="ValidationError",
                        message="Request body must be JSON"
                    ).model_dump()), 400
                
                # Validate and parse
                validated_data = schema_class(**json_data)
                g.validated_data = validated_data
                return f(*args, **kwargs)
                
            except ValidationError as e:
                logger.warning(f"Validation error: {e}")
                return jsonify(ErrorResponse(
                    error="ValidationError",
                    message="Request validation failed",
                    details={"errors": e.errors()}
                ).model_dump()), 400
                
            except Exception as e:
                logger.error(f"Request processing error: {e}")
                return jsonify(ErrorResponse(
                    error="RequestError",
                    message=str(e)
                ).model_dump()), 400
        
        return wrapper
    return decorator


def log_audit(event_type: str, action: str):
    """Decorator to log audit events"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Log audit entry
            try:
                config = get_config()
                if config.ENABLE_AUDIT_LOG:
                    patient_id = getattr(g, 'patient_id', 'unknown')
                    prediction_id = getattr(g, 'prediction_id', None)
                    
                    audit_entry = AuditLogEntry(
                        timestamp=datetime.utcnow(),
                        event_type=event_type,
                        patient_id=patient_id,
                        user_id=request.headers.get('X-User-ID'),
                        prediction_id=prediction_id,
                        action=action,
                        outcome="success" if result[1] < 400 else "failure",
                        metadata={
                            "latency_ms": (time.time() - start_time) * 1000,
                            "endpoint": request.endpoint,
                            "method": request.method
                        }
                    )
                    
                    # In production, this would write to audit log storage
                    logger.info(f"AUDIT: {audit_entry.model_dump_json()}")
                    
            except Exception as e:
                logger.error(f"Audit logging failed: {e}")
            
            return result
        
        return wrapper
    return decorator


# ============ Health & Info Endpoints ============

@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "amr-prediction-api",
        "timestamp": datetime.utcnow().isoformat()
    })


@api_bp.route("/info", methods=["GET"])
def info():
    """Service information endpoint"""
    config = get_config()
    predictor = get_predictor()
    
    return jsonify({
        "service": "AMR Prediction API",
        "version": "1.0.0",
        "model_version": predictor.MODEL_VERSION,
        "supported_antibiotics": config.TARGET_ANTIBIOTICS,
        "max_latency_ms": config.MAX_INFERENCE_LATENCY_MS,
        "risk_thresholds": config.RISK_THRESHOLDS
    })


# ============ Image Analysis Endpoints ============

@api_bp.route("/predict_image", methods=["POST"])
@validate_request(ImagePredictionRequest)
@log_audit("image_analysis", "analyze_blood_smear")
def predict_image():
    """
    Analyze blood smear image and extract features.
    
    Request body:
    {
        "image_base64": "base64_encoded_image",
        "patient_id": "optional_patient_id"
    }
    
    Response:
    {
        "image_features": {...},
        "detections": [...],
        "processing_time_ms": 123.45,
        "quality_score": 0.95,
        "warnings": []
    }
    """
    try:
        req: ImagePredictionRequest = g.validated_data
        g.patient_id = req.patient_id or "anonymous"
        
        config = get_config()
        analyzer = get_analyzer(
            model_path=config.YOLO_MODEL_PATH,
            confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD,
            iou_threshold=config.YOLO_IOU_THRESHOLD
        )
        
        # Analyze image
        result = analyzer.analyze(req.image_base64)
        
        # Convert to response schema
        response = ImageAnalysisResponse(
            image_features=ImageFeatures(**result["image_features"]),
            detections=result["detections"],
            processing_time_ms=result["processing_time_ms"],
            quality_score=result["quality_score"],
            warnings=result["warnings"]
        )
        
        return jsonify(response.model_dump()), 200
        
    except ValueError as e:
        return jsonify(ErrorResponse(
            error="ImageProcessingError",
            message=str(e)
        ).model_dump()), 400
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return jsonify(ErrorResponse(
            error="InternalError",
            message="Image analysis failed"
        ).model_dump()), 500


# ============ AMR Prediction Endpoints ============

@api_bp.route("/predict_amr", methods=["POST"])
@validate_request(AMRPredictionRequest)
@log_audit("amr_prediction", "predict_resistance")
def predict_amr():
    """
    Predict antimicrobial resistance probabilities.
    
    Request body: AMRPredictionRequest schema
    
    Response: AMRPredictionResponse schema with:
    - Per-antibiotic probability predictions
    - Risk levels
    - SHAP-based feature explanations
    - Clinical summary
    """
    try:
        req: AMRPredictionRequest = g.validated_data
        g.patient_id = req.patient_id
        
        config = get_config()
        predictor = get_predictor(
            model_path=config.AMR_MODEL_PATH,
            calibrator_path=config.AMR_CALIBRATOR_PATH,
            config=config
        )
        
        # Generate prediction
        response = predictor.predict(req, include_explanations=True)
        
        g.prediction_id = response.prediction_id
        
        return jsonify(response.model_dump(mode='json')), 200
        
    except Exception as e:
        logger.error(f"AMR prediction failed: {e}")
        return jsonify(ErrorResponse(
            error="PredictionError",
            message="AMR prediction failed",
            details={"error": str(e)}
        ).model_dump()), 500


@api_bp.route("/predict_amr/batch", methods=["POST"])
@log_audit("amr_prediction", "batch_predict_resistance")
def predict_amr_batch():
    """
    Batch prediction for multiple patients.
    
    Request body:
    {
        "predictions": [AMRPredictionRequest, ...]
    }
    
    Response:
    {
        "results": [AMRPredictionResponse, ...],
        "total_processed": N,
        "total_latency_ms": 123.45
    }
    """
    try:
        json_data = request.get_json()
        if not json_data or "predictions" not in json_data:
            return jsonify(ErrorResponse(
                error="ValidationError",
                message="Request must contain 'predictions' array"
            ).model_dump()), 400
        
        start_time = time.time()
        
        # Validate each request
        requests = []
        for i, pred_data in enumerate(json_data["predictions"]):
            try:
                req = AMRPredictionRequest(**pred_data)
                requests.append(req)
            except ValidationError as e:
                return jsonify(ErrorResponse(
                    error="ValidationError",
                    message=f"Validation failed for prediction {i}",
                    details={"errors": e.errors()}
                ).model_dump()), 400
        
        config = get_config()
        predictor = get_predictor(config=config)
        
        # Generate predictions
        responses = predictor.predict_batch(requests, include_explanations=True)
        
        total_latency = (time.time() - start_time) * 1000
        
        return jsonify({
            "results": [r.model_dump(mode='json') for r in responses],
            "total_processed": len(responses),
            "total_latency_ms": total_latency
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify(ErrorResponse(
            error="BatchPredictionError",
            message="Batch prediction failed"
        ).model_dump()), 500


# ============ Explanation Endpoints ============

@api_bp.route("/explain/<prediction_id>", methods=["GET"])
def get_explanation(prediction_id: str):
    """
    Get detailed explanation for a previous prediction.
    
    This endpoint retrieves stored explanation data for a prediction.
    In production, this would query a database.
    """
    # In production, retrieve from database
    # For now, return mock data
    return jsonify({
        "prediction_id": prediction_id,
        "explanation_type": "shap",
        "message": "Detailed explanation retrieval not implemented in demo"
    }), 200


@api_bp.route("/thresholds", methods=["GET"])
def get_thresholds():
    """
    Get decision thresholds for clinical use.
    Shows sensitivity/specificity trade-offs.
    """
    config = get_config()
    
    # Mock threshold analysis data
    # In production, this comes from model validation
    threshold_analysis = {
        "antibiotics": {}
    }
    
    for ab in config.TARGET_ANTIBIOTICS:
        threshold_analysis["antibiotics"][ab] = {
            "thresholds": [
                {"threshold": 0.3, "sensitivity": 0.95, "specificity": 0.40, "ppv": 0.35, "npv": 0.96},
                {"threshold": 0.5, "sensitivity": 0.85, "specificity": 0.65, "ppv": 0.52, "npv": 0.91},
                {"threshold": 0.7, "sensitivity": 0.70, "specificity": 0.82, "ppv": 0.68, "npv": 0.83},
                {"threshold": 0.8, "sensitivity": 0.55, "specificity": 0.90, "ppv": 0.75, "npv": 0.78},
            ],
            "recommended_threshold": 0.5,
            "auc_roc": 0.82
        }
    
    return jsonify(threshold_analysis), 200


# ============ Model Info Endpoints ============

@api_bp.route("/model/features", methods=["GET"])
def get_model_features():
    """Get list of features used by the model"""
    config = get_config()
    
    return jsonify({
        "image_features": config.IMAGE_FEATURE_NAMES,
        "vital_features": config.VITAL_FEATURE_NAMES,
        "lab_features": config.LAB_FEATURE_NAMES,
        "history_features": config.HISTORY_FEATURE_NAMES,
        "demographic_features": config.DEMOGRAPHIC_FEATURE_NAMES,
        "context_features": config.CONTEXT_FEATURE_NAMES
    }), 200


@api_bp.route("/model/performance", methods=["GET"])
def get_model_performance():
    """
    Get model performance metrics.
    In production, this would return actual validation metrics.
    """
    config = get_config()
    
    # Mock performance data
    performance = {
        "model_version": "1.0.0",
        "validation_date": "2025-11-01",
        "metrics_by_antibiotic": {}
    }
    
    for ab in config.TARGET_ANTIBIOTICS:
        performance["metrics_by_antibiotic"][ab] = {
            "auc_roc": round(0.75 + 0.1 * hash(ab) % 10 / 10, 3),
            "auc_pr": round(0.65 + 0.1 * hash(ab) % 10 / 10, 3),
            "brier_score": round(0.15 + 0.05 * hash(ab) % 10 / 10, 3),
            "f1_score": round(0.70 + 0.1 * hash(ab) % 10 / 10, 3),
            "precision_at_80_recall": round(0.55 + 0.1 * hash(ab) % 10 / 10, 3),
            "n_validation_samples": 500 + hash(ab) % 200
        }
    
    return jsonify(performance), 200


# ============ Clinical Override Endpoints ============

@api_bp.route("/override", methods=["POST"])
@log_audit("clinical_override", "override_prediction")
def clinical_override():
    """
    Record a clinical override of model prediction.
    For audit and continuous learning.
    """
    try:
        json_data = request.get_json()
        
        required_fields = ["prediction_id", "override_decision", "reason"]
        for field in required_fields:
            if field not in json_data:
                return jsonify(ErrorResponse(
                    error="ValidationError",
                    message=f"Missing required field: {field}"
                ).model_dump()), 400
        
        # In production, store override in database
        override_record = {
            "prediction_id": json_data["prediction_id"],
            "override_decision": json_data["override_decision"],
            "reason": json_data["reason"],
            "clinician_id": request.headers.get("X-User-ID", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Clinical override recorded: {override_record}")
        
        return jsonify({
            "status": "recorded",
            "override_id": f"OVR_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "message": "Override recorded for audit trail"
        }), 200
        
    except Exception as e:
        logger.error(f"Override recording failed: {e}")
        return jsonify(ErrorResponse(
            error="OverrideError",
            message="Failed to record override"
        ).model_dump()), 500


# ============ Error Handlers ============

@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify(ErrorResponse(
        error="BadRequest",
        message=str(error)
    ).model_dump()), 400


@api_bp.errorhandler(404)
def not_found(error):
    return jsonify(ErrorResponse(
        error="NotFound",
        message="Endpoint not found"
    ).model_dump()), 404


@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify(ErrorResponse(
        error="InternalError",
        message="An internal error occurred"
    ).model_dump()), 500
