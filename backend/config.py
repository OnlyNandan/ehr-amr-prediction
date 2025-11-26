"""
Configuration settings for AMR Prediction System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models" / "trained"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = False
    TESTING = False
    
    # YOLO Model Settings
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(MODEL_DIR / "yolo_blood_smear.pt"))
    YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
    YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    
    # AMR Model Settings
    AMR_MODEL_PATH = os.getenv("AMR_MODEL_PATH", str(MODEL_DIR / "amr_model.joblib"))
    AMR_CALIBRATOR_PATH = os.getenv("AMR_CALIBRATOR_PATH", str(MODEL_DIR / "amr_calibrator.joblib"))
    
    # Target antibiotics for prediction
    TARGET_ANTIBIOTICS = [
        "ceftriaxone",
        "ciprofloxacin",
        "meropenem",
        "vancomycin",
        "piperacillin_tazobactam",
        "gentamicin",
        "ampicillin",
        "trimethoprim_sulfamethoxazole"
    ]
    
    # Feature configuration
    IMAGE_FEATURE_NAMES = [
        "neutrophil_count",
        "lymphocyte_count",
        "monocyte_count",
        "eosinophil_count",
        "basophil_count",
        "nlr",  # neutrophil-to-lymphocyte ratio
        "platelet_estimate",
        "rbc_count",
        "parasite_present",
        "bacterial_cluster_count",
        "mean_bacterial_bbox_confidence",
        "rbc_morphology_anisocytosis",
        "rbc_morphology_poikilocytosis",
        "rbc_morphology_hypochromia"
    ]
    
    VITAL_FEATURE_NAMES = [
        "temperature",
        "heart_rate",
        "respiratory_rate",
        "bp_systolic",
        "bp_diastolic",
        "oxygen_saturation"
    ]
    
    LAB_FEATURE_NAMES = [
        "wbc",
        "crp",
        "lactate",
        "creatinine",
        "ast",
        "alt",
        "hemoglobin",
        "platelets"
    ]
    
    HISTORY_FEATURE_NAMES = [
        "days_since_last_antibiotic",
        "prior_hospitalizations_30d",
        "antibiotic_exposure_count_90d",
        "prior_amr_positive"
    ]
    
    DEMOGRAPHIC_FEATURE_NAMES = [
        "age",
        "sex",
        "weight"
    ]
    
    CONTEXT_FEATURE_NAMES = [
        "is_icu",
        "hospital_id",
        "sample_source"
    ]
    
    # Latency requirement (milliseconds)
    MAX_INFERENCE_LATENCY_MS = 500
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8
    }
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///amr_predictions.db")
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Privacy/Compliance
    ENABLE_AUDIT_LOG = True
    PHI_FIELDS = ["patient_id", "name", "dob", "mrn"]


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    
    # More strict in production
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = "sqlite:///:memory:"


# Configuration mapping
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)()
