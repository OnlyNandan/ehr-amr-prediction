"""
Pydantic models for data validation and serialization
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class SexEnum(str, Enum):
    """Biological sex enumeration"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class SampleSourceEnum(str, Enum):
    """Sample source enumeration"""
    BLOOD = "blood"
    URINE = "urine"
    SPUTUM = "sputum"
    WOUND = "wound"
    CSF = "csf"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============ Image Features ============

class ImageFeatures(BaseModel):
    """Features extracted from blood smear YOLO analysis"""
    neutrophil_count: float = Field(ge=0, description="Neutrophil count per HPF")
    lymphocyte_count: float = Field(ge=0, description="Lymphocyte count per HPF")
    monocyte_count: float = Field(ge=0, default=0, description="Monocyte count per HPF")
    eosinophil_count: float = Field(ge=0, default=0, description="Eosinophil count per HPF")
    basophil_count: float = Field(ge=0, default=0, description="Basophil count per HPF")
    nlr: Optional[float] = Field(ge=0, default=None, description="Neutrophil-to-lymphocyte ratio")
    platelet_estimate: float = Field(ge=0, description="Estimated platelet count")
    rbc_count: Optional[float] = Field(ge=0, default=None, description="RBC count per HPF")
    parasite_present: bool = Field(default=False, description="Parasite presence flag")
    bacterial_cluster_count: int = Field(ge=0, default=0, description="Number of bacterial clusters detected")
    mean_bacterial_bbox_confidence: float = Field(ge=0, le=1, default=0, description="Mean confidence of bacterial detections")
    rbc_morphology_anisocytosis: bool = Field(default=False, description="Anisocytosis flag")
    rbc_morphology_poikilocytosis: bool = Field(default=False, description="Poikilocytosis flag")
    rbc_morphology_hypochromia: bool = Field(default=False, description="Hypochromia flag")
    
    @validator('nlr', pre=True, always=True)
    def compute_nlr(cls, v, values):
        if v is not None:
            return v
        lymph = values.get('lymphocyte_count', 0)
        neut = values.get('neutrophil_count', 0)
        if lymph > 0:
            return neut / lymph
        return None

    class Config:
        json_schema_extra = {
            "example": {
                "neutrophil_count": 12.5,
                "lymphocyte_count": 3.2,
                "monocyte_count": 0.8,
                "eosinophil_count": 0.1,
                "basophil_count": 0.05,
                "platelet_estimate": 180000,
                "parasite_present": False,
                "bacterial_cluster_count": 2,
                "mean_bacterial_bbox_confidence": 0.85
            }
        }


# ============ Vitals ============

class Vitals(BaseModel):
    """Patient vital signs"""
    temperature: float = Field(ge=30, le=45, description="Temperature in Celsius")
    heart_rate: int = Field(ge=20, le=300, description="Heart rate BPM")
    respiratory_rate: int = Field(ge=5, le=60, description="Respiratory rate per minute")
    bp_systolic: int = Field(ge=50, le=300, description="Systolic blood pressure mmHg")
    bp_diastolic: int = Field(ge=20, le=200, description="Diastolic blood pressure mmHg")
    oxygen_saturation: float = Field(ge=50, le=100, description="SpO2 percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 38.5,
                "heart_rate": 95,
                "respiratory_rate": 20,
                "bp_systolic": 125,
                "bp_diastolic": 82,
                "oxygen_saturation": 96.0
            }
        }


# ============ Labs ============

class LabResults(BaseModel):
    """Laboratory test results"""
    wbc: float = Field(ge=0, description="White blood cell count (×10³/µL)")
    crp: Optional[float] = Field(ge=0, default=None, description="C-reactive protein (mg/L)")
    lactate: Optional[float] = Field(ge=0, default=None, description="Lactate (mmol/L)")
    creatinine: Optional[float] = Field(ge=0, default=None, description="Creatinine (mg/dL)")
    ast: Optional[float] = Field(ge=0, default=None, description="AST (U/L)")
    alt: Optional[float] = Field(ge=0, default=None, description="ALT (U/L)")
    hemoglobin: Optional[float] = Field(ge=0, default=None, description="Hemoglobin (g/dL)")
    platelets: Optional[float] = Field(ge=0, default=None, description="Platelet count (×10³/µL)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "wbc": 14.2,
                "crp": 85.0,
                "lactate": 2.1,
                "creatinine": 1.2,
                "ast": 45,
                "alt": 38,
                "hemoglobin": 12.5,
                "platelets": 220
            }
        }


# ============ Patient History ============

class AntibioticExposure(BaseModel):
    """Single antibiotic exposure record"""
    antibiotic_name: str = Field(description="Name of antibiotic")
    days_ago: int = Field(ge=0, description="Days since exposure")
    duration_days: int = Field(ge=1, default=1, description="Duration of treatment")
    route: Optional[str] = Field(default="oral", description="Administration route")


class PatientHistory(BaseModel):
    """Patient medical history relevant to AMR prediction"""
    prior_antibiotics: List[AntibioticExposure] = Field(default_factory=list, description="Antibiotic exposures in last 90 days")
    days_since_last_antibiotic: Optional[int] = Field(ge=0, default=None, description="Days since last antibiotic")
    prior_hospitalizations_30d: int = Field(ge=0, default=0, description="Number of hospitalizations in last 30 days")
    antibiotic_exposure_count_90d: int = Field(ge=0, default=0, description="Number of antibiotic courses in last 90 days")
    prior_amr_positive: bool = Field(default=False, description="Previous AMR positive culture")
    comorbidities: List[str] = Field(default_factory=list, description="List of relevant comorbidities")
    immunocompromised: bool = Field(default=False, description="Immunocompromised status")
    chronic_kidney_disease: bool = Field(default=False, description="CKD status")
    diabetes: bool = Field(default=False, description="Diabetes status")
    
    @validator('antibiotic_exposure_count_90d', pre=True, always=True)
    def compute_exposure_count(cls, v, values):
        if v is not None and v > 0:
            return v
        return len(values.get('prior_antibiotics', []))

    class Config:
        json_schema_extra = {
            "example": {
                "prior_antibiotics": [
                    {"antibiotic_name": "amoxicillin", "days_ago": 15, "duration_days": 7}
                ],
                "days_since_last_antibiotic": 8,
                "prior_hospitalizations_30d": 1,
                "prior_amr_positive": False,
                "comorbidities": ["hypertension", "diabetes"]
            }
        }


# ============ Demographics ============

class Demographics(BaseModel):
    """Patient demographics"""
    age: int = Field(ge=0, le=150, description="Patient age in years")
    sex: SexEnum = Field(description="Biological sex")
    weight: Optional[float] = Field(ge=0, le=500, default=None, description="Weight in kg")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "sex": "male",
                "weight": 78.5
            }
        }


# ============ Context ============

class ClinicalContext(BaseModel):
    """Clinical context information"""
    is_icu: bool = Field(default=False, description="Whether patient is in ICU")
    hospital_id: Optional[str] = Field(default=None, description="Hospital identifier")
    ward: Optional[str] = Field(default=None, description="Ward/unit name")
    sample_source: SampleSourceEnum = Field(default=SampleSourceEnum.BLOOD, description="Sample source")
    admission_date: Optional[datetime] = Field(default=None, description="Admission date")
    days_since_admission: Optional[int] = Field(ge=0, default=None, description="Days since admission")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_icu": True,
                "hospital_id": "HOSP001",
                "ward": "ICU-A",
                "sample_source": "blood",
                "days_since_admission": 3
            }
        }


# ============ Request Models ============

class ImagePredictionRequest(BaseModel):
    """Request model for image-based prediction"""
    image_base64: str = Field(description="Base64 encoded blood smear image")
    patient_id: Optional[str] = Field(default=None, description="Anonymized patient ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "base64_encoded_image_data...",
                "patient_id": "PT_HASH_12345"
            }
        }


class AMRPredictionRequest(BaseModel):
    """Complete request model for AMR prediction"""
    patient_id: str = Field(description="Anonymized patient ID")
    sample_time: datetime = Field(description="Sample collection time")
    target_antibiotics: Optional[List[str]] = Field(default=None, description="Specific antibiotics to predict (if None, predict all)")
    
    image_features: ImageFeatures = Field(description="Features from blood smear analysis")
    vitals: Vitals = Field(description="Current vital signs")
    labs: LabResults = Field(description="Laboratory results")
    history: PatientHistory = Field(description="Patient history")
    demographics: Demographics = Field(description="Patient demographics")
    context: ClinicalContext = Field(description="Clinical context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PT_HASH_12345",
                "sample_time": "2025-11-26T10:30:00Z",
                "target_antibiotics": ["ceftriaxone", "ciprofloxacin"]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    predictions: List[AMRPredictionRequest] = Field(description="List of prediction requests")


# ============ Response Models ============

class FeatureContribution(BaseModel):
    """Individual feature's contribution to prediction"""
    feature_name: str = Field(description="Name of the feature")
    feature_value: Any = Field(description="Value of the feature")
    shap_value: float = Field(description="SHAP contribution value")
    category: str = Field(description="Feature category (image/vitals/labs/history/demographics/context)")


class AntibioticPrediction(BaseModel):
    """Prediction result for a single antibiotic"""
    antibiotic: str = Field(description="Antibiotic name")
    probability: float = Field(ge=0, le=1, description="Probability of resistance")
    risk_level: RiskLevel = Field(description="Categorical risk level")
    confidence_interval: Optional[List[float]] = Field(default=None, description="95% CI [lower, upper]")
    calibrated: bool = Field(default=True, description="Whether probability is calibrated")
    top_contributing_features: List[FeatureContribution] = Field(description="Top features driving prediction")


class AMRPredictionResponse(BaseModel):
    """Complete response for AMR prediction"""
    patient_id: str = Field(description="Patient ID")
    prediction_id: str = Field(description="Unique prediction identifier for audit")
    prediction_time: datetime = Field(description="Time of prediction")
    model_version: str = Field(description="Model version used")
    
    predictions: List[AntibioticPrediction] = Field(description="Predictions per antibiotic")
    
    overall_risk_level: RiskLevel = Field(description="Overall risk assessment")
    clinical_summary: str = Field(description="Human-readable clinical summary")
    
    # Metadata
    inference_latency_ms: float = Field(description="Total inference time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or caveats")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PT_HASH_12345",
                "prediction_id": "PRED_20251126_001",
                "prediction_time": "2025-11-26T10:30:15Z",
                "model_version": "1.2.0",
                "predictions": [
                    {
                        "antibiotic": "ceftriaxone",
                        "probability": 0.73,
                        "risk_level": "high",
                        "calibrated": True,
                        "top_contributing_features": [
                            {"feature_name": "bacterial_cluster_count", "feature_value": 5, "shap_value": 0.15, "category": "image"},
                            {"feature_name": "crp", "feature_value": 120, "shap_value": 0.12, "category": "labs"}
                        ]
                    }
                ],
                "overall_risk_level": "high",
                "clinical_summary": "High probability of ceftriaxone resistance detected. Key contributors: bacterial clusters in smear, elevated CRP.",
                "inference_latency_ms": 127.5
            }
        }


class ImageAnalysisResponse(BaseModel):
    """Response from image analysis endpoint"""
    image_features: ImageFeatures = Field(description="Extracted features")
    detections: List[Dict[str, Any]] = Field(description="Raw YOLO detections")
    processing_time_ms: float = Field(description="Image processing time")
    quality_score: float = Field(ge=0, le=1, description="Image quality assessment")
    warnings: List[str] = Field(default_factory=list, description="Image quality warnings")


# ============ Error Models ============

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request ID for debugging")


# ============ Audit Models ============

class AuditLogEntry(BaseModel):
    """Audit log entry for compliance"""
    timestamp: datetime = Field(description="Event timestamp")
    event_type: str = Field(description="Type of event")
    patient_id: str = Field(description="Patient ID (hashed)")
    user_id: Optional[str] = Field(default=None, description="User who triggered event")
    prediction_id: Optional[str] = Field(default=None, description="Associated prediction ID")
    action: str = Field(description="Action performed")
    outcome: str = Field(description="Outcome of action")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
