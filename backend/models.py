from pydantic import BaseModel
from typing import List, Optional, Dict

class PatientData(BaseModel):
    patient_id: str
    name: Optional[str] = "Unknown"
    age: int
    gender: str
    wbc_count: float
    heart_rate: float = 80.0
    temperature: float = 37.0
    systolic_bp: float = 120.0
    prior_antibiotics_days: int
    device_use: bool
    suspected_bacterium: str
    candidate_antibiotic: str
    history: Optional[List[Dict[str, float]]] = []

class RiskFeature(BaseModel):
    name: str
    value: str
    risk_contribution: float
    description: str

class PredictionResult(BaseModel):
    risk_score: float  # 0.0 to 1.0
    risk_level: str    # "Low", "Moderate", "High"
    confidence_interval: List[float] # [lower, upper]
    symbolic_rule: str
    risk_features: List[RiskFeature]
    recommendation: str

class HospitalStats(BaseModel):
    heatmap_data: List[Dict[str, float]] # Mock data for map
