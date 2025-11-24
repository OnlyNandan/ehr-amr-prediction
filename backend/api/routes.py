from fastapi import APIRouter, HTTPException
from typing import List, Optional
from models import PatientData, PredictionResult, HospitalStats
from ml_engine import engine

router = APIRouter()

@router.post("/predict", response_model=PredictionResult)
async def predict_resistance(data: PatientData):
    try:
        result = engine.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hospital/stats", response_model=HospitalStats)
async def get_hospital_stats():
    # Mock heatmap data
    return HospitalStats(heatmap_data=[
        {"location": "ICU-A", "resistance_rate": 0.75},
        {"location": "Ward-3", "resistance_rate": 0.20},
        {"location": "ER", "resistance_rate": 0.45}
    ])

from patient_store import store

@router.get("/patients", response_model=List[PatientData])
async def get_patients(query: Optional[str] = None):
    if query:
        return store.search(query)
    return store.get_all()

@router.get("/patients/{patient_id}", response_model=PatientData)
async def get_patient(patient_id: str):
    patient = store.get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.post("/explain")
async def explain_prediction(data: PatientData):
    """
    Generate SHAP-based explanation for a prediction
    """
    try:
        explanation = engine.explain_prediction(data)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

