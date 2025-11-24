from fastapi import APIRouter, HTTPException
from typing import Dict
from pydantic import BaseModel
from sepsis_engine import sepsis_engine, SepsisFeatures
from utils.clinical_calculators import ClinicalCalculators, SOFAInput, qSOFAInput

router = APIRouter(
    prefix="/api/sepsis",
    tags=["Sepsis Prediction"]
)

class SepsisPredictionRequest(BaseModel):
    """Request for sepsis risk prediction"""
    # Vitals
    heart_rate: float
    temperature: float
    systolic_bp: float
    respiratory_rate: float = 18.0
    spo2: float = 98.0
    
    # Labs
    wbc_count: float
    lactate: float = 2.0
    creatinine: float = 1.0
    platelets: float = 200.0
    bilirubin: float = 0.8
    
    # Clinical
    age: int
    gcs: int = 15
    vasopressors: bool = False
    prior_sepsis: bool = False
    immunosuppressed: bool = False

class SOFARequest(BaseModel):
    """Request for SOFA score calculation"""
    pao2_fio2: float = None
    platelets: float = None
    bilirubin: float = None
    map_mmhg: float = None
    vasopressors: bool = False
    dopamine_dose: float = 0.0
    gcs: int = None
    creatinine: float = None
    urine_output: float = None

class qSOFARequest(BaseModel):
    """Request for qSOFA score"""
    respiratory_rate: float
    systolic_bp: float
    gcs: int

@router.post("/predict")
async def predict_sepsis(request: SepsisPredictionRequest) -> Dict:
    """
    Predict sepsis risk at 4/12/24/48 hour horizons
    """
    try:
        features = SepsisFeatures(
            heart_rate=request.heart_rate,
            temperature=request.temperature,
            systolic_bp=request.systolic_bp,
            respiratory_rate=request.respiratory_rate,
            spo2=request.spo2,
            wbc_count=request.wbc_count,
            lactate=request.lactate,
            creatinine=request.creatinine,
            platelets=request.platelets,
            bilirubin=request.bilirubin,
            age=request.age,
            gcs=request.gcs,
            vasopressors=request.vasopressors,
            prior_sepsis=request.prior_sepsis,
            immunosuppressed=request.immunosuppressed
        )
        
        prediction = sepsis_engine.predict_sepsis_risk(features)
        return prediction
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/sofa")
async def calculate_sofa(request: SOFARequest) -> Dict:
    """Calculate SOFA score"""
    try:
        sofa_input = SOFAInput(
            pao2_fio2=request.pao2_fio2,
            platelets=request.platelets,
            bilirubin=request.bilirubin,
            map_mmhg=request.map_mmhg,
            vasopressors=request.vasopressors,
            dopamine_dose=request.dopamine_dose,
            gcs=request.gcs,
            creatinine=request.creatinine,
            urine_output=request.urine_output
        )
        
        result = ClinicalCalculators.calculate_sofa_score(sofa_input)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOFA calculation error: {str(e)}")

@router.post("/qsofa")
async def calculate_qsofa(request: qSOFARequest) -> Dict:
    """Calculate quick SOFA score"""
    try:
        qsofa_input = qSOFAInput(
            respiratory_rate=request.respiratory_rate,
            systolic_bp=request.systolic_bp,
            gcs=request.gcs
        )
        
        result = ClinicalCalculators.calculate_qsofa_score(qsofa_input)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qSOFA calculation error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for sepsis prediction service"""
    return {"status": "healthy", "service": "sepsis-prediction"}
