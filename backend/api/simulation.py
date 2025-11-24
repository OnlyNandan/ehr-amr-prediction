from fastapi import APIRouter
from pydantic import BaseModel
import random
import numpy as np

router = APIRouter()

class SimulationState(BaseModel):
    heart_rate: float
    temperature: float
    systolic_bp: float
    wbc_count: float
    time_step: int

@router.post("/simulate/stream")
async def simulate_stream(state: SimulationState):
    # Random walk simulation
    # Add some drift and noise
    
    # Heart Rate: fluctuate around 80-100, spike occasionally
    hr_noise = np.random.normal(0, 2)
    new_hr = state.heart_rate + hr_noise
    # Tendency to return to mean if not spiking
    if new_hr > 120: new_hr -= 1
    if new_hr < 60: new_hr += 1
    
    # Temperature: slow drift
    temp_noise = np.random.normal(0, 0.05)
    new_temp = state.temperature + temp_noise
    
    # BP: correlate slightly with HR
    bp_noise = np.random.normal(0, 1)
    new_bp = state.systolic_bp + bp_noise
    
    # WBC: very slow drift
    wbc_noise = np.random.normal(0, 50)
    new_wbc = state.wbc_count + wbc_noise
    
    return {
        "heart_rate": round(new_hr, 1),
        "temperature": round(new_temp, 1),
        "systolic_bp": round(new_bp, 0),
        "wbc_count": round(new_wbc, 0),
        "time_step": state.time_step + 1
    }
