"""
Clinical Calculators for Sepsis Assessment
Implements SOFA and qSOFA scoring systems
"""
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

class OrganSystem(Enum):
    RESPIRATORY = "respiratory"
    COAGULATION = "coagulation"
    LIVER = "liver"
    CARDIOVASCULAR = "cardiovascular"
    CNS = "cns"
    RENAL = "renal"

@dataclass
class SOFAInput:
    """Input data for SOFA score calculation"""
    # Respiratory
    pao2_fio2: Optional[float] = None  # PaO2/FiO2 ratio
    
    # Coagulation
    platelets: Optional[float] = None  # x10^3/μL
    
    # Liver
    bilirubin: Optional[float] = None  # mg/dL
    
    # Cardiovascular
    map_mmhg: Optional[float] = None  # Mean arterial pressure
    vasopressors: bool = False
    dopamine_dose: float = 0.0  # μg/kg/min
    
    # CNS
    gcs: Optional[int] = None  # Glasgow Coma Scale (3-15)
    
    # Renal
    creatinine: Optional[float] = None  # mg/dL
    urine_output: Optional[float] = None  # mL/day

@dataclass
class qSOFAInput:
    """Input for quick SOFA screening"""
    respiratory_rate: float  # breaths/min
    systolic_bp: float  # mmHg
    gcs: int  # Glasgow Coma Scale

class ClinicalCalculators:
    """Sepsis clinical scoring systems"""
    
    @staticmethod
    def calculate_sofa_score(input_data: SOFAInput) -> Dict:
        """
        Calculate Sequential Organ Failure Assessment (SOFA) score
        Range: 0-24 (higher = worse)
        
        Returns: {
            'total_score': int,
            'component_scores': dict,
            'interpretation': str
        }
        """
        scores = {}
        
        # Respiratory (PaO2/FiO2)
        if input_data.pao2_fio2 is not None:
            if input_data.pao2_fio2 >= 400:
                scores[OrganSystem.RESPIRATORY] = 0
            elif input_data.pao2_fio2 >= 300:
                scores[OrganSystem.RESPIRATORY] = 1
            elif input_data.pao2_fio2 >= 200:
                scores[OrganSystem.RESPIRATORY] = 2
            elif input_data.pao2_fio2 >= 100:
                scores[OrganSystem.RESPIRATORY] = 3
            else:
                scores[OrganSystem.RESPIRATORY] = 4
        else:
            scores[OrganSystem.RESPIRATORY] = 0
        
        # Coagulation (Platelets)
        if input_data.platelets is not None:
            if input_data.platelets >= 150:
                scores[OrganSystem.COAGULATION] = 0
            elif input_data.platelets >= 100:
                scores[OrganSystem.COAGULATION] = 1
            elif input_data.platelets >= 50:
                scores[OrganSystem.COAGULATION] = 2
            elif input_data.platelets >= 20:
                scores[OrganSystem.COAGULATION] = 3
            else:
                scores[OrganSystem.COAGULATION] = 4
        else:
            scores[OrganSystem.COAGULATION] = 0
        
        # Liver (Bilirubin)
        if input_data.bilirubin is not None:
            if input_data.bilirubin < 1.2:
                scores[OrganSystem.LIVER] = 0
            elif input_data.bilirubin < 2.0:
                scores[OrganSystem.LIVER] = 1
            elif input_data.bilirubin < 6.0:
                scores[OrganSystem.LIVER] = 2
            elif input_data.bilirubin < 12.0:
                scores[OrganSystem.LIVER] = 3
            else:
                scores[OrganSystem.LIVER] = 4
        else:
            scores[OrganSystem.LIVER] = 0
        
        # Cardiovascular (MAP and vasopressors)
        if input_data.map_mmhg is not None:
            if input_data.map_mmhg >= 70:
                scores[OrganSystem.CARDIOVASCULAR] = 0
            elif input_data.map_mmhg < 70 and not input_data.vasopressors:
                scores[OrganSystem.CARDIOVASCULAR] = 1
            elif input_data.dopamine_dose <= 5:
                scores[OrganSystem.CARDIOVASCULAR] = 2
            elif input_data.dopamine_dose <= 15:
                scores[OrganSystem.CARDIOVASCULAR] = 3
            else:
                scores[OrganSystem.CARDIOVASCULAR] = 4
        else:
            scores[OrganSystem.CARDIOVASCULAR] = 0
        
        # CNS (Glasgow Coma Scale)
        if input_data.gcs is not None:
            if input_data.gcs == 15:
                scores[OrganSystem.CNS] = 0
            elif input_data.gcs >= 13:
                scores[OrganSystem.CNS] = 1
            elif input_data.gcs >= 10:
                scores[OrganSystem.CNS] = 2
            elif input_data.gcs >= 6:
                scores[OrganSystem.CNS] = 3
            else:
                scores[OrganSystem.CNS] = 4
        else:
            scores[OrganSystem.CNS] = 0
        
        # Renal (Creatinine and urine output)
        if input_data.creatinine is not None:
            if input_data.creatinine < 1.2:
                scores[OrganSystem.RENAL] = 0
            elif input_data.creatinine < 2.0:
                scores[OrganSystem.RENAL] = 1
            elif input_data.creatinine < 3.5:
                scores[OrganSystem.RENAL] = 2
            elif input_data.creatinine < 5.0 or (input_data.urine_output and input_data.urine_output < 500):
                scores[OrganSystem.RENAL] = 3
            else:
                scores[OrganSystem.RENAL] = 4
        else:
            scores[OrganSystem.RENAL] = 0
        
        total_score = sum(scores.values())
        
        # Interpretation
        if total_score >= 15:
            interpretation = "Severe organ dysfunction - High mortality risk (>50%)"
        elif total_score >= 10:
            interpretation = "Moderate organ dysfunction - ICU care likely needed"
        elif total_score >= 5:
            interpretation = "Mild organ dysfunction - Monitor closely"
        else:
            interpretation = "No significant organ dysfunction"
        
        return {
            "total_score": total_score,
            "component_scores": {k.value: v for k, v in scores.items()},
            "interpretation": interpretation,
            "mortality_risk": ClinicalCalculators._estimate_mortality(total_score)
        }
    
    @staticmethod
    def calculate_qsofa_score(input_data: qSOFAInput) -> Dict:
        """
        Calculate quick SOFA (qSOFA) - bedside screening tool
        Range: 0-3 (≥2 = high risk for sepsis)
        
        Criteria:
        - Respiratory rate ≥22/min
        - Altered mentation (GCS <15)
        - Systolic BP ≤100 mmHg
        """
        score = 0
        criteria_met = []
        
        if input_data.respiratory_rate >= 22:
            score += 1
            criteria_met.append("Tachypnea (RR ≥22)")
        
        if input_data.gcs < 15:
            score += 1
            criteria_met.append(f"Altered mentation (GCS={input_data.gcs})")
        
        if input_data.systolic_bp <= 100:
            score += 1
            criteria_met.append("Hypotension (SBP ≤100)")
        
        return {
            "score": score,
            "high_risk": score >= 2,
            "criteria_met": criteria_met,
            "recommendation": "Consider sepsis - obtain labs, cultures, initiate treatment" if score >= 2 else "Low risk for sepsis"
        }
    
    @staticmethod
    def _estimate_mortality(sofa_score: int) -> float:
        """Estimate ICU mortality based on SOFA score"""
        # Based on Vincent JL et al. JAMA 1996
        mortality_map = {
            0: 0.0,
            1: 0.01,
            2: 0.02,
            3: 0.05,
            4: 0.08,
            5: 0.12,
            6: 0.18,
            7: 0.25,
            8: 0.33,
            9: 0.40,
            10: 0.50,
            11: 0.60,
            12: 0.65,
            13: 0.70,
            14: 0.75,
            15: 0.80,
        }
        return mortality_map.get(min(sofa_score, 15), 0.95)
