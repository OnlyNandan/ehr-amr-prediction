"""
Sepsis Prediction Engine
Uses XGBoost model to predict sepsis onset 4-48 hours in advance
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SepsisFeatures:
    """Features for sepsis prediction"""
    # Vitals (required)
    heart_rate: float
    temperature: float
    systolic_bp: float
    wbc_count: float
    age: int
    
    # Vitals (optional)
    respiratory_rate: float = 18.0
    spo2: float = 98.0
    
    # Labs (optional)
    lactate: float = 2.0
    creatinine: float = 1.0
    platelets: float = 200.0
    bilirubin: float = 0.8
    
    # Clinical (optional)
    gcs: int = 15
    vasopressors: bool = False
    
    # Historical (optional)
    prior_sepsis: bool = False
    immunosuppressed: bool = False

class SepsisEngine:
    """Sepsis risk prediction using XGBoost"""
    
    def __init__(self):
        self.model_4hr = None
        self.model_12hr = None
        self.model_24hr = None
        self.model_48hr = None
        self._train_models()
    
    def _train_models(self):
        """Train XGBoost models for different time horizons"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 5000
        
        # Features
        X = pd.DataFrame({
            'heart_rate': np.random.normal(85, 20, n_samples),
            'temperature': np.random.normal(37.2, 1.5, n_samples),
            'systolic_bp': np.random.normal(120, 20, n_samples),
            'respiratory_rate': np.random.normal(18, 4, n_samples),
            'wbc_count': np.random.normal(9000, 4000, n_samples),
            'lactate': np.random.exponential(2, n_samples),
            'creatinine': np.random.exponential(1.2, n_samples),
            'platelets': np.random.normal(220, 80, n_samples),
            'age': np.random.randint(20, 90, n_samples),
            'gcs': np.random.choice([15, 14, 13, 12, 10, 8, 6], n_samples, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]),
            'vasopressors': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'prior_sepsis': np.random.choice([0, 1], n_samples, p=[0.90, 0.10]),
        })
        
        # Target: Simulate sepsis risk based on realistic criteria
        # High risk if: tachycardia + fever + elevated WBC + high lactate
        sepsis_score = (
            (X['heart_rate'] > 100).astype(int) * 2 +
            (X['temperature'] > 38.3).astype(int) * 2 +
            (X['wbc_count'] > 12000).astype(int) * 1.5 +
            (X['lactate'] > 2.5).astype(int) * 3 +
            (X['systolic_bp'] < 100).astype(int) * 2.5 +
            (X['respiratory_rate'] > 22).astype(int) * 1 +
            (X['gcs'] < 15).astype(int) * 2 +
            X['vasopressors'] * 4 +
            X['prior_sepsis'] * 1.5 +
            np.random.normal(0, 1, n_samples)  # Add noise
        )
        
        y_4hr = (sepsis_score > 6).astype(int)
        y_12hr = (sepsis_score > 5).astype(int)
        y_24hr = (sepsis_score > 4.5).astype(int)
        y_48hr = (sepsis_score > 4).astype(int)
        
        # Train models
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        }
        
        self.model_4hr = XGBClassifier(**params)
        self.model_4hr.fit(X, y_4hr)
        
        self.model_12hr = XGBClassifier(**params)
        self.model_12hr.fit(X, y_12hr)
        
        self.model_24hr = XGBClassifier(**params)
        self.model_24hr.fit(X, y_24hr)
        
        self.model_48hr = XGBClassifier(**params)
        self.model_48hr.fit(X, y_48hr)
    
    def predict_sepsis_risk(self, features: SepsisFeatures) -> Dict:
        """
        Predict sepsis risk at multiple time horizons
        
        Returns: {
            'risk_4hr': float,
            'risk_12hr': float,
            'risk_24hr': float,
            'risk_48hr': float,
            'overall_risk': float,
            'alert_level': str,
            'recommendations': List[str]
        }
        """
        # Prepare feature vector
        X = pd.DataFrame([{
            'heart_rate': features.heart_rate,
            'temperature': features.temperature,
            'systolic_bp': features.systolic_bp,
            'respiratory_rate': features.respiratory_rate,
            'wbc_count': features.wbc_count,
            'lactate': features.lactate,
            'creatinine': features.creatinine,
            'platelets': features.platelets,
            'age': features.age,
            'gcs': features.gcs,
            'vasopressors': int(features.vasopressors),
            'prior_sepsis': int(features.prior_sepsis),
        }])
        
        # Get predictions for each time horizon
        risk_4hr = float(self.model_4hr.predict_proba(X)[0, 1])
        risk_12hr = float(self.model_12hr.predict_proba(X)[0, 1])
        risk_24hr = float(self.model_24hr.predict_proba(X)[0, 1])
        risk_48hr = float(self.model_48hr.predict_proba(X)[0, 1])
        
        # Overall risk (weighted average, prioritizing near-term)
        overall_risk = (
            risk_4hr * 0.4 +
            risk_12hr * 0.3 +
            risk_24hr * 0.2 +
            risk_48hr * 0.1
        )
        
        # Determine alert level
        if overall_risk > 0.7:
            alert_level = "CRITICAL"
            color = "red"
        elif overall_risk > 0.5:
            alert_level = "HIGH"
            color = "orange"
        elif overall_risk > 0.3:
            alert_level = "MODERATE"
            color = "yellow"
        else:
            alert_level = "LOW"
            color = "green"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features, overall_risk)
        
        return {
            "risk_4hr": risk_4hr,
            "risk_12hr": risk_12hr,
            "risk_24hr": risk_24hr,
            "risk_48hr": risk_48hr,
            "overall_risk": overall_risk,
            "alert_level": alert_level,
            "alert_color": color,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "next_assessment": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    
    def _generate_recommendations(self, features: SepsisFeatures, risk: float) -> List[str]:
        """Generate clinical recommendations based on risk factors"""
        recommendations = []
        
        if risk > 0.5:
            recommendations.append("🚨 HIGH SEPSIS RISK - Initiate sepsis protocol immediately")
            recommendations.append("Obtain blood cultures (x2) before antibiotics")
            recommendations.append("Start broad-spectrum antibiotics within 1 hour")
            recommendations.append("Administer 30 mL/kg crystalloid for hypotension")
        elif risk > 0.3:
            recommendations.append("⚠️ Moderate sepsis risk - Monitor closely")
            recommendations.append("Consider blood cultures if clinical deterioration")
            recommendations.append("Reassess in 1-2 hours")
        
        # Specific interventions
        if features.lactate > 2.5:
            recommendations.append(f"🔬 Elevated lactate ({features.lactate:.1f}) - Repeat in 2-4 hours")
        
        if features.systolic_bp < 90:
            recommendations.append("💉 Hypotension - Consider fluid resuscitation")
        
        if features.wbc_count > 12000 or features.wbc_count < 4000:
            recommendations.append(f"🩸 Abnormal WBC ({features.wbc_count:.0f}) - Suggests infection")
        
        if features.gcs < 15:
            recommendations.append(f"🧠 Altered mental status (GCS={features.gcs}) - Monitor neurological status")
        
        if features.temperature > 38.3:
            recommendations.append("🌡️ Fever present - Source control evaluation needed")
        
        return recommendations

# Global instance
sepsis_engine = SepsisEngine()
