import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple
from .models import PatientData, RiskFeature, PredictionResult

class MLEngine:
    def __init__(self):
        self.model = None
        self.feature_names = [
            "age", "wbc_count", "prior_antibiotics_days", "device_use", 
            "heart_rate", "temperature", "systolic_bp",
            "gender_enc", "bacterium_enc", "antibiotic_enc"
        ]
        self.encoders = {}
        self._train_mock_model()

    def _train_mock_model(self):
        # Generate synthetic data for training
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            "age": np.random.randint(18, 90, n_samples),
            "wbc_count": np.random.normal(8000, 3000, n_samples),
            "prior_antibiotics_days": np.random.exponential(5, n_samples),
            "device_use": np.random.choice([0, 1], n_samples),
            "heart_rate": np.random.normal(80, 15, n_samples),
            "temperature": np.random.normal(37, 0.8, n_samples),
            "systolic_bp": np.random.normal(120, 15, n_samples),
            "gender": np.random.choice(["M", "F"], n_samples),
            "suspected_bacterium": np.random.choice(["E. coli", "S. aureus", "K. pneumoniae", "P. aeruginosa"], n_samples),
            "candidate_antibiotic": np.random.choice(["Ciprofloxacin", "Levofloxacin", "Ceftriaxone", "Meropenem"], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Define logic for "ground truth" resistance (to make model learnable)
        # Risk increases with: High WBC, Long prior Abx, Device use, High Temp, High HR
        risk_score = (
            (df["wbc_count"] > 11000).astype(int) * 2 +
            (df["prior_antibiotics_days"] / 10) +
            (df["device_use"] * 1.5) +
            ((df["temperature"] > 37.8).astype(int) * 1.5) +
            ((df["heart_rate"] > 100).astype(int) * 1.0)
        )
        
        # Add noise
        risk_score += np.random.normal(0, 0.5, n_samples)
        
        # Target: 1 if risk > threshold
        df["target"] = (risk_score > 2.5).astype(int)
        
        # Encoding
        le_gender = LabelEncoder()
        df["gender_enc"] = le_gender.fit_transform(df["gender"])
        self.encoders["gender"] = le_gender
        
        le_bact = LabelEncoder()
        df["bacterium_enc"] = le_bact.fit_transform(df["suspected_bacterium"])
        self.encoders["suspected_bacterium"] = le_bact
        
        le_abx = LabelEncoder()
        df["antibiotic_enc"] = le_abx.fit_transform(df["candidate_antibiotic"])
        self.encoders["candidate_antibiotic"] = le_abx
        
        X = df[self.feature_names]
        y = df["target"]
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X, y)

    def predict(self, data: PatientData) -> PredictionResult:
        # Prepare input vector
        try:
            gender_enc = self.encoders["gender"].transform([data.gender])[0]
        except: gender_enc = 0
            
        try:
            bact_enc = self.encoders["suspected_bacterium"].transform([data.suspected_bacterium])[0]
        except: bact_enc = 0
            
        try:
            abx_enc = self.encoders["candidate_antibiotic"].transform([data.candidate_antibiotic])[0]
        except: abx_enc = 0
        
        features = np.array([[
            data.age, data.wbc_count, data.prior_antibiotics_days, int(data.device_use),
            data.heart_rate, data.temperature, data.systolic_bp,
            gender_enc, bact_enc, abx_enc
        ]])
        
        # Predict probability
        risk_score = self.model.predict_proba(features)[0][1]
        
        # Determine Level
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Confidence Interval (using tree variance approximation)
        # Simple heuristic for demo: closer to 0.5 = higher uncertainty
        uncertainty = 0.1 - (abs(risk_score - 0.5) * 0.1)
        lower_bound = max(0.0, risk_score - uncertainty)
        upper_bound = min(1.0, risk_score + uncertainty)

        # Feature Importance (Global from model, but contextualized)
        importances = self.model.feature_importances_
        feature_impacts = []
        
        # Map back to readable names and values
        val_map = {
            "age": f"{data.age} yrs",
            "wbc_count": f"{data.wbc_count}",
            "prior_antibiotics_days": f"{data.prior_antibiotics_days} days",
            "device_use": "Yes" if data.device_use else "No",
            "heart_rate": f"{data.heart_rate} bpm",
            "temperature": f"{data.temperature}°C",
            "systolic_bp": f"{data.systolic_bp} mmHg",
            "gender_enc": data.gender,
            "bacterium_enc": data.suspected_bacterium,
            "antibiotic_enc": data.candidate_antibiotic
        }
        
        desc_map = {
            "wbc_count": "High WBC indicates infection load.",
            "prior_antibiotics_days": "Recent exposure drives selection pressure.",
            "device_use": "Devices are colonization hubs.",
            "temperature": "Fever correlates with systemic infection.",
            "heart_rate": "Tachycardia is a sign of sepsis risk."
        }

        for name, imp in zip(self.feature_names, importances):
            if name in desc_map: # Only show key clinical features
                feature_impacts.append(RiskFeature(
                    name=name.replace("_", " ").title(),
                    value=val_map[name],
                    risk_contribution=float(imp) * (2.0 if risk_score > 0.5 else 0.5), # Scale for visual effect
                    description=desc_map.get(name, "Clinical factor.")
                ))
        
        feature_impacts.sort(key=lambda x: x.risk_contribution, reverse=True)
        feature_impacts = feature_impacts[:3] # Top 3

        # Symbolic Rule (Heuristic based on dominant factors)
        rule_parts = []
        if data.prior_antibiotics_days > 7: rule_parts.append("Prior Abx > 7d")
        if data.wbc_count > 12000: rule_parts.append("WBC > 12k")
        if data.temperature > 38.0: rule_parts.append("Temp > 38°C")
        if data.device_use: rule_parts.append("Device = Yes")
        
        symbolic_rule = "IF " + " AND ".join(rule_parts) + " THEN High Risk" if rule_parts else "No critical risk factors active."

        recommendation = "Monitor vitals."
        if risk_level == "High":
            recommendation = "Escalate to broad-spectrum & consult ID."
        elif risk_level == "Moderate":
            recommendation = "Await culture results before escalation."

        return PredictionResult(
            risk_score=float(risk_score),
            risk_level=risk_level,
            confidence_interval=[float(lower_bound), float(upper_bound)],
            symbolic_rule=symbolic_rule,
            risk_features=feature_impacts,
            recommendation=recommendation
        )

engine = MLEngine()
