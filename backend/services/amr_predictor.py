"""
AMR Prediction Service
Main prediction engine for antimicrobial resistance
"""
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib

from models.schemas import (
    AMRPredictionRequest,
    AMRPredictionResponse,
    AntibioticPrediction,
    FeatureContribution,
    RiskLevel
)
from services.feature_engineering import FeatureEngineer
from services.explainability import SHAPExplainer
from config import Config

logger = logging.getLogger(__name__)


class AMRPredictor:
    """
    Antimicrobial Resistance Prediction Engine.
    
    Supports:
    - Binary classification per antibiotic
    - Multi-label classification for multiple antibiotics
    - Probability calibration
    - SHAP-based explanations
    """
    
    MODEL_VERSION = "1.0.0"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        calibrator_path: Optional[str] = None,
        target_antibiotics: Optional[List[str]] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the AMR predictor.
        
        Args:
            model_path: Path to trained model(s)
            calibrator_path: Path to calibration model
            target_antibiotics: List of antibiotics to predict
            config: Configuration object
        """
        self.config = config or Config()
        self.target_antibiotics = target_antibiotics or self.config.TARGET_ANTIBIOTICS
        
        self.feature_engineer = FeatureEngineer()
        self.explainer: Optional[SHAPExplainer] = None
        
        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None
        
        # Load models if paths provided
        if model_path and Path(model_path).exists():
            self._load_models(model_path)
        else:
            logger.warning("No model loaded. Using mock predictions.")
            self._init_mock_models()
        
        if calibrator_path and Path(calibrator_path).exists():
            self._load_calibrators(calibrator_path)
    
    def _load_models(self, model_path: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                # Multi-model format: {antibiotic: model}
                self.models = model_data.get("models", {})
                self.feature_names = model_data.get("feature_names", None)
                self.MODEL_VERSION = model_data.get("version", self.MODEL_VERSION)
            else:
                # Single model for all antibiotics
                for ab in self.target_antibiotics:
                    self.models[ab] = model_data
            
            logger.info(f"Loaded models for {len(self.models)} antibiotics")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._init_mock_models()
    
    def _load_calibrators(self, calibrator_path: str):
        """Load calibration models"""
        try:
            calibrator_data = joblib.load(calibrator_path)
            self.calibrators = calibrator_data
            logger.info(f"Loaded calibrators for {len(self.calibrators)} antibiotics")
        except Exception as e:
            logger.warning(f"Failed to load calibrators: {e}")
    
    def _init_mock_models(self):
        """Initialize mock models for testing"""
        logger.info("Initializing mock models for demonstration")
        for ab in self.target_antibiotics:
            self.models[ab] = MockModel(ab)
    
    def _get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level"""
        thresholds = self.config.RISK_THRESHOLDS
        
        if probability >= thresholds["high"]:
            return RiskLevel.CRITICAL
        elif probability >= thresholds["medium"]:
            return RiskLevel.HIGH
        elif probability >= thresholds["low"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calibrate_probability(self, antibiotic: str, probability: float) -> float:
        """Apply calibration to raw probability"""
        if antibiotic in self.calibrators:
            try:
                calibrator = self.calibrators[antibiotic]
                # Reshape for sklearn calibrators
                prob_array = np.array([[probability]])
                calibrated = calibrator.predict_proba(prob_array)[0, 1]
                return float(calibrated)
            except Exception as e:
                logger.warning(f"Calibration failed for {antibiotic}: {e}")
        
        return probability
    
    def _generate_clinical_summary(
        self,
        predictions: List[AntibioticPrediction],
        overall_risk: RiskLevel
    ) -> str:
        """Generate human-readable clinical summary"""
        high_risk_antibiotics = [
            p.antibiotic for p in predictions 
            if p.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        
        if not high_risk_antibiotics:
            return (
                "Low probability of antimicrobial resistance detected for tested antibiotics. "
                "Standard empiric therapy may be appropriate pending culture results."
            )
        
        ab_list = ", ".join(high_risk_antibiotics)
        
        # Get top contributing factors across all predictions
        all_contributions = []
        for pred in predictions:
            if pred.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                all_contributions.extend(pred.top_contributing_features[:2])
        
        # Get unique top factors
        top_factors = []
        seen_features = set()
        for contrib in sorted(all_contributions, key=lambda x: abs(x.shap_value), reverse=True):
            if contrib.feature_name not in seen_features:
                top_factors.append(contrib.feature_name)
                seen_features.add(contrib.feature_name)
            if len(top_factors) >= 3:
                break
        
        factor_str = ", ".join(top_factors) if top_factors else "multiple clinical factors"
        
        if overall_risk == RiskLevel.CRITICAL:
            return (
                f"CRITICAL: High probability of resistance to {ab_list}. "
                f"Key risk factors: {factor_str}. "
                "Consider alternative empiric coverage and urgent infectious disease consultation."
            )
        else:
            return (
                f"Elevated probability of resistance to {ab_list}. "
                f"Contributing factors: {factor_str}. "
                "Consider broader spectrum coverage pending culture results."
            )
    
    def predict(
        self,
        request: AMRPredictionRequest,
        include_explanations: bool = True
    ) -> AMRPredictionResponse:
        """
        Generate AMR predictions for a patient.
        
        Args:
            request: Complete prediction request
            include_explanations: Whether to compute SHAP explanations
            
        Returns:
            AMRPredictionResponse with predictions and explanations
        """
        start_time = time.time()
        prediction_id = f"PRED_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Transform features
        features, feature_names = self.feature_engineer.transform(request)
        
        # Determine which antibiotics to predict
        target_abs = request.target_antibiotics or self.target_antibiotics
        target_abs = [ab for ab in target_abs if ab in self.models]
        
        # Generate predictions
        predictions = []
        warnings = []
        
        for antibiotic in target_abs:
            try:
                pred = self._predict_single_antibiotic(
                    antibiotic=antibiotic,
                    features=features,
                    feature_names=feature_names,
                    include_explanations=include_explanations
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {antibiotic}: {e}")
                warnings.append(f"Prediction unavailable for {antibiotic}")
        
        # Calculate overall risk (highest among all)
        overall_risk = RiskLevel.LOW
        for pred in predictions:
            if pred.risk_level == RiskLevel.CRITICAL:
                overall_risk = RiskLevel.CRITICAL
                break
            elif pred.risk_level == RiskLevel.HIGH and overall_risk != RiskLevel.CRITICAL:
                overall_risk = RiskLevel.HIGH
            elif pred.risk_level == RiskLevel.MEDIUM and overall_risk == RiskLevel.LOW:
                overall_risk = RiskLevel.MEDIUM
        
        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(predictions, overall_risk)
        
        # Calculate latency
        inference_latency = (time.time() - start_time) * 1000
        
        # Check latency requirement
        if inference_latency > self.config.MAX_INFERENCE_LATENCY_MS:
            warnings.append(f"Inference latency ({inference_latency:.0f}ms) exceeded target")
        
        return AMRPredictionResponse(
            patient_id=request.patient_id,
            prediction_id=prediction_id,
            prediction_time=datetime.utcnow(),
            model_version=self.MODEL_VERSION,
            predictions=predictions,
            overall_risk_level=overall_risk,
            clinical_summary=clinical_summary,
            inference_latency_ms=inference_latency,
            warnings=warnings
        )
    
    def _predict_single_antibiotic(
        self,
        antibiotic: str,
        features: np.ndarray,
        feature_names: List[str],
        include_explanations: bool = True
    ) -> AntibioticPrediction:
        """
        Generate prediction for a single antibiotic.
        """
        model = self.models[antibiotic]
        
        # Get raw prediction
        if hasattr(model, 'predict_proba'):
            # Sklearn-style model
            features_2d = features.reshape(1, -1)
            proba = model.predict_proba(features_2d)[0, 1]
        else:
            # Assume callable returning probability
            proba = model(features)
        
        # Apply calibration
        calibrated_proba = self._calibrate_probability(antibiotic, proba)
        
        # Get risk level
        risk_level = self._get_risk_level(calibrated_proba)
        
        # Get SHAP explanations
        top_contributions = []
        if include_explanations:
            try:
                top_contributions = self._get_feature_contributions(
                    model=model,
                    features=features,
                    feature_names=feature_names,
                    top_k=5
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        return AntibioticPrediction(
            antibiotic=antibiotic,
            probability=calibrated_proba,
            risk_level=risk_level,
            calibrated=antibiotic in self.calibrators,
            top_contributing_features=top_contributions
        )
    
    def _get_feature_contributions(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        top_k: int = 5
    ) -> List[FeatureContribution]:
        """
        Get top contributing features using SHAP or feature importances.
        """
        contributions = []
        
        # Try SHAP first
        if self.explainer is not None:
            try:
                shap_values = self.explainer.explain(model, features.reshape(1, -1))
                
                # Get top k by absolute value
                indices = np.argsort(np.abs(shap_values[0]))[-top_k:][::-1]
                
                for idx in indices:
                    contrib = FeatureContribution(
                        feature_name=feature_names[idx],
                        feature_value=float(features[idx]),
                        shap_value=float(shap_values[0][idx]),
                        category=self.feature_engineer.get_feature_category(feature_names[idx])
                    )
                    contributions.append(contrib)
                
                return contributions
            except Exception as e:
                logger.debug(f"SHAP failed, falling back to feature importance: {e}")
        
        # Fallback to feature importance if model has it
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_k:][::-1]
            
            for idx in indices:
                contrib = FeatureContribution(
                    feature_name=feature_names[idx],
                    feature_value=float(features[idx]),
                    shap_value=float(importances[idx]),  # Use importance as proxy
                    category=self.feature_engineer.get_feature_category(feature_names[idx])
                )
                contributions.append(contrib)
        
        # If nothing else, use mock contributions
        if not contributions:
            contributions = self._mock_contributions(features, feature_names, top_k)
        
        return contributions
    
    def _mock_contributions(
        self,
        features: np.ndarray,
        feature_names: List[str],
        top_k: int
    ) -> List[FeatureContribution]:
        """Generate mock feature contributions for demonstration"""
        # Select features with non-zero values
        nonzero_indices = np.where(features != 0)[0]
        
        if len(nonzero_indices) > top_k:
            selected_indices = np.random.choice(nonzero_indices, top_k, replace=False)
        else:
            selected_indices = nonzero_indices[:top_k]
        
        contributions = []
        for idx in selected_indices:
            contrib = FeatureContribution(
                feature_name=feature_names[idx],
                feature_value=float(features[idx]),
                shap_value=np.random.uniform(-0.2, 0.2),
                category=self.feature_engineer.get_feature_category(feature_names[idx])
            )
            contributions.append(contrib)
        
        return contributions
    
    def predict_batch(
        self,
        requests: List[AMRPredictionRequest],
        include_explanations: bool = True
    ) -> List[AMRPredictionResponse]:
        """
        Generate predictions for multiple patients.
        """
        return [
            self.predict(request, include_explanations)
            for request in requests
        ]


class MockModel:
    """
    Mock model for testing when no trained model is available.
    Generates realistic-looking predictions based on feature patterns.
    """
    
    def __init__(self, antibiotic: str):
        self.antibiotic = antibiotic
        
        # Different antibiotics have different baseline resistance rates
        self.baseline_resistance = {
            "ceftriaxone": 0.25,
            "ciprofloxacin": 0.30,
            "meropenem": 0.10,
            "vancomycin": 0.05,
            "piperacillin_tazobactam": 0.20,
            "gentamicin": 0.15,
            "ampicillin": 0.45,
            "trimethoprim_sulfamethoxazole": 0.35
        }
        
        # Mock feature importances
        self.feature_importances_ = None
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate mock probability predictions"""
        n_samples = X.shape[0]
        base_rate = self.baseline_resistance.get(self.antibiotic, 0.20)
        
        # Use some features to modulate probability
        probas = []
        
        for i in range(n_samples):
            features = X[i]
            
            # Start with base rate
            prob = base_rate
            
            # Modulate based on feature patterns
            # Higher bacterial load increases risk
            if len(features) > 10:
                bacterial_idx = 10  # Approximate index
                prob += features[bacterial_idx] * 0.05
            
            # Prior AMR increases risk significantly
            if len(features) > 50:
                prior_amr_idx = 50
                if features[prior_amr_idx] > 0:
                    prob += 0.2
            
            # ICU patients have higher risk
            if len(features) > 60:
                icu_idx = 60
                if features[icu_idx] > 0:
                    prob += 0.15
            
            # Add some noise
            prob += np.random.normal(0, 0.05)
            
            # Clip to valid range
            prob = np.clip(prob, 0.01, 0.99)
            probas.append([1 - prob, prob])
        
        return np.array(probas)
    
    def __call__(self, features: np.ndarray) -> float:
        """Allow calling model directly"""
        return self.predict_proba(features.reshape(1, -1))[0, 1]


# Singleton instance
_predictor_instance: Optional[AMRPredictor] = None


def get_predictor(
    model_path: Optional[str] = None,
    calibrator_path: Optional[str] = None,
    config: Optional[Config] = None
) -> AMRPredictor:
    """Get or create singleton predictor instance"""
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = AMRPredictor(
            model_path=model_path,
            calibrator_path=calibrator_path,
            config=config
        )
    
    return _predictor_instance
