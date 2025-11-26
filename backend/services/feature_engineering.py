"""
Feature Engineering Module for AMR Prediction
Transforms raw data into model-ready features
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.schemas import (
    AMRPredictionRequest,
    ImageFeatures,
    Vitals,
    LabResults,
    PatientHistory,
    Demographics,
    ClinicalContext
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for AMR prediction.
    
    Handles:
    - Normalization of cell counts
    - Temporal features for antibiotic exposure
    - Interaction features (NLR * CRP, etc.)
    - Encoding of categorical variables
    - Missing value imputation
    """
    
    # Reference ranges for lab normalization
    LAB_REFERENCE_RANGES = {
        "wbc": (4.5, 11.0),  # ×10³/µL
        "crp": (0, 10),  # mg/L
        "lactate": (0.5, 2.0),  # mmol/L
        "creatinine": (0.6, 1.2),  # mg/dL (adult)
        "ast": (10, 40),  # U/L
        "alt": (7, 56),  # U/L
        "hemoglobin": (12, 17),  # g/dL
        "platelets": (150, 400),  # ×10³/µL
    }
    
    # Vital sign reference ranges
    VITAL_REFERENCE_RANGES = {
        "temperature": (36.1, 37.2),  # Celsius
        "heart_rate": (60, 100),  # BPM
        "respiratory_rate": (12, 20),  # per minute
        "bp_systolic": (90, 120),  # mmHg
        "bp_diastolic": (60, 80),  # mmHg
        "oxygen_saturation": (95, 100),  # %
    }
    
    # Antibiotic class mapping for encoding
    ANTIBIOTIC_CLASSES = {
        "penicillins": ["amoxicillin", "ampicillin", "penicillin", "piperacillin"],
        "cephalosporins": ["ceftriaxone", "cefazolin", "ceftazidime", "cefepime", "cephalexin"],
        "carbapenems": ["meropenem", "imipenem", "ertapenem"],
        "fluoroquinolones": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
        "aminoglycosides": ["gentamicin", "tobramycin", "amikacin"],
        "glycopeptides": ["vancomycin", "teicoplanin"],
        "macrolides": ["azithromycin", "erythromycin", "clarithromycin"],
        "tetracyclines": ["doxycycline", "tetracycline"],
        "sulfonamides": ["trimethoprim_sulfamethoxazole", "sulfamethoxazole"],
        "other": []
    }
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self._init_antibiotic_lookup()
    
    def _init_antibiotic_lookup(self):
        """Create antibiotic to class lookup"""
        self.antibiotic_to_class = {}
        for cls, antibiotics in self.ANTIBIOTIC_CLASSES.items():
            for ab in antibiotics:
                self.antibiotic_to_class[ab.lower()] = cls
    
    def get_antibiotic_class(self, antibiotic: str) -> str:
        """Get antibiotic class for an antibiotic name"""
        return self.antibiotic_to_class.get(antibiotic.lower(), "other")
    
    def normalize_to_reference(self, value: float, ref_range: Tuple[float, float]) -> float:
        """
        Normalize a value based on reference range.
        Returns how many standard deviations from midpoint.
        """
        if value is None:
            return 0.0
        
        low, high = ref_range
        mid = (low + high) / 2
        spread = (high - low) / 2
        
        if spread == 0:
            return 0.0
        
        return (value - mid) / spread
    
    def compute_image_features(self, image_features: ImageFeatures) -> Dict[str, float]:
        """
        Process and engineer image-derived features.
        """
        features = {}
        
        # Raw counts (normalized per HPF)
        features["img_neutrophil_count"] = image_features.neutrophil_count
        features["img_lymphocyte_count"] = image_features.lymphocyte_count
        features["img_monocyte_count"] = image_features.monocyte_count
        features["img_eosinophil_count"] = image_features.eosinophil_count
        features["img_basophil_count"] = image_features.basophil_count
        
        # NLR (critical for infection/sepsis)
        features["img_nlr"] = image_features.nlr or 0
        
        # NLR categories (clinical significance)
        nlr = features["img_nlr"]
        features["img_nlr_elevated"] = float(nlr > 3)
        features["img_nlr_high"] = float(nlr > 6)
        features["img_nlr_critical"] = float(nlr > 10)
        
        # Platelet features
        features["img_platelet_estimate"] = image_features.platelet_estimate
        features["img_thrombocytopenia"] = float(image_features.platelet_estimate < 150000)
        
        # Bacterial presence (critical for AMR)
        features["img_bacterial_cluster_count"] = image_features.bacterial_cluster_count
        features["img_bacteria_present"] = float(image_features.bacterial_cluster_count > 0)
        features["img_bacterial_confidence"] = image_features.mean_bacterial_bbox_confidence
        
        # High bacterial load indicator
        features["img_high_bacterial_load"] = float(image_features.bacterial_cluster_count > 3)
        
        # Parasite presence
        features["img_parasite_present"] = float(image_features.parasite_present)
        
        # RBC morphology flags
        features["img_anisocytosis"] = float(image_features.rbc_morphology_anisocytosis)
        features["img_poikilocytosis"] = float(image_features.rbc_morphology_poikilocytosis)
        features["img_hypochromia"] = float(image_features.rbc_morphology_hypochromia)
        
        # Morphology score (composite)
        features["img_morphology_abnormal_score"] = (
            features["img_anisocytosis"] +
            features["img_poikilocytosis"] +
            features["img_hypochromia"]
        ) / 3
        
        # Total WBC from image
        features["img_total_wbc"] = (
            features["img_neutrophil_count"] +
            features["img_lymphocyte_count"] +
            features["img_monocyte_count"] +
            features["img_eosinophil_count"] +
            features["img_basophil_count"]
        )
        
        return features
    
    def compute_vital_features(self, vitals: Vitals) -> Dict[str, float]:
        """
        Process and engineer vital sign features.
        """
        features = {}
        
        # Raw vitals (normalized to reference)
        features["vital_temp"] = vitals.temperature
        features["vital_temp_norm"] = self.normalize_to_reference(
            vitals.temperature, self.VITAL_REFERENCE_RANGES["temperature"]
        )
        
        features["vital_hr"] = vitals.heart_rate
        features["vital_hr_norm"] = self.normalize_to_reference(
            vitals.heart_rate, self.VITAL_REFERENCE_RANGES["heart_rate"]
        )
        
        features["vital_rr"] = vitals.respiratory_rate
        features["vital_rr_norm"] = self.normalize_to_reference(
            vitals.respiratory_rate, self.VITAL_REFERENCE_RANGES["respiratory_rate"]
        )
        
        features["vital_bp_sys"] = vitals.bp_systolic
        features["vital_bp_dia"] = vitals.bp_diastolic
        features["vital_map"] = (vitals.bp_systolic + 2 * vitals.bp_diastolic) / 3  # Mean arterial pressure
        
        features["vital_o2sat"] = vitals.oxygen_saturation
        
        # Clinical flags
        features["vital_fever"] = float(vitals.temperature >= 38.0)
        features["vital_high_fever"] = float(vitals.temperature >= 39.0)
        features["vital_hypothermia"] = float(vitals.temperature < 36.0)
        
        features["vital_tachycardia"] = float(vitals.heart_rate > 100)
        features["vital_severe_tachycardia"] = float(vitals.heart_rate > 130)
        
        features["vital_tachypnea"] = float(vitals.respiratory_rate > 22)
        
        features["vital_hypotension"] = float(vitals.bp_systolic < 90)
        features["vital_hypoxia"] = float(vitals.oxygen_saturation < 92)
        
        # SIRS criteria count (Systemic Inflammatory Response Syndrome)
        sirs_count = (
            int(vitals.temperature > 38 or vitals.temperature < 36) +
            int(vitals.heart_rate > 90) +
            int(vitals.respiratory_rate > 20)
        )
        features["vital_sirs_criteria"] = sirs_count
        features["vital_sirs_positive"] = float(sirs_count >= 2)
        
        return features
    
    def compute_lab_features(self, labs: LabResults) -> Dict[str, float]:
        """
        Process and engineer laboratory features.
        """
        features = {}
        
        # WBC
        features["lab_wbc"] = labs.wbc
        features["lab_wbc_norm"] = self.normalize_to_reference(
            labs.wbc, self.LAB_REFERENCE_RANGES["wbc"]
        )
        features["lab_leukocytosis"] = float(labs.wbc > 12)
        features["lab_severe_leukocytosis"] = float(labs.wbc > 20)
        features["lab_leukopenia"] = float(labs.wbc < 4)
        
        # CRP (inflammation marker)
        crp = labs.crp or 0
        features["lab_crp"] = crp
        features["lab_crp_elevated"] = float(crp > 10)
        features["lab_crp_high"] = float(crp > 50)
        features["lab_crp_critical"] = float(crp > 100)
        
        # Lactate (sepsis marker)
        lactate = labs.lactate or 0
        features["lab_lactate"] = lactate
        features["lab_lactate_elevated"] = float(lactate > 2)
        features["lab_lactate_high"] = float(lactate > 4)
        
        # Creatinine (renal function - affects antibiotic dosing)
        creatinine = labs.creatinine or 1.0
        features["lab_creatinine"] = creatinine
        features["lab_aki"] = float(creatinine > 1.5)  # Simplified AKI flag
        
        # Liver function
        ast = labs.ast or 0
        alt = labs.alt or 0
        features["lab_ast"] = ast
        features["lab_alt"] = alt
        features["lab_liver_elevated"] = float(ast > 80 or alt > 80)
        
        # Hemoglobin
        hgb = labs.hemoglobin or 13
        features["lab_hemoglobin"] = hgb
        features["lab_anemia"] = float(hgb < 10)
        
        # Platelets (from lab, if available)
        plt = labs.platelets or 200
        features["lab_platelets"] = plt
        features["lab_thrombocytopenia"] = float(plt < 150)
        
        return features
    
    def compute_history_features(self, history: PatientHistory) -> Dict[str, float]:
        """
        Process and engineer patient history features.
        """
        features = {}
        
        # Antibiotic exposure
        features["hist_days_since_antibiotic"] = history.days_since_last_antibiotic or 365
        features["hist_recent_antibiotic"] = float((history.days_since_last_antibiotic or 365) < 30)
        features["hist_very_recent_antibiotic"] = float((history.days_since_last_antibiotic or 365) < 7)
        
        features["hist_antibiotic_count_90d"] = history.antibiotic_exposure_count_90d
        features["hist_multiple_antibiotics"] = float(history.antibiotic_exposure_count_90d > 1)
        features["hist_heavy_antibiotic_exposure"] = float(history.antibiotic_exposure_count_90d > 3)
        
        # Encode antibiotic class exposures
        exposed_classes = set()
        for exposure in history.prior_antibiotics:
            ab_class = self.get_antibiotic_class(exposure.antibiotic_name)
            exposed_classes.add(ab_class)
        
        for ab_class in self.ANTIBIOTIC_CLASSES.keys():
            features[f"hist_exposed_{ab_class}"] = float(ab_class in exposed_classes)
        
        # Prior hospitalizations
        features["hist_hospitalizations_30d"] = history.prior_hospitalizations_30d
        features["hist_recent_hospitalization"] = float(history.prior_hospitalizations_30d > 0)
        features["hist_frequent_hospitalization"] = float(history.prior_hospitalizations_30d > 2)
        
        # Prior AMR
        features["hist_prior_amr"] = float(history.prior_amr_positive)
        
        # Comorbidities
        features["hist_comorbidity_count"] = len(history.comorbidities)
        features["hist_immunocompromised"] = float(history.immunocompromised)
        features["hist_ckd"] = float(history.chronic_kidney_disease)
        features["hist_diabetes"] = float(history.diabetes)
        
        return features
    
    def compute_demographic_features(self, demographics: Demographics) -> Dict[str, float]:
        """
        Process and engineer demographic features.
        """
        features = {}
        
        # Age
        features["demo_age"] = demographics.age
        features["demo_elderly"] = float(demographics.age >= 65)
        features["demo_very_elderly"] = float(demographics.age >= 80)
        features["demo_pediatric"] = float(demographics.age < 18)
        
        # Age groups (one-hot)
        features["demo_age_0_18"] = float(demographics.age < 18)
        features["demo_age_18_40"] = float(18 <= demographics.age < 40)
        features["demo_age_40_65"] = float(40 <= demographics.age < 65)
        features["demo_age_65_80"] = float(65 <= demographics.age < 80)
        features["demo_age_80_plus"] = float(demographics.age >= 80)
        
        # Sex (binary encoding)
        features["demo_male"] = float(demographics.sex.value == "male")
        features["demo_female"] = float(demographics.sex.value == "female")
        
        # Weight
        weight = demographics.weight or 70
        features["demo_weight"] = weight
        
        return features
    
    def compute_context_features(self, context: ClinicalContext) -> Dict[str, float]:
        """
        Process and engineer clinical context features.
        """
        features = {}
        
        # ICU status (major risk factor)
        features["ctx_is_icu"] = float(context.is_icu)
        
        # Days since admission
        days = context.days_since_admission or 0
        features["ctx_days_since_admission"] = days
        features["ctx_prolonged_stay"] = float(days > 7)
        features["ctx_very_prolonged_stay"] = float(days > 14)
        
        # Sample source encoding
        source_map = {
            "blood": 0,
            "urine": 1,
            "sputum": 2,
            "wound": 3,
            "csf": 4,
            "other": 5
        }
        features["ctx_sample_source_code"] = source_map.get(context.sample_source.value, 5)
        
        # One-hot encode sample source
        for source in source_map.keys():
            features[f"ctx_source_{source}"] = float(context.sample_source.value == source)
        
        return features
    
    def compute_interaction_features(self, all_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute interaction features between different feature groups.
        These capture important clinical relationships.
        """
        interactions = {}
        
        # NLR * CRP interaction (inflammation markers)
        nlr = all_features.get("img_nlr", 0)
        crp = all_features.get("lab_crp", 0)
        interactions["int_nlr_crp"] = nlr * crp / 100  # Scaled
        
        # Bacterial load * WBC interaction
        bacterial = all_features.get("img_bacterial_cluster_count", 0)
        wbc = all_features.get("lab_wbc", 0)
        interactions["int_bacterial_wbc"] = bacterial * wbc
        
        # Fever * CRP interaction
        fever = all_features.get("vital_fever", 0)
        interactions["int_fever_crp"] = fever * crp
        
        # ICU * antibiotic exposure
        icu = all_features.get("ctx_is_icu", 0)
        ab_count = all_features.get("hist_antibiotic_count_90d", 0)
        interactions["int_icu_antibiotics"] = icu * ab_count
        
        # Age * comorbidities
        age = all_features.get("demo_age", 0)
        comorbidities = all_features.get("hist_comorbidity_count", 0)
        interactions["int_age_comorbidity"] = (age / 100) * comorbidities
        
        # Prior AMR * recent antibiotic
        prior_amr = all_features.get("hist_prior_amr", 0)
        recent_ab = all_features.get("hist_recent_antibiotic", 0)
        interactions["int_prior_amr_recent_ab"] = prior_amr * recent_ab
        
        # SIRS * lactate (sepsis indicator)
        sirs = all_features.get("vital_sirs_criteria", 0)
        lactate = all_features.get("lab_lactate", 0)
        interactions["int_sirs_lactate"] = sirs * lactate
        
        # Hospital stay * bacterial load
        stay = all_features.get("ctx_days_since_admission", 0)
        interactions["int_stay_bacterial"] = (stay / 10) * bacterial
        
        return interactions
    
    def transform(self, request: AMRPredictionRequest) -> Tuple[np.ndarray, List[str]]:
        """
        Transform a prediction request into model-ready features.
        
        Args:
            request: AMRPredictionRequest with all patient data
            
        Returns:
            Tuple of (feature_array, feature_names)
        """
        all_features = {}
        
        # Compute features from each data source
        all_features.update(self.compute_image_features(request.image_features))
        all_features.update(self.compute_vital_features(request.vitals))
        all_features.update(self.compute_lab_features(request.labs))
        all_features.update(self.compute_history_features(request.history))
        all_features.update(self.compute_demographic_features(request.demographics))
        all_features.update(self.compute_context_features(request.context))
        
        # Compute interaction features
        all_features.update(self.compute_interaction_features(all_features))
        
        # Sort features by name for consistent ordering
        feature_names = sorted(all_features.keys())
        feature_values = np.array([all_features[name] for name in feature_names])
        
        return feature_values, feature_names
    
    def transform_batch(self, requests: List[AMRPredictionRequest]) -> Tuple[np.ndarray, List[str]]:
        """
        Transform multiple requests into a feature matrix.
        
        Args:
            requests: List of AMRPredictionRequest objects
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not requests:
            return np.array([]), []
        
        feature_matrix = []
        feature_names = None
        
        for request in requests:
            features, names = self.transform(request)
            feature_matrix.append(features)
            
            if feature_names is None:
                feature_names = names
        
        return np.array(feature_matrix), feature_names
    
    def get_feature_category(self, feature_name: str) -> str:
        """Get the category of a feature based on its prefix"""
        prefix_map = {
            "img_": "image",
            "vital_": "vitals",
            "lab_": "labs",
            "hist_": "history",
            "demo_": "demographics",
            "ctx_": "context",
            "int_": "interaction"
        }
        
        for prefix, category in prefix_map.items():
            if feature_name.startswith(prefix):
                return category
        
        return "unknown"
