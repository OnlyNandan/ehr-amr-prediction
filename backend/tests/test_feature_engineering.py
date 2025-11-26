"""
Tests for Feature Engineering Module
"""
import pytest
import numpy as np

from services.feature_engineering import FeatureEngineer
from models.schemas import (
    ImageFeatures,
    Vitals,
    LabResults,
    PatientHistory,
    Demographics,
    ClinicalContext,
    AMRPredictionRequest,
    SexEnum,
    SampleSourceEnum
)
from datetime import datetime


@pytest.fixture
def feature_engineer():
    """Create feature engineer instance"""
    return FeatureEngineer()


@pytest.fixture
def sample_request():
    """Create sample prediction request"""
    return AMRPredictionRequest(
        patient_id="TEST_001",
        sample_time=datetime.utcnow(),
        image_features=ImageFeatures(
            neutrophil_count=12.0,
            lymphocyte_count=3.0,
            monocyte_count=1.0,
            eosinophil_count=0.5,
            basophil_count=0.1,
            platelet_estimate=200000,
            parasite_present=False,
            bacterial_cluster_count=2,
            mean_bacterial_bbox_confidence=0.85,
            rbc_morphology_anisocytosis=False,
            rbc_morphology_poikilocytosis=False,
            rbc_morphology_hypochromia=False
        ),
        vitals=Vitals(
            temperature=38.5,
            heart_rate=95,
            respiratory_rate=20,
            bp_systolic=120,
            bp_diastolic=80,
            oxygen_saturation=96.0
        ),
        labs=LabResults(
            wbc=14.0,
            crp=85.0,
            lactate=2.0,
            creatinine=1.2,
            ast=45,
            alt=38,
            hemoglobin=12.5,
            platelets=200
        ),
        history=PatientHistory(
            prior_antibiotics=[],
            days_since_last_antibiotic=15,
            prior_hospitalizations_30d=1,
            antibiotic_exposure_count_90d=2,
            prior_amr_positive=False,
            comorbidities=["hypertension"],
            immunocompromised=False,
            chronic_kidney_disease=False,
            diabetes=True
        ),
        demographics=Demographics(
            age=65,
            sex=SexEnum.MALE,
            weight=78.0
        ),
        context=ClinicalContext(
            is_icu=True,
            hospital_id="HOSP001",
            ward="ICU-A",
            sample_source=SampleSourceEnum.BLOOD,
            days_since_admission=3
        )
    )


class TestImageFeatures:
    """Test image feature computation"""
    
    def test_nlr_computation(self, feature_engineer, sample_request):
        """Test NLR is correctly computed"""
        features = feature_engineer.compute_image_features(sample_request.image_features)
        
        expected_nlr = 12.0 / 3.0  # neutrophil / lymphocyte
        assert features['img_nlr'] == pytest.approx(expected_nlr, rel=0.01)
    
    def test_nlr_categories(self, feature_engineer, sample_request):
        """Test NLR category flags"""
        features = feature_engineer.compute_image_features(sample_request.image_features)
        
        # NLR = 4.0, should be elevated but not critical
        assert features['img_nlr_elevated'] == 1.0  # > 3
        assert features['img_nlr_high'] == 0.0  # not > 6
        assert features['img_nlr_critical'] == 0.0  # not > 10
    
    def test_bacterial_presence(self, feature_engineer, sample_request):
        """Test bacterial presence detection"""
        features = feature_engineer.compute_image_features(sample_request.image_features)
        
        assert features['img_bacteria_present'] == 1.0
        assert features['img_bacterial_cluster_count'] == 2


class TestVitalFeatures:
    """Test vital signs feature computation"""
    
    def test_fever_detection(self, feature_engineer, sample_request):
        """Test fever flags are set correctly"""
        features = feature_engineer.compute_vital_features(sample_request.vitals)
        
        assert features['vital_fever'] == 1.0  # temp >= 38.0
        assert features['vital_high_fever'] == 0.0  # temp < 39.0
    
    def test_sirs_criteria(self, feature_engineer, sample_request):
        """Test SIRS criteria calculation"""
        features = feature_engineer.compute_vital_features(sample_request.vitals)
        
        # temp > 38, hr > 90, rr == 20
        # Should have at least 2 SIRS criteria
        assert features['vital_sirs_criteria'] >= 2
        assert features['vital_sirs_positive'] == 1.0
    
    def test_map_calculation(self, feature_engineer, sample_request):
        """Test mean arterial pressure calculation"""
        features = feature_engineer.compute_vital_features(sample_request.vitals)
        
        # MAP = (SBP + 2*DBP) / 3 = (120 + 160) / 3
        expected_map = (120 + 2 * 80) / 3
        assert features['vital_map'] == pytest.approx(expected_map, rel=0.01)


class TestLabFeatures:
    """Test laboratory feature computation"""
    
    def test_crp_categories(self, feature_engineer, sample_request):
        """Test CRP category flags"""
        features = feature_engineer.compute_lab_features(sample_request.labs)
        
        # CRP = 85, should be elevated and high
        assert features['lab_crp_elevated'] == 1.0  # > 10
        assert features['lab_crp_high'] == 1.0  # > 50
        assert features['lab_crp_critical'] == 0.0  # not > 100
    
    def test_leukocytosis(self, feature_engineer, sample_request):
        """Test leukocytosis detection"""
        features = feature_engineer.compute_lab_features(sample_request.labs)
        
        # WBC = 14, should be leukocytosis but not severe
        assert features['lab_leukocytosis'] == 1.0  # > 12
        assert features['lab_severe_leukocytosis'] == 0.0  # not > 20


class TestHistoryFeatures:
    """Test patient history feature computation"""
    
    def test_antibiotic_exposure(self, feature_engineer, sample_request):
        """Test antibiotic exposure features"""
        features = feature_engineer.compute_history_features(sample_request.history)
        
        assert features['hist_days_since_antibiotic'] == 15
        assert features['hist_recent_antibiotic'] == 1.0  # < 30 days
        assert features['hist_antibiotic_count_90d'] == 2
        assert features['hist_multiple_antibiotics'] == 1.0
    
    def test_comorbidities(self, feature_engineer, sample_request):
        """Test comorbidity features"""
        features = feature_engineer.compute_history_features(sample_request.history)
        
        assert features['hist_diabetes'] == 1.0
        assert features['hist_immunocompromised'] == 0.0
        assert features['hist_comorbidity_count'] == 1


class TestDemographicFeatures:
    """Test demographic feature computation"""
    
    def test_age_categories(self, feature_engineer, sample_request):
        """Test age category flags"""
        features = feature_engineer.compute_demographic_features(sample_request.demographics)
        
        # Age = 65
        assert features['demo_elderly'] == 1.0  # >= 65
        assert features['demo_very_elderly'] == 0.0  # not >= 80
        assert features['demo_age_65_80'] == 1.0
    
    def test_sex_encoding(self, feature_engineer, sample_request):
        """Test sex encoding"""
        features = feature_engineer.compute_demographic_features(sample_request.demographics)
        
        assert features['demo_male'] == 1.0
        assert features['demo_female'] == 0.0


class TestContextFeatures:
    """Test clinical context feature computation"""
    
    def test_icu_flag(self, feature_engineer, sample_request):
        """Test ICU status"""
        features = feature_engineer.compute_context_features(sample_request.context)
        
        assert features['ctx_is_icu'] == 1.0
    
    def test_sample_source_encoding(self, feature_engineer, sample_request):
        """Test sample source encoding"""
        features = feature_engineer.compute_context_features(sample_request.context)
        
        assert features['ctx_source_blood'] == 1.0
        assert features['ctx_source_urine'] == 0.0


class TestInteractionFeatures:
    """Test interaction feature computation"""
    
    def test_nlr_crp_interaction(self, feature_engineer, sample_request):
        """Test NLR * CRP interaction"""
        features, _ = feature_engineer.transform(sample_request)
        feature_dict = dict(zip(_, features)) if isinstance(features, np.ndarray) else features
        
        # This tests the full transform which includes interactions
        assert len(features) > 50  # Should have many features
    
    def test_transform_output_shape(self, feature_engineer, sample_request):
        """Test transform output is consistent"""
        features, names = feature_engineer.transform(sample_request)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == len(names)
        assert all(isinstance(n, str) for n in names)


class TestFeatureCategories:
    """Test feature category detection"""
    
    def test_category_detection(self, feature_engineer):
        """Test feature category is correctly identified"""
        assert feature_engineer.get_feature_category('img_nlr') == 'image'
        assert feature_engineer.get_feature_category('vital_temp') == 'vitals'
        assert feature_engineer.get_feature_category('lab_wbc') == 'labs'
        assert feature_engineer.get_feature_category('hist_prior_amr') == 'history'
        assert feature_engineer.get_feature_category('demo_age') == 'demographics'
        assert feature_engineer.get_feature_category('ctx_is_icu') == 'context'
        assert feature_engineer.get_feature_category('int_nlr_crp') == 'interaction'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
