"""
Tests for AMR Prediction API
"""
import pytest
import json
from datetime import datetime

from app import create_app
from config import TestingConfig


@pytest.fixture
def app():
    """Create application for testing"""
    app = create_app(TestingConfig())
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test main health endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
    
    def test_api_health(self, client):
        """Test API health endpoint"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
    
    def test_info_endpoint(self, client):
        """Test info endpoint"""
        response = client.get('/api/v1/info')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'model_version' in data
        assert 'supported_antibiotics' in data


class TestImageAnalysis:
    """Test image analysis endpoints"""
    
    def test_predict_image_missing_data(self, client):
        """Test image prediction with missing data"""
        response = client.post('/api/v1/predict_image',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_predict_image_invalid_base64(self, client):
        """Test image prediction with invalid base64"""
        response = client.post('/api/v1/predict_image',
            data=json.dumps({'image_base64': 'not-valid-base64!@#'}),
            content_type='application/json'
        )
        assert response.status_code == 400


class TestAMRPrediction:
    """Test AMR prediction endpoints"""
    
    @pytest.fixture
    def valid_prediction_request(self):
        """Create valid prediction request"""
        return {
            "patient_id": "TEST_PT_001",
            "sample_time": datetime.utcnow().isoformat() + "Z",
            "target_antibiotics": ["ceftriaxone"],
            "image_features": {
                "neutrophil_count": 10.0,
                "lymphocyte_count": 3.0,
                "monocyte_count": 1.0,
                "eosinophil_count": 0.5,
                "basophil_count": 0.1,
                "platelet_estimate": 200000,
                "parasite_present": False,
                "bacterial_cluster_count": 1,
                "mean_bacterial_bbox_confidence": 0.8,
                "rbc_morphology_anisocytosis": False,
                "rbc_morphology_poikilocytosis": False,
                "rbc_morphology_hypochromia": False
            },
            "vitals": {
                "temperature": 38.0,
                "heart_rate": 90,
                "respiratory_rate": 18,
                "bp_systolic": 120,
                "bp_diastolic": 80,
                "oxygen_saturation": 97.0
            },
            "labs": {
                "wbc": 12.0,
                "crp": 50.0,
                "lactate": 1.5,
                "creatinine": 1.0,
                "ast": 30,
                "alt": 25,
                "hemoglobin": 13.0,
                "platelets": 200
            },
            "history": {
                "prior_antibiotics": [],
                "days_since_last_antibiotic": 30,
                "prior_hospitalizations_30d": 0,
                "antibiotic_exposure_count_90d": 1,
                "prior_amr_positive": False,
                "comorbidities": [],
                "immunocompromised": False,
                "chronic_kidney_disease": False,
                "diabetes": False
            },
            "demographics": {
                "age": 55,
                "sex": "male",
                "weight": 75.0
            },
            "context": {
                "is_icu": False,
                "hospital_id": "TEST_HOSP",
                "ward": "General",
                "sample_source": "blood",
                "days_since_admission": 1
            }
        }
    
    def test_predict_amr_success(self, client, valid_prediction_request):
        """Test successful AMR prediction"""
        response = client.post('/api/v1/predict_amr',
            data=json.dumps(valid_prediction_request),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert 'prediction_id' in data
        assert 'predictions' in data
        assert 'overall_risk_level' in data
        assert 'clinical_summary' in data
        assert 'inference_latency_ms' in data
        
        # Check predictions
        assert len(data['predictions']) > 0
        for pred in data['predictions']:
            assert 'antibiotic' in pred
            assert 'probability' in pred
            assert 0 <= pred['probability'] <= 1
            assert 'risk_level' in pred
    
    def test_predict_amr_missing_fields(self, client):
        """Test AMR prediction with missing required fields"""
        response = client.post('/api/v1/predict_amr',
            data=json.dumps({'patient_id': 'test'}),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_predict_amr_invalid_vitals(self, client, valid_prediction_request):
        """Test AMR prediction with invalid vital signs"""
        valid_prediction_request['vitals']['temperature'] = 50  # Invalid temperature
        response = client.post('/api/v1/predict_amr',
            data=json.dumps(valid_prediction_request),
            content_type='application/json'
        )
        assert response.status_code == 400


class TestThresholds:
    """Test threshold endpoints"""
    
    def test_get_thresholds(self, client):
        """Test getting decision thresholds"""
        response = client.get('/api/v1/thresholds')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'antibiotics' in data


class TestModelInfo:
    """Test model information endpoints"""
    
    def test_get_features(self, client):
        """Test getting model features"""
        response = client.get('/api/v1/model/features')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'image_features' in data
        assert 'vital_features' in data
        assert 'lab_features' in data
    
    def test_get_performance(self, client):
        """Test getting model performance"""
        response = client.get('/api/v1/model/performance')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'model_version' in data
        assert 'metrics_by_antibiotic' in data


class TestClinicalOverride:
    """Test clinical override functionality"""
    
    def test_record_override(self, client):
        """Test recording clinical override"""
        response = client.post('/api/v1/override',
            data=json.dumps({
                'prediction_id': 'PRED_TEST_001',
                'override_decision': 'prescribe_alternative',
                'reason': 'Clinical judgment based on patient history'
            }),
            content_type='application/json',
            headers={'X-User-ID': 'DR_001'}
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'recorded'
    
    def test_override_missing_fields(self, client):
        """Test override with missing required fields"""
        response = client.post('/api/v1/override',
            data=json.dumps({'prediction_id': 'PRED_TEST_001'}),
            content_type='application/json'
        )
        assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
