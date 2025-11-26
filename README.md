# Multimodal AMR Prediction System

A comprehensive system for predicting Antimicrobial Resistance (AMR) using multimodal data combining blood smear image analysis with Electronic Health Records (EHR).

## Overview

This system predicts the probability of antimicrobial resistance for target antibiotics using:

1. **Blood Smear Image Features** - YOLO-based cell detection (neutrophils, lymphocytes, bacterial clusters, parasites)
2. **Admission Vitals** - Temperature, heart rate, respiratory rate, blood pressure, SpO2
3. **Laboratory Results** - WBC, CRP, lactate, creatinine, liver enzymes
4. **Patient History** - Prior antibiotic exposures, hospitalizations, AMR history
5. **Demographics** - Age, sex, weight
6. **Clinical Context** - ICU status, ward, sample source

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│                    AMR Prediction Dashboard                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ REST API
┌─────────────────────────▼───────────────────────────────────────┐
│                    Flask API Gateway                            │
│              /predict_image  /predict_amr                       │
└───────┬─────────────────────────────────────────┬───────────────┘
        │                                         │
┌───────▼───────┐                        ┌────────▼────────┐
│ YOLO Service  │                        │  AMR Predictor  │
│ Blood Smear   │                        │   XGBoost/MLP   │
│ Analysis      │                        │   + Calibrator  │
└───────────────┘                        └────────┬────────┘
                                                  │
                                         ┌────────▼────────┐
                                         │ SHAP Explainer  │
                                         │ Feature Import. │
                                         └─────────────────┘
```

## Features

### Machine Learning
- **Multiple model support**: XGBoost, LightGBM, MLP, Stacking ensemble
- **Patient-wise train/val/test splits** to prevent data leakage
- **Temporal holdout validation** for realistic performance estimation
- **Probability calibration** using Isotonic or Platt scaling
- **SHAP-based explanations** for interpretable predictions

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict_image` | POST | Analyze blood smear image |
| `/api/v1/predict_amr` | POST | Predict AMR probabilities |
| `/api/v1/predict_amr/batch` | POST | Batch prediction |
| `/api/v1/thresholds` | GET | Decision threshold analysis |
| `/api/v1/model/performance` | GET | Model performance metrics |
| `/api/v1/override` | POST | Record clinical override |
| `/metrics` | GET | Prometheus metrics |

### Monitoring & Compliance
- Prometheus metrics for latency, prediction distribution
- Data drift detection (PSI, KS tests)
- HIPAA/GDPR compliant audit logging
- Clinical override tracking

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (optional)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f amr-api

# Stop services
docker-compose down
```

## API Usage

### Predict AMR Risk

```python
import requests

response = requests.post("http://localhost:5000/api/v1/predict_amr", json={
    "patient_id": "PT_12345",
    "sample_time": "2025-11-26T10:30:00Z",
    "target_antibiotics": ["ceftriaxone", "ciprofloxacin"],
    
    "image_features": {
        "neutrophil_count": 12.5,
        "lymphocyte_count": 3.2,
        "bacterial_cluster_count": 2,
        "mean_bacterial_bbox_confidence": 0.85,
        "platelet_estimate": 180000,
        "parasite_present": False
    },
    
    "vitals": {
        "temperature": 38.5,
        "heart_rate": 95,
        "respiratory_rate": 20,
        "bp_systolic": 125,
        "bp_diastolic": 82,
        "oxygen_saturation": 96.0
    },
    
    "labs": {
        "wbc": 14.2,
        "crp": 85.0,
        "lactate": 2.1,
        "creatinine": 1.2
    },
    
    "history": {
        "days_since_last_antibiotic": 8,
        "prior_hospitalizations_30d": 1,
        "antibiotic_exposure_count_90d": 2,
        "prior_amr_positive": False
    },
    
    "demographics": {
        "age": 65,
        "sex": "male",
        "weight": 78.5
    },
    
    "context": {
        "is_icu": True,
        "sample_source": "blood",
        "days_since_admission": 3
    }
})

result = response.json()
print(f"Risk Level: {result['overall_risk_level']}")
for pred in result['predictions']:
    print(f"  {pred['antibiotic']}: {pred['probability']:.1%} ({pred['risk_level']})")
```

### Response Format

```json
{
  "patient_id": "PT_12345",
  "prediction_id": "PRED_20251126103015_abc123",
  "prediction_time": "2025-11-26T10:30:15Z",
  "model_version": "1.0.0",
  "overall_risk_level": "high",
  "clinical_summary": "High probability of ceftriaxone resistance detected...",
  "predictions": [
    {
      "antibiotic": "ceftriaxone",
      "probability": 0.73,
      "risk_level": "high",
      "calibrated": true,
      "top_contributing_features": [
        {"feature_name": "bacterial_cluster_count", "shap_value": 0.15, "category": "image"},
        {"feature_name": "crp", "shap_value": 0.12, "category": "labs"}
      ]
    }
  ],
  "inference_latency_ms": 127.5
}
```

## Model Training

```python
from training.train_pipeline import AMRTrainingPipeline, create_synthetic_training_data

# Create synthetic data for testing
df = create_synthetic_training_data(n_samples=1000)

# Initialize pipeline
pipeline = AMRTrainingPipeline(
    target_antibiotics=["ceftriaxone", "ciprofloxacin", "meropenem"],
    model_type="xgboost"  # or "lightgbm", "mlp", "stacking"
)

# Prepare data with patient-wise split
splits = pipeline.prepare_data(
    df,
    label_column="resistant",
    patient_id_column="patient_id",
    temporal_split=False
)

# Train models with calibration
pipeline.train_all(splits, calibrate=True)

# Generate report
report = pipeline.generate_report()
print(f"Mean AUC-ROC: {report['summary']['auc_roc']['mean']:.3f}")

# Save models
pipeline.save_models(version="1.0.0")
```

## Project Structure

```
ehr-amr-prediction/
├── backend/
│   ├── api/
│   │   ├── routes.py          # Flask API endpoints
│   │   └── middleware.py      # Request/response handling
│   ├── models/
│   │   └── schemas.py         # Pydantic data models
│   ├── services/
│   │   ├── yolo_inference.py  # Blood smear analysis
│   │   ├── feature_engineering.py
│   │   ├── amr_predictor.py   # Main prediction engine
│   │   └── explainability.py  # SHAP explanations
│   ├── training/
│   │   └── train_pipeline.py  # Model training
│   ├── utils/
│   │   ├── logging_config.py
│   │   └── monitoring.py
│   ├── app.py                 # Flask application
│   ├── config.py              # Configuration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── PredictionForm.jsx
│   │   │   └── FeatureContributionChart.jsx
│   │   └── services/
│   │       └── api.js
│   └── package.json
├── monitoring/
│   └── prometheus.yml
├── docker-compose.yml
└── README.md
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | development | Environment mode |
| `SECRET_KEY` | - | Flask secret key |
| `YOLO_MODEL_PATH` | models/trained/yolo_blood_smear.pt | YOLO model path |
| `AMR_MODEL_PATH` | models/trained/amr_model.joblib | AMR model path |
| `LOG_LEVEL` | INFO | Logging level |
| `CORS_ORIGINS` | * | Allowed CORS origins |

## Evaluation Metrics

The system reports comprehensive metrics:

- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve
- **Brier Score**: Calibration quality
- **Precision@K**: Precision at fixed recall levels
- **F1, Precision, Recall**: Classification metrics

## Clinical Considerations

⚠️ **This system is designed as a clinical decision support tool, not a replacement for clinical judgment.**

- Always confirm with culture and sensitivity results
- Consider patient-specific factors not captured by the model
- Use clinical override functionality when appropriate
- Consult infectious disease specialists for complex cases

## Safety & Governance

- Model outputs should be validated retrospectively before clinical deployment
- Audit logs track all predictions and overrides
- Data drift monitoring alerts to potential performance degradation
- PHI de-identification ensures HIPAA/GDPR compliance

## License

This project is for educational/research purposes. Clinical deployment requires appropriate regulatory approval.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
