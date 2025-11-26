-- AMR Prediction Database Schema
-- PostgreSQL initialization script

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(100) UNIQUE NOT NULL,
    patient_id VARCHAR(100) NOT NULL,
    sample_time TIMESTAMP NOT NULL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) NOT NULL,
    overall_risk_level VARCHAR(20) NOT NULL,
    clinical_summary TEXT,
    inference_latency_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Antibiotic predictions (one per antibiotic per prediction)
CREATE TABLE IF NOT EXISTS antibiotic_predictions (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(100) REFERENCES predictions(prediction_id),
    antibiotic VARCHAR(100) NOT NULL,
    probability FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    calibrated BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature contributions (SHAP values)
CREATE TABLE IF NOT EXISTS feature_contributions (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(100) REFERENCES predictions(prediction_id),
    antibiotic VARCHAR(100) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value FLOAT,
    shap_value FLOAT NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    patient_id VARCHAR(100),
    user_id VARCHAR(100),
    prediction_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    metadata JSONB,
    ip_address VARCHAR(50)
);

-- Clinical overrides
CREATE TABLE IF NOT EXISTS clinical_overrides (
    id SERIAL PRIMARY KEY,
    override_id VARCHAR(100) UNIQUE NOT NULL,
    prediction_id VARCHAR(100) REFERENCES predictions(prediction_id),
    original_decision VARCHAR(100),
    override_decision VARCHAR(100) NOT NULL,
    reason TEXT NOT NULL,
    clinician_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    antibiotic VARCHAR(100) NOT NULL,
    evaluation_date DATE NOT NULL,
    auc_roc FLOAT,
    auc_pr FLOAT,
    f1_score FLOAT,
    brier_score FLOAT,
    n_samples INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ground truth outcomes (for retrospective analysis)
CREATE TABLE IF NOT EXISTS ground_truth (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(100) REFERENCES predictions(prediction_id),
    patient_id VARCHAR(100) NOT NULL,
    antibiotic VARCHAR(100) NOT NULL,
    culture_result VARCHAR(50),  -- 'resistant', 'susceptible', 'intermediate'
    organism VARCHAR(200),
    culture_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_antibiotic_pred ON antibiotic_predictions(prediction_id, antibiotic);
CREATE INDEX IF NOT EXISTS idx_ground_truth_pred ON ground_truth(prediction_id);

-- Views for reporting
CREATE OR REPLACE VIEW v_prediction_summary AS
SELECT 
    p.prediction_id,
    p.patient_id,
    p.prediction_time,
    p.overall_risk_level,
    COUNT(ap.id) as antibiotics_tested,
    AVG(ap.probability) as avg_probability,
    MAX(ap.probability) as max_probability
FROM predictions p
LEFT JOIN antibiotic_predictions ap ON p.prediction_id = ap.prediction_id
GROUP BY p.prediction_id, p.patient_id, p.prediction_time, p.overall_risk_level;

CREATE OR REPLACE VIEW v_model_accuracy AS
SELECT 
    ap.antibiotic,
    COUNT(*) as total_predictions,
    AVG(CASE WHEN (ap.probability >= 0.5 AND gt.culture_result = 'resistant') 
             OR (ap.probability < 0.5 AND gt.culture_result = 'susceptible') 
        THEN 1.0 ELSE 0.0 END) as accuracy
FROM antibiotic_predictions ap
JOIN ground_truth gt ON ap.prediction_id = gt.prediction_id AND ap.antibiotic = gt.antibiotic
WHERE gt.culture_result IN ('resistant', 'susceptible')
GROUP BY ap.antibiotic;
