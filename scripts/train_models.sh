#!/bin/bash
# Train models script

set -e

# Activate virtual environment
if [ -f "backend/venv/bin/activate" ]; then
    source backend/venv/bin/activate
fi

cd backend

# Create output directory
mkdir -p models/trained

# Run training
echo "Starting model training..."
python -c "
from training.train_pipeline import AMRTrainingPipeline, create_synthetic_training_data
import pandas as pd

print('Creating synthetic training data...')
df = create_synthetic_training_data(n_samples=2000)
print(f'Dataset size: {len(df)} samples')

# Train for each antibiotic
antibiotics = ['ceftriaxone', 'ciprofloxacin', 'meropenem', 'vancomycin', 
               'piperacillin_tazobactam', 'gentamicin', 'azithromycin', 'cefepime']

pipeline = AMRTrainingPipeline(
    target_antibiotics=antibiotics,
    model_type='xgboost'
)

print('Preparing data with patient-wise splits...')
splits = pipeline.prepare_data(
    df,
    label_column='resistant',
    patient_id_column='patient_id',
    temporal_split=False
)

print('Training models...')
pipeline.train_all(splits, calibrate=True)

print('Generating report...')
report = pipeline.generate_report()
print('\\n=== Training Complete ===')
print(f\"Mean AUC-ROC: {report['summary']['auc_roc']['mean']:.3f} ± {report['summary']['auc_roc']['std']:.3f}\")
print(f\"Mean AUC-PR: {report['summary']['auc_pr']['mean']:.3f} ± {report['summary']['auc_pr']['std']:.3f}\")

print('\\nSaving models...')
pipeline.save_models(version='1.0.0')
print('Models saved to backend/models/trained/')
"

echo "Training complete!"
