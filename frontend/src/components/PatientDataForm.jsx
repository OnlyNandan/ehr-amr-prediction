import React, { useState } from 'react';
import { Box, Paper, Typography, Grid, TextField, Button, Chip, Switch, FormControlLabel, CircularProgress, MenuItem } from '@mui/material';

const API_URL = 'http://localhost:5050/api/v1';

const PatientDataForm = ({ imageFeatures, onPredictions }) => {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    age: 45, sex: 'male', weight: 70,
    temperature: 37, heartRate: 80, oxygenSaturation: 98,
    wbc: 8, crp: 5, lactate: 1, creatinine: 1,
    isIcu: false, priorAmrPositive: false,
  });
  const [selectedAntibiotics, setSelectedAntibiotics] = useState(['ceftriaxone', 'ciprofloxacin', 'meropenem']);

  const antibiotics = [
    { id: 'ceftriaxone', label: 'Ceftriaxone' },
    { id: 'ciprofloxacin', label: 'Ciprofloxacin' },
    { id: 'meropenem', label: 'Meropenem' },
    { id: 'vancomycin', label: 'Vancomycin' },
    { id: 'piperacillin_tazobactam', label: 'Piperacillin/Tazobactam' },
    { id: 'gentamicin', label: 'Gentamicin' },
  ];

  const handleChange = (field) => (e) => {
    setFormData(prev => ({ ...prev, [field]: e.target.type === 'checkbox' ? e.target.checked : e.target.value }));
  };

  const toggleAntibiotic = (id) => {
    setSelectedAntibiotics(prev => prev.includes(id) ? prev.filter(a => a !== id) : [...prev, id]);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const request = {
        patient_id: `PT_${Date.now()}`,
        sample_time: new Date().toISOString(),
        target_antibiotics: selectedAntibiotics,
        image_features: imageFeatures || {
          neutrophil_count: 10, lymphocyte_count: 3, monocyte_count: 1,
          eosinophil_count: 0.5, basophil_count: 0.1, platelet_estimate: 200000,
          parasite_present: false, bacterial_cluster_count: 0,
          mean_bacterial_bbox_confidence: 0,
          rbc_morphology_anisocytosis: false,
          rbc_morphology_poikilocytosis: false,
          rbc_morphology_hypochromia: false,
        },
        vitals: {
          temperature: parseFloat(formData.temperature),
          heart_rate: parseInt(formData.heartRate),
          respiratory_rate: 16,
          bp_systolic: 120, bp_diastolic: 80,
          oxygen_saturation: parseFloat(formData.oxygenSaturation),
        },
        labs: {
          wbc: parseFloat(formData.wbc), crp: parseFloat(formData.crp),
          lactate: parseFloat(formData.lactate), creatinine: parseFloat(formData.creatinine),
          ast: 25, alt: 20, hemoglobin: 14, platelets: 250,
        },
        history: {
          prior_antibiotics: [], days_since_last_antibiotic: 30,
          prior_hospitalizations_30d: 0, antibiotic_exposure_count_90d: 0,
          prior_amr_positive: formData.priorAmrPositive, comorbidities: [],
          immunocompromised: false, chronic_kidney_disease: false, diabetes: false,
        },
        demographics: { age: parseInt(formData.age), sex: formData.sex, weight: parseFloat(formData.weight) },
        context: {
          is_icu: formData.isIcu, hospital_id: 'HOSP001', ward: 'General',
          sample_source: 'blood', days_since_admission: 1,
        },
      };

      const response = await fetch(`${API_URL}/predict_amr`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) throw new Error('Prediction failed');
      const result = await response.json();
      onPredictions(result);
    } catch (error) {
      onPredictions({ error: true, message: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 3, bgcolor: '#fff' }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#333', fontWeight: 600 }}>
        Patient Information
      </Typography>
      <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
        Enter clinical data for AMR risk prediction
      </Typography>

      {/* Antibiotics */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: '#333' }}>Target Antibiotics</Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
        {antibiotics.map((ab) => (
          <Chip
            key={ab.id}
            label={ab.label}
            onClick={() => toggleAntibiotic(ab.id)}
            color={selectedAntibiotics.includes(ab.id) ? 'primary' : 'default'}
            variant={selectedAntibiotics.includes(ab.id) ? 'filled' : 'outlined'}
          />
        ))}
      </Box>

      {/* Demographics */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: '#333' }}>Demographics</Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={4}>
          <TextField label="Age" type="number" value={formData.age} onChange={handleChange('age')} size="small" fullWidth />
        </Grid>
        <Grid item xs={4}>
          <TextField label="Sex" select value={formData.sex} onChange={handleChange('sex')} size="small" fullWidth>
            <MenuItem value="male">Male</MenuItem>
            <MenuItem value="female">Female</MenuItem>
          </TextField>
        </Grid>
        <Grid item xs={4}>
          <TextField label="Weight (kg)" type="number" value={formData.weight} onChange={handleChange('weight')} size="small" fullWidth />
        </Grid>
      </Grid>

      {/* Vitals */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: '#333' }}>Vital Signs</Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={4}>
          <TextField label="Temp (°C)" type="number" value={formData.temperature} onChange={handleChange('temperature')} size="small" fullWidth />
        </Grid>
        <Grid item xs={4}>
          <TextField label="HR (bpm)" type="number" value={formData.heartRate} onChange={handleChange('heartRate')} size="small" fullWidth />
        </Grid>
        <Grid item xs={4}>
          <TextField label="SpO2 (%)" type="number" value={formData.oxygenSaturation} onChange={handleChange('oxygenSaturation')} size="small" fullWidth />
        </Grid>
      </Grid>

      {/* Labs */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: '#333' }}>Labs</Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={3}>
          <TextField label="WBC (K/μL)" type="number" value={formData.wbc} onChange={handleChange('wbc')} size="small" fullWidth />
        </Grid>
        <Grid item xs={3}>
          <TextField label="CRP (mg/L)" type="number" value={formData.crp} onChange={handleChange('crp')} size="small" fullWidth />
        </Grid>
        <Grid item xs={3}>
          <TextField label="Lactate" type="number" value={formData.lactate} onChange={handleChange('lactate')} size="small" fullWidth />
        </Grid>
        <Grid item xs={3}>
          <TextField label="Creatinine" type="number" value={formData.creatinine} onChange={handleChange('creatinine')} size="small" fullWidth />
        </Grid>
      </Grid>

      {/* Switches */}
      <Box sx={{ display: 'flex', gap: 3, mb: 3 }}>
        <FormControlLabel
          control={<Switch checked={formData.isIcu} onChange={handleChange('isIcu')} />}
          label="ICU Patient"
        />
        <FormControlLabel
          control={<Switch checked={formData.priorAmrPositive} onChange={handleChange('priorAmrPositive')} />}
          label="Prior AMR+"
        />
      </Box>

      <Button variant="contained" fullWidth onClick={handleSubmit} disabled={loading} size="large">
        {loading ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
        Predict AMR Risk
      </Button>
    </Paper>
  );
};

export default PatientDataForm;
