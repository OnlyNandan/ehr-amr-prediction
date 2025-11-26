const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api/v1';

/**
 * Analyze blood smear image for cell counting
 */
export const analyzeBloodSmear = async (imageFile) => {
  try {
    // Convert image to base64
    const base64 = await fileToBase64(imageFile);

    const response = await fetch(`${API_BASE_URL}/predict_image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_base64: base64.split(',')[1], // Remove data:image/xxx;base64, prefix
        filename: imageFile.name,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    // Transform the response to our expected format
    return {
      rbc_count: data.rbc_count || data.cell_counts?.rbc || 0,
      wbc_count: data.wbc_count || data.cell_counts?.wbc || 0,
      total_rbc: data.total_rbc || data.cell_counts?.total_rbc || 0,
      total_wbc: data.total_wbc || data.cell_counts?.total_wbc || 0,
      platelet_count: data.platelet_count || data.cell_counts?.platelets || 0,
      platelets: data.platelets || data.platelet_estimate || 0,
      platelet_estimate: data.platelet_estimate || 0,

      // WBC Differential
      neutrophil_count: data.neutrophil_count || data.differential?.neutrophils || 0,
      lymphocyte_count: data.lymphocyte_count || data.differential?.lymphocytes || 0,
      monocyte_count: data.monocyte_count || data.differential?.monocytes || 0,
      eosinophil_count: data.eosinophil_count || data.differential?.eosinophils || 0,
      basophil_count: data.basophil_count || data.differential?.basophils || 0,

      // Abnormalities
      bacterial_cluster_count: data.bacterial_cluster_count || data.abnormalities?.bacteria || 0,
      bacteria_count: data.bacteria_count || 0,
      parasite_count: data.parasite_count || data.abnormalities?.parasites || 0,
      parasite_present: data.parasite_present || (data.parasite_count > 0) || false,

      // Morphology
      mean_bacterial_bbox_confidence: data.mean_bacterial_bbox_confidence || 0,
      rbc_morphology_anisocytosis: data.rbc_morphology_anisocytosis || false,
      rbc_morphology_poikilocytosis: data.rbc_morphology_poikilocytosis || false,
      rbc_morphology_hypochromia: data.rbc_morphology_hypochromia || false,

      // Quality
      image_quality: data.image_quality || 'good',
      analysis_confidence: data.confidence || 0.85,

      // Raw response for debugging
      _raw: data,
    };
  } catch (error) {
    console.error('Blood smear analysis failed:', error);

    // Return mock data for demo purposes when API is unavailable
    if (error.message.includes('fetch') || error.message.includes('NetworkError') || error.message.includes('Failed')) {
      console.warn('API unavailable, returning simulated analysis');
      return generateMockAnalysis();
    }

    throw error;
  }
};

/**
 * Predict AMR risk
 */
export const predictAMR = async (requestData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict_amr`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': `req_${Date.now()}`,
      },
      body: JSON.stringify(requestData),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('AMR prediction failed:', error);

    // Return mock prediction for demo when API unavailable
    if (error.message.includes('fetch') || error.message.includes('NetworkError') || error.message.includes('Failed')) {
      console.warn('API unavailable, returning simulated prediction');
      return generateMockPrediction(requestData);
    }

    throw error;
  }
};

/**
 * Get model thresholds
 */
export const getThresholds = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/thresholds`);
    if (!response.ok) throw new Error('Failed to fetch thresholds');
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch thresholds:', error);
    return null;
  }
};

/**
 * Health check
 */
export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
};

// Helper functions

const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });
};

const generateMockAnalysis = () => {
  // Simulate realistic blood smear analysis
  const neutrophils = 8 + Math.random() * 6;
  const lymphocytes = 2 + Math.random() * 3;
  const monocytes = 0.5 + Math.random() * 1;
  const eosinophils = 0.1 + Math.random() * 0.5;
  const basophils = 0.05 + Math.random() * 0.15;

  return {
    rbc_count: Math.floor(4.2 + Math.random() * 1.5) * 1000000,
    wbc_count: Math.floor((neutrophils + lymphocytes + monocytes + eosinophils + basophils) * 1000),
    total_rbc: Math.floor(4.2 + Math.random() * 1.5),
    total_wbc: (neutrophils + lymphocytes + monocytes + eosinophils + basophils).toFixed(1),
    platelet_count: Math.floor(150 + Math.random() * 250),
    platelets: Math.floor(150 + Math.random() * 250),
    platelet_estimate: Math.floor(150000 + Math.random() * 250000),

    neutrophil_count: parseFloat(neutrophils.toFixed(1)),
    lymphocyte_count: parseFloat(lymphocytes.toFixed(1)),
    monocyte_count: parseFloat(monocytes.toFixed(1)),
    eosinophil_count: parseFloat(eosinophils.toFixed(2)),
    basophil_count: parseFloat(basophils.toFixed(2)),

    bacterial_cluster_count: Math.random() > 0.7 ? Math.floor(Math.random() * 3) : 0,
    bacteria_count: Math.random() > 0.7 ? Math.floor(Math.random() * 5) : 0,
    parasite_count: Math.random() > 0.95 ? 1 : 0,
    parasite_present: Math.random() > 0.95,

    mean_bacterial_bbox_confidence: 0.75 + Math.random() * 0.2,
    rbc_morphology_anisocytosis: Math.random() > 0.8,
    rbc_morphology_poikilocytosis: Math.random() > 0.85,
    rbc_morphology_hypochromia: Math.random() > 0.9,

    image_quality: 'good',
    analysis_confidence: 0.82 + Math.random() * 0.15,
    _simulated: true,
  };
};

const generateMockPrediction = (requestData) => {
  const antibiotics = requestData.target_antibiotics || ['ceftriaxone', 'ciprofloxacin', 'meropenem'];

  const predictions = antibiotics.map((antibiotic) => {
    const probability = Math.random() * 0.8 + 0.1; // 10-90%
    const riskLevel = probability < 0.3 ? 'low' : probability < 0.6 ? 'moderate' : 'high';

    return {
      antibiotic,
      probability: parseFloat(probability.toFixed(3)),
      risk_level: riskLevel,
      calibrated: true,
      top_contributing_features: [
        { feature_name: 'crp_elevated', shap_value: 0.08 + Math.random() * 0.1, category: 'labs' },
        { feature_name: 'bacterial_cluster_count', shap_value: 0.05 + Math.random() * 0.08, category: 'image' },
        { feature_name: 'prior_antibiotic_exposure', shap_value: 0.04 + Math.random() * 0.06, category: 'history' },
      ],
    };
  });

  const maxProb = Math.max(...predictions.map((p) => p.probability));
  const overallRisk = maxProb < 0.3 ? 'low' : maxProb < 0.6 ? 'moderate' : 'high';

  return {
    patient_id: requestData.patient_id,
    prediction_id: `PRED_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
    prediction_time: new Date().toISOString(),
    model_version: '2.0.0-demo',
    overall_risk_level: overallRisk,
    clinical_summary: `Based on the provided clinical data and blood smear analysis, ${overallRisk === 'high'
      ? 'elevated risk of antimicrobial resistance detected. Consider alternative empiric therapy or await culture results.'
      : overallRisk === 'moderate'
        ? 'moderate risk profile observed. Standard empiric coverage may be appropriate with close monitoring.'
        : 'low resistance risk predicted. Standard empiric therapy is likely appropriate.'
      }`,
    predictions,
    inference_latency_ms: 85 + Math.random() * 50,
    _simulated: true,
  };
};

export default {
  analyzeBloodSmear,
  predictAMR,
  getThresholds,
  checkHealth,
};
