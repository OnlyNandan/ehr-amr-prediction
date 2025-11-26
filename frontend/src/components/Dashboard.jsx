import React, { useState, useCallback } from 'react';
import {
  Box,
  Container,
  Grid,
  Typography,
  AppBar,
  Toolbar,
  alpha,
} from '@mui/material';
import BloodSmearUpload from './BloodSmearUpload';
import PatientDataForm from './PatientDataForm';
import PredictionResults from './PredictionResults';
import CellCountDisplay from './CellCountDisplay';

const Dashboard = () => {
  const [imageAnalysis, setImageAnalysis] = useState(null);
  const [patientData, setPatientData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const handleImageAnalyzed = useCallback((analysis) => {
    setImageAnalysis(analysis);
  }, []);

  const handlePatientDataSubmit = useCallback((data) => {
    setPatientData(data);
  }, []);

  const handlePrediction = useCallback((result) => {
    setPrediction(result);
  }, []);

  return (
    <Box
      sx={{
        minHeight: '100vh',
        backgroundColor: '#0a0a14',
      }}
    >
      {/* Header */}
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          backgroundColor: '#0a0a14',
          borderBottom: '1px solid #1e1e2e',
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Typography variant="h6" fontWeight={600} sx={{ color: '#fff' }}>
              AMR Prediction System
            </Typography>
            <Typography 
              variant="caption" 
              sx={{ 
                color: '#6b7280',
                borderLeft: '1px solid #374151',
                pl: 1.5,
                ml: 0.5,
              }}
            >
              Clinical Decision Support
            </Typography>
          </Box>

          <Typography variant="caption" sx={{ color: '#6b7280' }}>
            v1.0.0
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ py: 3 }}>
        <Grid container spacing={3}>
          {/* Left Column - Image Analysis */}
          <Grid item xs={12} lg={5}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <BloodSmearUpload
                onAnalysisComplete={handleImageAnalyzed}
                isAnalyzing={isAnalyzing}
                setIsAnalyzing={setIsAnalyzing}
              />
              
              {imageAnalysis && (
                <CellCountDisplay cellCounts={imageAnalysis} />
              )}
            </Box>
          </Grid>

          {/* Right Column - Patient Data & Results */}
          <Grid item xs={12} lg={7}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <PatientDataForm
                imageFeatures={imageAnalysis}
                onSubmit={handlePatientDataSubmit}
                onPrediction={handlePrediction}
                isPredicting={isPredicting}
                setIsPredicting={setIsPredicting}
              />
              
              {prediction && (
                <PredictionResults prediction={prediction} />
              )}
            </Box>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Dashboard;
