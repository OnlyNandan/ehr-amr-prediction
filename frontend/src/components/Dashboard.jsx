import React, { useState } from 'react';
import { Container, Grid, Box, AppBar, Toolbar, Typography } from '@mui/material';
import BloodSmearUpload from './BloodSmearUpload';
import CellCountDisplay from './CellCountDisplay';
import PatientDataForm from './PatientDataForm';
import PredictionResults from './PredictionResults';

function Dashboard() {
  const [imageFeatures, setImageFeatures] = useState(null);
  const [predictions, setPredictions] = useState(null);

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f7fa' }}>
      <AppBar position="static" elevation={0} sx={{ bgcolor: '#fff', borderBottom: '1px solid #e0e0e0' }}>
        <Toolbar>
          <Typography variant="h6" sx={{ color: '#333', fontWeight: 600 }}>
            AMR Prediction System
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <BloodSmearUpload onAnalysisComplete={setImageFeatures} />
            {imageFeatures && (
              <Box sx={{ mt: 3 }}>
                <CellCountDisplay features={imageFeatures} />
              </Box>
            )}
          </Grid>

          <Grid item xs={12} md={6}>
            <PatientDataForm imageFeatures={imageFeatures} onPredictions={setPredictions} />
            {predictions && (
              <Box sx={{ mt: 3 }}>
                <PredictionResults predictions={predictions} />
              </Box>
            )}
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default Dashboard;
