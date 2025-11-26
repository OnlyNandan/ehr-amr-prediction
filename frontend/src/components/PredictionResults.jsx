import React from 'react';
import { Box, Paper, Typography, Grid, Alert, LinearProgress } from '@mui/material';

const PredictionResults = ({ predictions }) => {
  if (!predictions) return null;

  if (predictions.error) {
    return (
      <Paper sx={{ p: 3, bgcolor: '#fff' }}>
        <Alert severity="error">{predictions.message || 'Prediction failed'}</Alert>
      </Paper>
    );
  }

  const getRiskColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return '#4caf50';
      case 'moderate': return '#ff9800';
      case 'high': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  const getRiskBg = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return '#e8f5e9';
      case 'moderate': return '#fff3e0';
      case 'high': return '#ffebee';
      default: return '#f5f5f5';
    }
  };

  return (
    <Paper sx={{ p: 3, bgcolor: '#fff' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ color: '#333', fontWeight: 600 }}>
          AMR Risk Assessment
        </Typography>
        <Box sx={{ 
          px: 2, py: 0.5, borderRadius: 1, 
          bgcolor: getRiskBg(predictions.overall_risk_level),
          color: getRiskColor(predictions.overall_risk_level),
          fontWeight: 600,
          fontSize: '0.875rem',
        }}>
          {predictions.overall_risk_level?.toUpperCase() || 'UNKNOWN'} RISK
        </Box>
      </Box>

      {predictions.clinical_summary && (
        <Box sx={{ p: 2, mb: 3, bgcolor: '#f5f5f5', borderRadius: 1 }}>
          <Typography variant="body2" color="textSecondary">
            {predictions.clinical_summary}
          </Typography>
        </Box>
      )}

      <Typography variant="subtitle2" sx={{ mb: 2, color: '#666' }}>
        Resistance Probability by Antibiotic
      </Typography>

      <Grid container spacing={2}>
        {predictions.predictions?.map((pred, idx) => {
          const pct = Math.round((pred.probability || 0) * 100);
          const color = getRiskColor(pred.risk_level);
          const bg = getRiskBg(pred.risk_level);
          
          return (
            <Grid item xs={6} key={pred.antibiotic || idx}>
              <Box sx={{ p: 2, bgcolor: bg, borderRadius: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 500, mb: 1, color: '#333' }}>
                  {pred.antibiotic?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ flex: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={pct} 
                      sx={{ 
                        height: 8, 
                        borderRadius: 4,
                        bgcolor: 'rgba(0,0,0,0.1)',
                        '& .MuiLinearProgress-bar': { bgcolor: color, borderRadius: 4 }
                      }} 
                    />
                  </Box>
                  <Typography variant="body2" sx={{ fontWeight: 700, color, minWidth: 45 }}>
                    {pct}%
                  </Typography>
                </Box>
                <Typography variant="caption" sx={{ color: color, fontWeight: 500 }}>
                  {pred.risk_level?.toUpperCase()}
                </Typography>
              </Box>
            </Grid>
          );
        })}
      </Grid>

      {predictions.model_version && (
        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 2, textAlign: 'right' }}>
          Model: {predictions.model_version}
        </Typography>
      )}
    </Paper>
  );
};

export default PredictionResults;
