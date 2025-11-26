import React from 'react';
import { Box, Paper, Typography, Grid } from '@mui/material';

const CellCountDisplay = ({ features }) => {
  if (!features) return null;

  const counts = [
    { label: 'WBC', value: features.wbc_count || 0, color: '#1976d2' },
    { label: 'RBC', value: features.rbc_count || 0, color: '#d32f2f' },
  ];

  return (
    <Paper sx={{ p: 3, bgcolor: '#fff' }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#333', fontWeight: 600 }}>
        Cell Counts
      </Typography>
      <Grid container spacing={2}>
        {counts.map((item) => (
          <Grid item xs={6} key={item.label}>
            <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="textSecondary">{item.label}</Typography>
              <Typography variant="h4" sx={{ color: item.color, fontWeight: 700 }}>
                {item.value}
              </Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default CellCountDisplay;
