import React, { useState, useRef, useEffect } from 'react';
import { Box, Paper, Typography, Button, CircularProgress, Alert } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const API_URL = 'http://localhost:5050/api/v1';

function BloodSmearUpload({ onAnalysisComplete }) {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setImageUrl(URL.createObjectURL(file));
      setDetections([]);
      setError(null);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    try {
      const base64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(image);
      });

      const response = await fetch(`${API_URL}/predict_image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64 }),
      });

      if (!response.ok) throw new Error('Analysis failed');

      const result = await response.json();
      setDetections(result.detections || []);
      
      if (onAnalysisComplete) {
        onAnalysisComplete({
          ...result.image_features,
          wbc_count: result.metadata?.wbc_count || 0,
          rbc_count: result.metadata?.rbc_count || 0,
          detections: result.detections,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Draw bounding boxes
  useEffect(() => {
    if (!canvasRef.current || !imageRef.current || !imageUrl || detections.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;

    const drawBoxes = () => {
      const displayW = img.clientWidth;
      const displayH = img.clientHeight;
      const naturalW = img.naturalWidth;
      const naturalH = img.naturalHeight;

      canvas.width = displayW;
      canvas.height = displayH;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const scaleX = displayW / naturalW;
      const scaleY = displayH / naturalH;

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
        const w = (x2 - x1) * scaleX, h = (y2 - y1) * scaleY;

        const color = det.class_name === 'wbc' ? '#2196F3' : '#e53935';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, w, h);

        const label = det.class_name.toUpperCase();
        ctx.font = 'bold 10px Arial';
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(sx1, sy1 - 14, textW + 6, 14);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, sx1 + 3, sy1 - 3);
      });
    };

    if (img.complete) drawBoxes();
    else img.onload = drawBoxes;
    
    window.addEventListener('resize', drawBoxes);
    return () => window.removeEventListener('resize', drawBoxes);
  }, [detections, imageUrl]);

  const wbcCount = detections.filter(d => d.class_name === 'wbc').length;
  const rbcCount = detections.filter(d => d.class_name === 'rbc').length;

  return (
    <Paper sx={{ p: 3, bgcolor: '#fff' }}>
      <Typography variant="h6" gutterBottom sx={{ color: '#333', fontWeight: 600 }}>
        Blood Smear Analysis
      </Typography>

      {!imageUrl ? (
        <Box
          onClick={() => document.getElementById('file-input').click()}
          sx={{
            border: '2px dashed #ccc',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            '&:hover': { borderColor: '#1976d2', bgcolor: '#f5f5f5' },
          }}
        >
          <input id="file-input" type="file" accept="image/*" onChange={handleFileChange} hidden />
          <CloudUploadIcon sx={{ fontSize: 48, color: '#999', mb: 1 }} />
          <Typography color="textSecondary">Click to upload blood smear image</Typography>
        </Box>
      ) : (
        <Box>
          <Box sx={{ position: 'relative', mb: 2 }}>
            <img
              ref={imageRef}
              src={imageUrl}
              alt="Blood smear"
              style={{ width: '100%', display: 'block', borderRadius: 8 }}
            />
            <canvas
              ref={canvasRef}
              style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
            />
          </Box>

          {detections.length > 0 && (
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <Box sx={{ px: 2, py: 1, bgcolor: '#e3f2fd', borderRadius: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 12, height: 12, bgcolor: '#2196F3', borderRadius: '50%' }} />
                <Typography variant="body2">WBC: <strong>{wbcCount}</strong></Typography>
              </Box>
              <Box sx={{ px: 2, py: 1, bgcolor: '#ffebee', borderRadius: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 12, height: 12, bgcolor: '#e53935', borderRadius: '50%' }} />
                <Typography variant="body2">RBC: <strong>{rbcCount}</strong></Typography>
              </Box>
            </Box>
          )}

          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button variant="contained" onClick={analyzeImage} disabled={loading}>
              {loading ? <CircularProgress size={20} sx={{ mr: 1 }} /> : null}
              {loading ? 'Analyzing...' : 'Analyze Image'}
            </Button>
            <Button variant="outlined" onClick={() => { setImage(null); setImageUrl(null); setDetections([]); }}>
              Clear
            </Button>
          </Box>
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
    </Paper>
  );
}

export default BloodSmearUpload;
