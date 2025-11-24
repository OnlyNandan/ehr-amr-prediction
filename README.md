# EHR AMR Prediction System

A real-time, AI-powered system for predicting antimicrobial resistance (AMR) risk using Electronic Health Record (EHR) data.

## Features

- **Real-time Risk Prediction**: Uses a Random Forest model trained on synthetic clinical data to predict resistance probability.
- **Interactive Dashboard**: Role-based UI for clinicians, nurses, and admins.
- **"What-If" Analysis**: Simulate changes in patient vitals (WBC, Heart Rate, Temp) and antibiotic choices to see impact on risk.
- **Live Vitals Monitor**: Simulates real-time streaming of patient data (HR, BP, Temp) and updates risk assessment dynamically.
- **Hospital Surveillance Map**: Visual heatmap of resistance hotspots within the facility.
- **Explainable AI**: Provides symbolic rules and feature importance to explain predictions.

## Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, NumPy, Pandas.
- **Frontend**: React, Vite, Tailwind CSS, Recharts, Leaflet.
- **Architecture**: Monorepo with REST API communication.

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+

### Backend Setup
1. Navigate to `backend/`:
   ```bash
   cd backend
   ```
2. Create virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup
1. Navigate to `frontend/`:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## Usage
1. Open `http://localhost:5173` in your browser.
2. Login as **Clinician / Doctor**.
3. Use the **Clinical Dashboard** to view patient risk.
4. Go to **Live Monitor** to start a real-time simulation.
5. Use the **What-If Panel** to experiment with different clinical scenarios.
