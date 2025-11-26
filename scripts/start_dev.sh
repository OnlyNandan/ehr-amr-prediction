#!/bin/bash
# Local development startup script

# Check if Python virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "Creating Python virtual environment..."
    cd backend
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
else
    source backend/venv/bin/activate
fi

# Start backend in background
echo "Starting backend server..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start frontend
echo "Starting frontend dev server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "==================================="
echo "AMR Prediction System Running"
echo "==================================="
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:5173"
echo "API Docs: http://localhost:5000/api/v1/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo "==================================="

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
