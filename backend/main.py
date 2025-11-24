from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import routes

app = FastAPI(
    title="EHR AMR Prediction API",
    description="Real-time antimicrobial resistance prediction system",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:5173",  # Vite default port
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)

from api import simulation
app.include_router(simulation.router)

from api import image_analysis
app.include_router(image_analysis.router)

@app.get("/")
async def root():
    return {"message": "EHR AMR Prediction API is running"}

