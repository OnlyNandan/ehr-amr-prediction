from typing import List, Optional
from .models import PatientData

class PatientStore:
    def __init__(self):
        self.patients = [
            PatientData(
                patient_id="P-1024",
                name="John Doe",
                age=65,
                gender="M",
                wbc_count=11500,
                heart_rate=92.0,
                temperature=37.8,
                systolic_bp=135.0,
                prior_antibiotics_days=5,
                device_use=True,
                suspected_bacterium="E. coli",
                candidate_antibiotic="Ciprofloxacin",
                history=[
                    {"day": 1, "wbc": 9000, "temp": 37.2},
                    {"day": 2, "wbc": 10500, "temp": 37.8},
                    {"day": 3, "wbc": 11500, "temp": 38.1},
                ]
            ),
            PatientData(
                patient_id="P-1025",
                name="Jane Smith",
                age=42,
                gender="F",
                wbc_count=8500,
                prior_antibiotics_days=0,
                device_use=False,
                suspected_bacterium="S. aureus",
                candidate_antibiotic="Ceftriaxone",
                history=[
                    {"day": 1, "wbc": 8200, "temp": 36.8},
                    {"day": 2, "wbc": 8400, "temp": 37.0},
                    {"day": 3, "wbc": 8500, "temp": 37.1},
                ]
            ),
            PatientData(
                patient_id="P-1026",
                name="Robert Brown",
                age=78,
                gender="M",
                wbc_count=18000,
                prior_antibiotics_days=12,
                device_use=True,
                suspected_bacterium="K. pneumoniae",
                candidate_antibiotic="Meropenem",
                history=[
                    {"day": 1, "wbc": 14000, "temp": 38.0},
                    {"day": 2, "wbc": 16500, "temp": 38.5},
                    {"day": 3, "wbc": 18000, "temp": 39.2},
                ]
            ),
             PatientData(
                patient_id="P-1027",
                name="Emily Davis",
                age=29,
                gender="F",
                wbc_count=6000,
                prior_antibiotics_days=2,
                device_use=False,
                suspected_bacterium="E. coli",
                candidate_antibiotic="Levofloxacin",
                history=[
                    {"day": 1, "wbc": 6200, "temp": 36.9},
                    {"day": 2, "wbc": 6100, "temp": 36.8},
                    {"day": 3, "wbc": 6000, "temp": 36.7},
                ]
            ),
        ]

    def get_all(self) -> List[PatientData]:
        return self.patients

    def get_by_id(self, patient_id: str) -> Optional[PatientData]:
        for p in self.patients:
            if p.patient_id == patient_id:
                return p
        return None

    def search(self, query: str) -> List[PatientData]:
        query = query.lower()
        return [p for p in self.patients if query in p.name.lower() or query in p.patient_id.lower()]

store = PatientStore()
