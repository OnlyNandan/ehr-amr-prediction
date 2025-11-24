import React, { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import PatientHeader from './PatientHeader';
import StatCard from './StatCard';
import PredictionCard from './PredictionCard';
import WhatIfPanel from './WhatIfPanel';
import HospitalMap from './HospitalMap';
import PatientHistory from './PatientHistory';
import LiveMonitor from './LiveMonitor';
import NotificationToast from './NotificationToast';
import BloodCellAnalyzer from './BloodCellAnalyzer';
import RiskExplainer from './RiskExplainer';
import SepsisEarlyWarning from './SepsisEarlyWarning';
import { Search } from 'lucide-react';

const Dashboard = ({ userRole }) => {
    const [activeTab, setActiveTab] = useState('prediction');
    const [collapsed, setCollapsed] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [notification, setNotification] = useState(null);

    // Default patient
    const [patientData, setPatientData] = useState({
        patient_id: "P-1024",
        name: "John Doe",
        age: 65,
        gender: "M",
        wbc_count: 11500,
        heart_rate: 80,
        temperature: 37.0,
        systolic_bp: 120,
        prior_antibiotics_days: 5,
        device_use: true,
        suspected_bacterium: "E. coli",
        candidate_antibiotic: "Ciprofloxacin",
        history: []
    });

    // Fetch initial patient data
    useEffect(() => {
        fetchPatient("P-1024");
    }, []);

    const fetchPatient = async (id) => {
        try {
            const response = await fetch(`http://localhost:8000/patients/${id}`);
            if (response.ok) {
                const data = await response.json();
                setPatientData(data);
                handlePredict(data);
            }
        } catch (error) {
            console.error("Error fetching patient:", error);
        }
    };

    const handlePredict = async (data) => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const result = await response.json();

            // Check for high risk to trigger notification
            if (result.risk_score > 0.8 && (!prediction || prediction.risk_score <= 0.8)) {
                setNotification({ type: 'error', message: 'CRITICAL: High resistance risk detected!' });
            }

            setTimeout(() => {
                setPrediction(result);
                setLoading(false);
            }, 400);
        } catch (error) {
            console.error("Error fetching prediction:", error);
            setLoading(false);
        }
    };

    return (
        <div className="flex h-screen bg-slate-50 font-sans overflow-hidden">
            <Sidebar
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                userRole={userRole}
                collapsed={collapsed}
                setCollapsed={setCollapsed}
            />

            <div className="flex-1 flex flex-col h-full overflow-hidden relative">
                <PatientHeader patient={patientData} />

                <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
                    {/* Stat Cards Row */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                        <StatCard
                            title="Heart Rate"
                            value={patientData.heart_rate || 80}
                            unit="bpm"
                            color="red"
                            trend="up"
                            trendValue="+2%"
                        />
                        <StatCard
                            title="Blood Pressure"
                            value={`${patientData.systolic_bp || 120}/80`}
                            unit="mmHg"
                            color="blue"
                            trend="stable"
                            trendValue="0%"
                        />
                        <StatCard
                            title="Temperature"
                            value={patientData.temperature || 37.0}
                            unit="°C"
                            color="amber"
                            trend="up"
                            trendValue="+0.4"
                        />
                        <StatCard
                            title="WBC Count"
                            value={(patientData.wbc_count / 1000).toFixed(1)}
                            unit="k/µL"
                            color="emerald"
                            trend="down"
                            trendValue="-5%"
                        />
                    </div>

                    {activeTab === 'prediction' && (
                        <div className="grid grid-cols-12 gap-6">
                            {/* Main Prediction Area */}
                            <div className="col-span-12 lg:col-span-8 space-y-6">
                                <PredictionCard prediction={prediction} loading={loading} />
                                <RiskExplainer patientData={patientData} />
                            </div>

                            {/* Sidebar Controls */}
                            <div className="col-span-12 lg:col-span-4 space-y-6">
                                <SepsisEarlyWarning patientData={patientData} />
                                <BloodCellAnalyzer
                                    onAnalysisComplete={(counts) => {
                                        const updated = { ...patientData, wbc_count: counts.WBC * 1000 };
                                        setPatientData(updated);
                                        handlePredict(updated);
                                    }}
                                />
                                <WhatIfPanel
                                    data={patientData}
                                    onChange={(newData) => {
                                        setPatientData(newData);
                                        handlePredict(newData);
                                    }}
                                />
                            </div>
                        </div>
                    )}

                    {activeTab === 'monitor' && (
                        <LiveMonitor
                            initialData={patientData}
                            onDataUpdate={(newVitals) => {
                                const updated = { ...patientData, ...newVitals };
                                setPatientData(updated);
                                handlePredict(updated);
                            }}
                        />
                    )}

                    {activeTab === 'history' && (
                        <PatientHistory history={patientData.history} />
                    )}

                    {activeTab === 'map' && (
                        <div className="h-[700px] bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden relative">
                            <HospitalMap />
                        </div>
                    )}
                </main>

                {notification && (
                    <NotificationToast
                        message={notification.message}
                        type={notification.type}
                        onClose={() => setNotification(null)}
                    />
                )}
            </div>
        </div>
    );
};

export default Dashboard;
