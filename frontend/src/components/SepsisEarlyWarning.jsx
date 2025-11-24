import React, { useState, useEffect } from 'react';
import { AlertTriangle, Activity, Thermometer, Heart, Wind } from 'lucide-react';

const SepsisEarlyWarning = ({ patientData }) => {
    const [sepsisRisk, setSepsisRisk] = useState(null);
    const [loading, setLoading] = useState(false);
    const [lastUpdate, setLastUpdate] = useState(null);

    useEffect(() => {
        if (patientData) {
            fetchSepsisRisk();
        }
    }, [patientData]);

    const fetchSepsisRisk = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/api/sepsis/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    heart_rate: patientData.heart_rate || 80,
                    temperature: patientData.temperature || 37.0,
                    systolic_bp: patientData.systolic_bp || 120,
                    respiratory_rate: 18,
                    wbc_count: patientData.wbc_count || 9000,
                    lactate: 2.0,
                    creatinine: 1.0,
                    platelets: 200,
                    age: patientData.age || 65,
                    gcs: 15,
                    vasopressors: false,
                    prior_sepsis: false
                })
            });
            const data = await response.json();
            setSepsisRisk(data);
            setLastUpdate(new Date());
        } catch (error) {
            console.error('Failed to fetch sepsis risk:', error);
        } finally {
            setLoading(false);
        }
    };

    if (!sepsisRisk && !loading) {
        return (
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <div className="flex items-center gap-2 mb-4">
                    <Activity className="w-5 h-5 text-red-600" />
                    <h3 className="text-lg font-semibold text-slate-900">Sepsis Early Warning</h3>
                </div>
                <button
                    onClick={fetchSepsisRisk}
                    className="w-full bg-red-600 text-white py-2 rounded-lg font-medium hover:bg-red-700"
                >
                    Run Sepsis Assessment
                </button>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <div className="flex items-center justify-center h-48">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600"></div>
                </div>
            </div>
        );
    }

    const getAlertColor = () => {
        switch (sepsisRisk.alert_level) {
            case 'CRITICAL': return 'bg-red-500';
            case 'HIGH': return 'bg-orange-500';
            case 'MODERATE': return 'bg-yellow-500';
            default: return 'bg-green-500';
        }
    };

    const getAlertTextColor = () => {
        switch (sepsisRisk.alert_level) {
            case 'CRITICAL': return 'text-red-600';
            case 'HIGH': return 'text-orange-600';
            case 'MODERATE': return 'text-yellow-600';
            default: return 'text-green-600';
        }
    };

    return (
        <div className={`bg-white rounded-xl shadow-lg border-2 ${sepsisRisk.alert_level === 'CRITICAL' ? 'border-red-500 animate-pulse' : 'border-slate-200'}`}>
            {/* Alert Banner */}
            {sepsisRisk.alert_level !== 'LOW' && (
                <div className={`${getAlertColor()} text-white px-6 py-3 rounded-t-xl flex items-center gap-2`}>
                    <AlertTriangle className="w-5 h-5" />
                    <span className="font-semibold">{sepsisRisk.alert_level} SEPSIS RISK DETECTED</span>
                </div>
            )}

            <div className="p-6">
                {/* Overall Risk Gauge */}
                <div className="text-center mb-6">
                    <div className="text-sm text-slate-500 mb-2">Overall Sepsis Risk</div>
                    <div className={`text-6xl font-bold ${getAlertTextColor()}`}>
                        {(sepsisRisk.overall_risk * 100).toFixed(0)}%
                    </div>
                    <div className="mt-2">
                        <div className="w-full bg-slate-200 rounded-full h-3">
                            <div
                                className={`h-3 rounded-full ${getAlertColor()}`}
                                style={{ width: `${sepsisRisk.overall_risk * 100}%` }}
                            ></div>
                        </div>
                    </div>
                </div>

                {/* Time Horizons */}
                <div className="grid grid-cols-4 gap-3 mb-6">
                    <div className="text-center">
                        <div className="text-xs text-slate-500 mb-1">4 Hours</div>
                        <div className="text-2xl font-bold text-red-600">
                            {(sepsisRisk.risk_4hr * 100).toFixed(0)}%
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-slate-500 mb-1">12 Hours</div>
                        <div className="text-2xl font-bold text-orange-600">
                            {(sepsisRisk.risk_12hr * 100).toFixed(0)}%
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-slate-500 mb-1">24 Hours</div>
                        <div className="text-2xl font-bold text-yellow-600">
                            {(sepsisRisk.risk_24hr * 100).toFixed(0)}%
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-slate-500 mb-1">48 Hours</div>
                        <div className="text-2xl font-bold text-blue-600">
                            {(sepsisRisk.risk_48hr * 100).toFixed(0)}%
                        </div>
                    </div>
                </div>

                {/* Current Vitals */}
                <div className="grid grid-cols-3 gap-3 mb-6 p-4 bg-slate-50 rounded-lg">
                    <div className="flex items-center gap-2">
                        <Heart className="w-4 h-4 text-red-500" />
                        <div>
                            <div className="text-xs text-slate-500">HR</div>
                            <div className="font-semibold">{patientData.heart_rate}</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <Thermometer className="w-4 h-4 text-orange-500" />
                        <div>
                            <div className="text-xs text-slate-500">Temp</div>
                            <div className="font-semibold">{patientData.temperature}°C</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-blue-500" />
                        <div>
                            <div className="text-xs text-slate-500">BP</div>
                            <div className="font-semibold">{patientData.systolic_bp}/80</div>
                        </div>
                    </div>
                </div>

                {/* Recommendations */}
                <div className="space-y-2 mb-4">
                    <h4 className="font-semibold text-slate-900 text-sm">Clinical Recommendations:</h4>
                    {sepsisRisk.recommendations.map((rec, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-sm">
                            <div className="text-red-500 mt-0.5">•</div>
                            <div className="text-slate-700">{rec}</div>
                        </div>
                    ))}
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                    <button
                        onClick={fetchSepsisRisk}
                        className="flex-1 bg-slate-100 text-slate-700 py-2 rounded-lg font-medium hover:bg-slate-200 text-sm"
                    >
                        Refresh Assessment
                    </button>
                    {sepsisRisk.alert_level === 'CRITICAL' && (
                        <button className="flex-1 bg-red-600 text-white py-2 rounded-lg font-medium hover:bg-red-700 text-sm">
                            Activate Sepsis Protocol
                        </button>
                    )}
                </div>

                {lastUpdate && (
                    <div className="text-xs text-slate-400 mt-3 text-center">
                        Last updated: {lastUpdate.toLocaleTimeString()}
                    </div>
                )}
            </div>
        </div>
    );
};

export default SepsisEarlyWarning;
