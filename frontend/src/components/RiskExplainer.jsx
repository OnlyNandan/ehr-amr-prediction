import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertCircle } from 'lucide-react';

const RiskExplainer = ({ patientData }) => {
    const [shapData, setShapData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (patientData) {
            fetchExplanation();
        }
    }, [patientData]);

    const fetchExplanation = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patientData)
            });
            const data = await response.json();
            setShapData(data);
        } catch (error) {
            console.error('Failed to fetch SHAP explanation:', error);
        } finally {
            setLoading(false);
        }
    };

    if (!shapData || loading) {
        return (
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Risk Explanation (SHAP)</h3>
                <div className="flex items-center justify-center h-48 text-slate-400">
                    {loading ? 'Loading explanation...' : 'No explanation data available'}
                </div>
            </div>
        );
    }

    // Transform SHAP data for visualization
    const chartData = Object.entries(shapData.contributions || {})
        .map(([feature, data]) => ({
            feature: feature.replace(/_/g, ' ').replace(/enc/g, '').trim(),
            impact: data.shap_value,
            value: data.value
        }))
        .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
        .slice(0, 7); // Top 7 features

    const friendlyNames = {
        'age': 'Age',
        'wbc count': 'WBC Count',
        'prior antibiotics days': 'Prior Antibiotic Use',
        'device use': 'Device in Use',
        'heart rate': 'Heart Rate',
        'temperature': 'Temperature',
        'systolic bp': 'Blood Pressure',
        'gender': 'Gender',
        'bacterium': 'Suspected Bacterium',
        'antibiotic': 'Candidate Antibiotic'
    };

    const displayData = chartData.map(d => ({
        ...d,
        feature: friendlyNames[d.feature.toLowerCase()] || d.feature
    }));

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <div className="flex items-center gap-2 mb-4">
                <AlertCircle className="w-5 h-5 text-blue-600" />
                <h3 className="text-lg font-semibold text-slate-900">Risk Explanation (SHAP Values)</h3>
            </div>

            <div className="mb-4">
                <p className="text-sm text-slate-600">
                    This chart shows how each clinical factor contributes to the overall resistance risk prediction.
                    Positive values (red) increase risk, negative values (blue) decrease risk.
                </p>
            </div>

            <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={displayData} layout="vertical" margin={{ left: 120, right: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="feature" width={120} />
                        <Tooltip
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                        <div className="bg-white p-3 rounded-lg shadow-lg border border-slate-200">
                                            <p className="font-semibold text-slate-900">{data.feature}</p>
                                            <p className="text-sm text-slate-600 mt-1">
                                                Value: <span className="font-medium">{data.value.toFixed(2)}</span>
                                            </p>
                                            <p className="text-sm text-slate-600">
                                                Impact: <span className={`font-medium ${data.impact > 0 ? 'text-red-600' : 'text-blue-600'}`}>
                                                    {data.impact > 0 ? '+' : ''}{data.impact.toFixed(4)}
                                                </span>
                                            </p>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                            {displayData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.impact > 0 ? '#ef4444' : '#3b82f6'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-4 flex items-center gap-4 text-xs text-slate-600">
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-500 rounded"></div>
                    <span>Increases Risk</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-blue-500 rounded"></div>
                    <span>Decreases Risk</span>
                </div>
            </div>

            <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-xs text-blue-900">
                    <span className="font-semibold">Base Prediction:</span> {(shapData.base_value * 100).toFixed(1)}% |
                    <span className="font-semibold ml-2">Final Prediction:</span> {(shapData.prediction * 100).toFixed(1)}%
                </p>
            </div>
        </div>
    );
};

export default RiskExplainer;
