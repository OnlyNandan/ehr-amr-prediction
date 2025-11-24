import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Play, Square, Activity } from 'lucide-react';

const LiveMonitor = ({ initialData, onDataUpdate }) => {
    const [isSimulating, setIsSimulating] = useState(false);
    const [dataPoints, setDataPoints] = useState([]);
    const [currentVitals, setCurrentVitals] = useState({
        heart_rate: initialData.heart_rate || 80,
        temperature: initialData.temperature || 37,
        systolic_bp: initialData.systolic_bp || 120,
        wbc_count: initialData.wbc_count || 8000,
        time_step: 0
    });

    const intervalRef = useRef(null);

    // Initialize chart with some empty history
    useEffect(() => {
        const initialPoints = Array.from({ length: 20 }, (_, i) => ({
            time: i,
            heart_rate: initialData.heart_rate || 80,
            temperature: initialData.temperature || 37,
            systolic_bp: initialData.systolic_bp || 120
        }));
        setDataPoints(initialPoints);
    }, []);

    const toggleSimulation = () => {
        if (isSimulating) {
            clearInterval(intervalRef.current);
            setIsSimulating(false);
        } else {
            setIsSimulating(true);
            intervalRef.current = setInterval(fetchNextStep, 1000);
        }
    };

    const fetchNextStep = async () => {
        try {
            const response = await fetch('http://localhost:8000/simulate/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentVitals),
            });
            const nextStep = await response.json();

            setCurrentVitals(nextStep);

            setDataPoints(prev => {
                const newPoints = [...prev, {
                    time: nextStep.time_step,
                    heart_rate: nextStep.heart_rate,
                    temperature: nextStep.temperature,
                    systolic_bp: nextStep.systolic_bp
                }];
                return newPoints.slice(-30); // Keep last 30 points
            });

            // Notify parent to update risk prediction occasionally
            if (nextStep.time_step % 3 === 0) {
                onDataUpdate(nextStep);
            }

        } catch (error) {
            console.error("Simulation error:", error);
            setIsSimulating(false);
        }
    };

    useEffect(() => {
        return () => clearInterval(intervalRef.current);
    }, []);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-full ${isSimulating ? 'bg-green-100 text-green-600 animate-pulse' : 'bg-slate-100 text-slate-500'}`}>
                        <Activity size={24} />
                    </div>
                    <div>
                        <h3 className="font-bold text-slate-900">Live Vitals Monitor</h3>
                        <p className="text-sm text-slate-500">{isSimulating ? 'Streaming real-time data...' : 'Simulation paused'}</p>
                    </div>
                </div>
                <button
                    onClick={toggleSimulation}
                    className={`flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-colors ${isSimulating
                            ? 'bg-red-50 text-red-600 hover:bg-red-100'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                >
                    {isSimulating ? <><Square size={18} /> Stop Simulation</> : <><Play size={18} /> Start Simulation</>}
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Heart Rate */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                    <div className="flex justify-between items-end mb-4">
                        <h4 className="font-medium text-slate-500">Heart Rate</h4>
                        <span className="text-2xl font-bold text-slate-900">{currentVitals.heart_rate} <span className="text-sm font-normal text-slate-400">bpm</span></span>
                    </div>
                    <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={dataPoints}>
                                <Line type="monotone" dataKey="heart_rate" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                                <YAxis domain={['dataMin - 5', 'dataMax + 5']} hide />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Temperature */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                    <div className="flex justify-between items-end mb-4">
                        <h4 className="font-medium text-slate-500">Temperature</h4>
                        <span className="text-2xl font-bold text-slate-900">{currentVitals.temperature} <span className="text-sm font-normal text-slate-400">°C</span></span>
                    </div>
                    <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={dataPoints}>
                                <Line type="monotone" dataKey="temperature" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                                <YAxis domain={['dataMin - 0.5', 'dataMax + 0.5']} hide />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* BP */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                    <div className="flex justify-between items-end mb-4">
                        <h4 className="font-medium text-slate-500">Systolic BP</h4>
                        <span className="text-2xl font-bold text-slate-900">{currentVitals.systolic_bp} <span className="text-sm font-normal text-slate-400">mmHg</span></span>
                    </div>
                    <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={dataPoints}>
                                <Line type="monotone" dataKey="systolic_bp" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                                <YAxis domain={['dataMin - 10', 'dataMax + 10']} hide />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiveMonitor;
