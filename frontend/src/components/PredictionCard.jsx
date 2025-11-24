import React from 'react';
import { AlertTriangle, CheckCircle, Info, ShieldAlert, Activity } from 'lucide-react';

const PredictionCard = ({ prediction, loading }) => {
    if (loading || !prediction) {
        return (
            <div className="h-full min-h-[400px] flex flex-col items-center justify-center bg-white rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 to-transparent"></div>
                <div className="relative z-10 flex flex-col items-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-100 border-t-blue-600"></div>
                    <p className="mt-6 text-slate-500 font-medium animate-pulse">Analyzing clinical parameters...</p>
                    <p className="text-xs text-slate-400 mt-2">Running Ensemble Models & Symbolic Regression</p>
                </div>
            </div>
        );
    }

    const { risk_score, risk_level, confidence_interval, symbolic_rule, risk_features, recommendation } = prediction;

    const getRiskColor = (level) => {
        switch (level) {
            case 'Low': return 'text-emerald-700 bg-emerald-50 border-emerald-200 ring-emerald-100';
            case 'Moderate': return 'text-amber-700 bg-amber-50 border-amber-200 ring-amber-100';
            case 'High': return 'text-rose-700 bg-rose-50 border-rose-200 ring-rose-100';
            default: return 'text-slate-600 bg-slate-50';
        }
    };

    const getRiskIcon = (level) => {
        switch (level) {
            case 'Low': return <CheckCircle className="w-6 h-6" />;
            case 'Moderate': return <AlertTriangle className="w-6 h-6" />;
            case 'High': return <ShieldAlert className="w-6 h-6" />;
            default: return null;
        }
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Top Card: Main Risk Score */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-6 opacity-10">
                    <Activity size={120} />
                </div>

                <div className="relative z-10">
                    <div className="flex items-start justify-between mb-6">
                        <div>
                            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider">Predicted Resistance Risk</h3>
                            <div className="mt-3 flex items-baseline gap-3">
                                <span className="text-6xl font-extrabold text-slate-900 tracking-tight">{(risk_score * 100).toFixed(1)}%</span>
                                <div className="flex flex-col">
                                    <span className="text-sm font-medium text-slate-500">Confidence Interval</span>
                                    <span className="text-sm font-bold text-slate-700">
                                        {(confidence_interval[0] * 100).toFixed(1)}% - {(confidence_interval[1] * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div className={`px-5 py-3 rounded-xl border ring-4 flex items-center gap-3 shadow-sm ${getRiskColor(risk_level)}`}>
                            {getRiskIcon(risk_level)}
                            <span className="font-bold text-lg">{risk_level} Risk</span>
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="relative pt-2">
                        <div className="overflow-hidden h-3 mb-4 text-xs flex rounded-full bg-slate-100 shadow-inner">
                            <div style={{ width: `${risk_score * 100}%` }} className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-1000 ease-out ${risk_level === 'High' ? 'bg-gradient-to-r from-rose-500 to-red-600' : risk_level === 'Moderate' ? 'bg-gradient-to-r from-amber-400 to-amber-500' : 'bg-gradient-to-r from-emerald-400 to-emerald-500'
                                }`}></div>
                        </div>
                    </div>

                    <div className="mt-6 p-5 bg-slate-50/80 backdrop-blur rounded-xl border border-slate-200/60">
                        <h4 className="text-xs font-bold text-slate-500 uppercase mb-3 flex items-center gap-2">
                            <Info size={14} /> Symbolic Rule Logic
                        </h4>
                        <code className="text-sm text-blue-900 font-mono block leading-relaxed">
                            {symbolic_rule}
                        </code>
                    </div>
                </div>
            </div>

            {/* Bottom Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Feature Importance */}
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
                    <h3 className="text-lg font-bold text-slate-900 mb-6">Key Risk Drivers</h3>
                    <div className="space-y-5">
                        {risk_features.map((feature, idx) => (
                            <div key={idx} className="group relative">
                                <div className="flex justify-between text-sm mb-2">
                                    <span className="font-medium text-slate-700">{feature.name}</span>
                                    <span className="text-slate-500 font-mono">{feature.value}</span>
                                </div>
                                <div className="w-full bg-slate-100 rounded-full h-2">
                                    <div
                                        className="bg-blue-600 h-2 rounded-full transition-all duration-700 ease-out"
                                        style={{ width: `${feature.risk_contribution * 100}%` }}
                                    ></div>
                                </div>
                                {/* Tooltip */}
                                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-slate-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-all w-56 text-center pointer-events-none z-20 shadow-xl">
                                    {feature.description}
                                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-slate-900"></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Recommendation */}
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 flex flex-col">
                    <h3 className="text-lg font-bold text-slate-900 mb-6">AI Recommendation</h3>
                    <div className="flex-1 bg-blue-50/50 rounded-xl p-6 border border-blue-100 flex items-center justify-center text-center relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-400 to-blue-600"></div>
                        <p className="text-blue-900 font-medium text-lg leading-relaxed">
                            {recommendation}
                        </p>
                    </div>
                    <div className="mt-6 text-xs text-slate-400 text-center flex items-center justify-center gap-2">
                        <ShieldAlert size={12} />
                        AI-generated suggestion. Verify with clinical guidelines.
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PredictionCard;
