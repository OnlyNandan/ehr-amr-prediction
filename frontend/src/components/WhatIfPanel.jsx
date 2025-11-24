import React from 'react';
import { Sliders, RefreshCw, Zap } from 'lucide-react';

const WhatIfPanel = ({ data, onChange }) => {
    const handleChange = (key, value) => {
        onChange({ ...data, [key]: value });
    };

    return (
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 h-full overflow-y-auto max-h-[800px]">
            <div className="flex items-center justify-between mb-8">
                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                    <div className="bg-violet-100 p-1.5 rounded-lg text-violet-600">
                        <Zap size={18} />
                    </div>
                    Simulation
                </h3>
                <button
                    onClick={() => onChange({
                        ...data,
                        prior_antibiotics_days: 5,
                        wbc_count: 11500,
                        heart_rate: 80,
                        temperature: 37.0,
                        device_use: true,
                        candidate_antibiotic: "Ciprofloxacin"
                    })}
                    className="text-xs font-medium text-slate-500 hover:text-blue-600 flex items-center gap-1.5 transition-colors bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-100 hover:border-blue-200"
                >
                    <RefreshCw size={12} /> Reset
                </button>
            </div>

            <div className="space-y-6">
                {/* Prior Abx */}
                <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                        Prior Antibiotic Use
                    </label>
                    <input
                        type="range"
                        min="0"
                        max="30"
                        value={data.prior_antibiotics_days}
                        onChange={(e) => handleChange('prior_antibiotics_days', parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                        <span>0d</span>
                        <span className="text-blue-600 font-bold">{data.prior_antibiotics_days} days</span>
                        <span>30d</span>
                    </div>
                </div>

                {/* WBC */}
                <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                        WBC Count (/µL)
                    </label>
                    <input
                        type="range"
                        min="4000"
                        max="25000"
                        step="500"
                        value={data.wbc_count}
                        onChange={(e) => handleChange('wbc_count', parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                        <span>4k</span>
                        <span className="text-blue-600 font-bold">{data.wbc_count.toLocaleString()}</span>
                        <span>25k</span>
                    </div>
                </div>

                {/* Heart Rate */}
                <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                        Heart Rate (bpm)
                    </label>
                    <input
                        type="range"
                        min="40"
                        max="180"
                        value={data.heart_rate || 80}
                        onChange={(e) => handleChange('heart_rate', parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-red-500"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                        <span>40</span>
                        <span className="text-red-600 font-bold">{data.heart_rate}</span>
                        <span>180</span>
                    </div>
                </div>

                {/* Temperature */}
                <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                        Temperature (°C)
                    </label>
                    <input
                        type="range"
                        min="35.0"
                        max="42.0"
                        step="0.1"
                        value={data.temperature || 37.0}
                        onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                        className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-amber-500"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                        <span>35.0</span>
                        <span className="text-amber-600 font-bold">{data.temperature}</span>
                        <span>42.0</span>
                    </div>
                </div>

                {/* Device Use */}
                <div className="flex items-center justify-between py-3 border-t border-b border-slate-100">
                    <label className="text-sm font-semibold text-slate-700">
                        Indwelling Device?
                    </label>
                    <button
                        onClick={() => handleChange('device_use', !data.device_use)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${data.device_use ? 'bg-blue-600' : 'bg-slate-200'}`}
                    >
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${data.device_use ? 'translate-x-6' : 'translate-x-1'}`} />
                    </button>
                </div>

                {/* Antibiotic */}
                <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                        Candidate Antibiotic
                    </label>
                    <select
                        value={data.candidate_antibiotic}
                        onChange={(e) => handleChange('candidate_antibiotic', e.target.value)}
                        className="block w-full rounded-lg border-slate-200 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2.5 border bg-slate-50"
                    >
                        <option>Ciprofloxacin</option>
                        <option>Levofloxacin</option>
                        <option>Ceftriaxone</option>
                        <option>Meropenem</option>
                    </select>
                </div>
            </div>
        </div>
    );
};

export default WhatIfPanel;
