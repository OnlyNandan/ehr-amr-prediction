import React from 'react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

const StatCard = ({ title, value, unit, trend, trendValue, data, color }) => {
    // Generate mock sparkline data if none provided
    const chartData = data || Array.from({ length: 10 }, (_, i) => ({ val: Math.random() * 100 }));

    const getColor = (c) => {
        switch (c) {
            case 'blue': return '#3b82f6';
            case 'red': return '#ef4444';
            case 'amber': return '#f59e0b';
            case 'emerald': return '#10b981';
            default: return '#64748b';
        }
    };

    const strokeColor = getColor(color);

    return (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start mb-2">
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">{title}</h3>
                <div className={`flex items-center text-xs font-medium ${trend === 'up' ? 'text-red-500' : trend === 'down' ? 'text-emerald-500' : 'text-slate-400'}`}>
                    {trend === 'up' && <ArrowUpRight size={14} />}
                    {trend === 'down' && <ArrowDownRight size={14} />}
                    {trend === 'stable' && <Minus size={14} />}
                    <span className="ml-1">{trendValue}</span>
                </div>
            </div>

            <div className="flex items-end justify-between">
                <div>
                    <div className="flex items-baseline gap-1">
                        <span className="text-2xl font-bold text-slate-900">{value}</span>
                        <span className="text-sm font-medium text-slate-400">{unit}</span>
                    </div>
                </div>

                <div className="h-10 w-24">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <Line
                                type="monotone"
                                dataKey="val"
                                stroke={strokeColor}
                                strokeWidth={2}
                                dot={false}
                                isAnimationActive={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default StatCard;
