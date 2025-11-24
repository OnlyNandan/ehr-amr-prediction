import React from 'react';
import { User, Calendar, AlertCircle, Heart } from 'lucide-react';

const PatientHeader = ({ patient }) => {
    return (
        <div className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between sticky top-0 z-20 shadow-sm/50 backdrop-blur-sm bg-white/90">
            <div className="flex items-center gap-6">
                <div className="relative">
                    <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-600 font-bold text-lg border-2 border-white shadow-sm">
                        {patient.name.charAt(0)}
                    </div>
                    <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 border-2 border-white rounded-full"></div>
                </div>

                <div>
                    <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                        {patient.name}
                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-500 border border-slate-200">
                            {patient.patient_id}
                        </span>
                    </h1>
                    <div className="flex items-center gap-4 text-sm text-slate-500 mt-1">
                        <span className="flex items-center gap-1"><User size={14} /> {patient.age} yrs, {patient.gender}</span>
                        <span className="w-1 h-1 rounded-full bg-slate-300"></span>
                        <span className="flex items-center gap-1"><Calendar size={14} /> Admitted: 3 days ago</span>
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-6">
                <div className="text-right">
                    <p className="text-xs font-semibold text-slate-400 uppercase">Allergies</p>
                    <div className="flex items-center gap-1 text-sm font-medium text-red-600">
                        <AlertCircle size={14} /> Penicillin
                    </div>
                </div>
                <div className="h-8 w-px bg-slate-200"></div>
                <div className="text-right">
                    <p className="text-xs font-semibold text-slate-400 uppercase">Code Status</p>
                    <div className="flex items-center gap-1 text-sm font-medium text-slate-700">
                        <Heart size={14} /> Full Code
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PatientHeader;
