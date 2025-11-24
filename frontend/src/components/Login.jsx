import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, User, Activity } from 'lucide-react';

const Login = ({ setUserRole }) => {
    const navigate = useNavigate();

    const handleLogin = (role) => {
        setUserRole(role);
        navigate('/dashboard');
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-slate-100">
            <div className="w-full max-w-md p-8 space-y-8 bg-white rounded-xl shadow-lg">
                <div className="text-center">
                    <div className="flex justify-center mx-auto bg-blue-100 p-3 rounded-full w-16 h-16 items-center">
                        <ShieldCheck className="w-10 h-10 text-blue-600" />
                    </div>
                    <h2 className="mt-6 text-3xl font-extrabold text-slate-900">
                        EHR AMR Prediction
                    </h2>
                    <p className="mt-2 text-sm text-slate-600">
                        Select your role to access the system
                    </p>
                </div>
                <div className="mt-8 space-y-4">
                    <button
                        onClick={() => handleLogin('doctor')}
                        className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
                    >
                        <span className="absolute left-0 inset-y-0 flex items-center pl-3">
                            <User className="h-5 w-5 text-blue-500 group-hover:text-blue-400" aria-hidden="true" />
                        </span>
                        Clinician / Doctor
                    </button>
                    <button
                        onClick={() => handleLogin('nurse')}
                        className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500 transition-colors"
                    >
                        <span className="absolute left-0 inset-y-0 flex items-center pl-3">
                            <Activity className="h-5 w-5 text-emerald-500 group-hover:text-emerald-400" aria-hidden="true" />
                        </span>
                        Infection Control Nurse
                    </button>
                    <button
                        onClick={() => handleLogin('admin')}
                        className="group relative w-full flex justify-center py-3 px-4 border border-gray-300 text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
                    >
                        Admin
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Login;
