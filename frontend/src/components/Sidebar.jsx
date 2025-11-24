import React, { useState } from 'react';
import { LayoutDashboard, Map as MapIcon, FileText, Activity, LogOut, ChevronLeft, ChevronRight, Settings, Bell } from 'lucide-react';

const Sidebar = ({ activeTab, setActiveTab, userRole, collapsed, setCollapsed }) => {
    const menuItems = [
        { id: 'prediction', label: 'Dashboard', icon: LayoutDashboard },
        { id: 'monitor', label: 'Live Monitor', icon: Activity },
        { id: 'history', label: 'Patient History', icon: FileText },
        { id: 'map', label: 'Surveillance', icon: MapIcon },
    ];

    return (
        <aside
            className={`bg-slate-900 text-white flex flex-col transition-all duration-300 ease-in-out relative z-30 ${collapsed ? 'w-20' : 'w-64'}`}
        >
            {/* Brand */}
            <div className="h-16 flex items-center px-6 border-b border-slate-800">
                <div className="flex items-center gap-3">
                    <div className="bg-blue-600 p-1.5 rounded-lg shrink-0">
                        <Activity className="text-white w-5 h-5" />
                    </div>
                    {!collapsed && (
                        <span className="font-bold text-lg tracking-tight whitespace-nowrap">
                            AMR<span className="text-blue-400">Guard</span>
                        </span>
                    )}
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 py-6 px-3 space-y-1">
                {menuItems.map((item) => (
                    <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-200 group ${activeTab === item.id
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                            }`}
                        title={collapsed ? item.label : ''}
                    >
                        <item.icon size={20} className={`shrink-0 ${activeTab === item.id ? 'text-white' : 'text-slate-400 group-hover:text-white'}`} />
                        {!collapsed && <span className="text-sm font-medium">{item.label}</span>}

                        {/* Active Indicator */}
                        {activeTab === item.id && !collapsed && (
                            <div className="ml-auto w-1.5 h-1.5 rounded-full bg-white"></div>
                        )}
                    </button>
                ))}
            </nav>

            {/* Bottom Actions */}
            <div className="p-4 border-t border-slate-800 space-y-2">
                <button className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white transition-colors ${collapsed ? 'justify-center' : ''}`}>
                    <Settings size={20} />
                    {!collapsed && <span className="text-sm font-medium">Settings</span>}
                </button>
                <button className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-slate-400 hover:bg-red-900/20 hover:text-red-400 transition-colors ${collapsed ? 'justify-center' : ''}`}>
                    <LogOut size={20} />
                    {!collapsed && <span className="text-sm font-medium">Sign Out</span>}
                </button>
            </div>

            {/* Collapse Toggle */}
            <button
                onClick={() => setCollapsed(!collapsed)}
                className="absolute -right-3 top-20 bg-slate-800 text-slate-400 border border-slate-700 rounded-full p-1 hover:text-white hover:bg-blue-600 transition-colors shadow-md"
            >
                {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
            </button>
        </aside>
    );
};

export default Sidebar;
