import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const HospitalMap = () => {
    // Mock locations for the hospital map
    const locations = [
        { id: 1, name: "ICU Wing A", lat: 51.505, lng: -0.09, risk: 0.8, cases: 12 },
        { id: 2, name: "General Ward 3", lat: 51.51, lng: -0.1, risk: 0.3, cases: 4 },
        { id: 3, name: "Emergency Dept", lat: 51.51, lng: -0.08, risk: 0.5, cases: 8 },
    ];

    const getColor = (risk) => {
        if (risk > 0.7) return '#ef4444'; // Red
        if (risk > 0.4) return '#f59e0b'; // Amber
        return '#10b981'; // Green
    };

    return (
        <div className="h-full w-full relative z-0">
            <MapContainer center={[51.505, -0.09]} zoom={13} scrollWheelZoom={false} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {locations.map(loc => (
                    <CircleMarker
                        key={loc.id}
                        center={[loc.lat, loc.lng]}
                        pathOptions={{ color: getColor(loc.risk), fillColor: getColor(loc.risk), fillOpacity: 0.6 }}
                        radius={20}
                    >
                        <Popup>
                            <div className="p-2">
                                <h3 className="font-bold text-slate-900">{loc.name}</h3>
                                <p className="text-sm text-slate-600">Resistance Risk: {(loc.risk * 100).toFixed(0)}%</p>
                                <p className="text-sm text-slate-600">Active Cases: {loc.cases}</p>
                            </div>
                        </Popup>
                    </CircleMarker>
                ))}
            </MapContainer>
            <div className="absolute bottom-4 right-4 bg-white p-4 rounded-lg shadow-lg z-[1000]">
                <h4 className="font-bold text-sm mb-2">Resistance Heatmap</h4>
                <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-red-500"></div> High Risk (>70%)</div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-amber-500"></div> Moderate Risk</div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-emerald-500"></div> Low Risk</div>
                </div>
            </div>
        </div>
    );
};

export default HospitalMap;
