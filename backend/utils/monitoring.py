"""
Monitoring utilities for AMR Prediction System
Tracks model performance, data drift, and system health
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor model predictions for drift and performance degradation.
    
    Tracks:
    - Prediction distribution
    - Feature distribution drift
    - Latency metrics
    - Alert thresholds
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize model monitor.
        
        Args:
            window_size: Number of predictions to track
            drift_threshold: Threshold for drift alerts
            alert_callback: Function to call when alert triggered
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.alert_callback = alert_callback
        
        # Tracking storage
        self.predictions: Dict[str, List[float]] = defaultdict(list)
        self.features: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.latencies: List[float] = []
        self.timestamps: List[datetime] = []
        
        # Baseline statistics (set during calibration)
        self.baseline_stats: Dict[str, Any] = {}
        
        # Alert history
        self.alerts: List[Dict] = []
    
    def record_prediction(
        self,
        antibiotic: str,
        probability: float,
        features: Optional[np.ndarray] = None,
        latency_ms: Optional[float] = None
    ):
        """Record a prediction for monitoring"""
        self.predictions[antibiotic].append(probability)
        self.timestamps.append(datetime.utcnow())
        
        if features is not None:
            self.features[antibiotic].append(features)
        
        if latency_ms is not None:
            self.latencies.append(latency_ms)
        
        # Trim to window size
        if len(self.predictions[antibiotic]) > self.window_size:
            self.predictions[antibiotic] = self.predictions[antibiotic][-self.window_size:]
        
        if len(self.features[antibiotic]) > self.window_size:
            self.features[antibiotic] = self.features[antibiotic][-self.window_size:]
        
        if len(self.latencies) > self.window_size:
            self.latencies = self.latencies[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
        
        # Check for drift
        self._check_prediction_drift(antibiotic)
        self._check_feature_drift(antibiotic)
    
    def set_baseline(self, antibiotic: str, predictions: List[float], features: Optional[np.ndarray] = None):
        """Set baseline statistics for drift detection"""
        self.baseline_stats[antibiotic] = {
            "pred_mean": np.mean(predictions),
            "pred_std": np.std(predictions),
            "pred_quantiles": np.percentile(predictions, [25, 50, 75])
        }
        
        if features is not None:
            self.baseline_stats[antibiotic]["feature_means"] = np.mean(features, axis=0)
            self.baseline_stats[antibiotic]["feature_stds"] = np.std(features, axis=0)
    
    def _check_prediction_drift(self, antibiotic: str):
        """Check for drift in prediction distribution"""
        if antibiotic not in self.baseline_stats:
            return
        
        if len(self.predictions[antibiotic]) < 100:
            return
        
        baseline = self.baseline_stats[antibiotic]
        current_mean = np.mean(self.predictions[antibiotic][-100:])
        
        # Check if mean shifted significantly
        shift = abs(current_mean - baseline["pred_mean"]) / (baseline["pred_std"] + 1e-6)
        
        if shift > self.drift_threshold * 10:  # More than 1 std shift
            self._trigger_alert(
                alert_type="prediction_drift",
                antibiotic=antibiotic,
                message=f"Prediction distribution shifted: mean {baseline['pred_mean']:.3f} -> {current_mean:.3f}",
                severity="warning" if shift < self.drift_threshold * 20 else "critical"
            )
    
    def _check_feature_drift(self, antibiotic: str):
        """Check for drift in feature distributions"""
        if antibiotic not in self.baseline_stats:
            return
        
        if "feature_means" not in self.baseline_stats[antibiotic]:
            return
        
        if len(self.features[antibiotic]) < 100:
            return
        
        baseline = self.baseline_stats[antibiotic]
        recent_features = np.array(self.features[antibiotic][-100:])
        current_means = np.mean(recent_features, axis=0)
        
        # Calculate normalized drift per feature
        drifts = np.abs(current_means - baseline["feature_means"]) / (baseline["feature_stds"] + 1e-6)
        max_drift = np.max(drifts)
        max_drift_idx = np.argmax(drifts)
        
        if max_drift > self.drift_threshold * 10:
            self._trigger_alert(
                alert_type="feature_drift",
                antibiotic=antibiotic,
                message=f"Feature {max_drift_idx} drifted significantly (z-score: {max_drift:.2f})",
                severity="warning"
            )
    
    def _trigger_alert(
        self,
        alert_type: str,
        antibiotic: str,
        message: str,
        severity: str = "warning"
    ):
        """Trigger an alert"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "antibiotic": antibiotic,
            "message": message,
            "severity": severity
        }
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
        
        if self.alert_callback:
            self.alert_callback(alert)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "predictions_tracked": {ab: len(preds) for ab, preds in self.predictions.items()},
            "latency": {
                "mean_ms": np.mean(self.latencies) if self.latencies else 0,
                "p95_ms": np.percentile(self.latencies, 95) if len(self.latencies) > 10 else 0,
                "p99_ms": np.percentile(self.latencies, 99) if len(self.latencies) > 100 else 0
            },
            "prediction_distributions": {},
            "recent_alerts": self.alerts[-10:]  # Last 10 alerts
        }
        
        for ab, preds in self.predictions.items():
            if len(preds) > 0:
                stats["prediction_distributions"][ab] = {
                    "mean": float(np.mean(preds)),
                    "std": float(np.std(preds)),
                    "min": float(np.min(preds)),
                    "max": float(np.max(preds)),
                    "high_risk_rate": float(np.mean(np.array(preds) > 0.5))
                }
        
        return stats
    
    def get_performance_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate performance report for a time period"""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.utcnow()
        
        # Filter by time
        mask = [
            start_time <= ts <= end_time
            for ts in self.timestamps[-len(self.latencies):]
        ]
        
        filtered_latencies = [
            l for l, m in zip(self.latencies[-len(mask):], mask) if m
        ]
        
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_predictions": sum(len(p) for p in self.predictions.values()),
            "latency_stats": {
                "mean_ms": np.mean(filtered_latencies) if filtered_latencies else 0,
                "median_ms": np.median(filtered_latencies) if filtered_latencies else 0,
                "p95_ms": np.percentile(filtered_latencies, 95) if len(filtered_latencies) > 10 else 0,
                "max_ms": max(filtered_latencies) if filtered_latencies else 0
            },
            "alerts_in_period": [
                a for a in self.alerts
                if start_time.isoformat() <= a["timestamp"] <= end_time.isoformat()
            ],
            "health_status": "healthy"
        }
        
        # Determine health status
        if len(report["alerts_in_period"]) > 10:
            report["health_status"] = "degraded"
        
        critical_alerts = [a for a in report["alerts_in_period"] if a.get("severity") == "critical"]
        if critical_alerts:
            report["health_status"] = "critical"
        
        return report


class DataDriftDetector:
    """
    Detect data drift using statistical tests.
    Supports Population Stability Index (PSI) and Kolmogorov-Smirnov test.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize drift detector.
        
        Args:
            n_bins: Number of bins for PSI calculation
        """
        self.n_bins = n_bins
        self.reference_distributions: Dict[str, np.ndarray] = {}
    
    def set_reference(self, feature_name: str, data: np.ndarray):
        """Set reference distribution for a feature"""
        # Create histogram bins
        self.reference_distributions[feature_name] = {
            "data": data,
            "hist": np.histogram(data, bins=self.n_bins)[0] / len(data),
            "edges": np.histogram(data, bins=self.n_bins)[1]
        }
    
    def calculate_psi(self, feature_name: str, current_data: np.ndarray) -> float:
        """
        Calculate Population Stability Index.
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        if feature_name not in self.reference_distributions:
            return 0.0
        
        ref = self.reference_distributions[feature_name]
        
        # Use same bins as reference
        current_hist = np.histogram(current_data, bins=ref["edges"])[0] / len(current_data)
        
        # Avoid division by zero
        ref_hist = np.clip(ref["hist"], 1e-6, 1)
        current_hist = np.clip(current_hist, 1e-6, 1)
        
        # PSI formula
        psi = np.sum((current_hist - ref_hist) * np.log(current_hist / ref_hist))
        
        return float(psi)
    
    def calculate_ks_statistic(self, feature_name: str, current_data: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        if feature_name not in self.reference_distributions:
            return 0.0
        
        from scipy import stats
        
        ref_data = self.reference_distributions[feature_name]["data"]
        ks_stat, _ = stats.ks_2samp(ref_data, current_data)
        
        return float(ks_stat)
    
    def detect_drift(
        self,
        feature_name: str,
        current_data: np.ndarray,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect drift using both PSI and KS tests.
        
        Returns:
            Dictionary with drift detection results
        """
        psi = self.calculate_psi(feature_name, current_data)
        ks = self.calculate_ks_statistic(feature_name, current_data)
        
        return {
            "feature": feature_name,
            "psi": psi,
            "ks_statistic": ks,
            "psi_drift_detected": psi >= psi_threshold,
            "ks_drift_detected": ks >= ks_threshold,
            "drift_detected": psi >= psi_threshold or ks >= ks_threshold
        }


# Singleton monitor instance
_monitor_instance: Optional[ModelMonitor] = None


def get_monitor() -> ModelMonitor:
    """Get singleton monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance
