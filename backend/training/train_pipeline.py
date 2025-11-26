"""
Model Training Pipeline for AMR Prediction
Handles data preparation, model training, evaluation, and calibration
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AMRTrainingPipeline:
    """
    Complete training pipeline for AMR prediction models.
    
    Supports:
    - Patient-wise train/val/test split
    - Temporal holdout validation
    - XGBoost, LightGBM, and MLP models
    - Model stacking
    - Calibration (isotonic and Platt)
    - Comprehensive evaluation metrics
    """
    
    def __init__(
        self,
        target_antibiotics: List[str],
        model_type: str = "xgboost",
        random_state: int = 42,
        output_dir: Optional[str] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            target_antibiotics: List of antibiotics to train models for
            model_type: Type of model ('xgboost', 'lightgbm', 'mlp', 'stacking')
            random_state: Random seed for reproducibility
            output_dir: Directory to save trained models
        """
        self.target_antibiotics = target_antibiotics
        self.model_type = model_type
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else Path("models/trained")
        
        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Dict[str, float]] = {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        label_column: str = "resistant",
        patient_id_column: str = "patient_id",
        time_column: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        temporal_split: bool = False
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data with patient-wise or temporal split.
        
        Args:
            df: DataFrame with features and labels
            label_column: Name of label column
            patient_id_column: Column with patient IDs
            time_column: Column with timestamps (for temporal split)
            test_size: Proportion for test set
            val_size: Proportion for validation set
            temporal_split: Whether to use temporal holdout
            
        Returns:
            Dictionary with train/val/test splits per antibiotic
        """
        splits = {}
        
        # Identify feature columns (exclude metadata)
        exclude_cols = [label_column, patient_id_column, time_column, 'antibiotic_target']
        exclude_cols = [c for c in exclude_cols if c is not None and c in df.columns]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_names = feature_cols
        
        for antibiotic in self.target_antibiotics:
            # Filter data for this antibiotic
            if 'antibiotic_target' in df.columns:
                ab_df = df[df['antibiotic_target'] == antibiotic].copy()
            else:
                ab_df = df.copy()
            
            if len(ab_df) == 0:
                logger.warning(f"No data for {antibiotic}, skipping")
                continue
            
            X = ab_df[feature_cols].values
            y = ab_df[label_column].values
            patients = ab_df[patient_id_column].values
            
            if temporal_split and time_column:
                # Temporal holdout
                ab_df = ab_df.sort_values(time_column)
                n = len(ab_df)
                
                train_end = int(n * (1 - test_size - val_size))
                val_end = int(n * (1 - test_size))
                
                X_train, y_train = X[:train_end], y[:train_end]
                X_val, y_val = X[train_end:val_end], y[train_end:val_end]
                X_test, y_test = X[val_end:], y[val_end:]
            else:
                # Patient-wise split
                unique_patients = np.unique(patients)
                np.random.seed(self.random_state)
                np.random.shuffle(unique_patients)
                
                n_patients = len(unique_patients)
                train_end = int(n_patients * (1 - test_size - val_size))
                val_end = int(n_patients * (1 - test_size))
                
                train_patients = set(unique_patients[:train_end])
                val_patients = set(unique_patients[train_end:val_end])
                test_patients = set(unique_patients[val_end:])
                
                train_mask = np.array([p in train_patients for p in patients])
                val_mask = np.array([p in val_patients for p in patients])
                test_mask = np.array([p in test_patients for p in patients])
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_val, y_val = X[val_mask], y[val_mask]
                X_test, y_test = X[test_mask], y[test_mask]
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            self.scalers[antibiotic] = scaler
            
            splits[antibiotic] = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test)
            }
            
            logger.info(
                f"{antibiotic}: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
                f"pos_rate_train={y_train.mean():.3f}"
            )
        
        return splits
    
    def create_model(self, antibiotic: str) -> Any:
        """Create model based on model_type"""
        if self.model_type == "xgboost":
            return self._create_xgboost_model()
        elif self.model_type == "lightgbm":
            return self._create_lightgbm_model()
        elif self.model_type == "mlp":
            return self._create_mlp_model()
        elif self.model_type == "stacking":
            return self._create_stacking_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_xgboost_model(self):
        """Create XGBoost classifier"""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=1,  # Adjusted during training
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=20
            )
        except ImportError:
            logger.warning("XGBoost not installed, falling back to sklearn")
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
    
    def _create_lightgbm_model(self):
        """Create LightGBM classifier"""
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1
            )
        except ImportError:
            logger.warning("LightGBM not installed, falling back to XGBoost")
            return self._create_xgboost_model()
    
    def _create_mlp_model(self):
        """Create MLP classifier"""
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=self.random_state
        )
    
    def _create_stacking_model(self):
        """Create stacking ensemble"""
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        estimators = [
            ('xgb', self._create_xgboost_model()),
            ('lgb', self._create_lightgbm_model()),
            ('mlp', self._create_mlp_model())
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
    
    def train_single_antibiotic(
        self,
        antibiotic: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Any:
        """Train model for a single antibiotic"""
        logger.info(f"Training model for {antibiotic}")
        
        model = self.create_model(antibiotic)
        
        # Handle class imbalance
        pos_rate = y_train.mean()
        if pos_rate < 0.5:
            scale_pos_weight = (1 - pos_rate) / pos_rate
            if hasattr(model, 'scale_pos_weight'):
                model.scale_pos_weight = scale_pos_weight
        
        # Train with early stopping if validation data provided
        if X_val is not None and hasattr(model, 'fit'):
            try:
                # XGBoost style
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            except TypeError:
                # Fallback for other models
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        self.models[antibiotic] = model
        return model
    
    def calibrate_model(
        self,
        antibiotic: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "isotonic"
    ):
        """
        Calibrate model probabilities.
        
        Args:
            antibiotic: Antibiotic name
            X_val: Validation features
            y_val: Validation labels
            method: Calibration method ('isotonic' or 'sigmoid')
        """
        model = self.models.get(antibiotic)
        if model is None:
            logger.warning(f"No model for {antibiotic}, skipping calibration")
            return
        
        logger.info(f"Calibrating {antibiotic} model using {method}")
        
        calibrator = CalibratedClassifierCV(
            model,
            method=method,
            cv='prefit'
        )
        
        calibrator.fit(X_val, y_val)
        self.calibrators[antibiotic] = calibrator
    
    def evaluate(
        self,
        antibiotic: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_calibrated: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns comprehensive metrics including:
        - AUC-ROC, AUC-PR
        - F1, Precision, Recall
        - Brier score (calibration)
        - Precision at fixed recall levels
        """
        model = self.calibrators.get(antibiotic) if use_calibrated else self.models.get(antibiotic)
        if model is None:
            model = self.models.get(antibiotic)
        
        if model is None:
            logger.warning(f"No model for {antibiotic}")
            return {}
        
        # Get predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # ROC-AUC
        metrics['auc_roc'] = roc_auc_score(y_test, y_prob)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        metrics['auc_pr'] = auc(recall, precision)
        
        # Classification metrics
        metrics['f1'] = f1_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        
        # Calibration (Brier score)
        metrics['brier_score'] = brier_score_loss(y_test, y_prob)
        
        # Precision at fixed recall (clinical utility)
        for target_recall in [0.8, 0.9, 0.95]:
            # Find threshold for target recall
            recall_thresholds = recall[::-1]
            precision_at_recall = precision[::-1]
            
            for i, r in enumerate(recall_thresholds):
                if r >= target_recall:
                    metrics[f'precision_at_{int(target_recall*100)}_recall'] = precision_at_recall[i]
                    break
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        self.metrics[antibiotic] = metrics
        
        logger.info(f"{antibiotic} - AUC-ROC: {metrics['auc_roc']:.3f}, F1: {metrics['f1']:.3f}, Brier: {metrics['brier_score']:.3f}")
        
        return metrics
    
    def train_all(
        self,
        splits: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        calibrate: bool = True
    ):
        """Train models for all antibiotics"""
        for antibiotic, data in splits.items():
            X_train, y_train = data["train"]
            X_val, y_val = data["val"]
            X_test, y_test = data["test"]
            
            # Train
            self.train_single_antibiotic(antibiotic, X_train, y_train, X_val, y_val)
            
            # Calibrate
            if calibrate:
                self.calibrate_model(antibiotic, X_val, y_val)
            
            # Evaluate
            self.evaluate(antibiotic, X_test, y_test, use_calibrated=calibrate)
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        label_column: str = "resistant",
        patient_id_column: str = "patient_id",
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform patient-wise cross-validation.
        
        Args:
            df: DataFrame with features and labels
            label_column: Name of label column
            patient_id_column: Column with patient IDs
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of metrics per antibiotic
        """
        cv_results = {ab: {"auc_roc": [], "f1": [], "brier": []} for ab in self.target_antibiotics}
        
        exclude_cols = [label_column, patient_id_column, 'antibiotic_target']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        for antibiotic in self.target_antibiotics:
            if 'antibiotic_target' in df.columns:
                ab_df = df[df['antibiotic_target'] == antibiotic].copy()
            else:
                ab_df = df.copy()
            
            if len(ab_df) < n_splits * 10:
                logger.warning(f"Insufficient data for {antibiotic} CV")
                continue
            
            X = ab_df[feature_cols].values
            y = ab_df[label_column].values
            groups = ab_df[patient_id_column].values
            
            gkf = GroupKFold(n_splits=n_splits)
            
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Train
                model = self.create_model(antibiotic)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                
                cv_results[antibiotic]["auc_roc"].append(roc_auc_score(y_test, y_prob))
                cv_results[antibiotic]["f1"].append(f1_score(y_test, y_pred))
                cv_results[antibiotic]["brier"].append(brier_score_loss(y_test, y_prob))
            
            # Log summary
            auc_mean = np.mean(cv_results[antibiotic]["auc_roc"])
            auc_std = np.std(cv_results[antibiotic]["auc_roc"])
            logger.info(f"{antibiotic} CV AUC-ROC: {auc_mean:.3f} ± {auc_std:.3f}")
        
        return cv_results
    
    def save_models(self, version: str = "1.0.0"):
        """Save trained models to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main models
        model_data = {
            "models": self.models,
            "feature_names": self.feature_names,
            "version": version,
            "model_type": self.model_type,
            "target_antibiotics": self.target_antibiotics,
            "trained_at": timestamp
        }
        
        model_path = self.output_dir / f"amr_model_{timestamp}.joblib"
        joblib.dump(model_data, model_path)
        logger.info(f"Saved models to {model_path}")
        
        # Save calibrators
        if self.calibrators:
            cal_path = self.output_dir / f"amr_calibrator_{timestamp}.joblib"
            joblib.dump(self.calibrators, cal_path)
            logger.info(f"Saved calibrators to {cal_path}")
        
        # Save scalers
        if self.scalers:
            scaler_path = self.output_dir / f"amr_scalers_{timestamp}.joblib"
            joblib.dump(self.scalers, scaler_path)
            logger.info(f"Saved scalers to {scaler_path}")
        
        # Save metrics
        metrics_path = self.output_dir / f"amr_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Create symlinks to latest
        for name, path in [
            ("amr_model.joblib", model_path),
            ("amr_calibrator.joblib", self.output_dir / f"amr_calibrator_{timestamp}.joblib"),
            ("amr_scalers.joblib", scaler_path),
            ("amr_metrics.json", metrics_path)
        ]:
            if path.exists():
                link_path = self.output_dir / name
                if link_path.exists():
                    link_path.unlink()
                try:
                    link_path.symlink_to(path.name)
                except OSError:
                    # Windows fallback
                    import shutil
                    shutil.copy(path, link_path)
        
        return str(model_path)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            "model_type": self.model_type,
            "target_antibiotics": self.target_antibiotics,
            "feature_count": len(self.feature_names),
            "metrics_by_antibiotic": self.metrics,
            "summary": {}
        }
        
        # Calculate summary statistics
        for metric in ["auc_roc", "f1", "brier_score"]:
            values = [m.get(metric, 0) for m in self.metrics.values() if metric in m]
            if values:
                report["summary"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return report


def create_synthetic_training_data(
    n_samples: int = 1000,
    n_antibiotics: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create synthetic training data for testing the pipeline.
    
    This generates realistic-looking but fake data.
    """
    np.random.seed(random_state)
    
    antibiotics = [
        "ceftriaxone", "ciprofloxacin", "meropenem", 
        "vancomycin", "gentamicin"
    ][:n_antibiotics]
    
    data = []
    
    for _ in range(n_samples):
        patient_id = f"PT_{np.random.randint(1000, 9999)}"
        
        for ab in antibiotics:
            # Image features
            row = {
                "patient_id": patient_id,
                "antibiotic_target": ab,
                "img_neutrophil_count": np.random.lognormal(2, 0.5),
                "img_lymphocyte_count": np.random.lognormal(1.5, 0.5),
                "img_nlr": np.random.lognormal(1, 0.8),
                "img_bacterial_cluster_count": np.random.poisson(0.5),
                "img_bacterial_confidence": np.random.uniform(0, 1),
                
                # Vitals
                "vital_temp": np.random.normal(37.5, 1.2),
                "vital_hr": np.random.normal(85, 15),
                "vital_rr": np.random.normal(18, 4),
                "vital_bp_sys": np.random.normal(120, 20),
                "vital_sirs_criteria": np.random.randint(0, 4),
                
                # Labs
                "lab_wbc": np.random.lognormal(2.2, 0.4),
                "lab_crp": np.random.lognormal(2, 1.5),
                "lab_lactate": np.random.lognormal(0.5, 0.5),
                "lab_creatinine": np.random.lognormal(0.1, 0.3),
                
                # History
                "hist_antibiotic_count_90d": np.random.poisson(1),
                "hist_prior_amr": np.random.binomial(1, 0.15),
                "hist_hospitalizations_30d": np.random.poisson(0.3),
                
                # Demographics
                "demo_age": np.random.normal(55, 18),
                "demo_male": np.random.binomial(1, 0.5),
                
                # Context
                "ctx_is_icu": np.random.binomial(1, 0.2),
                "ctx_days_since_admission": np.random.poisson(3),
            }
            
            # Generate label based on features (simulate real relationship)
            base_prob = 0.2
            prob = base_prob
            
            # Risk factors increase resistance probability
            prob += 0.1 * row["hist_prior_amr"]
            prob += 0.02 * row["hist_antibiotic_count_90d"]
            prob += 0.05 * row["ctx_is_icu"]
            prob += 0.001 * row["lab_crp"]
            prob += 0.01 * row["img_bacterial_cluster_count"]
            prob += 0.05 * (row["demo_age"] > 65)
            
            # Antibiotic-specific adjustments
            if ab == "ampicillin":
                prob += 0.2  # Higher resistance
            elif ab == "meropenem":
                prob -= 0.1  # Lower resistance
            
            prob = np.clip(prob, 0.05, 0.95)
            row["resistant"] = np.random.binomial(1, prob)
            
            data.append(row)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    df = create_synthetic_training_data(n_samples=500)
    
    # Initialize pipeline
    pipeline = AMRTrainingPipeline(
        target_antibiotics=["ceftriaxone", "ciprofloxacin", "meropenem"],
        model_type="xgboost"
    )
    
    # Prepare data
    splits = pipeline.prepare_data(df)
    
    # Train models
    pipeline.train_all(splits)
    
    # Generate report
    report = pipeline.generate_report()
    print(json.dumps(report, indent=2, default=str))
    
    # Save models
    pipeline.save_models()
