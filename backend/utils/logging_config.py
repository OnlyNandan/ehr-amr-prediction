"""
Logging Configuration for AMR Prediction System
Structured logging with JSON output for production
"""
import logging
import sys
from typing import Optional
from pathlib import Path

import structlog
from flask import Flask


def setup_logging(app: Flask, log_level: Optional[str] = None):
    """
    Setup structured logging for the Flask application.
    
    Args:
        app: Flask application
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    level = log_level or app.config.get("LOG_LEVEL", "INFO")
    log_dir = Path(app.config.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not app.debug else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Set level for all loggers
    logging.getLogger().setLevel(getattr(logging, level.upper()))
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    
    # Setup file handler for production
    if not app.debug:
        file_handler = logging.FileHandler(log_dir / "amr_api.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        ))
        logging.getLogger().addHandler(file_handler)
        
        # Separate error log
        error_handler = logging.FileHandler(log_dir / "amr_api_errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        ))
        logging.getLogger().addHandler(error_handler)
    
    return structlog.get_logger()


class AuditLogger:
    """
    Specialized logger for audit events.
    Ensures compliance with HIPAA/GDPR requirements.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.logger = structlog.get_logger("audit")
        self.log_path = log_path or Path("logs/audit.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler for audit log
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter('%(message)s'))
        
        audit_logger = logging.getLogger("audit")
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
    
    def log_prediction(
        self,
        patient_id: str,
        prediction_id: str,
        antibiotics: list,
        risk_levels: list,
        user_id: Optional[str] = None
    ):
        """Log a prediction event"""
        self.logger.info(
            "prediction_made",
            patient_id=patient_id,
            prediction_id=prediction_id,
            antibiotics=antibiotics,
            risk_levels=risk_levels,
            user_id=user_id
        )
    
    def log_data_access(
        self,
        patient_id: str,
        data_type: str,
        user_id: Optional[str] = None,
        purpose: Optional[str] = None
    ):
        """Log a data access event"""
        self.logger.info(
            "data_accessed",
            patient_id=patient_id,
            data_type=data_type,
            user_id=user_id,
            purpose=purpose
        )
    
    def log_override(
        self,
        prediction_id: str,
        original_decision: str,
        override_decision: str,
        reason: str,
        clinician_id: str
    ):
        """Log a clinical override"""
        self.logger.info(
            "clinical_override",
            prediction_id=prediction_id,
            original_decision=original_decision,
            override_decision=override_decision,
            reason=reason,
            clinician_id=clinician_id
        )
    
    def log_model_deployment(
        self,
        model_version: str,
        deployment_type: str,
        deployed_by: str
    ):
        """Log model deployment event"""
        self.logger.info(
            "model_deployed",
            model_version=model_version,
            deployment_type=deployment_type,
            deployed_by=deployed_by
        )
