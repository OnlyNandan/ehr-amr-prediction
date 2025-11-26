"""
Flask Middleware for AMR Prediction API
Handles monitoring, logging, and request processing
"""
import logging
import time
import uuid
from functools import wraps

from flask import Flask, request, g
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'amr_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'amr_api_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

PREDICTION_COUNT = Counter(
    'amr_predictions_total',
    'Total AMR predictions',
    ['antibiotic', 'risk_level']
)

MODEL_INFERENCE_LATENCY = Histogram(
    'amr_model_inference_latency_seconds',
    'Model inference latency',
    ['model_type'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)


def setup_middleware(app: Flask):
    """Setup middleware for the Flask application"""
    
    @app.before_request
    def before_request():
        """Pre-request processing"""
        # Generate request ID
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        g.start_time = time.time()
        
        # Log request
        logger.debug(
            f"Request started",
            extra={
                "request_id": g.request_id,
                "method": request.method,
                "path": request.path,
                "remote_addr": request.remote_addr
            }
        )
    
    @app.after_request
    def after_request(response):
        """Post-request processing"""
        # Calculate latency
        latency = time.time() - g.get('start_time', time.time())
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
        response.headers['X-Response-Time'] = f"{latency * 1000:.2f}ms"
        
        # Record metrics
        endpoint = request.endpoint or 'unknown'
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(latency)
        
        # Log response
        logger.debug(
            f"Request completed",
            extra={
                "request_id": g.get('request_id'),
                "status": response.status_code,
                "latency_ms": latency * 1000
            }
        )
        
        return response
    
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        from flask import Response
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    
    # Add CORS headers
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Request-ID, X-User-ID'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        return response
    
    # Handle OPTIONS requests for CORS preflight
    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def options_handler(path):
        from flask import Response
        return Response(status=200)


def record_prediction_metrics(antibiotic: str, risk_level: str):
    """Record metrics for a prediction"""
    PREDICTION_COUNT.labels(
        antibiotic=antibiotic,
        risk_level=risk_level
    ).inc()


def record_inference_latency(model_type: str, latency_seconds: float):
    """Record model inference latency"""
    MODEL_INFERENCE_LATENCY.labels(
        model_type=model_type
    ).observe(latency_seconds)
