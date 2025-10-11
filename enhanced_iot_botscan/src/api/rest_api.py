"""
REST API Implementation for Enhanced IoT BotScan
Implements SRS-CI-001, SRS-CI-002 requirements

Author: Kotiwale Sumesh Singh (160124862043)
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import jwt
import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.ensemble.hybrid_ensemble import HybridEnsemble
from core.drift_detection.drift_detector import DriftDetector
from utils.config_manager import ConfigManager
from utils.logger import get_logger

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600))

# Enable CORS (SRS-UI-001)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Rate limiting (SRS-CI-001)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per minute"],
    storage_uri="memory://"
)

# Logger
logger = get_logger(__name__)

# Global model instance
model = None
drift_detector = None
config_manager = None

def init_system():
    """Initialize system components"""
    global model, drift_detector, config_manager
    
    try:
        config_manager = ConfigManager()
        model = HybridEnsemble()
        drift_detector = DriftDetector(config_manager.get_drift_config())
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

# Authentication decorator (SRS-NF-006)
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# API Routes (SRS-CI-002)

@app.route('/api/v1/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """
    Authenticate user and return JWT token
    
    Request Body:
    {
        "username": "string",
        "password": "string"
    }
    """
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # TODO: Implement proper authentication against database
        # This is a simplified version for demonstration
        if username and password:
            # Generate JWT token
            token = jwt.encode({
                'user': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(
                    seconds=app.config['JWT_ACCESS_TOKEN_EXPIRES']
                )
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            return jsonify({
                'token': token,
                'expires_in': app.config['JWT_ACCESS_TOKEN_EXPIRES']
            }), 200
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Authentication failed'}), 500

@app.route('/api/v1/detect', methods=['POST'])
@token_required
@limiter.limit("1000 per minute")
def detect(current_user):
    """
    Real-time botnet detection endpoint (SRS-CI-002)
    
    Request Body:
    {
        "features": [[...]], // Network flow features
        "metadata": {...}   // Optional metadata
    }
    
    Response:
    {
        "predictions": [0, 1, ...],
        "confidence_scores": [0.95, 0.87, ...],
        "attack_types": ["Benign", "Mirai", ...],
        "processing_time": 0.123,
        "drift_detected": false
    }
    """
    try:
        if not model or not model.is_trained:
            return jsonify({'error': 'Model not trained'}), 503
        
        data = request.get_json()
        features = data.get('features')
        
        if not features:
            return jsonify({'error': 'Features are required'}), 400
        
        start_time = datetime.datetime.now()
        
        # Convert to DataFrame
        X = pd.DataFrame(features)
        
        # Make predictions (SRS-HDE-001)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Get confidence scores (SRS-HDE-004)
        confidence_scores = np.max(probabilities, axis=1).tolist()
        
        # Detect drift (SRS-CDD-001)
        drift_result = drift_detector.detect_drift(X_new=X.values)
        
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Map predictions to attack types
        label_mapping = {0: 'Benign', 1: 'Mirai', 2: 'Gafgyt', 3: 'Bashlite', 4: 'Other'}
        attack_types = [label_mapping.get(p, 'Unknown') for p in predictions]
        
        response = {
            'predictions': predictions.tolist(),
            'confidence_scores': confidence_scores,
            'attack_types': attack_types,
            'probabilities': probabilities.tolist(),
            'processing_time': processing_time,
            'drift_detected': drift_result.get('drift_detected', False),
            'timestamp': datetime.datetime.now().isoformat(),
            'user': current_user
        }
        
        logger.info(f"Detection request processed for {len(features)} samples in {processing_time:.3f}s")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/status', methods=['GET'])
@token_required
def get_status(current_user):
    """
    System health and status endpoint (SRS-CI-002, SRS-NF-011)
    
    Response:
    {
        "status": "healthy",
        "model_loaded": true,
        "model_trained": true,
        "uptime": 123456,
        "version": "1.0.0",
        "performance_metrics": {...}
    }
    """
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'model_trained': model.is_trained if model else False,
            'version': '1.0.0',
            'timestamp': datetime.datetime.now().isoformat(),
            'drift_detector_active': drift_detector is not None,
            'user': current_user
        }
        
        if model and model.is_trained:
            status['model_info'] = model.get_model_info()
        
        if drift_detector:
            status['drift_statistics'] = drift_detector.get_comprehensive_statistics()
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/v1/analytics', methods=['GET'])
@token_required
def get_analytics(current_user):
    """
    Historical analysis and analytics endpoint (SRS-CI-002)
    
    Query Parameters:
    - start_date: ISO format datetime
    - end_date: ISO format datetime
    - metric: specific metric to retrieve
    
    Response:
    {
        "analytics": {...},
        "period": {...},
        "statistics": {...}
    }
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        metric = request.args.get('metric', 'all')
        
        # TODO: Implement actual analytics retrieval from database
        analytics = {
            'total_detections': 0,
            'threat_breakdown': {},
            'detection_accuracy': 0.0,
            'false_positive_rate': 0.0,
            'processing_statistics': {},
            'period': {
                'start': start_date,
                'end': end_date
            }
        }
        
        if drift_detector:
            analytics['drift_statistics'] = drift_detector.get_comprehensive_statistics()
        
        return jsonify(analytics), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/config', methods=['GET', 'PUT'])
@token_required
def manage_config(current_user):
    """
    System configuration endpoint (SRS-CI-002)
    
    GET - Retrieve current configuration
    PUT - Update configuration
    """
    try:
        if request.method == 'GET':
            if config_manager:
                return jsonify(config_manager.config), 200
            return jsonify({'error': 'Configuration not available'}), 500
        
        elif request.method == 'PUT':
            new_config = request.get_json()
            
            if config_manager:
                config_manager.update_config(new_config)
                config_manager.save_config()
                
                logger.info(f"Configuration updated by {current_user}")
                
                return jsonify({
                    'message': 'Configuration updated successfully',
                    'timestamp': datetime.datetime.now().isoformat()
                }), 200
            
            return jsonify({'error': 'Configuration manager not available'}), 500
            
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/model/train', methods=['POST'])
@token_required
@limiter.limit("1 per hour")
def train_model(current_user):
    """
    Trigger model training
    
    Request Body:
    {
        "dataset": "n_baiot",
        "parameters": {...}
    }
    """
    try:
        data = request.get_json()
        dataset = data.get('dataset', 'n_baiot')
        
        # TODO: Implement async training
        return jsonify({
            'message': 'Training initiated',
            'dataset': dataset,
            'status': 'in_progress',
            'timestamp': datetime.datetime.now().isoformat()
        }), 202
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint (no authentication required)
    Implements SRS-NF-011
    """
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.datetime.now().isoformat()
    }), 200

@app.route('/api/v1/metrics', methods=['GET'])
def metrics():
    """
    Prometheus-compatible metrics endpoint
    Implements SRS-NF-021
    """
    # TODO: Implement Prometheus metrics
    return "# Metrics endpoint\n", 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def ratelimit_handler(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

def run_api_server(host='0.0.0.0', port=8000, debug=False):
    """
    Run the API server
    Implements SRS-CI-001
    """
    init_system()
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_api_server()
