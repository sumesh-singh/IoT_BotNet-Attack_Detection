"""
Mock Database/Data Layer for Enhanced IoT BotScan
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ThreatData:
    id: str
    timestamp: datetime
    threat_type: str
    confidence: float
    source_ip: str
    destination_ip: str
    port: int
    protocol: str
    severity: str
    model_used: str
    false_positive: bool

@dataclass
class ModelData:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    last_updated: datetime
    status: str

class MockDatabase:
    def __init__(self):
        self.threats_db: List[ThreatData] = []
        self.models_db: List[ModelData] = []
        self.datasets_db: List[Dict] = []
        self.adversarial_tests_db: List[Dict] = []
        self.concept_drifts_db: List[Dict] = []
        self.alerts_db: List[Dict] = []
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        # Sample threats
        self.threats_db = [
            ThreatData(
                id="threat_001",
                timestamp=datetime.now() - timedelta(minutes=5),
                threat_type="botnet",
                confidence=0.95,
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                port=80,
                protocol="TCP",
                severity="high",
                model_used="ensemble",
                false_positive=False
            ),
            ThreatData(
                id="threat_002",
                timestamp=datetime.now() - timedelta(minutes=15),
                threat_type="ddos",
                confidence=0.87,
                source_ip="192.168.1.101",
                destination_ip="10.0.0.2",
                port=443,
                protocol="TCP",
                severity="medium",
                model_used="xgboost",
                false_positive=False
            )
        ]

        # Sample models
        self.models_db = [
            ModelData(
                name="Random Forest",
                accuracy=0.92,
                precision=0.89,
                recall=0.87,
                f1_score=0.88,
                roc_auc=0.94,
                training_time=120.5,
                last_updated=datetime.now() - timedelta(hours=2),
                status="active"
            ),
            ModelData(
                name="XGBoost",
                accuracy=0.94,
                precision=0.91,
                recall=0.89,
                f1_score=0.90,
                roc_auc=0.96,
                training_time=95.2,
                last_updated=datetime.now() - timedelta(hours=1),
                status="active"
            ),
            ModelData(
                name="LightGBM",
                accuracy=0.93,
                precision=0.90,
                recall=0.88,
                f1_score=0.89,
                roc_auc=0.95,
                training_time=78.8,
                last_updated=datetime.now() - timedelta(minutes=30),
                status="active"
            ),
            ModelData(
                name="Ensemble",
                accuracy=0.96,
                precision=0.94,
                recall=0.92,
                f1_score=0.93,
                roc_auc=0.98,
                training_time=180.3,
                last_updated=datetime.now() - timedelta(minutes=15),
                status="active"
            )
        ]

        # Sample datasets
        self.datasets_db = [
            {
                "name": "N-BaIoT",
                "type": "training",
                "size": 1000000,
                "features": 115,
                "last_updated": datetime.now() - timedelta(days=1),
                "status": "active",
                "description": "N-BaIoT dataset for IoT botnet detection"
            },
            {
                "name": "IoT-23",
                "type": "validation",
                "size": 500000,
                "features": 115,
                "last_updated": datetime.now() - timedelta(hours=6),
                "status": "active",
                "description": "IoT-23 dataset for validation"
            },
            {
                "name": "BoT-IoT",
                "type": "test",
                "size": 750000,
                "features": 115,
                "last_updated": datetime.now() - timedelta(hours=12),
                "status": "active",
                "description": "BoT-IoT dataset for testing"
            }
        ]

        # Sample adversarial tests
        self.adversarial_tests_db = [
            {
                "test_id": "adv_001",
                "attack_type": "FGSM",
                "model_name": "Random Forest",
                "success_rate": 0.15,
                "accuracy_drop": 0.08,
                "perturbation_norm": 0.05,
                "test_timestamp": datetime.now() - timedelta(hours=1),
                "status": "completed"
            },
            {
                "test_id": "adv_002",
                "attack_type": "PGD",
                "model_name": "XGBoost",
                "success_rate": 0.22,
                "accuracy_drop": 0.12,
                "perturbation_norm": 0.08,
                "test_timestamp": datetime.now() - timedelta(minutes=30),
                "status": "completed"
            }
        ]

        # Sample concept drifts
        self.concept_drifts_db = [
            {
                "drift_id": "drift_001",
                "drift_type": "covariate_shift",
                "confidence": 0.85,
                "severity": "medium",
                "detection_time": 2.3,
                "performance_drop": 0.05,
                "detected_at": datetime.now() - timedelta(hours=2),
                "resolved": False
            }
        ]

        # Sample alerts
        self.alerts_db = [
            {
                "alert_id": "alert_001",
                "alert_type": "drift_detection",
                "severity": "medium",
                "message": "Concept drift detected in network traffic patterns",
                "timestamp": datetime.now() - timedelta(minutes=10),
                "resolved": False,
                "acknowledged": False
            },
            {
                "alert_id": "alert_002",
                "alert_type": "high_false_positive",
                "severity": "high",
                "message": "High false positive rate detected in Random Forest model",
                "timestamp": datetime.now() - timedelta(minutes=5),
                "resolved": False,
                "acknowledged": True
            }
        ]

# Global instance
db = MockDatabase()
