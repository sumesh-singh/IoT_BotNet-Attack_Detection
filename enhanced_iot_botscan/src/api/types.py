"""
GraphQL Types for Enhanced IoT BotScan
"""
import strawberry
from typing import List, Optional
from datetime import datetime

@strawberry.type
class ThreatType:
    """Threat detection result type."""
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

@strawberry.type
class ModelPerformanceType:
    """Model performance metrics type."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    last_updated: datetime
    status: str

@strawberry.type
class DatasetType:
    """Dataset information type."""
    name: str
    type: str
    size: int
    features: int
    last_updated: datetime
    status: str
    description: str

@strawberry.type
class AdversarialTestType:
    """Adversarial testing result type."""
    test_id: str
    attack_type: str
    model_name: str
    success_rate: float
    accuracy_drop: float
    perturbation_norm: float
    test_timestamp: datetime
    status: str

@strawberry.type
class ConceptDriftType:
    """Concept drift detection result type."""
    drift_id: str
    drift_type: str
    confidence: float
    severity: str
    detection_time: float
    performance_drop: float
    detected_at: datetime
    resolved: bool

@strawberry.type
class SystemMetricsType:
    """System performance metrics type."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_load: float
    timestamp: datetime

@strawberry.type
class AlertType:
    """System alert type."""
    alert_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool
    acknowledged: bool

@strawberry.type
class DashboardSummaryType:
    """Dashboard summary data type."""
    total_threats: int
    active_models: int
    average_accuracy: float
    drift_alerts: int
    system_status: str
    last_updated: datetime

@strawberry.type
class AnalyticsDataType:
    """Analytics data type."""
    performance_metrics: List[ModelPerformanceType]
    robustness_metrics: List[AdversarialTestType]
    drift_metrics: List[ConceptDriftType]
    time_range: str
    generated_at: datetime

@strawberry.input
class ThreatInput:
    """Input type for threat detection."""
    source_ip: str
    destination_ip: str
    port: int
    protocol: str
    features: List[float]

@strawberry.input
class ModelTrainingInput:
    """Input type for model training."""
    model_name: str
    dataset_name: str
    hyperparameters: Optional[str] = None # JSON string
    validation_split: Optional[float] = None

@strawberry.input
class AdversarialTestInput:
    """Input type for adversarial testing."""
    model_name: str
    attack_type: str
    epsilon: Optional[float] = None
    num_iterations: Optional[int] = None
    test_samples: Optional[int] = None

@strawberry.type
class ThreatDetectionResult:
    success: bool
    threat: Optional[ThreatType]
    error: Optional[str]

@strawberry.type
class ModelTrainingResult:
    success: bool
    model: Optional[ModelPerformanceType]
    error: Optional[str]

@strawberry.type
class AdversarialTestResult:
    success: bool
    test_result: Optional[AdversarialTestType]
    error: Optional[str]

@strawberry.type
class AlertAcknowledgmentResult:
    success: bool
    alert: Optional[AlertType]
    error: Optional[str]
