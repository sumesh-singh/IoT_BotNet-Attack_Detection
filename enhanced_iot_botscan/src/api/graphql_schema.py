"""
GraphQL Schema for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Defines the GraphQL schema for the IoT botnet detection system API.
"""

from graphene import ObjectType, String, Int, Float, Boolean, List, Field, Mutation, InputObjectType
from graphene.types.datetime import DateTime
import graphene


class ThreatType(ObjectType):
    """Threat detection result type."""
    id = String()
    timestamp = DateTime()
    threat_type = String()
    confidence = Float()
    source_ip = String()
    destination_ip = String()
    port = Int()
    protocol = String()
    severity = String()
    model_used = String()
    false_positive = Boolean()


class ModelPerformanceType(ObjectType):
    """Model performance metrics type."""
    model_name = String()
    accuracy = Float()
    precision = Float()
    recall = Float()
    f1_score = Float()
    roc_auc = Float()
    training_time = Float()
    last_updated = DateTime()
    status = String()


class DatasetType(ObjectType):
    """Dataset information type."""
    name = String()
    type = String()
    size = Int()
    features = Int()
    last_updated = DateTime()
    status = String()
    description = String()


class AdversarialTestType(ObjectType):
    """Adversarial testing result type."""
    test_id = String()
    attack_type = String()
    model_name = String()
    success_rate = Float()
    accuracy_drop = Float()
    perturbation_norm = Float()
    test_timestamp = DateTime()
    status = String()


class ConceptDriftType(ObjectType):
    """Concept drift detection result type."""
    drift_id = String()
    drift_type = String()
    confidence = Float()
    severity = String()
    detection_time = Float()
    performance_drop = Float()
    detected_at = DateTime()
    resolved = Boolean()


class SystemMetricsType(ObjectType):
    """System performance metrics type."""
    cpu_usage = Float()
    memory_usage = Float()
    disk_usage = Float()
    network_load = Float()
    timestamp = DateTime()


class AlertType(ObjectType):
    """System alert type."""
    alert_id = String()
    alert_type = String()
    severity = String()
    message = String()
    timestamp = DateTime()
    resolved = Boolean()
    acknowledged = Boolean()


class ThreatInput(InputObjectType):
    """Input type for threat detection."""
    source_ip = String(required=True)
    destination_ip = String(required=True)
    port = Int(required=True)
    protocol = String(required=True)
    features = List(Float, required=True)


class ModelTrainingInput(InputObjectType):
    """Input type for model training."""
    model_name = String(required=True)
    dataset_name = String(required=True)
    hyperparameters = String()  # JSON string
    validation_split = Float()


class AdversarialTestInput(InputObjectType):
    """Input type for adversarial testing."""
    model_name = String(required=True)
    attack_type = String(required=True)
    epsilon = Float()
    num_iterations = Int()
    test_samples = Int()



class DashboardSummaryType(ObjectType):
    """Dashboard summary data type."""
    total_threats = Int()
    active_models = Int()
    average_accuracy = Float()
    drift_alerts = Int()
    system_status = String()
    last_updated = DateTime()


class AnalyticsDataType(ObjectType):
    """Analytics data type."""
    performance_metrics = List(ModelPerformanceType)
    robustness_metrics = List(AdversarialTestType)
    drift_metrics = List(ConceptDriftType)
    time_range = String()
    generated_at = DateTime()


class Query(ObjectType):
    """Root query type."""

    # Threat detection queries
    threats = List(ThreatType, limit=Int(), offset=Int(), threat_type=String())
    threat = Field(ThreatType, threat_id=String(required=True))

    # Model performance queries
    models = List(ModelPerformanceType)
    model = Field(ModelPerformanceType, model_name=String(required=True))
    model_performance_history = List(
        ModelPerformanceType, model_name=String(required=True), days=Int())

    # Dataset queries
    datasets = List(DatasetType)
    dataset = Field(DatasetType, dataset_name=String(required=True))

    # Adversarial testing queries
    adversarial_tests = List(
        AdversarialTestType, model_name=String(), attack_type=String())
    adversarial_test = Field(
        AdversarialTestType, test_id=String(required=True))

    # Concept drift queries
    concept_drifts = List(ConceptDriftType, resolved=Boolean())
    concept_drift = Field(ConceptDriftType, drift_id=String(required=True))

    # System metrics queries
    system_metrics = Field(SystemMetricsType)
    system_metrics_history = List(SystemMetricsType, hours=Int())

    # Alert queries
    alerts = List(AlertType, severity=String(), resolved=Boolean())
    alert = Field(AlertType, alert_id=String(required=True))

    # Dashboard queries
    dashboard_summary = Field(DashboardSummaryType)
    analytics_data = Field(AnalyticsDataType,
                           time_range=String(), metric=String())







class ThreatDetectionMutation(Mutation):
    """Mutation for threat detection."""
    class Arguments:
        threat_data = ThreatInput(required=True)

    success = Boolean()
    threat = Field(ThreatType)
    error = String()

    def mutate(self, info, threat_data):
        # Implement threat detection logic
        try:
            # This would call the actual threat detection system
            threat = ThreatType(
                id="threat_123",
                timestamp=DateTime.now(),
                threat_type="botnet",
                confidence=0.95,
                source_ip=threat_data.source_ip,
                destination_ip=threat_data.destination_ip,
                port=threat_data.port,
                protocol=threat_data.protocol,
                severity="high",
                model_used="ensemble",
                false_positive=False
            )
            return ThreatDetectionMutation(success=True, threat=threat)
        except Exception as e:
            return ThreatDetectionMutation(success=False, error=str(e))


class ModelTrainingMutation(Mutation):
    """Mutation for model training."""
    class Arguments:
        training_data = ModelTrainingInput(required=True)

    success = Boolean()
    model = Field(ModelPerformanceType)
    error = String()

    def mutate(self, info, training_data):
        # Implement model training logic
        try:
            # This would call the actual model training system
            model = ModelPerformanceType(
                model_name=training_data.model_name,
                accuracy=0.95,
                precision=0.93,
                recall=0.91,
                f1_score=0.92,
                roc_auc=0.97,
                training_time=120.5,
                last_updated=DateTime.now(),
                status="trained"
            )
            return ModelTrainingMutation(success=True, model=model)
        except Exception as e:
            return ModelTrainingMutation(success=False, error=str(e))


class AdversarialTestingMutation(Mutation):
    """Mutation for adversarial testing."""
    class Arguments:
        test_data = AdversarialTestInput(required=True)

    success = Boolean()
    test_result = Field(AdversarialTestType)
    error = String()

    def mutate(self, info, test_data):
        # Implement adversarial testing logic
        try:
            # This would call the actual adversarial testing system
            test_result = AdversarialTestType(
                test_id="test_123",
                attack_type=test_data.attack_type,
                model_name=test_data.model_name,
                success_rate=0.15,
                accuracy_drop=0.08,
                perturbation_norm=0.05,
                test_timestamp=DateTime.now(),
                status="completed"
            )
            return AdversarialTestingMutation(success=True, test_result=test_result)
        except Exception as e:
            return AdversarialTestingMutation(success=False, error=str(e))


class AlertAcknowledgmentMutation(Mutation):
    """Mutation for acknowledging alerts."""
    class Arguments:
        alert_id = String(required=True)

    success = Boolean()
    alert = Field(AlertType)
    error = String()

    def mutate(self, info, alert_id):
        # Implement alert acknowledgment logic
        try:
            # This would update the alert status
            alert = AlertType(
                alert_id=alert_id,
                alert_type="drift_detection",
                severity="medium",
                message="Concept drift detected",
                timestamp=DateTime.now(),
                resolved=False,
                acknowledged=True
            )
            return AlertAcknowledgmentMutation(success=True, alert=alert)
        except Exception as e:
            return AlertAcknowledgmentMutation(success=False, error=str(e))


class Mutation(ObjectType):
    """Root mutation type."""
    detect_threat = ThreatDetectionMutation.Field()
    train_model = ModelTrainingMutation.Field()
    run_adversarial_test = AdversarialTestingMutation.Field()
    acknowledge_alert = AlertAcknowledgmentMutation.Field()


class Subscription(ObjectType):
    """Root subscription type for real-time updates."""

    threat_detected = Field(ThreatType)
    model_performance_updated = Field(
        ModelPerformanceType, model_name=String())
    concept_drift_detected = Field(ConceptDriftType)
    system_metrics_updated = Field(SystemMetricsType)
    alert_created = Field(AlertType)

    def resolve_threat_detected(self, info):
        # This would be connected to a real-time threat detection system
        pass

    def resolve_model_performance_updated(self, info, model_name=None):
        # This would be connected to model performance monitoring
        pass

    def resolve_concept_drift_detected(self, info):
        # This would be connected to concept drift detection system
        pass

    def resolve_system_metrics_updated(self, info):
        # This would be connected to system monitoring
        pass

    def resolve_alert_created(self, info):
        # This would be connected to alerting system
        pass


# Create the schema
schema = graphene.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)
