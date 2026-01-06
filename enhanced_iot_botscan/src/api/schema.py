"""
GraphQL Schema and Resolvers for Enhanced IoT BotScan
"""
import strawberry
from typing import List, Optional, Any
from datetime import datetime, timedelta
import logging

from .types import (
    ThreatType, ModelPerformanceType, DatasetType, AdversarialTestType,
    ConceptDriftType, SystemMetricsType, AlertType, DashboardSummaryType,
    AnalyticsDataType, ThreatInput, ModelTrainingInput, AdversarialTestInput,
    ThreatDetectionResult, ModelTrainingResult, AdversarialTestResult, AlertAcknowledgmentResult
)
from .mock_data import db, ThreatData, ModelData

logger = logging.getLogger(__name__)

@strawberry.type
class Query:
    @strawberry.field
    def threats(self, limit: Optional[int] = None, offset: Optional[int] = None,
                threat_type: Optional[str] = None) -> List[ThreatType]:
        results = db.threats_db
        if threat_type:
            results = [t for t in results if t.threat_type == threat_type]
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]
        
        return [ThreatType(
            id=t.id, timestamp=t.timestamp, threat_type=t.threat_type,
            confidence=t.confidence, source_ip=t.source_ip, destination_ip=t.destination_ip,
            port=t.port, protocol=t.protocol, severity=t.severity,
            model_used=t.model_used, false_positive=t.false_positive
        ) for t in results]

    @strawberry.field
    def threat(self, threat_id: str) -> Optional[ThreatType]:
        t = next((t for t in db.threats_db if t.id == threat_id), None)
        if t:
            return ThreatType(
                id=t.id, timestamp=t.timestamp, threat_type=t.threat_type,
                confidence=t.confidence, source_ip=t.source_ip, destination_ip=t.destination_ip,
                port=t.port, protocol=t.protocol, severity=t.severity,
                model_used=t.model_used, false_positive=t.false_positive
            )
        return None

    @strawberry.field
    def models(self) -> List[ModelPerformanceType]:
        return [ModelPerformanceType(
            model_name=m.name, accuracy=m.accuracy, precision=m.precision,
            recall=m.recall, f1_score=m.f1_score, roc_auc=m.roc_auc,
            training_time=m.training_time, last_updated=m.last_updated, status=m.status
        ) for m in db.models_db]

    @strawberry.field
    def model(self, model_name: str) -> Optional[ModelPerformanceType]:
        m = next((m for m in db.models_db if m.name == model_name), None)
        if m:
            return ModelPerformanceType(
                model_name=m.name, accuracy=m.accuracy, precision=m.precision,
                recall=m.recall, f1_score=m.f1_score, roc_auc=m.roc_auc,
                training_time=m.training_time, last_updated=m.last_updated, status=m.status
            )
        return None

    @strawberry.field
    def datasets(self) -> List[DatasetType]:
        return [DatasetType(**d) for d in db.datasets_db]

    @strawberry.field
    def adversarial_tests(self, model_name: Optional[str] = None, attack_type: Optional[str] = None) -> List[AdversarialTestType]:
        tests = db.adversarial_tests_db
        if model_name:
            tests = [t for t in tests if t["model_name"] == model_name]
        if attack_type:
            tests = [t for t in tests if t["attack_type"] == attack_type]
        return [AdversarialTestType(**t) for t in tests]

    @strawberry.field
    def concept_drifts(self, resolved: Optional[bool] = None) -> List[ConceptDriftType]:
        drifts = db.concept_drifts_db
        if resolved is not None:
            drifts = [d for d in drifts if d["resolved"] == resolved]
        return [ConceptDriftType(**d) for d in drifts]

    @strawberry.field
    def system_metrics(self) -> SystemMetricsType:
        return SystemMetricsType(
            cpu_usage=45.2,
            memory_usage=67.8,
            disk_usage=23.1,
            network_load=12.5,
            timestamp=datetime.now()
        )

    @strawberry.field
    def alerts(self, severity: Optional[str] = None, resolved: Optional[bool] = None) -> List[AlertType]:
        alerts = db.alerts_db
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        if resolved is not None:
            alerts = [a for a in alerts if a["resolved"] == resolved]
        return [AlertType(**a) for a in alerts]

    @strawberry.field
    def dashboard_summary(self) -> DashboardSummaryType:
        return DashboardSummaryType(
            total_threats=len(db.threats_db),
            active_models=len([m for m in db.models_db if m.status == "active"]),
            average_accuracy=sum(m.accuracy for m in db.models_db) / len(db.models_db) if db.models_db else 0.0,
            drift_alerts=len([a for a in db.alerts_db if a["alert_type"] == "drift_detection"]),
            system_status="operational",
            last_updated=datetime.now()
        )

    @strawberry.field
    def analytics_data(self, time_range: str = "24h", metric: str = "accuracy") -> AnalyticsDataType:
        return AnalyticsDataType(
            performance_metrics=[ModelPerformanceType(
                model_name=m.name, accuracy=m.accuracy, precision=m.precision,
                recall=m.recall, f1_score=m.f1_score, roc_auc=m.roc_auc,
                training_time=m.training_time, last_updated=m.last_updated, status=m.status
            ) for m in db.models_db],
            robustness_metrics=[AdversarialTestType(**t) for t in db.adversarial_tests_db],
            drift_metrics=[ConceptDriftType(**d) for d in db.concept_drifts_db],
            time_range=time_range,
            generated_at=datetime.now()
        )

@strawberry.type
class Mutation:
    @strawberry.mutation
    def detect_threat(self, threat_data: ThreatInput) -> ThreatDetectionResult:
        try:
            threat = ThreatData(
                id=f"threat_{len(db.threats_db) + 1:03d}",
                timestamp=datetime.now(),
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
            db.threats_db.append(threat)
            
            return ThreatDetectionResult(
                success=True,
                threat=ThreatType(
                    id=threat.id, timestamp=threat.timestamp, threat_type=threat.threat_type,
                    confidence=threat.confidence, source_ip=threat.source_ip, destination_ip=threat.destination_ip,
                    port=threat.port, protocol=threat.protocol, severity=threat.severity,
                    model_used=threat.model_used, false_positive=threat.false_positive
                ),
                error=None
            )
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return ThreatDetectionResult(success=False, threat=None, error=str(e))

    @strawberry.mutation
    def train_model(self, training_data: ModelTrainingInput) -> ModelTrainingResult:
        try:
            model = ModelData(
                name=training_data.model_name,
                accuracy=0.95,
                precision=0.93,
                recall=0.91,
                f1_score=0.92,
                roc_auc=0.97,
                training_time=120.5,
                last_updated=datetime.now(),
                status="trained"
            )
            
            existing = next((m for m in db.models_db if m.name == model.name), None)
            if existing:
                existing.accuracy = model.accuracy
                # ... update other fields if needed
            else:
                db.models_db.append(model)
                
            return ModelTrainingResult(
                success=True,
                model=ModelPerformanceType(
                    model_name=model.name, accuracy=model.accuracy, precision=model.precision,
                    recall=model.recall, f1_score=model.f1_score, roc_auc=model.roc_auc,
                    training_time=model.training_time, last_updated=model.last_updated, status=model.status
                ),
                error=None
            )
        except Exception as e:
             return ModelTrainingResult(success=False, model=None, error=str(e))

    @strawberry.mutation
    def run_adversarial_test(self, test_data: AdversarialTestInput) -> AdversarialTestResult:
        try:
            test_result = {
                "test_id": f"adv_{len(db.adversarial_tests_db) + 1:03d}",
                "attack_type": test_data.attack_type,
                "model_name": test_data.model_name,
                "success_rate": 0.15,
                "accuracy_drop": 0.08,
                "perturbation_norm": test_data.epsilon or 0.05,
                "test_timestamp": datetime.now(),
                "status": "completed"
            }
            db.adversarial_tests_db.append(test_result)
            return AdversarialTestResult(success=True, test_result=AdversarialTestType(**test_result), error=None)
        except Exception as e:
            return AdversarialTestResult(success=False, test_result=None, error=str(e))

    @strawberry.mutation
    def acknowledge_alert(self, alert_id: str) -> AlertAcknowledgmentResult:
        try:
            alert = next((a for a in db.alerts_db if a["alert_id"] == alert_id), None)
            if alert:
                alert["acknowledged"] = True
                return AlertAcknowledgmentResult(success=True, alert=AlertType(**alert), error=None)
            return AlertAcknowledgmentResult(success=False, alert=None, error="Alert not found")
        except Exception as e:
            return AlertAcknowledgmentResult(success=False, alert=None, error=str(e))

schema = strawberry.Schema(query=Query, mutation=Mutation)
