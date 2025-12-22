"""
GraphQL Schema for Enhanced IoT BotScan
Migrated from graphene to strawberry-graphql
"""

import strawberry
from typing import List, Optional
from datetime import datetime

# ============================================
# Type Definitions
# ============================================

@strawberry.type
class Device:
    """IoT Device representation"""
    id: strawberry.ID
    name: str
    ip_address: str
    mac_address: str
    device_type: str
    status: str
    last_seen: datetime
    threat_level: Optional[str] = None

@strawberry.type
class ThreatDetection:
    """Threat detection result"""
    id: strawberry.ID
    device_id: str
    threat_type: str
    severity: str
    confidence: float
    detected_at: datetime
    description: str

@strawberry.type
class ModelMetrics:
    """ML Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_trained: datetime

@strawberry.type
class SystemStatus:
    """System health status"""
    status: str
    active_devices: int
    threats_detected: int
    models_active: int
    uptime_seconds: int

# ============================================
# Input Types
# ============================================

@strawberry.input
class DeviceFilterInput:
    """Filter options for device queries"""
    device_type: Optional[str] = None
    status: Optional[str] = None
    threat_level: Optional[str] = None

@strawberry.input
class CreateDeviceInput:
    """Input for creating a new device"""
    name: str
    ip_address: str
    mac_address: str
    device_type: str

# ============================================
# Query Resolvers
# ============================================

@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        """Test query"""
        return "Enhanced IoT BotScan API - GraphQL Endpoint"
    
    @strawberry.field
    def system_status(self) -> SystemStatus:
        """Get current system status"""
        # TODO: Implement actual logic
        return SystemStatus(
            status="operational",
            active_devices=0,
            threats_detected=0,
            models_active=0,
            uptime_seconds=0
        )
    
    @strawberry.field
    def devices(
        self,
        filters: Optional[DeviceFilterInput] = None,
        limit: int = 100
    ) -> List[Device]:
        """Get list of monitored devices"""
        # TODO: Implement actual database query
        return []
    
    @strawberry.field
    def device(self, device_id: strawberry.ID) -> Optional[Device]:
        """Get single device by ID"""
        # TODO: Implement actual database query
        return None
    
    @strawberry.field
    def threats(
        self,
        device_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[ThreatDetection]:
        """Get threat detections"""
        # TODO: Implement actual database query
        return []
    
    @strawberry.field
    def model_metrics(self) -> List[ModelMetrics]:
        """Get ML model performance metrics"""
        # TODO: Implement actual logic
        return []

# ============================================
# Mutation Resolvers
# ============================================

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_device(self, input: CreateDeviceInput) -> Device:
        """Register a new device"""
        # TODO: Implement actual database insert
        return Device(
            id=strawberry.ID("temp-id"),
            name=input.name,
            ip_address=input.ip_address,
            mac_address=input.mac_address,
            device_type=input.device_type,
            status="active",
            last_seen=datetime.now()
        )
    
    @strawberry.mutation
    def train_model(
        self,
        model_name: str,
        dataset_path: str
    ) -> bool:
        """Trigger model training"""
        # TODO: Implement actual training logic
        return True
    
    @strawberry.mutation
    def scan_device(self, device_id: strawberry.ID) -> bool:
        """Trigger security scan for a device"""
        # TODO: Implement actual scan logic
        return True

# ============================================
# Schema
# ============================================

schema = strawberry.Schema(query=Query, mutation=Mutation)
