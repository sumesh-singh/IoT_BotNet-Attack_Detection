# Enhanced IoT BotScan

**Author:** Kotiwale Sumesh Singh (160124862043)  
**Project:** Enhanced IoT Botnet Detection System with Hybrid Ensemble Learning

## Overview

Enhanced IoT BotScan is a comprehensive IoT botnet detection system that combines multiple machine learning models with advanced techniques for robust and real-time threat detection. The system features hybrid ensemble learning, adversarial robustness testing, concept drift detection, and a modern web-based dashboard.

## Key Features

### 🤖 **Hybrid Ensemble Learning**

- **Random Forest**: Robust tree-based classifier
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **LightGBM**: Fast gradient boosting for large datasets
- **Meta-Learner**: Stacking ensemble for optimal performance
- **Hybrid Ensemble**: Combines all models with intelligent weighting

### 🛡️ **Adversarial Robustness**

- **FGSM Attack**: Fast Gradient Sign Method implementation
- **PGD Attack**: Projected Gradient Descent with multiple iterations
- **C&W Attack**: Carlini & Wagner attack for strong adversarial examples
- **Robust Training**: Adversarial training pipeline
- **Defense Mechanisms**: Gradient masking, feature squeezing, input validation

### 📊 **Concept Drift Detection**

- **Kolmogorov-Smirnov Test**: Statistical drift detection
- **Page-Hinkley Test**: Sequential change detection
- **Adaptive Learning**: Automatic model retraining
- **Performance Monitoring**: Continuous accuracy tracking

### 🔍 **Advanced Preprocessing**

- **Data Cleaning**: Missing value handling, outlier detection, duplicate removal
- **Feature Engineering**: Statistical features, polynomial features, domain-specific features
- **Dimensionality Reduction**: PCA, ICA, SVD, t-SNE, Isomap
- **Scaling**: Multiple scaling techniques (Standard, MinMax, Robust, Quantile, Power)

### 📈 **Comprehensive Evaluation**

- **Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC, log-loss
- **Cross-Validation**: Stratified, time-series, nested CV
- **Robustness Testing**: Noise, outlier, missing value, corruption resistance
- **Multi-Dataset Validation**: Cross-dataset evaluation and transfer learning

### 🌐 **Modern Web Interface**

- **Real-time Dashboard**: Live threat monitoring and system status
- **Analytics Page**: Performance metrics, robustness analysis, drift detection
- **GraphQL API**: Flexible data querying and mutations
- **WebSocket**: Real-time updates and notifications

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │   Analytics     │    │   Mobile App    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      FastAPI Server        │
                    │  ┌─────────────────────┐   │
                    │  │   GraphQL API       │   │
                    │  └─────────────────────┘   │
                    │  ┌─────────────────────┐   │
                    │  │   WebSocket Handler │   │
                    │  └─────────────────────┘   │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │    Core ML System        │
                    │  ┌─────────────────────┐ │
                    │  │  Hybrid Ensemble    │ │
                    │  └─────────────────────┘ │
                    │  ┌─────────────────────┐ │
                    │  │  Adversarial Tests  │ │
                    │  └─────────────────────┘ │
                    │  ┌─────────────────────┐ │
                    │  │  Drift Detection    │ │
                    │  └─────────────────────┘ │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      Data Layer           │
                    │  ┌─────────────────────┐ │
                    │  │  Dataset Manager    │ │
                    │  └─────────────────────┘ │
                    │  ┌─────────────────────┐ │
                    │  │  Validation Handler │ │
                    │  └─────────────────────┘ │
                    └───────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Kubernetes (optional)
- Git

### Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd enhanced_iot_botscan
   ```

2. **Build and deploy with Docker Compose**

   ```bash
   ./scripts/deploy.sh deploy-docker
   ```

3. **Access the application**
   - Dashboard: http://localhost:8000/dashboard
   - Analytics: http://localhost:8000/analytics
   - GraphQL API: http://localhost:8000/graphql
   - WebSocket: ws://localhost:8000/ws

### Manual Installation

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   python -m src.api.main
   ```

## Usage

### Training Models

```python
from src.core.ensemble.hybrid_ensemble import HybridEnsemble
from src.core.data.dataset_manager import DatasetManager

# Load dataset
manager = DatasetManager()
df = manager.load_n_baiot_dataset('data/n_baiot.csv')

# Create train/test split
splits = manager.create_train_test_split('n_baiot', stratify_column='label')

# Initialize ensemble
ensemble = HybridEnsemble({
    'use_stacking': True,
    'stacking_cv_folds': 5
})

# Train ensemble
results = ensemble.train(
    splits['train'].drop('label', axis=1),
    splits['train']['label'],
    validation_data=(splits['validation'].drop('label', axis=1), splits['validation']['label'])
)
```

### Adversarial Testing

```python
from src.core.adversarial.attack_generator import AttackGenerator
from src.core.evaluation.robustness_tester import RobustnessTester

# Generate adversarial attacks
attack_generator = AttackGenerator()
attacks = attack_generator.generate_all_attacks(model, X_test, y_test)

# Test robustness
tester = RobustnessTester()
robustness_results = tester.test_comprehensive_robustness(model, X, y)
```

### Concept Drift Detection

```python
from src.core.drift_detection.kolmogorov_smirnov import KolmogorovSmirnovDriftDetector

# Initialize drift detector
detector = KolmogorovSmirnovDriftDetector({
    'alpha': 0.05,
    'min_samples': 100
})

# Set reference data
detector.set_reference_data(X_reference, y_reference)

# Detect drift in new data
drift_results = detector.detect_drift(X_new, y_new)
```

## API Documentation

### GraphQL Queries

```graphql
# Get model performance
query {
  models {
    modelName
    accuracy
    precision
    recall
    f1Score
    status
  }
}

# Get recent threats
query {
  threats(limit: 10) {
    id
    threatType
    confidence
    timestamp
    severity
  }
}

# Get system metrics
query {
  systemMetrics {
    cpuUsage
    memoryUsage
    diskUsage
    networkLoad
  }
}
```

### GraphQL Mutations

```graphql
# Detect threat
mutation {
  detectThreat(threatData: {
    sourceIp: "192.168.1.100"
    destinationIp: "10.0.0.1"
    port: 80
    protocol: "TCP"
    features: [1.2, 3.4, 5.6, ...]
  }) {
    success
    threat {
      id
      threatType
      confidence
    }
  }
}

# Train model
mutation {
  trainModel(trainingData: {
    modelName: "Random Forest"
    datasetName: "N-BaIoT"
    hyperparameters: "{\"n_estimators\": 100}"
  }) {
    success
    model {
      modelName
      accuracy
      status
    }
  }
}
```

### WebSocket Events

```javascript
// Connect to WebSocket
const ws = new WebSocket("ws://localhost:8000/ws");

// Subscribe to channels
ws.send(
  JSON.stringify({
    type: "subscribe",
    channels: ["threats", "models", "drift", "alerts"],
  })
);

// Handle real-time updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case "threat_detected":
      console.log("New threat:", data.data);
      break;
    case "model_performance_updated":
      console.log("Model updated:", data.data);
      break;
    case "concept_drift_detected":
      console.log("Drift detected:", data.data);
      break;
  }
};
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/iot_botscan

# Redis
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO

# Model Configuration
USE_STACKING=true
STACKING_CV_FOLDS=5
OPTIMIZE_BASE_MODELS=false

# Adversarial Testing
EPSILON_RANGE=0.01,0.05,0.1,0.2
ATTACK_ITERATIONS=10

# Drift Detection
DRIFT_ALPHA=0.05
MIN_SAMPLES=100
```

### Model Configuration

```python
config = {
    'ensemble': {
        'use_stacking': True,
        'stacking_cv_folds': 5,
        'optimize_base_models': False
    },
    'preprocessing': {
        'create_statistical_features': True,
        'create_polynomial_features': False,
        'feature_selection_method': 'mutual_info',
        'n_features_select': 50
    },
    'adversarial': {
        'epsilon_range': [0.01, 0.05, 0.1, 0.2],
        'num_iterations': 10,
        'norms': ['inf', '2']
    },
    'drift_detection': {
        'alpha': 0.05,
        'min_samples': 100,
        'window_size': 1000
    }
}
```

## Deployment

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f iot-botscan

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml

# Check deployment
kubectl get pods -n iot-botscan

# Access via port-forward
kubectl port-forward -n iot-botscan service/iot-botscan-service 8000:8000
```

### Production Deployment

1. **Configure SSL/TLS**

   ```bash
   # Generate SSL certificates
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   ```

2. **Set up monitoring**

   ```bash
   # Deploy Prometheus and Grafana
   kubectl apply -f monitoring/prometheus.yaml
   kubectl apply -f monitoring/grafana.yaml
   ```

3. **Configure logging**
   ```bash
   # Deploy ELK stack
   kubectl apply -f logging/elasticsearch.yaml
   kubectl apply -f logging/kibana.yaml
   ```

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models/ -v
python -m pytest tests/test_adversarial/ -v
python -m pytest tests/test_drift/ -v
```

### Integration Tests

```bash
# Test API endpoints
python -m pytest tests/test_api/ -v

# Test WebSocket connections
python -m pytest tests/test_websocket/ -v
```

### Performance Tests

```bash
# Load testing
python -m pytest tests/test_performance/ -v

# Benchmark models
python scripts/benchmark.py
```

## Monitoring

### Metrics

- **Model Performance**: Accuracy, precision, recall, F1-score
- **System Metrics**: CPU, memory, disk usage
- **Threat Detection**: Detection rate, false positive rate
- **Drift Detection**: Drift frequency, adaptation time
- **Adversarial Robustness**: Attack success rate, accuracy drop

### Dashboards

- **Grafana**: System and application metrics
- **Kibana**: Log analysis and visualization
- **Custom Dashboard**: Real-time threat monitoring

### Alerts

- **High False Positive Rate**: Alert when FPR > 5%
- **Concept Drift**: Alert when drift detected
- **System Issues**: Alert on high resource usage
- **Model Degradation**: Alert on performance drop

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Datasets**: N-BaIoT, IoT-23, BoT-IoT
- **Libraries**: scikit-learn, XGBoost, LightGBM, PyTorch
- **Frameworks**: FastAPI, GraphQL, WebSocket
- **Infrastructure**: Docker, Kubernetes, Prometheus, Grafana

## Contact

**Author:** Kotiwale Sumesh Singh  
**Student ID:** 160124862043  
**Email:** [your-email@example.com]  
**Project:** Enhanced IoT Botnet Detection System

---

_This project is part of academic research in IoT security and machine learning for cybersecurity applications._
