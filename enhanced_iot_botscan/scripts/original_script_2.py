# Create comprehensive configuration files

# 1. Main configuration file (config.yaml)
config_yaml_content = """# Enhanced IoT BotScan System Configuration
# Author: Kotiwale Sumesh Singh (160124862043)

system:
  name: "Enhanced IoT BotScan Defense System"
  version: "1.0.0"
  environment: "development"  # development, staging, production
  debug_mode: true
  
database:
  primary:
    type: "postgresql"
    host: "${DB_HOST}"
    port: "${DB_PORT}"
    database: "${DB_NAME}"
    username: "${DB_USER}"
    password: "${DB_PASSWORD}"
    pool_size: 20
    max_overflow: 10
    
  secondary:
    type: "mongodb"
    host: "${MONGO_HOST}"
    port: "${MONGO_PORT}"
    database: "${MONGO_DB}"
    
  cache:
    type: "redis"
    host: "${REDIS_HOST}"
    port: "${REDIS_PORT}"
    database: "${REDIS_DB}"
    ttl: 3600

machine_learning:
  ensemble:
    algorithms:
      - name: "random_forest"
        enabled: true
        n_estimators: 100
        max_depth: 10
        random_state: 42
        n_jobs: -1
      - name: "xgboost"
        enabled: true
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        random_state: 42
      - name: "lightgbm"
        enabled: true
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        random_state: 42
        
  meta_learner:
    algorithm: "logistic_regression"
    cross_validation_folds: 5
    
  feature_engineering:
    scaling_method: "standard"  # standard, minmax, robust
    dimensionality_reduction:
      method: "pca"
      variance_threshold: 0.95
    feature_selection:
      method: "recursive_feature_elimination"
      n_features: 50

adversarial_training:
  enabled: true
  training_ratio: 0.3  # 30% adversarial examples
  attacks:
    fgsm:
      enabled: true
      epsilon: 0.1
      clip_min: 0.0
      clip_max: 1.0
    pgd:
      enabled: true
      epsilon: 0.1
      alpha: 0.01
      num_iter: 10
      clip_min: 0.0
      clip_max: 1.0
    cw:
      enabled: true
      confidence: 0.0
      learning_rate: 0.01
      max_iter: 1000

concept_drift:
  detection:
    enabled: true
    methods:
      - "kolmogorov_smirnov"
      - "page_hinkley"
    threshold: 0.05
    window_size: 1000
    monitoring_interval: 300  # seconds
    
  adaptation:
    enabled: true
    retraining_threshold: 0.05  # performance drop
    incremental_learning: true
    backup_models: 3

datasets:
  validation:
    n_baiot:
      path: "./data/raw/n_baiot/"
      enabled: true
      device_types: ["Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell", 
                    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
                    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
                    "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"]
    iot_23:
      path: "./data/raw/iot_23/"
      enabled: true
    bot_iot:
      path: "./data/raw/bot_iot/"
      enabled: true

api:
  rest:
    host: "${API_HOST}"
    port: "${API_PORT}"
    debug: false
    cors_enabled: true
    rate_limiting:
      enabled: true
      requests_per_minute: 100
      
  graphql:
    enabled: true
    endpoint: "/graphql"
    
  websocket:
    enabled: true
    endpoint: "/ws"
    max_connections: 1000

security:
  authentication:
    method: "jwt"  # jwt, oauth2, basic
    secret_key: "${JWT_SECRET_KEY}"
    token_expiry: 3600  # seconds
    
  authorization:
    rbac_enabled: true
    roles:
      - "admin"
      - "analyst"
      - "operator"
      - "viewer"
      
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_days: 90

monitoring:
  logging:
    level: "${LOG_LEVEL}"
    format: "${LOG_FORMAT}"
    file: "${LOG_FILE}"
    max_size_mb: 100
    backup_count: 10
    
  metrics:
    enabled: true
    endpoint: "/metrics"
    port: 9090
    
  health_checks:
    enabled: true
    interval: 30  # seconds
    endpoint: "/health"

performance:
  processing:
    batch_size: "${ML_BATCH_SIZE}"
    max_workers: "${MAX_WORKERS}"
    timeout: 300  # seconds
    
  caching:
    enabled: true
    ttl: 3600  # seconds
    max_size: 1000  # entries
    
  optimization:
    gpu_enabled: false
    parallel_processing: true
    memory_optimization: true

deployment:
  mode: "standalone"  # standalone, docker, kubernetes
  scaling:
    auto_scaling: false
    min_instances: 1
    max_instances: 10
    cpu_threshold: 80  # percent
    memory_threshold: 80  # percent
"""

with open('./enhanced_iot_botscan/config/config.yaml', 'w') as f:
    f.write(config_yaml_content)

print("✅ Created config/config.yaml with system configuration")

# 2. Logging configuration
logging_config_content = """# Logging Configuration for Enhanced IoT BotScan
# Structured logging with different levels and formatters

version: 1
disable_existing_loggers: False

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
    
  detailed:
    format: "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s (%(filename)s:%(funcName)s)"
    datefmt: "%Y-%m-%d %H:%M:%S"
    
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
    datefmt: "%Y-%m-%dT%H:%M:%S"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: ./logs/iot_botscan.log
    maxBytes: 104857600  # 100MB
    backupCount: 10
    encoding: utf8
    
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: ./logs/iot_botscan_errors.log
    maxBytes: 104857600  # 100MB
    backupCount: 5
    encoding: utf8
    
  json_file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: ./logs/iot_botscan_structured.log
    maxBytes: 104857600  # 100MB
    backupCount: 10
    encoding: utf8

loggers:
  enhanced_iot_botscan:
    level: DEBUG
    handlers: [console, file, error_file, json_file]
    propagate: False
    
  enhanced_iot_botscan.core:
    level: DEBUG
    handlers: [console, file, json_file]
    propagate: False
    
  enhanced_iot_botscan.adversarial:
    level: INFO
    handlers: [console, file, json_file]
    propagate: False
    
  enhanced_iot_botscan.api:
    level: INFO
    handlers: [console, file]
    propagate: False
    
  sqlalchemy.engine:
    level: WARNING
    handlers: [file]
    propagate: False
    
  werkzeug:
    level: WARNING
    handlers: [file]
    propagate: False

root:
  level: INFO
  handlers: [console, file]
"""

with open('./enhanced_iot_botscan/config/logging_config.yaml', 'w') as f:
    f.write(logging_config_content)

print("✅ Created config/logging_config.yaml with logging configuration")

# 3. Model-specific configuration (JSON)
model_config_content = """{
  "model_metadata": {
    "version": "1.0.0",
    "created_by": "Kotiwale Sumesh Singh",
    "created_date": "2025-10-04",
    "description": "Enhanced IoT BotScan with Hybrid Ensemble Learning",
    "license": "MIT"
  },
  
  "training_config": {
    "cross_validation": {
      "folds": 10,
      "stratified": true,
      "random_state": 42,
      "shuffle": true
    },
    
    "hyperparameter_tuning": {
      "method": "grid_search",
      "cv_folds": 5,
      "n_jobs": -1,
      "scoring": "f1_weighted"
    },
    
    "early_stopping": {
      "enabled": true,
      "patience": 10,
      "min_delta": 0.001
    }
  },
  
  "ensemble_config": {
    "stacking": {
      "cv_folds": 5,
      "shuffle": true,
      "random_state": 42,
      "passthrough": false
    },
    
    "base_models": {
      "random_forest": {
        "hyperparameter_space": {
          "n_estimators": [50, 100, 200],
          "max_depth": [5, 10, 15, null],
          "min_samples_split": [2, 5, 10],
          "min_samples_leaf": [1, 2, 4],
          "max_features": ["sqrt", "log2", null]
        }
      },
      
      "xgboost": {
        "hyperparameter_space": {
          "n_estimators": [50, 100, 200],
          "max_depth": [3, 6, 9],
          "learning_rate": [0.01, 0.1, 0.2],
          "subsample": [0.8, 0.9, 1.0],
          "colsample_bytree": [0.8, 0.9, 1.0]
        }
      },
      
      "lightgbm": {
        "hyperparameter_space": {
          "n_estimators": [50, 100, 200],
          "max_depth": [3, 6, 9],
          "learning_rate": [0.01, 0.1, 0.2],
          "num_leaves": [31, 127, 255],
          "feature_fraction": [0.8, 0.9, 1.0]
        }
      }
    },
    
    "meta_learner": {
      "algorithm": "logistic_regression",
      "hyperparameter_space": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["liblinear", "saga"]
      }
    }
  },
  
  "feature_config": {
    "preprocessing": {
      "handle_missing": "median",
      "handle_outliers": "iqr",
      "encoding_categorical": "one_hot"
    },
    
    "feature_selection": {
      "method": "recursive_feature_elimination",
      "n_features_to_select": 50,
      "step": 1,
      "cv": 5
    },
    
    "scaling": {
      "method": "standard",
      "with_mean": true,
      "with_std": true
    },
    
    "dimensionality_reduction": {
      "method": "pca",
      "n_components": null,
      "variance_threshold": 0.95,
      "whiten": false
    }
  },
  
  "adversarial_config": {
    "attack_methods": {
      "fgsm": {
        "epsilon_range": [0.01, 0.05, 0.1, 0.15, 0.2],
        "norm": "inf",
        "targeted": false
      },
      
      "pgd": {
        "epsilon_range": [0.01, 0.05, 0.1, 0.15, 0.2],
        "alpha_ratio": 0.1,
        "num_iter_range": [10, 20, 40],
        "norm": "inf",
        "targeted": false
      },
      
      "cw": {
        "c_range": [0.1, 1.0, 10.0],
        "kappa": 0.0,
        "max_iter": 1000,
        "learning_rate": 0.01,
        "binary_search_steps": 10
      }
    },
    
    "defense_methods": {
      "adversarial_training": {
        "adversarial_ratio": 0.3,
        "epochs": 50,
        "batch_size": 128
      },
      
      "gradient_masking": {
        "enabled": true,
        "noise_scale": 0.1
      },
      
      "input_transformation": {
        "methods": ["feature_squeezing", "bit_depth_reduction"],
        "parameters": {
          "bit_depth": 4,
          "spatial_smoothing": 2
        }
      }
    }
  },
  
  "drift_detection_config": {
    "statistical_tests": {
      "kolmogorov_smirnov": {
        "alpha": 0.05,
        "alternative": "two-sided"
      },
      
      "page_hinkley": {
        "threshold": 50,
        "alpha": 0.005
      },
      
      "adwin": {
        "delta": 0.002,
        "max_buckets": 5
      }
    },
    
    "performance_monitoring": {
      "metrics": ["accuracy", "precision", "recall", "f1_score"],
      "window_size": 1000,
      "threshold_degradation": 0.05
    },
    
    "adaptation_strategy": {
      "method": "incremental_learning",
      "retraining_frequency": "on_drift_detection",
      "batch_size": 1000,
      "learning_rate_decay": 0.95
    }
  },
  
  "evaluation_config": {
    "metrics": {
      "classification": [
        "accuracy", "precision", "recall", "f1_score", 
        "roc_auc", "confusion_matrix", "classification_report"
      ],
      "robustness": [
        "adversarial_accuracy", "certified_accuracy",
        "attack_success_rate", "perturbation_budget"
      ]
    },
    
    "validation": {
      "test_size": 0.2,
      "random_state": 42,
      "stratify": true
    },
    
    "cross_dataset_validation": {
      "datasets": ["n_baiot", "iot_23", "bot_iot"],
      "evaluation_protocol": "leave_one_dataset_out"
    }
  }
}"""

with open('./enhanced_iot_botscan/config/model_config.json', 'w') as f:
    f.write(model_config_content)

print("✅ Created config/model_config.json with ML configuration")

# 4. Deployment configuration
deployment_config_content = """# Deployment Configuration for Enhanced IoT BotScan
# Supports multiple deployment scenarios

development:
  database:
    type: sqlite
    path: ./data/dev.db
  api:
    host: localhost
    port: 8000
    debug: true
  web:
    host: localhost
    port: 5000
    debug: true
  logging:
    level: DEBUG
    console: true
    file: false

testing:
  database:
    type: sqlite
    path: ":memory:"
  api:
    host: localhost
    port: 8001
    debug: false
  web:
    host: localhost
    port: 5001
    debug: false
  logging:
    level: INFO
    console: true
    file: true

staging:
  database:
    type: postgresql
    host: staging-db
    port: 5432
    name: iot_botscan_staging
  api:
    host: 0.0.0.0
    port: 8000
    debug: false
    workers: 2
  web:
    host: 0.0.0.0
    port: 5000
    debug: false
  logging:
    level: INFO
    console: true
    file: true
  security:
    https: true
    cors_origins: ["https://staging.iot-botscan.com"]

production:
  database:
    type: postgresql
    host: prod-db-cluster
    port: 5432
    name: iot_botscan_prod
    ssl_mode: require
  api:
    host: 0.0.0.0
    port: 8000
    debug: false
    workers: 4
    max_requests: 1000
    timeout: 300
  web:
    host: 0.0.0.0
    port: 5000
    debug: false
  logging:
    level: WARNING
    console: false
    file: true
    structured: true
  security:
    https: true
    hsts: true
    cors_origins: ["https://iot-botscan.com"]
    rate_limiting: true
  monitoring:
    prometheus: true
    grafana: true
    health_checks: true
  scaling:
    auto_scaling: true
    min_replicas: 2
    max_replicas: 10
    cpu_threshold: 70
    memory_threshold: 80

docker:
  base_image: python:3.10-slim
  working_dir: /app
  expose_ports: [8000, 5000, 9090]
  volumes:
    - ./data:/app/data
    - ./logs:/app/logs
    - ./config:/app/config
  environment:
    - PYTHONPATH=/app/src
    - FLASK_ENV=production
  health_check:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3

kubernetes:
  namespace: iot-botscan
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi
  replicas: 3
  strategy:
    type: RollingUpdate
    rolling_update:
      max_surge: 1
      max_unavailable: 0
  service:
    type: ClusterIP
    ports:
      - name: api
        port: 8000
        target_port: 8000
      - name: web
        port: 5000
        target_port: 5000
  ingress:
    enabled: true
    annotations:
      kubernetes.io/ingress.class: nginx
      cert-manager.io/cluster-issuer: letsencrypt-prod
    hosts:
      - host: iot-botscan.example.com
        paths:
          - path: /
            backend:
              service_name: iot-botscan-web
              service_port: 5000
          - path: /api
            backend:
              service_name: iot-botscan-api
              service_port: 8000
    tls:
      - secret_name: iot-botscan-tls
        hosts:
          - iot-botscan.example.com
"""

with open('./enhanced_iot_botscan/config/deployment_config.yaml', 'w') as f:
    f.write(deployment_config_content)

print("✅ Created config/deployment_config.yaml with deployment settings")

print("\n🔧 Configuration files created successfully!")
print("📋 Summary of configuration files:")
print("   - config.yaml: Main system configuration")
print("   - model_config.json: ML model parameters")
print("   - logging_config.yaml: Logging setup")
print("   - deployment_config.yaml: Deployment settings")