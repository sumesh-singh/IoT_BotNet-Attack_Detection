# File Tree: enhanced_iot_botscan

Generated on: 10/9/2025, 9:43:27 PM
Root path: `<repo-root>/enhanced_iot_botscan` (repo-relative)

```
├── 📁 config/
│   ├── ⚙️ config.yaml
│   ├── ⚙️ deployment_config.yaml
│   ├── ⚙️ logging_config.yaml
│   └── 📄 model_config.json
├── 📁 data/
│   ├── 📁 models/
│   ├── 📁 processed/
│   ├── 📁 raw/
│   │   ├── 📁 bot_iot/
│   │   ├── 📁 iot_23/
│   │   └── 📁 n_baiot/
│   └── 📁 results/
├── 📁 deployment/
│   ├── 📁 docker/
│   │   ├── 🐳 Dockerfile
│   │   ├── ⚙️ docker-compose.yml
│   │   └── 📄 requirements-docker.txt
│   ├── 📁 kubernetes/
│   │   ├── ⚙️ configmap.yaml
│   │   ├── ⚙️ deployment.yaml
│   │   └── ⚙️ service.yaml
│   └── 📁 scripts/
│       ├── 🐚 backup.sh
│       ├── 🐚 install.sh
│       └── 🐚 start.sh
├── 📁 docs/
│   ├── 📁 api/
│   │   ├── 📝 api_documentation.md
│   │   └── ⚙️ swagger.yaml
│   ├── 📁 developer_guide/
│   │   ├── 📝 architecture.md
│   │   ├── 📝 coding_standards.md
│   │   └── 📝 contribution_guide.md
│   └── 📁 user_guide/
│       ├── 📝 installation_guide.md
│       ├── 📝 troubleshooting.md
│       └── 📝 user_manual.md
├── 📁 enhanced_iot_botscan/
│   ├── 📁 config/
│   ├── 📁 data/
│   │   ├── 📁 models/
│   │   ├── 📁 processed/
│   │   ├── 📁 raw/
│   │   └── 📁 results/
│   ├── 📁 notebooks/
│   ├── 📁 scripts/
│   ├── 📁 src/
│   │   ├── 📁 api/
│   │   ├── 📁 core/
│   │   │   ├── 📁 adversarial/
│   │   │   ├── 📁 drift_detection/
│   │   │   ├── 📁 ensemble/
│   │   │   └── 📁 preprocessing/
│   │   ├── 📁 evaluation/
│   │   └── 📁 utils/
│   ├── 📁 tests/
│   └── 📁 web/
├── 📁 notebooks/
│   ├── 📓 01_data_exploration.ipynb
│   ├── 📓 02_baseline_reproduction.ipynb
│   ├── 📓 03_ensemble_development.ipynb
│   ├── 📓 04_adversarial_training.ipynb
│   ├── 📓 05_concept_drift_testing.ipynb
│   └── 📓 06_multi_dataset_validation.ipynb
├── 📁 scripts/
│   ├── 🐍 deploy.py
│   ├── 🐍 download_datasets.py
│   ├── 🐍 evaluate_models.py
│   ├── 🐍 original_script_1.py
│   ├── 🐍 original_script_2.py
│   ├── 🐍 original_script_3.py
│   ├── 🐍 original_script_4.py
│   ├── 🐍 script.py
│   ├── 🐍 script_1.py
│   ├── 🐍 script_10.py
│   ├── 🐍 script_2.py
│   ├── 🐍 script_3.py
│   ├── 🐍 script_4.py
│   ├── 🐍 script_5.py
│   ├── 🐍 script_6.py
│   ├── 🐍 script_7.py
│   ├── 🐍 script_8.py
│   ├── 🐍 script_9.py
│   └── 🐍 train_models.py
├── 📁 src/
│   ├── 📁 api/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 auth.py
│   │   ├── 🐍 graphql_api.py
│   │   ├── 🐍 rest_api.py
│   │   └── 🐍 websocket_handler.py
│   ├── 📁 core/
│   │   ├── 📁 adversarial/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 adversarial_trainer.py
│   │   │   ├── 🐍 attack_generator.py
│   │   │   ├── 🐍 cw_attack.py
│   │   │   ├── 🐍 defense_mechanisms.py
│   │   │   ├── 🐍 fgsm_attack.py
│   │   │   └── 🐍 pgd_attack.py
│   │   ├── 📁 drift_detection/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 adaptive_learner.py
│   │   │   ├── 🐍 drift_detector.py
│   │   │   ├── 🐍 kolmogorov_smirnov.py
│   │   │   ├── 🐍 page_hinkley.py
│   │   │   └── 🐍 performance_monitor.py
│   │   ├── 📁 ensemble/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 hybrid_ensemble.py
│   │   │   ├── 🐍 lightgbm_model.py
│   │   │   ├── 🐍 meta_learner.py
│   │   │   ├── 🐍 random_forest_model.py
│   │   │   └── 🐍 xgboost_model.py
│   │   ├── 📁 preprocessing/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 data_cleaner.py
│   │   │   ├── 🐍 dimensionality_reducer.py
│   │   │   ├── 🐍 feature_engineer.py
│   │   │   └── 🐍 scaler.py
│   │   └── 🐍 __init__.py
│   ├── 📁 data/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 data_loader.py
│   │   ├── 🐍 dataset_manager.py
│   │   └── 🐍 validation_datasets.py
│   ├── 📁 evaluation/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 cross_validator.py
│   │   ├── 🐍 metrics.py
│   │   ├── 🐍 performance_evaluator.py
│   │   └── 🐍 robustness_tester.py
│   ├── 📁 utils/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config_manager.py
│   │   ├── 🐍 file_utils.py
│   │   ├── 🐍 logger.py
│   │   └── 🐍 visualization.py
│   └── 🐍 __init__.py
├── 📁 tests/
│   ├── 📁 integration/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 test_api_integration.py
│   │   └── 🐍 test_end_to_end.py
│   ├── 📁 performance/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 test_load.py
│   │   └── 🐍 test_scalability.py
│   ├── 📁 unit/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 test_adversarial.py
│   │   ├── 🐍 test_drift_detection.py
│   │   ├── 🐍 test_ensemble.py
│   │   └── 🐍 test_preprocessing.py
│   └── 🐍 __init__.py
├── 📁 web/
│   ├── 📁 static/
│   │   ├── 📁 css/
│   │   │   ├── 🎨 dashboard.css
│   │   │   └── 🎨 style.css
│   │   ├── 📁 img/
│   │   └── 📁 js/
│   │       ├── 📄 charts.js
│   │       ├── 📄 dashboard.js
│   │       └── 📄 main.js
│   └── 📁 templates/
│       ├── 🌐 analytics.html
│       ├── 🌐 dashboard.html
│       └── 🌐 index.html
├── 📄 .env.example
├── 🚫 .gitignore
├── 📜 LICENSE
├── 📖 README.md
├── 📄 requirements.txt
└── 🐍 setup.py
```

---
