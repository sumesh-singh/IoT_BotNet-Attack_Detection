# Now let's create the essential configuration and setup files with proper content

# 1. requirements.txt - All necessary Python packages
requirements_content = """# Core Machine Learning
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=3.3.0
numpy>=1.21.0
pandas>=1.5.0
scipy>=1.9.0

# Deep Learning & Adversarial
tensorflow>=2.10.0
torch>=1.12.0
torchvision>=0.13.0
cleverhans>=4.0.0
foolbox>=3.3.0
adversarial-robustness-toolbox>=1.13.0

# Data Processing & Visualization
matplotlib>=3.6.0
seaborn>=0.11.0
plotly>=5.10.0
dash>=2.6.0
streamlit>=1.12.0

# Web Framework & APIs
flask>=2.2.0
fastapi>=0.85.0
uvicorn>=0.18.0
websockets>=10.3
graphene>=3.0.0

# Database & Storage
psycopg2-binary>=2.9.0
pymongo>=4.2.0
redis>=4.3.0
sqlalchemy>=1.4.0

# Monitoring & Logging
prometheus-client>=0.14.0
structlog>=22.1.0
colorlog>=6.7.0

# Configuration & Utilities
pyyaml>=6.0
python-dotenv>=0.20.0
click>=8.1.0
tqdm>=4.64.0
joblib>=1.2.0

# Testing & Quality
pytest>=7.1.0
pytest-cov>=3.0.0
pytest-mock>=3.8.0
black>=22.6.0
flake8>=5.0.0
mypy>=0.971

# Deployment & Containerization
docker>=6.0.0
kubernetes>=24.2.0
gunicorn>=20.1.0

# Security & Authentication
cryptography>=37.0.0
pyjwt>=2.4.0
bcrypt>=4.0.0

# Network Analysis
scapy>=2.4.0
pyshark>=0.6.0
netaddr>=0.8.0

# Statistical Analysis
statsmodels>=0.13.0
pingouin>=0.5.0

# Jupyter & Development
jupyter>=1.0.0
jupyterlab>=3.4.0
ipywidgets>=8.0.0
notebook>=6.4.0
"""

with open('./enhanced_iot_botscan/requirements.txt', 'w') as f:
    f.write(requirements_content)

print("✅ Created requirements.txt with all necessary dependencies")

# 2. setup.py - Package setup configuration
setup_content = '''"""
Setup script for Enhanced IoT BotScan Defense System
Author: Kotiwale Sumesh Singh (160124862043)
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-iot-botscan",
    version="1.0.0",
    author="Kotiwale Sumesh Singh",
    author_email="sumesh.kotiwale@student.bits-pilani.ac.in",
    description="Enhanced IoT BotScan Defense System with Hybrid Ensemble Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sumeshkotiwale/enhanced-iot-botscan",
    project_urls={
        "Bug Tracker": "https://github.com/sumeshkotiwale/enhanced-iot-botscan/issues",
        "Documentation": "https://github.com/sumeshkotiwale/enhanced-iot-botscan/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Internet",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
            "torch-audio>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "iot-botscan=src.main:main",
            "iot-botscan-train=scripts.train_models:main",
            "iot-botscan-evaluate=scripts.evaluate_models:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
'''

with open('./enhanced_iot_botscan/setup.py', 'w') as f:
    f.write(setup_content)

print("✅ Created setup.py with package configuration")

# 3. .gitignore - Git ignore file
gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project-specific
data/raw/
data/processed/
data/models/
data/results/
logs/
*.pkl
*.joblib
*.h5
*.hdf5
*.model

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Temporary files
tmp/
temp/
*.tmp
*.temp
"""

with open('./enhanced_iot_botscan/.gitignore', 'w') as f:
    f.write(gitignore_content)

print("✅ Created .gitignore file")

# 4. .env.example - Environment variables template
env_example_content = """# Enhanced IoT BotScan Configuration
# Copy this file to .env and update with your actual values

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=iot_botscan
DB_USER=postgres
DB_PASSWORD=your_password_here

# MongoDB Configuration (for unstructured data)
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=iot_botscan_logs

# Redis Configuration (for caching)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ACCESS_TOKEN_EXPIRES=3600

# Machine Learning Configuration
ML_MODEL_PATH=./data/models/
ML_BATCH_SIZE=1000
ML_N_JOBS=-1

# Adversarial Training Configuration
ADVERSARIAL_RATIO=0.3
FGSM_EPSILON=0.1
PGD_EPSILON=0.1
PGD_ALPHA=0.01
PGD_STEPS=10

# Concept Drift Configuration
DRIFT_DETECTION_THRESHOLD=0.05
DRIFT_WINDOW_SIZE=1000
PERFORMANCE_THRESHOLD=0.95

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=./logs/iot_botscan.log

# Web Interface Configuration
WEB_HOST=0.0.0.0
WEB_PORT=5000
WEB_DEBUG=False

# External Services
THREAT_INTEL_API_KEY=your_threat_intel_api_key
SIEM_ENDPOINT=http://your-siem-server:8089
SIEM_API_KEY=your_siem_api_key

# Performance Configuration
MAX_WORKERS=4
BATCH_PROCESSING_SIZE=10000
REAL_TIME_BUFFER_SIZE=1000

# Security Configuration
ENABLE_HTTPS=True
SSL_CERT_PATH=./certs/cert.pem
SSL_KEY_PATH=./certs/key.pem
RATE_LIMIT_PER_MINUTE=100

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ENABLE_METRICS=True

# Development Configuration
DEVELOPMENT_MODE=True
DEBUG_MODE=False
TESTING_MODE=False
"""

with open('./enhanced_iot_botscan/.env.example', 'w') as f:
    f.write(env_example_content)

print("✅ Created .env.example file")

print("\n📁 Created essential configuration files:")
print("   - requirements.txt (Python dependencies)")
print("   - setup.py (Package configuration)")
print("   - .gitignore (Version control ignores)")
print("   - .env.example (Environment variables template)")