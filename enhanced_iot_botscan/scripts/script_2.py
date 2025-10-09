# It seems the directory structure was not persisted. Let me recreate everything and implement the complete project

import os
import json

# Recreate the entire project structure with all components
def create_complete_project():
    base_path = "./enhanced_iot_botscan"
    
    # Create all directories
    directories = [
        "config",
        "src",
        "src/core",
        "src/core/ensemble",
        "src/core/adversarial", 
        "src/core/drift_detection",
        "src/core/preprocessing",
        "src/data",
        "src/evaluation",
        "src/api",
        "src/utils",
        "data/raw/n_baiot",
        "data/raw/iot_23", 
        "data/raw/bot_iot",
        "data/processed",
        "data/models",
        "data/results",
        "notebooks",
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "web/templates",
        "web/static/css",
        "web/static/js",
        "web/static/img",
        "deployment/docker",
        "deployment/kubernetes",
        "deployment/scripts",
        "docs/api",
        "docs/user_guide",
        "docs/developer_guide",
        "scripts",
        "logs"
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/core/__init__.py", 
        "src/core/ensemble/__init__.py",
        "src/core/adversarial/__init__.py",
        "src/core/drift_detection/__init__.py", 
        "src/core/preprocessing/__init__.py",
        "src/data/__init__.py",
        "src/evaluation/__init__.py",
        "src/api/__init__.py", 
        "src/utils/__init__.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/integration/__init__.py",
        "tests/performance/__init__.py"
    ]
    
    for init_file in init_files:
        full_path = os.path.join(base_path, init_file)
        with open(full_path, 'w') as f:
            f.write('"""Module initialization."""\n')
    
    print(f"✅ Created complete project structure at {base_path}")
    return base_path

# Create the project structure
project_path = create_complete_project()
print(f"📁 Project created at: {project_path}")

# Verify structure was created
def verify_structure():
    if os.path.exists("./enhanced_iot_botscan"):
        print("✅ Project structure verified!")
        # List some key directories
        key_dirs = [
            "./enhanced_iot_botscan/src/core/ensemble",
            "./enhanced_iot_botscan/src/core/adversarial",
            "./enhanced_iot_botscan/src/core/drift_detection"
        ]
        
        for dir_path in key_dirs:
            exists = os.path.exists(dir_path)
            print(f"{'✅' if exists else '❌'} {dir_path}")
        
        return True
    else:
        print("❌ Project structure not found")
        return False

verify_structure()