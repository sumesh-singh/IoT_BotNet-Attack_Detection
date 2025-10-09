# I need to check what files exist and create the missing core modules
import os

# Check what directories and files exist
def show_directory_structure(path, max_depth=3, current_depth=0):
    items = []
    if current_depth >= max_depth:
        return items
    
    try:
        for item in sorted(os.listdir(path)):
            if item.startswith('.'):
                continue
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                items.append(f"{'  ' * current_depth}📁 {item}/")
                items.extend(show_directory_structure(item_path, max_depth, current_depth + 1))
            else:
                items.append(f"{'  ' * current_depth}📄 {item}")
    except PermissionError:
        items.append(f"{'  ' * current_depth}❌ Permission denied")
    
    return items

print("🔍 Current directory structure:")
structure = show_directory_structure("./enhanced_iot_botscan", max_depth=4)
for item in structure[:50]:  # Show first 50 items
    print(item)

if len(structure) > 50:
    print(f"... and {len(structure) - 50} more items")

# Check if core directories exist
core_dirs = [
    "./enhanced_iot_botscan/src/core/ensemble",
    "./enhanced_iot_botscan/src/core/adversarial", 
    "./enhanced_iot_botscan/src/core/drift_detection",
    "./enhanced_iot_botscan/src/core/preprocessing"
]

print("\n📂 Checking core directories:")
for dir_path in core_dirs:
    exists = os.path.exists(dir_path)
    print(f"{'✅' if exists else '❌'} {dir_path}")
    if not exists:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   Created: {dir_path}")

# Ensure __init__.py files exist in all directories
init_files = [
    "./enhanced_iot_botscan/src/core/ensemble/__init__.py",
    "./enhanced_iot_botscan/src/core/adversarial/__init__.py", 
    "./enhanced_iot_botscan/src/core/drift_detection/__init__.py",
    "./enhanced_iot_botscan/src/core/preprocessing/__init__.py"
]

for init_file in init_files:
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Core module initialization."""\n')
        print(f"✅ Created {init_file}")