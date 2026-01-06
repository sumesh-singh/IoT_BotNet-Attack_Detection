
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"Failed to import numpy: {e}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch path: {torch.__file__}")
    print("PyTorch import successful")
except ImportError as e:
    print(f"Failed to import torch: {e}")
except Exception as e:
    print(f"An error occurred while importing torch: {e}")

print("\n--- Testing Adversarial Imports ---")
try:
    from src.core.adversarial.attack_generator import AdversarialAttackGenerator
    print("AdversarialAttackGenerator import successful")
except ImportError as e:
    print(f"Failed to import AdversarialAttackGenerator: {e}")
except Exception as e:
    print(f"An error occurred while importing AdversarialAttackGenerator: {e}")

try:
    from src.core.adversarial.adversarial_trainer import AdversarialTrainer
    print("AdversarialTrainer import successful")
except ImportError as e:
    print(f"Failed to import AdversarialTrainer: {e}")
except Exception as e:
    print(f"An error occurred while importing AdversarialTrainer: {e}")
