import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import core modules
from src.core.ensemble.hybrid_ensemble import HybridEnsemble
from src.core.preprocessing.data_cleaner import DataCleaner
from src.core.preprocessing.feature_engineer import FeatureEngineer
from src.data.data_loader import DataLoader
from src.core.drift_detection.drift_detector import DriftDetector
from src.core.drift_detection.monitoring_service import PerformanceMonitor

# Optional: Adversarial modules (requires PyTorch)
# Use lazy import pattern to avoid blocking app startup
ADVERSARIAL_AVAILABLE = None  # None = not checked yet, True/False = checked
ADVERSARIAL_ERROR = None
AdversarialTrainer = None
AdversarialAttackGenerator = None

def _check_adversarial_modules():
    """Lazy check for adversarial module availability."""
    global ADVERSARIAL_AVAILABLE, ADVERSARIAL_ERROR, AdversarialTrainer, AdversarialAttackGenerator
    
    if ADVERSARIAL_AVAILABLE is not None:
        return ADVERSARIAL_AVAILABLE
    
    try:
        # Windows-specific fix: Add torch DLL directory to search path
        # This fixes c10.dll loading issues in subprocess environments (like Streamlit)
        import os
        import sys
        if sys.platform == 'win32':
            # Try to find torch installation path and add its lib directory
            import importlib.util
            torch_spec = importlib.util.find_spec('torch')
            if torch_spec and torch_spec.origin:
                torch_lib_path = os.path.join(os.path.dirname(torch_spec.origin), 'lib')
                if os.path.exists(torch_lib_path):
                    # Python 3.8+ has os.add_dll_directory
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(torch_lib_path)
                        logger.info(f"Added DLL directory: {torch_lib_path}")
                    # Also add to PATH as fallback
                    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
        
        # First check if torch itself imports
        import torch
        logger.info(f"PyTorch imported successfully: {torch.__version__}")
        
        # Now try to import adversarial modules
        from src.core.adversarial.adversarial_trainer import AdversarialTrainer as _AdversarialTrainer
        from src.core.adversarial.attack_generator import AdversarialAttackGenerator as _AdversarialAttackGenerator
        
        AdversarialTrainer = _AdversarialTrainer
        AdversarialAttackGenerator = _AdversarialAttackGenerator
        ADVERSARIAL_AVAILABLE = True
        ADVERSARIAL_ERROR = None
        logger.info("Adversarial modules loaded successfully")
        return True
        
    except ImportError as e:
        ADVERSARIAL_AVAILABLE = False
        ADVERSARIAL_ERROR = f"ImportError: {e}"
        logger.warning(f"Adversarial modules not available (ImportError): {e}")
        return False
    except OSError as e:
        ADVERSARIAL_AVAILABLE = False
        ADVERSARIAL_ERROR = f"OSError (likely DLL issue): {e}"
        logger.warning(f"Adversarial modules not available (OSError): {e}")
        return False
    except Exception as e:
        ADVERSARIAL_AVAILABLE = False
        ADVERSARIAL_ERROR = f"Unexpected error: {type(e).__name__}: {e}"
        logger.error(f"Adversarial modules failed with unexpected error: {e}")
        return False

from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendInterface:
    """Singleton interface to manage backend logic for Streamlit."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackendInterface, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.model: Optional[HybridEnsemble] = None
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.data_loader = DataLoader({}) # Initialize with empty config for now
        self.model_path = os.path.join("models", "hybrid_ensemble.joblib")
        # Initialize Drift Detector
        # Initialize Drift Detector
        self.drift_detector = DriftDetector()
        self.monitor = PerformanceMonitor()
        
        self.training_history = []
        self.current_metrics = {}
        self.validation_results = {}
        
        # Ensure directories exist
        os.makedirs("models", exist_ok=True)
        
        self._load_model_if_exists()
        self._initialized = True

    def _load_model_if_exists(self):
        """Attempt to load an existing model."""
        if os.path.exists(self.model_path):
            try:
                self.model = HybridEnsemble()
                self.model.load_model(self.model_path)
                
                # Restore feature engineer state if available (Fix for feature mismatch)
                if hasattr(self.model, 'feature_engineer_state') and self.model.feature_engineer_state:
                    self.engineer.set_state(self.model.feature_engineer_state)
                    logger.info("Restored FeatureEngineer state from model.")
                
                self.current_metrics = self.model.get_model_info()
                
                # IMPORTANT: Restore training history and metrics from the loaded model
                # so the dashboard can display accuracy and training date correctly
                if hasattr(self.model, 'training_history') and self.model.training_history:
                    # Sync backend history with model history
                    self.training_history = []
                    for entry in self.model.training_history:
                        acc = entry.get('ensemble_validation_accuracy', 0.0)
                        ts = entry.get('training_timestamp', 'Unknown')
                        self.training_history.append({
                            'timestamp': ts,
                            'accuracy': acc,
                            'config': {}  # Config is saved elsewhere, leave empty for now
                        })
                    
                    # Ensure current_metrics has the latest validation accuracy for the UI
                    latest = self.model.training_history[-1]
                    if 'ensemble_validation_accuracy' in latest:
                        self.current_metrics['ensemble_validation_accuracy'] = latest['ensemble_validation_accuracy']
                    
                # Restore validation results for confusion matrix
                if hasattr(self.model, 'validation_results') and self.model.validation_results:
                    self.validation_results = self.model.validation_results
                    logger.info("Restored validation results for confusion matrix.")
                    
                logger.info("Loaded existing model.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None

    def train_model(self, data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full training pipeline: Load -> Clean -> Engineer -> Train.
        """
        try:
            # 1. Load Data
            # 1. Load Data
            if config.get('dataset_mode') == 'unified':
                logger.info("Loading unified multi-dataset...")
                # Update loader config if needed
                self.data_loader = DataLoader(config) 
                dataset_dict = self.data_loader.load_unified_dataset()
                
                # reconstruct dataframe for compatibility with existing pipeline
                # (Ideally pipeline should handle numpy arrays, but cleaner/engineer might expect DF)
                df = pd.DataFrame(dataset_dict['features'], columns=dataset_dict['feature_names'])
                y = pd.Series(dataset_dict['labels'])
                
                # Check for label encoding in unified dataset (0/1)
                # Ensure y is aligned
                df['label'] = y
                target_col = 'label'
                
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
            else:
                logger.info(f"Loading data from {data_path}")
                df = pd.read_csv(data_path)
                
                target_col = config.get('target_column', 'label')
                if target_col not in df.columns:
                    # Try to find a likely target column if not specified correctly
                    possible_targets = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'target' in c.lower()]
                    if possible_targets:
                        target_col = possible_targets[0]
                    else:
                        return {'status': 'error', 'message': f"Target column '{target_col}' not found."}
    
                X = df.drop(columns=[target_col])
                y = df[target_col]

            # 2. Clean Data
            logger.info("Cleaning data...")
            self.cleaner = DataCleaner(config)
            X_clean = self.cleaner.clean_dataset(X)
            # Align y with X_clean (in case rows were dropped)
            y = y.loc[X_clean.index]

            # 3. Feature Engineering
            logger.info("Engineering features...")
            self.engineer = FeatureEngineer(config)
            X_eng = self.engineer.engineer_features(X_clean, y)

            # 4. Train Model
            logger.info("Training model...")
            # 4. Train Model
            logger.info("Training model...")
            self.model = HybridEnsemble(config)
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.model.label_encoder = le # Store encoder in model
            
            # Split for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X_eng, y_encoded, test_size=0.2, random_state=42)
            
            # Create Series with correct indices to prevent alignment issues
            y_train_series = pd.Series(y_train, index=X_train.index)
            y_val_series = pd.Series(y_val, index=X_val.index)
            
            # CRITICAL FIX: Pass feature_engineer to train() for proper state management
            results = self.model.train(
                X_train, y_train_series, 
                validation_data=(X_val, y_val_series),
                feature_engineer=self.engineer  # Pass feature engineer for state saving
            )
            
            # Generate and store validation predictions for dashboard
            logger.info("Predicting on validation set...")
            y_pred_encoded = self.model.predict(X_val)
            logger.info(f"Prediction shape: {y_pred_encoded.shape if hasattr(y_pred_encoded, 'shape') else 'No shape'}, Type: {type(y_pred_encoded)}")
            
            y_pred = le.inverse_transform(y_pred_encoded)
            y_val_orig = le.inverse_transform(y_val) # Restore validation labels for storage/display
            logger.info("Saving ARM validation dataset...")
            # Save validation data for ARM
            val_df = X_val.copy()
            val_df['label'] = y_val_orig
            arm_val_path = "data/processed/arm_validation_data.csv"
            val_df.to_csv(arm_val_path, index=False)
            logger.info(f"ARM validation dataset saved to {arm_val_path}")

            self.validation_results = {
                'y_true': y_val_orig.tolist(),
                'y_pred': y_pred.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model with FeatureEngineer state (Fix for feature mismatch)
            feature_engineer_state = self.engineer.get_state()
            if not feature_engineer_state or not feature_engineer_state.get('selected_features'):
                logger.error("CRITICAL: Feature engineer state is empty or invalid!")
            else:
                 logger.info(f"Saving feature engineer state with {len(feature_engineer_state['selected_features'])} features")

            self.model.save_model(
                self.model_path, 
                feature_engineer_state=feature_engineer_state,
                validation_results=self.validation_results
            )

            # Set reference data for drift detection (using training data)
            # We use a subset for performance if dataset is large, but full data is ideal for distribution
            logger.info("Setting drift detection reference data...")
            # y_train is a numpy array (from train_test_split on y_encoded), so it doesn't have .values
            self.drift_detector.set_reference_data(X_train.values, y_train)
            
            # Update state
            self.current_metrics = results
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': results.get('ensemble_validation_accuracy', 0.0),
                'config': config
            })
            
            # Update active monitoring
            self.monitor.update_metrics(results)
            
            return {'status': 'success', 'results': results}

        except Exception as e:
            logger.error(f"Training pipeline failed at step: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def get_recent_predictions(self) -> Dict[str, List[Any]]:
        """Get recent validation predictions for visualization."""
        return self.validation_results

    def get_confusion_matrix(self) -> Dict[str, Any]:
        """Get confusion matrix data."""
        if not self.validation_results:
            return {}
            
        from sklearn.metrics import confusion_matrix
        # Ensure labels are strings to avoid "Mix of label input types" error
        y_true = [str(x) for x in self.validation_results['y_true']]
        # Ensure labels are strings to avoid "Mix of label input types" error
        y_true = [str(x) for x in self.validation_results['y_true']]
        y_pred = [str(x) for x in self.validation_results['y_pred']]
        
        # Determine labels from data if possible to ensure consistent order
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            'matrix': cm.tolist(),
            'classes': labels
        }

    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute inference pipeline: Clean -> Engineer -> Predict.
        """
        if not self.model or not self.model.is_trained:
            return {'status': 'error', 'message': "Model is not trained."}

        try:
            # 1. Clean
            X_clean = self.cleaner.clean_dataset(input_data)
            
            # 2. Engineer (Transform)
            X_eng = self.engineer.transform_new_data(X_clean)
            
            # 3. Predict
            predictions = self.model.predict(X_eng)
            # 3. Predict
            predictions_encoded = self.model.predict(X_eng)
            probabilities = self.model.predict_proba(X_eng)
            
            # Decode predictions if encoder exists
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                 predictions = self.model.label_encoder.inverse_transform(predictions_encoded)
            else:
                 predictions = predictions_encoded
            
            return {
                'status': 'success',
                'predictions': predictions,
                'probabilities': probabilities,
                'feature_importance': self.model.get_feature_importance(),
                'indices': X_clean.index.tolist()
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            'model_loaded': self.model is not None and self.model.is_trained,
            'model_type': 'Hybrid Ensemble' if self.model else 'None',
            'last_training': self.training_history[-1]['timestamp'] if self.training_history else 'Never',
            'accuracy': f"{self.current_metrics.get('ensemble_validation_accuracy', 0.0):.2%}" if self.current_metrics else 'N/A',
            'drift_status': self.monitor.check_degradation()
        }
        return status

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if self.model and self.model.is_trained:
            # Aggregate importance from base models (simplification)
            raw_importance = self.model.get_feature_importance()
            # Just take the first available model's importance for visualization or average them
            # Here we'll flatten it for the UI
            flat_importance = {}
            for model_name, imp_series in raw_importance.items():
                for feat, val in imp_series.items():
                    flat_importance[f"{model_name}_{feat}"] = val
            return flat_importance
        return {}

    # --- Drift Detection Methods ---

    def check_drift(self, data_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check for concept drift in a new dataset compared to the training data.
        """
        if not self.model or not self.model.is_trained:
            return {'status': 'error', 'message': "Model must be trained to establish a baseline."}

        try:
            # 1. Load New Data
            df = pd.read_csv(data_path)
            
            target_col = config.get('target_column', 'label') if config else 'label'
            if target_col not in df.columns:
                 # Try auto-detect if not found
                possible_targets = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
                target_col = possible_targets[0] if possible_targets else None

            if target_col:
                X_new = df.drop(columns=[target_col]).values
                y_new = df[target_col].values
            else:
                X_new = df.values
                y_new = None

            # 2. Run Drift Detection
            results = self.drift_detector.detect_drift(X_new, y_new)
            
            return {
                'status': 'success',
                'results': results,
                'statistics': self.drift_detector.get_drift_statistics()
            }

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_drift_status(self) -> Dict[str, Any]:
        """Get current drift status and statistics."""
        return self.drift_detector.get_drift_statistics()

    def get_drift_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance based on drift frequency (KS detector)."""
        if hasattr(self.drift_detector, 'detectors') and 'ks' in self.drift_detector.detectors:
            ks_detector = self.drift_detector.detectors['ks']
            if hasattr(ks_detector, 'get_feature_importance'):
                return ks_detector.get_feature_importance()
        return {}

    def retrain_model(self, data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically retrain the model using new data.
        Invokes the training pipeline.
        """
        logger.info("Starting automatic retraining on new data...")
        return self.train_model(data_path, config)

    def monitor_and_adapt(self, new_data_path: str, config: Dict[str, Any] = None,
                          auto_retrain: bool = True) -> Dict[str, Any]:
        """
        Monitor for concept drift and automatically adapt if configured.
        
        This implements the 'automated detection module that triggers timely 
        model adaptation' as described in the project abstract.
        
        Args:
            new_data_path: Path to new data for drift checking
            config: Optional training config for retraining
            auto_retrain: If True, automatically retrain when drift detected
            
        Returns:
            Status dict with drift detection and adaptation results
        """
        if config is None:
            config = {}
            
        logger.info(f"Running automated drift monitoring on: {new_data_path}")
        
        # Step 1: Check for drift
        drift_result = self.check_drift(new_data_path, config)
        
        if drift_result['status'] != 'success':
            return {
                'status': 'error',
                'message': f"Drift check failed: {drift_result.get('message', 'Unknown error')}",
                'drift_detected': False,
                'retraining_triggered': False
            }
        
        results = drift_result['results']
        drift_detected = results.get('drift_detected', False)
        drift_severity = results.get('severity', 'none')
        
        logger.info(f"Drift detection result: detected={drift_detected}, severity={drift_severity}")
        
        if drift_detected:
            logger.warning("⚠️ Concept drift detected in incoming data!")
            
            if auto_retrain:
                logger.info("🔄 Auto-retrain enabled. Triggering automatic model adaptation...")
                
                # Step 2: Retrain the model with new data
                retrain_result = self.retrain_model(new_data_path, config)
                
                if retrain_result['status'] == 'success':
                    logger.info("✅ Model successfully adapted to new data distribution")
                    return {
                        'status': 'adapted',
                        'message': 'Drift detected and model automatically retrained',
                        'drift_detected': True,
                        'retraining_triggered': True,
                        'drift_severity': drift_severity,
                        'drift_details': results,
                        'retraining_results': retrain_result['results']
                    }
                else:
                    logger.error(f"❌ Automatic retraining failed: {retrain_result.get('message')}")
                    return {
                        'status': 'error',
                        'message': f"Retraining failed: {retrain_result.get('message')}",
                        'drift_detected': True,
                        'retraining_triggered': True,
                        'drift_severity': drift_severity,
                        'drift_details': results
                    }
            else:
                logger.warning("Auto-retrain disabled. Manual intervention required.")
                return {
                    'status': 'drift_detected',
                    'message': 'Drift detected but auto-retrain is disabled. Manual retraining required.',
                    'drift_detected': True,
                    'retraining_triggered': False,
                    'drift_severity': drift_severity,
                    'drift_details': results
                }
        else:
            logger.info("✅ No significant drift detected. System stable.")
            return {
                'status': 'stable',
                'message': 'No drift detected. Model adaptation not required.',
                'drift_detected': False,
                'retraining_triggered': False,
                'drift_details': results
            }


    # --- Adversarial Methods ---

    def train_robust_model(self, data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a robust model using ARM-style data augmentation.
        Uses noise injection, feature masking, and burst patterns to augment training data.
        """
        try:
            import numpy as np
            from sklearn.model_selection import train_test_split
            from src.core.robustness.threat_generators.noise_injector import NoiseInjector
            from src.core.robustness.threat_generators.feature_masker import FeatureMasker
            
            logger.info(f"Loading data for robust training from {data_path}")
            df = pd.read_csv(data_path)
            
            target_col = config.get('target_column', 'label')
            if target_col not in df.columns:
                possible = [c for c in df.columns if 'label' in c.lower()]
                target_col = possible[0] if possible else df.columns[-1]
            
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Clean & Engineer
            self.cleaner = DataCleaner(config)
            X_clean = self.cleaner.clean_dataset(X)
            y = y.loc[X_clean.index]
            
            self.engineer = FeatureEngineer(config)
            X_eng = self.engineer.engineer_features(X_clean, y)
            
            # ARM-based data augmentation
            logger.info("Generating ARM-augmented training data...")
            noise_injector = NoiseInjector()
            feature_masker = FeatureMasker()
            
            X_np = X_eng.values
            y_np = y.values
            
            augmentation_ratio = config.get('augmentation_ratio', 0.3)
            n_augment = int(len(X_np) * augmentation_ratio)
            
            # Sample indices for augmentation
            aug_indices = np.random.choice(len(X_np), n_augment, replace=False)
            X_sample = X_np[aug_indices]
            y_sample = y_np[aug_indices]
            
            # Create augmented versions
            X_noisy = noise_injector.inject_gaussian_noise(X_sample, scale=0.1)
            X_masked = feature_masker.mask_random_features(X_sample, mask_rate=0.1)
            
            # Combine original + augmented data
            X_combined = np.vstack([X_np, X_noisy, X_masked])
            y_combined = np.hstack([y_np, y_sample, y_sample])
            
            logger.info(f"Training data augmented: {len(X_np)} -> {len(X_combined)} samples")
            
            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_combined, y_combined, test_size=0.2, random_state=42
            )
            
            # Train the model on augmented data
            logger.info("Training model on augmented data...")
            X_train_df = pd.DataFrame(X_train, columns=X_eng.columns)
            y_train_series = pd.Series(y_train)
            X_val_df = pd.DataFrame(X_val, columns=X_eng.columns)
            y_val_series = pd.Series(y_val)
            
            # Train on augmented data
            results = self.model.train(X_train_df, y_train_series, validation_data=(X_val_df, y_val_series))
            
            # CRITICAL FIX: Save with feature engineer state
            feature_engineer_state = self.engineer.get_state()
            self.model.save_model(self.model_path, feature_engineer_state=feature_engineer_state)
            self.current_metrics = results
            
            # Evaluate robustness improvement
            from src.core.robustness.robustness_monitor import AdaptiveRobustnessMonitor
            arm = AdaptiveRobustnessMonitor(config)
            arm.establish_baseline(self.model, X_val, y_val)
            robustness = arm.evaluate_comprehensive_robustness(self.model, X_val, y_val)
            
            results['robustness_after_training'] = robustness['aggregate_scores']['overall_robustness']
            results['augmentation_samples'] = len(X_combined) - len(X_np)
            
            logger.info(f"Robust training complete. New robustness: {results['robustness_after_training']:.2%}")
            
            return {'status': 'success', 'results': results}

        except Exception as e:
            logger.error(f"Robust training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def evaluate_robustness(self, data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate robustness on a dataset using ARM."""
        if not self.model or not self.model.is_trained:
            return {'status': 'error', 'message': "Model not trained."}

        if not data_path:
            return {'status': 'error', 'message': "No dataset selected for robustness evaluation."}

        try:
            # 1. Load Data
            logger.info(f"Loading data for robustness evaluation from {data_path}")
            df = pd.read_csv(data_path)
            
            target_col = config.get('target_column', 'label')
            
            # Auto-detect target column if not found
            if target_col not in df.columns:
                possible = [c for c in df.columns if any(x in c.lower() 
                           for x in ['label', 'class', 'target', 'attack'])]
                if possible:
                    target_col = possible[0]
                    logger.info(f"Auto-detected target column: {target_col}")
                else:
                    target_col = df.columns[-1]
                    logger.warning(f"Using last column as target: {target_col}")
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Handle float labels (convert to int)
            if y.dtype in ['float64', 'float32']:
                unique_vals = y.unique()
                if len(unique_vals) <= 10:
                    y = y.astype(int)
                    logger.info("Converted float labels to integers")
            
            # 2. Clean Data (CRITICAL: Must match training pipeline)
            logger.info("Cleaning data...")
            X_clean = self.cleaner.clean_dataset(X)
            y = y.loc[X_clean.index]
            
            # 3. CRITICAL FIX: Restore Feature Engineer State
            # Check if model has saved feature engineer state
            fe_state = None
            if hasattr(self.model, 'feature_engineer_state'):
                fe_state = self.model.feature_engineer_state
            elif hasattr(self.model, '_feature_engineer_state'):
                fe_state = self.model._feature_engineer_state
            
            if fe_state:
                logger.info("Restoring FeatureEngineer state from model...")
                # Create fresh engineer instance with restored state
                self.engineer = FeatureEngineer(config)
                self.engineer.set_state(fe_state)
                logger.info(f"FeatureEngineer restored. Selected features: {len(self.engineer.selected_features) if self.engineer.selected_features else 0}")
            else:
                logger.error("CRITICAL: No feature engineer state found in model!")
                return {
                    'status': 'error',
                    'message': "Feature engineer state not found. Model must be retrained with updated save logic."
                }
            
            # 4. Transform Data (Apply same feature engineering as training)
            logger.info("Applying feature engineering transformations...")
            X_eng = self.engineer.transform_new_data(X_clean)
            logger.info(f"Data transformed: {X_clean.shape} -> {X_eng.shape}")
            
            # 5. Feature Count Validation
            expected_features = None
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = len(self.model.feature_names_in_)
            elif hasattr(self.model, 'base_models') and self.model.base_models:
                first_model = list(self.model.base_models.values())[0] if self.model.base_models else None
                if first_model and hasattr(first_model, 'model') and hasattr(first_model.model, 'n_features_in_'):
                    expected_features = first_model.model.n_features_in_
            
            if expected_features:
                current_features = X_eng.shape[1]
                if current_features != expected_features:
                    return {
                        'status': 'error',
                        'message': f"Feature mismatch: Test data has {current_features} features, "
                                  f"but model expects {expected_features}. "
                                  f"Ensure test dataset is from the same source as training data."
                    }
                logger.info(f"Feature count validated: {current_features} features match expected")
            
            # 6. Convert to numpy arrays
            X_numpy = X_eng.values if hasattr(X_eng, 'values') else X_eng
            y_numpy = y.values if hasattr(y, 'values') else y
            
            # 7. Run ARM Evaluation
            logger.info("Starting ARM robustness evaluation...")
            from src.core.robustness.robustness_monitor import AdaptiveRobustnessMonitor
            
            arm = AdaptiveRobustnessMonitor(config)
            
            # Establish baseline
            arm.establish_baseline(self.model, X_numpy, y_numpy)
            
            # Comprehensive evaluation
            results = arm.evaluate_comprehensive_robustness(self.model, X_numpy, y_numpy)
            
            # Get report
            report = arm.get_robustness_report()
            
            # Combine results
            results['report'] = report
            results['data_info'] = {
                'n_samples': len(X_numpy),
                'n_features': X_numpy.shape[1],
                'n_classes': len(np.unique(y_numpy))
            }
            
            logger.info(f"ARM evaluation complete. Overall robustness: {results['aggregate_scores']['overall_robustness']:.2%}")
            
            # Add visualizations to the results as well
            from src.core.robustness.visualization import ARMVisualizer
            visualizer = ARMVisualizer()
            
            # Robustness Comparison Chart
            # We need previous results for comparison, but if not available we just show current
            # For now, let's create a placeholder comparison
            comparison_fig = visualizer.create_robustness_comparison(
                results['aggregate_scores'],
                {'overall_robustness': 0.8, 'noise_robustness': 0.8, 'masking_robustness': 0.8, 'burst_robustness': 0.8} # Placeholder baseline
            )
            
            # Threat Heatmap
            heatmap_fig = visualizer.create_threat_heatmap(results)
            
            # Scenario Bar Chart
            scenario_fig = visualizer.create_scenario_bar_chart(results)
            
            # Summary Gauge
            gauge_fig = visualizer.create_summary_gauge(results['aggregate_scores']['overall_robustness'])
            
            # Accuracy Impact Chart
            accuracy_fig = visualizer.create_accuracy_impact_chart(results)
            
            # Store figures in results (as JSON/dict for serializability if needed, or pass figure objects for Streamlit)
            # Streamlit handles figure objects directly, so we pass them.
            results['figures'] = {
                'comparison': comparison_fig,
                'heatmap': heatmap_fig,
                'scenario': scenario_fig,
                'gauge': gauge_fig,
                'accuracy_impact': accuracy_fig
            }
            
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"Robustness evaluation failed: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}
