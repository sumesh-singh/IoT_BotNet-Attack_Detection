import pandas as pd
import numpy as np
import os
import joblib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import core modules
from src.core.ensemble.hybrid_ensemble import HybridEnsemble
from src.core.ensemble.hybrid_ensemble import HybridEnsemble
from src.core.preprocessing.data_cleaner import DataCleaner
from src.core.preprocessing.feature_engineer import FeatureEngineer
from src.core.drift_detection.drift_detector import DriftDetector

# Optional: Adversarial modules (requires PyTorch)
try:
    from src.core.adversarial.adversarial_trainer import AdversarialTrainer
    from src.core.adversarial.attack_generator import AdversarialAttackGenerator
    ADVERSARIAL_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: Adversarial modules not available: {e}")
    AdversarialTrainer = None
    AdversarialAttackGenerator = None
    ADVERSARIAL_AVAILABLE = False

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
        self.model_path = os.path.join("models", "hybrid_ensemble.joblib")
        # Initialize Drift Detector
        self.drift_detector = DriftDetector()
        
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
                self.current_metrics = self.model.get_model_info()
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
            
            results = self.model.train(X_train, y_train_series, validation_data=(X_val, y_val_series))
            
            # Generate and store validation predictions for dashboard
            logger.info("Predicting on validation set...")
            y_pred_encoded = self.model.predict(X_val)
            logger.info(f"Prediction shape: {y_pred_encoded.shape if hasattr(y_pred_encoded, 'shape') else 'No shape'}, Type: {type(y_pred_encoded)}")
            
            y_pred = le.inverse_transform(y_pred_encoded)
            y_val_orig = le.inverse_transform(y_val) # Restore validation labels for storage/display
            
            self.validation_results = {
                'y_true': y_val_orig.tolist(),
                'y_pred': y_pred.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model
            self.model.save_model(self.model_path)

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
                'feature_importance': self.model.get_feature_importance()
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
            'accuracy': f"{self.current_metrics.get('ensemble_validation_accuracy', 0.0):.2%}" if self.current_metrics else 'N/A'
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

    # --- Adversarial Methods ---

    def train_robust_model(self, data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute adversarial training pipeline.
        """
        try:
             # Basic Data Loading (Similar to standard training)
            logger.info(f"Loading data for adversarial training from {data_path}")
            df = pd.read_csv(data_path)
            
            target_col = config.get('target_column', 'label')
            if target_col not in df.columns:
                 # Try simple fallback
                 possible = [c for c in df.columns if 'label' in c.lower()]
                 target_col = possible[0] if possible else 'label'
            
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Clean & Engineer
            self.cleaner = DataCleaner(config)
            X_clean = self.cleaner.clean_dataset(X)
            y = y.loc[X_clean.index]
            
            self.engineer = FeatureEngineer(config)
            X_eng = self.engineer.engineer_features(X_clean, y)
            
            # Split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X_eng, y, test_size=0.2, random_state=42)
            
            # Initialize Base Model (e.g., LogisticRegression or Random Forest for robustness base)
            # Note: Adversarial training usually works best with differentiable models or specific wrappers.
            # We'll use the HybridEnsemble if supported, or a base component.
            # For simplicity/robustness, we might start with the RF or simple model as 'base_model' 
            # but AdversarialTrainer expects a sklearn-like estimator.
            
            # Using the HybridEnsemble as the base model to train robustly
            # (Note: Standard ensembles are hard to adversarially train directly without specific attacks,
            # but our trainer supports 'black-box' style generation if attacks are transferable)
            model_to_train = HybridEnsemble(config) 
            
            trainer = AdversarialTrainer(config)
            results = trainer.train_robust_model(
                model_to_train, X_train.values, y_train.values, X_val.values, y_val.values
            )
            
            # Update the main model if successful
            if results.get('best_model'):
                self.model = results['best_model']
                self.model.save_model(self.model_path)
                self.current_metrics = {'ensemble_validation_accuracy': results.get('best_robustness')} # Using robustness as key metric here
            
            return {'status': 'success', 'results': results}

        except Exception as e:
            logger.error(f"Adversarial training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def evaluate_robustness(self, data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate robustness on a dataset."""
        if not self.model or not self.model.is_trained:
             return {'status': 'error', 'message': "Model not trained."}

        if not data_path:
            return {'status': 'error', 'message': "No dataset selected for robustness evaluation."}

        try:
            # 1. Load & Process Data similar to predict
            df = pd.read_csv(data_path)
            target_col = config.get('target_column', 'label')
            # ... (data loading logic reuse ideally) ...
            if target_col not in df.columns:
                 possible = [c for c in df.columns if 'label' in c.lower()]
                 target_col = possible[0] if possible else 'label'

            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            X_clean = self.cleaner.clean_dataset(X)
            y = y.loc[X_clean.index]
            X_eng = self.engineer.transform_new_data(X_clean)

            # 2. Evaluate
            # We need an attack generator instance
            attack_gen = AdversarialAttackGenerator(config)
            results = attack_gen.evaluate_robustness(self.model, X_eng.values, y.values)
            
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"Robustness evaluation failed: {e}")
            return {'status': 'error', 'message': str(e)}
