import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.core.ensemble.random_forest_model import RandomForestModel
from src.core.drift_detection.drift_detector import DriftDetector
from src.core.preprocessing.feature_engineer import FeatureEngineer
from src.core.preprocessing.scaler import Scaler

def load_data(benign_path, attack_path, max_samples=4000):
    try:
        benign_df = pd.read_csv(benign_path).sample(min(8000, max_samples), random_state=42)
        benign_df['label'] = 0
        
        attack_df = pd.read_csv(attack_path).sample(min(8000, max_samples), random_state=42)
        attack_df['label'] = 1
        
        df = pd.concat([benign_df, attack_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        return df.drop('label', axis=1), df['label']
    except Exception as e:
        print(f"Error loading {benign_path} or {attack_path}: {e}")
        return pd.DataFrame(), pd.Series()

def run_drift_eval():
    print("Loading Data (Device 1)...")
    base_dir = "data/raw/n_baiot"
    
    # Load all required files for Device 1
    # Train/Reference: Benign + Mirai
    X_ref, y_ref = load_data(os.path.join(base_dir, "1.benign.csv"), os.path.join(base_dir, "1.mirai.scan.csv"), 6000)
    
    # Split into Reference and Stable Stream
    X_train, y_train = X_ref.iloc[:4000], y_ref.iloc[:4000]
    X_stable, y_stable = X_ref.iloc[4000:], y_ref.iloc[4000:]
    
    # Drift Stream: Benign + Gafgyt
    X_drift, y_drift = load_data(os.path.join(base_dir, "1.benign.csv"), os.path.join(base_dir, "1.gafgyt.scan.csv"), 6000)
    # Ensure no overlap by taking a different slice or just accepting some benign overlap
    X_drift = X_drift.iloc[:4000]
    y_drift = y_drift.iloc[:4000]

    fe = FeatureEngineer()
    scaler = Scaler()

    print("Preprocessing Reference Data...")
    X_train_fe = fe.engineer_features(X_train)
    X_train_scaled, _ = scaler.fit_transform(X_train_fe, y_train)

    print("Training Initial Model...")
    model = RandomForestModel({'n_estimators': 30})
    model.train(X_train_scaled, y_train)

    print("Initializing Drift Detector...")
    detector_config = {
        'enabled_methods': ['ks', 'ph'], 
        'consensus_threshold': 0.5, # Revert back to reasonable thresholds
        'ks_config': {'alpha': 0.05, 'feature_threshold': 0.3},
        'ph_config': {'delta': 0.05, 'threshold': 100}
    }
    detector = DriftDetector(detector_config)
    detector.set_reference_data(X_train_scaled.values, y_train.values)

    # Prepare streaming data
    # Part 1: Stable (Benign + Mirai unseen)
    # Part 2: Drift Event (Benign + Gafgyt) -> Model hasn't seen Gafgyt
    print("Preparing streaming data...")

    X_stable_fe = fe.engineer_features(X_stable)
    X_stable_scaled, _ = scaler.transform(X_stable_fe, y_stable)

    X_drift_fe = fe.engineer_features(X_drift)
    X_drift_scaled, _ = scaler.transform(X_drift_fe, y_drift)

    stream_X = pd.concat([X_stable_scaled, X_drift_scaled])
    stream_y = pd.concat([y_stable, y_drift])

    chunk_size = 200
    n_chunks = len(stream_X) // chunk_size

    accuracies = []
    drift_flags = []
    retrain_events = []

    print(f"Starting stream evaluation with {n_chunks} chunks of size {chunk_size}...")
    
    latent_chunks = 0
    false_positives = 0
    drift_event_chunk = len(X_stable_scaled) // chunk_size # Chunk where drift starts

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        X_chunk = stream_X.iloc[start:end]
        y_chunk = stream_y.iloc[start:end]

        y_pred = model.predict(X_chunk)
        acc = accuracy_score(y_chunk, y_pred)
        accuracies.append(acc)

        # Detect drift
        drift_res = detector.detect_drift(X_chunk.values, y_chunk.values)
        drift_flag = drift_res['drift_detected']
        drift_flags.append(drift_flag)

        # Evaluate performance tracking
        if i < drift_event_chunk and drift_flag:
            false_positives += 1
        elif i >= drift_event_chunk and drift_flag and latent_chunks == 0:
            latent_chunks = i - drift_event_chunk

        # Automatic retraining if performance significantly drops or drift detected repeatedly
        # We simulate retraining by just retraining the model on the new distribution explicitly
        # and updating reference data.
        if drift_flag and i >= drift_event_chunk and len(retrain_events) == 0:
            print(f"Drift flagged at Chunk {i}. Triggering Automatic Retraining...")
            # Simulate historical window extraction for retrain 
            retrain_X = stream_X.iloc[start-400:end]
            retrain_y = stream_y.iloc[start-400:end]
            model.train(retrain_X, retrain_y)
            detector.update_reference_data(retrain_X.values, retrain_y.values)
            retrain_events.append(i)
            
        print(f"Chunk {i:02d} | True Class: {'STABLE' if i < drift_event_chunk else 'DRIFT '} | Acc: {acc:.4f} | Drift Flag: {drift_flag}")

    # Metrics Output
    fpr = false_positives / drift_event_chunk if drift_event_chunk > 0 else 0
    post_retrain_acc = np.mean(accuracies[retrain_events[0]+1:]) if retrain_events else 0

    print("\n--- Evaluation Metrics ---")
    print(f"Detection Latency: {latent_chunks} chunks ({latent_chunks*chunk_size} samples)")
    print(f"False Positive Rate (FPR): {fpr:.2%}")
    print(f"Performance Recovery (Post-Retrain Mean Acc): {post_retrain_acc:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_chunks), accuracies, label='Inference Accuracy', color='#1f77b4', marker='o')
    plt.axvline(x=drift_event_chunk, color='r', linestyle='--', label='Gafgyt Drift Event Starts')
    
    for r in retrain_events:
        plt.axvline(x=r, color='g', linestyle='-.', label='Automatic Retraining Triggered')

    plt.title('Enhanced IoT BotScan: Model Accuracy & Drift Detection Over Time')
    plt.xlabel('Streaming Sequence (Chunks of 200 samples)')
    plt.ylabel('Accuracy Score')
    plt.ylim([0, 1.05])
    
    # Avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/drift_accuracy.png', dpi=300)
    print("Graph saved to results/drift_accuracy.png")

if __name__ == "__main__":
    run_drift_eval()
