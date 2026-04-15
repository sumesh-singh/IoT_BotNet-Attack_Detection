import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

def compute_mi():
    base_dir = "data/raw/n_baiot"
    print("Loading data for MI extraction...")
    
    try:
        # Load Benign
        df_benign = pd.read_csv(os.path.join(base_dir, "1.benign.csv")).sample(5000, random_state=42)
        df_benign['label'] = 0
        
        # Load Mirai
        df_mirai = pd.read_csv(os.path.join(base_dir, "1.mirai.scan.csv")).sample(2500, random_state=42)
        df_mirai['label'] = 1
        
        # Load Gafgyt
        df_gafgyt = pd.read_csv(os.path.join(base_dir, "1.gafgyt.scan.csv")).sample(2500, random_state=42)
        df_gafgyt['label'] = 2
        
        df = pd.concat([df_benign, df_mirai, df_gafgyt])
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Use existing feature names, mostly N-BaIoT has long statistics names
        feature_names = X.columns.tolist()
        
        print("Computing Mutual Information...")
        mi_scores = mutual_info_classif(X.values, y.values, random_state=42)
        
        # Rank features
        mi_series = pd.Series(mi_scores, index=feature_names)
        top_20 = mi_series.sort_values(ascending=False).head(20)
        
        # Plot
        plt.figure(figsize=(12, 8))
        top_20.sort_values(ascending=True).plot(kind='barh', color='#2ca02c')
        plt.title('Top 20 Features by Mutual Information (N-BaIoT / Mirai / Gafgyt)')
        plt.xlabel('Mutual Information Score')
        plt.ylabel('Feature Name')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        out_path = os.path.join('results', 'top_20_features_mi.png')
        plt.savefig(out_path, dpi=300)
        print(f"Graph saved to {out_path}")
        
        print("\n--- TOP 20 FEATURES ---")
        for idx, (name, score) in enumerate(top_20.items(), 1):
            print(f"{idx:02d}. {score:.4f} | {name}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    compute_mi()
