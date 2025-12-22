import sys
import os
import pandas as pd
import numpy as np

def main():
    print("Preparing training dataset...")
    
    data_dir = './data/raw/n_baiot'
    output_path = './data/processed/training_data.csv'
    
    # Files to load (Danmini Doorbell)
    # 1.benign.csv -> Label 0
    # 1.mirai.scan.csv -> Label 1
    # 1.gafgyt.combo.csv -> Label 2
    
    files = [
        ('1.benign.csv', 0),
        ('1.mirai.scan.csv', 1),
        ('1.gafgyt.combo.csv', 2)
    ]
    
    all_data = []
    
    for filename, label in files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            print(f"Loading {filename}...")
            # Load only first 5000 rows to keep it fast for the user
            df = pd.read_csv(file_path, nrows=5000) 
            df['label'] = label
            all_data.append(df)
        else:
            print(f"Warning: {filename} not found.")
            
    if not all_data:
        print("No data loaded!")
        return

    print("Combining data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    
    print("Done! Dataset ready.")
    print(f"Target Column: 'label'")
    print(f"Classes: 0=Benign, 1=Mirai, 2=Gafgyt")

if __name__ == "__main__":
    main()
