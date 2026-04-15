import json
import os
import pandas as pd

def compute_overall(results, weights):
    return (
        weights['noise'] * results['arm_noise'] +
        weights['masking'] * results['arm_masking'] +
        weights['burst'] * results['arm_burst'] +
        weights['confidence'] * results['arm_confidence']
    )

def run_sensitivity_analysis():
    results_path = os.path.join("results", "ablation_results.json")
    if not os.path.exists(results_path):
        print(f"Results file not found at {results_path}")
        return

    with open(results_path, 'r') as f:
        ablation = json.load(f)

    weight_configs = {
        "Equal (25-25-25-25)": {'noise': 0.25, 'masking': 0.25, 'burst': 0.25, 'confidence': 0.25},
        "Current (30-30-20-20)": {'noise': 0.30, 'masking': 0.30, 'burst': 0.20, 'confidence': 0.20},
        "Noise-Heavy (60-20-10-10)": {'noise': 0.60, 'masking': 0.20, 'burst': 0.10, 'confidence': 0.10},
        "Masking-Heavy (20-60-10-10)": {'noise': 0.20, 'masking': 0.60, 'burst': 0.10, 'confidence': 0.10}
    }

    print("Weight Configurations:\n")
    data = []
    
    for cond_key, res in ablation.items():
        row = {'Condition': res['condition']}
        for w_name, w_dict in weight_configs.items():
            score = compute_overall(res, w_dict)
            row[w_name] = round(score, 4)
        data.append(row)

    # Manually build the markdown table
    header = "| Condition | " + " | ".join(weight_configs.keys()) + " |"
    separator = "|---|" + "---|".join(["" for _ in weight_configs]) + "|"
    lines = [header, separator]
    
    for row in data:
        line_vals = [str(row['Condition'])] + [str(row[k]) for k in weight_configs.keys()]
        lines.append("| " + " | ".join(line_vals) + " |")
        
    md_table = "\n".join(lines)
    print(md_table)
    
    with open(os.path.join("results", "weight_sensitivity.md"), "w") as f:
        f.write("### Weight Sensitivity Analysis\n\n")
        f.write(md_table)
        f.write("\n")

if __name__ == "__main__":
    run_sensitivity_analysis()
