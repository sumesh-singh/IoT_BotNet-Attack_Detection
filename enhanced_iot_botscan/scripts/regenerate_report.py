"""Regenerate the markdown report from saved JSON results."""
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))

# Import the report generator from the benchmark script
from scripts.benchmark_evaluation import generate_markdown_report

with open('results/benchmark_results.json') as f:
    all_results = json.load(f)

# Print summary
for ds_name, results in all_results.items():
    print(f"\n=== {ds_name.upper()} ===")
    for r in results:
        auc = f"{r['auc_roc']:.4f}" if r['auc_roc'] is not None else "N/A"
        print(f"  {r['model']:<25} Acc={r['accuracy']:.4f}  F1(W)={r['f1_weighted']:.4f}  AUC={auc}")

# Generate report
md = generate_markdown_report(all_results, 2733.0)
with open('results/benchmark_results.md', 'w', encoding='utf-8') as f:
    f.write(md)
print("\nReport saved to results/benchmark_results.md")
