from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

print("Surgical Skill Statistics and Gaze Analysis")
print("=" * 46)
print("Notebook:", ROOT / "analysis.ipynb")
print("Report:  ", ROOT / "report.pdf")
print()
print("Workflow")
print("- completion-time descriptive statistics")
print("- Mann-Whitney U tests for expert vs novice groups")
print("- procedural error scoring from step sequences")
print("- fixation-map sparsity analysis")
print()
print("Expected raw inputs")
for name in [
    "time_experts.csv",
    "time_novices.csv",
    "error_data.xlsx",
    "fixation_maps/experts/",
    "fixation_maps/novice/",
]:
    status = "present" if (ROOT / name).exists() else "missing"
    print(f"- {name}: {status}")
