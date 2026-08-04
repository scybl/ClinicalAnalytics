RESULTS = [
    ("Surgical skill classification", "Random Forest", "accuracy=0.8500, f1=0.8571"),
    ("COVID CT discriminant analysis", "QDA", "accuracy=0.7252, f1=0.7050"),
    ("COVID CT HoG + SVM", "RBF SVM", "accuracy=0.8097, f1=0.7942"),
    ("Imbalanced COVID CT", "SVM", "accuracy=0.7912, f1=0.8394"),
]

print("Clinical ML Benchmarks")
print("=" * 22)
for task, model, metrics in RESULTS:
    print(f"- {task}: {model} ({metrics})")
