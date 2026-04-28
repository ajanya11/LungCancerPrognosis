import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from fusion.full_fusion import full_fusion
from sklearn.metrics import accuracy_score, f1_score, classification_report

os.makedirs("results/fusion", exist_ok=True)


# -----------------------------
# DATA (SIMULATED)
# -----------------------------
def generate_data(n=300):
    data = []
    for _ in range(n):
        ct_logits = np.random.rand(4)
        clinical = np.random.rand()
        genomic = np.random.rand()

        label = 1 if (0.6*np.max(ct_logits) + 0.25*clinical + 0.15*genomic) > 0.5 else 0
        data.append((ct_logits, clinical, genomic, label))
    return data


# -----------------------------
# RUN
# -----------------------------
data = generate_data()

preds, targets = [], []

for ct_logits, clinical, genomic, label in data:
    result = full_fusion(ct_logits, clinical, genomic)
    preds.append(result["binary"])
    targets.append(label)


# -----------------------------
# DEBUG DISTRIBUTION
# -----------------------------
print("\n📊 Distribution")
print("Pred:", Counter(preds))
print("True:", Counter(targets))


# -----------------------------
# METRICS
# -----------------------------
acc = accuracy_score(targets, preds)
f1 = f1_score(targets, preds)

report = classification_report(
    targets,
    preds,
    output_dict=True,
    zero_division=0   # 🔥 FIX
)

print("\n📊 VALIDATION RESULTS")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")


# -----------------------------
# SAVE
# -----------------------------
with open("results/fusion/validation.json", "w") as f:
    json.dump(report, f, indent=4)

pd.DataFrame(report).transpose().to_csv("results/fusion/validation.csv")

with open("results/fusion/validation.txt", "w") as f:
    f.write(f"Accuracy: {acc}\nF1: {f1}\n")

print("✅ Fusion validation saved!")