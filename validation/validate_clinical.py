import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocessing.clinical_preprocess import ClinicalDataset
from models.clinical_model import ClinicalModel


# -----------------------------
# SETTINGS
# -----------------------------
CSV_PATH = "dataset/CLINICAL DATA/dataset.csv"
MODEL_PATH = "saved_models/clinical_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results/clinical", exist_ok=True)


# -----------------------------
# DATA
# -----------------------------
val_dataset = ClinicalDataset(CSV_PATH, split="val")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# -----------------------------
# MODEL
# -----------------------------
model = ClinicalModel(input_dim=val_dataset.X.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# VALIDATION
# -----------------------------
preds, targets = [], []

with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        _, outputs = model(X)

        probs = torch.sigmoid(outputs)
        pred = (probs > 0.5).float()

        preds.extend(pred.cpu().numpy())
        targets.extend(y.cpu().numpy())


# -----------------------------
# METRICS
# -----------------------------
acc = accuracy_score(targets, preds)
f1 = f1_score(targets, preds)

report = classification_report(targets, preds, output_dict=True)

print("\n📊 Validation Results")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")


# -----------------------------
# SAVE METRICS
# -----------------------------

# TXT
with open("results/clinical/validation.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write(classification_report(targets, preds)) # type: ignore

# JSON
with open("results/clinical/validation.json", "w") as f:
    json.dump(report, f, indent=4)

# CSV
df = pd.DataFrame(report).transpose()
df.to_csv("results/clinical/validation.csv")

print("✅ Validation metrics saved!")