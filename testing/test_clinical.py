import os
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from preprocessing.clinical_preprocess import ClinicalDataset
from models.clinical_model import ClinicalModel


# -----------------------------
# SETTINGS
# -----------------------------
CSV_PATH = "dataset/CLINICAL DATA/dataset.csv"
MODEL_PATH = "saved_models/clinical_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results/clinical", exist_ok=True)
os.makedirs("plots/clinical", exist_ok=True)


# -----------------------------
# DATA
# -----------------------------
test_dataset = ClinicalDataset(CSV_PATH, split="test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------
# MODEL
# -----------------------------
model = ClinicalModel(input_dim=test_dataset.X.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# TESTING
# -----------------------------
preds, targets = [], []

with torch.no_grad():
    for X, y in test_loader:
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

print("\n🧪 Test Results")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")


# -----------------------------
# SAVE METRICS
# -----------------------------

# TXT
with open("results/clinical/test.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write(classification_report(targets, preds)) # type: ignore

# JSON
with open("results/clinical/test.json", "w") as f:
    json.dump(report, f, indent=4)

# CSV
df = pd.DataFrame(report).transpose()
df.to_csv("results/clinical/test.csv")


# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(targets, preds)

plt.figure()
plt.imshow(cm)
plt.title("Clinical Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/clinical/confusion_matrix.png")
plt.close()

print("✅ Test metrics & confusion matrix saved!")