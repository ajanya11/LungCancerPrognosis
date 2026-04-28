import os
import json
import torch
import numpy as np
import pandas as pd

from models.genomic_vae import GenomicVAE


# -----------------------------
# SETTINGS
# -----------------------------
DEVICE = torch.device("cpu")

DATA_PATH = "genomic_tensor.pt"
MODEL_PATH = "checkpoints/genomic_vae_best.pth"

os.makedirs("results/genomic", exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------
data = torch.load(DATA_PATH, map_location=DEVICE)

# Split (70% train already used → take 15% validation)
val_split = int(0.85 * len(data))
val_data = data[int(0.7 * len(data)):val_split]


# -----------------------------
# LOAD MODEL
# -----------------------------
input_dim = data.shape[1]

model = GenomicVAE(input_dim=input_dim, latent_dim=512)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -----------------------------
# VALIDATION
# -----------------------------
errors = []

with torch.no_grad():
    for batch in val_data.split(32):
        batch = batch.to(DEVICE)

        recon, _, _ = model(batch)

        mse = torch.mean((recon - batch) ** 2, dim=1)
        errors.extend(mse.cpu().numpy())

errors = np.array(errors)


# -----------------------------
# METRICS
# -----------------------------
mean_err = errors.mean()
std_err = errors.std()
min_err = errors.min()
max_err = errors.max()

threshold = mean_err + 2 * std_err


print("\n📊 VALIDATION RESULTS")
print(f"Mean Error : {mean_err:.6f}")
print(f"Std Error  : {std_err:.6f}")
print(f"Threshold  : {threshold:.6f}")


# -----------------------------
# SAVE METRICS
# -----------------------------
metrics = {
    "mean_error": float(mean_err),
    "std_error": float(std_err),
    "min_error": float(min_err),
    "max_error": float(max_err),
    "threshold": float(threshold)
}

# JSON
with open("results/genomic/validation.json", "w") as f:
    json.dump(metrics, f, indent=4)

# CSV
pd.DataFrame([metrics]).to_csv("results/genomic/validation.csv", index=False)

# TXT
with open("results/genomic/validation.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

print("✅ Validation metrics saved!")