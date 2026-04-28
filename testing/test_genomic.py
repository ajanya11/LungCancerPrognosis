import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.genomic_vae import GenomicVAE


# -----------------------------
# SETTINGS
# -----------------------------
DEVICE = torch.device("cpu")

DATA_PATH = "genomic_tensor.pt"
MODEL_PATH = "checkpoints/genomic_vae_best.pth"

os.makedirs("results/genomic", exist_ok=True)
os.makedirs("plots/genomic", exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------
data = torch.load(DATA_PATH, map_location=DEVICE)

# Test split (last 15%)
test_data = data[int(0.85 * len(data)):]


# -----------------------------
# LOAD MODEL
# -----------------------------
input_dim = data.shape[1]

model = GenomicVAE(input_dim=input_dim, latent_dim=512)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -----------------------------
# TESTING
# -----------------------------
errors = []

with torch.no_grad():
    for batch in test_data.split(32):
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


print("\n🧪 TEST RESULTS")
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
with open("results/genomic/test.json", "w") as f:
    json.dump(metrics, f, indent=4)

# CSV
pd.DataFrame([metrics]).to_csv("results/genomic/test.csv", index=False)

# TXT
with open("results/genomic/test.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")


# -----------------------------
# PLOT ERROR DISTRIBUTION
# -----------------------------
plt.figure()
plt.hist(errors, bins=50)
plt.title("Test Error Distribution")
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.savefig("plots/genomic/test_distribution.png")
plt.close()

print("✅ Test metrics + plots saved!")