import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from models.genomic_vae import GenomicVAE


def compute_genomic_error_stats():

    # -----------------------------
    # DEVICE
    # -----------------------------
    device = torch.device("cpu")
    print(f"🖥️ Using device: {device}")

    # -----------------------------
    # BASE PATH (IMPORTANT FIX)
    # -----------------------------
    BASE_DIR = r"D:\Desktop\LungCancerPrognosis\LungCancerPrognosis"

    SAVE_DIR = os.path.join(BASE_DIR, "outputs", "genomic")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # -----------------------------
    # CORRECT FILE PATHS
    # -----------------------------
    data_path = os.path.join(BASE_DIR, "genomic_tensor.pt")
    model_path = os.path.join(CHECKPOINT_DIR, "genomic_vae_best.pth")

    # -----------------------------
    # DEBUG CHECK (VERY IMPORTANT)
    # -----------------------------
    print("📂 DATA PATH:", data_path, os.path.exists(data_path))
    print("📂 MODEL PATH:", model_path, os.path.exists(model_path))

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Missing file: {data_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing model: {model_path}")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    data = torch.load(data_path, map_location=device)

    if len(data.shape) != 2:
        raise ValueError("❌ Genomic tensor must be 2D (samples, features)")

    input_dim = data.shape[1]
    print(f"✅ Loaded Genomic Data: {data.shape}")

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = GenomicVAE(input_dim=input_dim, latent_dim=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ Model Loaded Successfully")

    # -----------------------------
    # COMPUTE RECON ERROR (FAST)
    # -----------------------------
    errors = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)

            recon, _, _ = model(batch)

            mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.extend(mse.cpu().numpy())

            print(f"⏳ Processed {min(i+batch_size, len(data))}/{len(data)}")

    errors = np.array(errors)

    # -----------------------------
    # STATISTICS
    # -----------------------------
    mean_err = errors.mean()
    std_err = errors.std()
    min_err = errors.min()
    max_err = errors.max()

    print("\n📊 GENOMIC ERROR STATS")
    print("----------------------")
    print(f"Mean Error : {mean_err:.6f}")
    print(f"Std Error  : {std_err:.6f}")
    print(f"Min Error  : {min_err:.6f}")
    print(f"Max Error  : {max_err:.6f}")

    # -----------------------------
    # SAVE STATS
    # -----------------------------
    with open(os.path.join(SAVE_DIR, "error_stats.txt"), "w") as f:
        f.write(
            f"Mean Error: {mean_err}\n"
            f"Std Error: {std_err}\n"
            f"Min Error: {min_err}\n"
            f"Max Error: {max_err}\n"
        )

    # -----------------------------
    # HISTOGRAM
    # -----------------------------
    plt.figure()
    plt.hist(errors, bins=50)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE Error")
    plt.ylabel("Frequency")

    plt.savefig(os.path.join(SAVE_DIR, "error_distribution.png"))
    plt.close()

    # -----------------------------
    # SORT ERRORS
    # -----------------------------
    sorted_indices = np.argsort(errors)
    low_indices = sorted_indices[:10]
    high_indices = sorted_indices[-10:]

    print("\n🔥 Top 10 High Error Samples:", high_indices)
    print("✅ Top 10 Low Error Samples:", low_indices)

    # -----------------------------
    # SAVE ARRAYS
    # -----------------------------
    np.save(os.path.join(SAVE_DIR, "error_values.npy"), errors)
    np.save(os.path.join(SAVE_DIR, "high_error_indices.npy"), high_indices)
    np.save(os.path.join(SAVE_DIR, "low_error_indices.npy"), low_indices)

    # -----------------------------
    # GENOMIC RISK SCORE (0–1)
    # -----------------------------
    risk_scores = (errors - min_err) / (max_err - min_err)
    np.save(os.path.join(SAVE_DIR, "genomic_risk_scores.npy"), risk_scores)

    print("🔥 Risk scores saved")

    # -----------------------------
    # THRESHOLD
    # -----------------------------
    threshold = mean_err + 2 * std_err

    with open(os.path.join(SAVE_DIR, "threshold.txt"), "w") as f:
        f.write(f"threshold: {threshold}")

    print(f"🚨 Threshold: {threshold:.6f}")

    print("\n🎯 DONE: Genomic analysis complete!")

    return errors


if __name__ == "__main__":
    compute_genomic_error_stats()