import os
import torch
import numpy as np
import torch.nn as nn


# =========================
# VAE MODEL (SAME AS TRAINING)
# =========================
class GenomicVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super(GenomicVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.mu = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            nn.ReLU(),

            nn.Linear(2048, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# =========================
# GENOMIC INFERENCE CLASS
# =========================
class GenomicInference:
    def __init__(self, model_path, input_dim, stats_path):
        self.device = torch.device("cpu")

        # Load model
        self.model = GenomicVAE(input_dim=input_dim, latent_dim=512)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print("✅ Model loaded")

        # Load stats
        self._load_stats(stats_path)

    def _load_stats(self, stats_path):
        errors = np.load(os.path.join(stats_path, "error_values.npy"))

        self.mean_err = errors.mean()
        self.std_err = errors.std()
        self.min_err = errors.min()
        self.max_err = errors.max()

        print("✅ Stats loaded")
        print(f"Mean: {self.mean_err:.6f}, Std: {self.std_err:.6f}")

    def predict(self, sample):
        if len(sample.shape) == 1:
            sample = np.expand_dims(sample, axis=0)

        x = torch.tensor(sample, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon, _, _ = self.model(x)
            mse = torch.mean((recon - x) ** 2, dim=1)
            error = mse.cpu().numpy()[0]

        # Normalize risk score
        risk_score = (error - self.min_err) / (self.max_err - self.min_err)
        risk_score = np.clip(risk_score, 0, 1)

        # 3-level classification
        low_th = self.mean_err
        high_th = self.mean_err + 2 * self.std_err

        if error < low_th:
            level = "Low Risk 🟢"
        elif error < high_th:
            level = "Moderate Risk 🟡"
        else:
            level = "High Risk 🔴"

        return {
            "error": float(error),
            "risk_score": float(risk_score),
            "risk_level": level
        }


# =========================
# MAIN RUN
# =========================
def main():

    # -----------------------------
    # PATHS (EDIT THESE)
    # -----------------------------
    MODEL_PATH = "checkpoints/genomic_vae_best.pth"
    STATS_PATH = "outputs/genomic"
    SAMPLE_PATH = "sample_genomic.npy"

    # -----------------------------
    # LOAD SAMPLE
    # -----------------------------
    if not os.path.exists(SAMPLE_PATH):
        raise FileNotFoundError(f"❌ Missing sample: {SAMPLE_PATH}")

    sample = np.load(SAMPLE_PATH)
    input_dim = sample.shape[0]

    print(f"✅ Sample loaded: {sample.shape}")

    # -----------------------------
    # INIT MODEL
    # -----------------------------
    infer = GenomicInference(
        model_path=MODEL_PATH,
        input_dim=input_dim,
        stats_path=STATS_PATH
    )

    # -----------------------------
    # PREDICT
    # -----------------------------
    result = infer.predict(sample)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\n🧬 GENOMIC ANALYSIS RESULT")
    print("----------------------------")
    print(f"Reconstruction Error : {result['error']:.6f}")
    print(f"Risk Score (0–1)     : {result['risk_score']:.4f}")
    print(f"Risk Level           : {result['risk_level']}")

    # -----------------------------
    # FUSION-READY OUTPUT
    # -----------------------------
    if "Low" in result["risk_level"]:
        fusion_value = 0.2
    elif "Moderate" in result["risk_level"]:
        fusion_value = 0.5
    else:
        fusion_value = 0.9

    print(f"Fusion Value         : {fusion_value:.2f}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()