import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =========================
# DATASET LOADER
# =========================
class GenomicDataset(Dataset):
    def __init__(self, tensor_path):
        """
        tensor_path: path to preprocessed tensor (genomic_tensor.pt)
        """
        self.data = torch.load(tensor_path)

        # Safety check
        if len(self.data.shape) != 2:
            raise ValueError("❌ Data must be 2D (samples, features)")

        print(f"✅ Loaded Genomic Dataset: {self.data.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


# =========================
# VAE MODEL
# =========================
class GenomicVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super(GenomicVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Latent space
        self.mu = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            nn.ReLU(),

            nn.Linear(2048, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# =========================
# LOSS FUNCTION
# =========================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_loss, recon_loss, kl_loss


# =========================
# QUICK TEST (optional)
# =========================
if __name__ == "__main__":
    dataset = GenomicDataset("genomic_tensor.pt")

    sample = dataset[0]
    input_dim = sample.shape[0]

    model = GenomicVAE(input_dim)

    x = sample.unsqueeze(0)  # add batch
    recon, mu, logvar = model(x)

    print("✅ Forward pass success")
    print("Input shape:", x.shape)
    print("Reconstruction shape:", recon.shape)
    print("Latent shape:", mu.shape)