import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.genomic_vae import GenomicDataset, GenomicVAE, vae_loss


def train():
    # =========================
    # CONFIG
    # =========================
    device = torch.device("cpu")

    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 1e-3

    # =========================
    # PATHS
    # =========================
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================
    dataset = GenomicDataset("genomic_tensor.pt")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = dataset[0].shape[0]

    # =========================
    # MODEL
    # =========================
    model = GenomicVAE(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n🚀 Training started on {device}")
    print(f"Input dim: {input_dim}")

    best_loss = float("inf")

    # =========================
    # STORE METRICS
    # =========================
    loss_history = []
    recon_history = []
    kl_history = []

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch in loader:
            batch = batch.to(device)

            recon, mu, logvar = model(batch)

            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()

            # Optional stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)

        loss_history.append(avg_loss)
        recon_history.append(avg_recon)
        kl_history.append(avg_kl)

        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

        # =========================
        # SAVE BEST MODEL
        # =========================
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(checkpoint_dir, "genomic_vae_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at {save_path}")

    # =========================
    # PLOT LOSSES
    # =========================
    plt.figure()
    plt.plot(loss_history, label="Total Loss")
    plt.plot(recon_history, label="Reconstruction Loss")
    plt.plot(kl_history, label="KL Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")

    plot_path = os.path.join(checkpoint_dir, "training_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"📊 Training plot saved at {plot_path}")
    print("\n🎯 Training Complete!")


if __name__ == "__main__":
    train()