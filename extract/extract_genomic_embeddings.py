import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from models.genomic_vae import GenomicDataset, GenomicVAE


def extract_embeddings():
    # =========================
    # CONFIG
    # =========================
    device = torch.device("cpu")

    tensor_path = "genomic_tensor.pt"   # adjust if needed
    model_path = "checkpoints/genomic_vae_best.pth"
    save_path = "checkpoints/genomic_embeddings.npy"

    BATCH_SIZE = 16

    # =========================
    # LOAD DATA
    # =========================
    dataset = GenomicDataset(tensor_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = dataset[0].shape[0]

    # =========================
    # LOAD MODEL
    # =========================
    model = GenomicVAE(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("✅ Model loaded for embedding extraction")

    # =========================
    # EXTRACT EMBEDDINGS
    # =========================
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            mu, logvar = model.encode(batch)

            # Use mu as embedding (standard practice)
            embeddings = mu

            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)

    # =========================
    # SAVE
    # =========================
    os.makedirs("data", exist_ok=True)
    np.save(save_path, all_embeddings)

    print(f"✅ Embeddings saved: {save_path}")
    print(f"📐 Shape: {all_embeddings.shape}")


if __name__ == "__main__":
    extract_embeddings()