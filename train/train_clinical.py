import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from preprocessing.clinical_preprocess import ClinicalDataset   # ✅ CORRECT
from models.clinical_model import ClinicalModel


# -----------------------------
# SETTINGS
# -----------------------------
CSV_PATH = "dataset/CLINICAL DATA/dataset.csv"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots/clinical", exist_ok=True)


# -----------------------------
# DATA
# -----------------------------
train_dataset = ClinicalDataset(CSV_PATH, split="train")
val_dataset = ClinicalDataset(CSV_PATH, split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# MODEL
# -----------------------------
model = ClinicalModel(input_dim=train_dataset.X.shape[1]).to(DEVICE)

# 🔥 CORRECT LOSS (VERY IMPORTANT)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# -----------------------------
# EARLY STOPPING
# -----------------------------
best_val_loss = float("inf")
patience = 7
counter = 0


# -----------------------------
# TRACKING
# -----------------------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []


# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    total_loss = 0
    preds, targets = [], []

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        _, outputs = model(X)   # features not needed here

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 APPLY SIGMOID ONLY FOR METRICS
        probs = torch.sigmoid(outputs)
        pred = (probs > 0.5).float()

        preds.extend(pred.cpu().numpy())
        targets.extend(y.cpu().numpy())

    train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(targets, preds)
    train_f1 = f1_score(targets, preds)

    # ===== VALIDATION =====
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            _, outputs = model(X)

            loss = criterion(outputs, y)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            pred = (probs > 0.5).float()

            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    val_acc = accuracy_score(targets, preds)
    val_f1 = f1_score(targets, preds)

    # ===== STORE =====
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    # ===== EARLY STOPPING =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "saved_models/clinical_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping triggered")
            break


# -----------------------------
# PLOTS
# -----------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Clinical Loss")
plt.savefig("plots/clinical/loss.png")
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Clinical Accuracy")
plt.savefig("plots/clinical/accuracy.png")
plt.close()

plt.figure()
plt.plot(train_f1s, label="Train F1")
plt.plot(val_f1s, label="Val F1")
plt.legend()
plt.title("Clinical F1")
plt.savefig("plots/clinical/f1.png")
plt.close()

print("✅ Clinical Model Training Complete")