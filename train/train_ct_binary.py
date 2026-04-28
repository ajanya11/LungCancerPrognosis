# =====================================================
# 🫁 CT BINARY TRAINING (FINAL FIXED VERSION)
# =====================================================

import os, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from preprocessing.ct_binary_dataset import CTBinaryDataset
from models.ct_model import CTModel


# =========================
# 🔥 SEED
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# MAIN
# =========================
def main():

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    print("Device:", device)

    # =========================
    # DATA PATHS
    # =========================
    iq_root = "dataset/ct scan/The IQ-OTHNCCD lung cancer dataset"
    chest_root = "dataset/chest ct scan lunng"

    samples = CTBinaryDataset.load_samples(iq_root, chest_root)

    if len(samples) == 0:
        raise ValueError("❌ No samples found")

    labels = [l for _, l in samples]
    print("Class distribution:", Counter(labels))
    print("Unique labels:", set(labels))  # MUST be {0,1}

    # =========================
    # SPLIT
    # =========================
    train, temp = train_test_split(
        samples, test_size=0.3, stratify=labels, random_state=42
    )

    temp_labels = [l for _, l in temp]

    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # =========================
    # TRANSFORMS
    # =========================
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # =========================
    # ✅ FIXED DATALOADERS
    # =========================
    train_dataset = CTBinaryDataset(train, train_tf)
    val_dataset   = CTBinaryDataset(val, eval_tf)
    test_dataset  = CTBinaryDataset(test, eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=2,
        pin_memory=use_cuda
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=2,
        pin_memory=use_cuda
    )

    # =========================
    # MODEL
    # =========================
    model = CTModel(num_classes=2).to(device)

    # =========================
    # CLASS WEIGHTS
    # =========================
    y = np.array([l for _, l in train])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0,1]),
        y=y
    )

    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class weights:", weights)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    # =========================
    # TRAIN
    # =========================
    best_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots/binary", exist_ok=True)

    for epoch in range(20):

        model.train()
        tl, correct, total = 0,0,0

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out,_ = model(x)
            loss = criterion(out,y)

            loss.backward()
            optimizer.step()

            tl += loss.item()*x.size(0)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

        train_loss = tl/total
        train_acc = correct/total

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        vl, correct, total = 0,0,0

        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)

                out,_ = model(x)
                loss = criterion(out,y)

                vl += loss.item()*x.size(0)
                pred = out.argmax(1)

                correct += (pred==y).sum().item()
                total += y.size(0)

        val_loss = vl/total
        val_acc = correct/total

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} | Train {train_acc:.3f} | Val {val_acc:.3f}")

        # =========================
        # SAVE BEST
        # =========================
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "checkpoints/best_ct_binary.pth")

    # =========================
    # TEST
    # =========================
    model.load_state_dict(best_model)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)

            out,_ = model(x)
            pred = out.argmax(1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)

    # =========================
    # PLOTS
    # =========================
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.legend()
    plt.title("Loss")
    plt.savefig("plots/binary/loss.png")
    plt.close()

    plt.plot(train_accs,label="train")
    plt.plot(val_accs,label="val")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("plots/binary/acc.png")
    plt.close()

    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("plots/binary/cm.png")
    plt.close()


if __name__ == "__main__":
    main()