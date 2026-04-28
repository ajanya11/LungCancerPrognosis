# =====================================================
# 🫁 CT TRAINING (FINAL STRONG VERSION)
# =====================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight


# =========================
# CONFIG
# =========================
DATA_DIR = "dataset/chest ct scan lunng"
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
DEVICE = torch.device("cpu")


# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# =========================
# DATASET
# =========================
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_transform)

num_classes = len(train_dataset.classes)

print("Classes:", train_dataset.classes)


# =========================
# CLASS BALANCING
# =========================
labels = [label for _, label in train_dataset.samples]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

sample_weights = [class_weights[label].item() for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))


# =========================
# DATALOADER
# =========================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# MODEL (STRONG BACKBONE)
# =========================
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)


# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)


# =========================
# TRAIN LOOP
# =========================
best_acc = 0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

    # =========================
    # SAVE BEST MODEL
    # =========================
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model, "checkpoints/best_ct_type.pth")
        print("✅ Best model saved!")


print("🔥 Training Complete")
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # =========================
    # REPORT
    # =========================
    print("\n🔥 Classification Report:\n")
    print(classification_report(all_labels, all_preds))

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)

    return cm, all_labels, all_preds
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names):

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/type/confusion_matrix.png")
