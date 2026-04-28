# =====================================================
# 🧠 CT VALIDATION (BINARY + TYPE)
# =====================================================

import os
import torch
from torchvision import transforms
from PIL import Image
from typing import cast

from models.ct_model import CTModel

DEVICE = "cpu"

# =========================
# 🔥 LOAD MODELS
# =========================

# Binary model (2-class)
binary_model = CTModel(num_classes=2)
binary_model.load_state_dict(
    torch.load("checkpoints/best_ct_binary.pth", map_location=DEVICE)
)
binary_model.eval()

# Type model (4-class)
type_model = CTModel(num_classes=4)
type_model.load_state_dict(
    torch.load("checkpoints/best_ct_type.pth", map_location=DEVICE)
)
type_model.eval()

# =========================
# 🔥 TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# 🔥 DATASET PATH
# =========================
DATASET_PATH = "dataset/chest ct scan lunng/train"

CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma",
    "normal"
]

# =========================
# 🔥 VALIDATION
# =========================
binary_correct = 0
binary_total = 0

type_correct = 0
type_total = 0

for class_idx, class_name in enumerate(CLASSES):

    folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(folder):
        continue

    for img_name in os.listdir(folder):

        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder, img_name)

        # Load image
        img_pil = Image.open(img_path).convert("RGB")

        img = transform(img_pil)
        img = cast(torch.Tensor, img).unsqueeze(0)

        with torch.no_grad():

            # =========================
            # 🔥 BINARY VALIDATION
            # =========================
            binary_out, _ = binary_model(img)
            binary_pred = int(torch.argmax(binary_out, dim=1).item())

            # Ground truth (binary)
            gt_binary = 0 if class_name == "normal" else 1

            if binary_pred == gt_binary:
                binary_correct += 1

            binary_total += 1

            # =========================
            # 🔥 TYPE VALIDATION
            # =========================
            if gt_binary == 1:  # only cancer cases

                type_out, _ = type_model(img)
                type_pred = int(torch.argmax(type_out, dim=1).item())

                if type_pred == class_idx:
                    type_correct += 1

                type_total += 1

# =========================
# 🔥 RESULTS
# =========================
binary_acc = binary_correct / binary_total if binary_total > 0 else 0
type_acc = type_correct / type_total if type_total > 0 else 0

print("\n==============================")
print("🧠 CT VALIDATION RESULTS")
print("==============================")
print(f"Binary Accuracy   : {binary_acc:.4f}")
print(f"Type Accuracy     : {type_acc:.4f}")
print("==============================")