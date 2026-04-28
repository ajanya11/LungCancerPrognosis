# =====================================================
# 🧠 CT BINARY DATASET (FINAL FIXED)
# =====================================================

import os
from PIL import Image
from torch.utils.data import Dataset


class CTBinaryDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    # =====================================================
    # 🔥 LOAD DATA (FIXED BINARY)
    # =====================================================
    @staticmethod
    def load_samples(iq_root, chest_root):

        samples = []

        # -------------------------
        # CHEST CT DATASET
        # -------------------------
        chest_train = os.path.join(chest_root, "train")

        for class_name in os.listdir(chest_train):

            class_path = os.path.join(chest_train, class_name)

            if not os.path.isdir(class_path):
                continue

            # 🔥 BINARY LABEL
            label = 0 if "normal" in class_name.lower() else 1

            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)

                if img_path.endswith((".png", ".jpg", ".jpeg")):
                    samples.append((img_path, label))

        # -------------------------
        # IQ DATASET
        # -------------------------
        for class_name in os.listdir(iq_root):

            class_path = os.path.join(iq_root, class_name)

            if not os.path.isdir(class_path):
                continue

            label = 0 if "normal" in class_name.lower() else 1

            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)

                if img_path.endswith((".png", ".jpg", ".jpeg")):
                    samples.append((img_path, label))

        return samples