# preprocessing/ct_type_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset


class CTTypeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

        self.idx_to_class = {
            0: "adenocarcinoma",
            1: "large.cell.carcinoma",
            2: "normal",
            3: "squamous.cell.carcinoma"
        }

        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"⚠️ Skipping broken image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def load_samples(root):

        samples = []

        if os.path.exists(os.path.join(root, "Data")):
            data_path = os.path.join(root, "Data")
        else:
            data_path = root

        class_map = {
            "adenocarcinoma": 0,
            "large.cell.carcinoma": 1,
            "normal": 2,
            "squamous.cell.carcinoma": 3
        }

        for split in ["train", "test", "valid"]:
            split_path = os.path.join(data_path, split)

            if not os.path.exists(split_path):
                continue

            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)

                if not os.path.isdir(class_path):
                    continue

                class_name = class_name.lower()

                if class_name not in class_map:
                    print(f"⚠️ Skipped: {class_name}")
                    continue

                label = class_map[class_name]

                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)

                    if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue

                    samples.append((img_path, label))

        print(f"🔥 Total Type Samples: {len(samples)}")

        if len(samples) == 0:
            raise ValueError("❌ No samples loaded!")

        return samples