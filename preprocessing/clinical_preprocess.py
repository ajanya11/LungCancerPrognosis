import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class ClinicalDataset(Dataset):

    def __init__(self, csv_path, split="train", scaler_path="clinical_scaler.pkl"):

        df = pd.read_csv(csv_path)

        # =============================
        # TARGET
        # =============================
        y = df["LUNG_CANCER"].map({"NO": 0.0, "YES": 1.0})

        # =============================
        # FEATURES
        # =============================
        X = df.drop(columns=["LUNG_CANCER"])

        self.feature_columns = [
            "AGE", "GENDER", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
            "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE",
            "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING",
            "COUGHING", "SHORTNESS_OF_BREATH",
            "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
        ]

        X = X[self.feature_columns]

        # =============================
        # 🔥 FIX: 1/2 → 0/1
        # =============================
        for col in X.columns:
            if col == "GENDER":
                X[col] = X[col].map({"M": 1, "F": 0})
            else:
                X[col] = X[col] - 1   # 🔥 KEY FIX

        X = X.fillna(0)

        # =============================
        # SPLIT
        # =============================
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # =============================
        # SCALER
        # =============================
        if split == "train":
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)

            joblib.dump({
                "scaler": self.scaler,
                "columns": self.feature_columns
            }, scaler_path)

        else:
            data = joblib.load(scaler_path)
            self.scaler = data["scaler"]

            X_train = self.scaler.transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        # =============================
        # SELECT SPLIT
        # =============================
        if split == "train":
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        elif split == "val":
            self.X = torch.tensor(X_val, dtype=torch.float32)
            self.y = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]