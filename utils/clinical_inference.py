import torch
import joblib
import numpy as np
from typing import List
from models.clinical_model import ClinicalModel


class ClinicalInference:
    def __init__(
        self,
        model_path: str = "saved_models/clinical_best.pth",
        scaler_path: str = "clinical_scaler.pkl"
    ):

        self.device = torch.device("cpu")

        # =========================
        # LOAD MODEL
        # =========================
        self.model = ClinicalModel(input_dim=15)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # =========================
        # LOAD SCALER
        # =========================
        data = joblib.load(scaler_path)
        self.scaler = data["scaler"]
        self.feature_columns = data["columns"]

        print("✅ Clinical model + scaler loaded")

    def predict(self, features: List[float]) -> float:

        # =========================
        # VALIDATE INPUT SIZE
        # =========================
        if len(features) != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {len(features)}"
            )

        # =========================
        # SCALE INPUT
        # =========================
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        x = self.scaler.transform(x)

        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        # =========================
        # INFERENCE
        # =========================
        with torch.no_grad():
            _, logits = self.model(x)
            prob = torch.sigmoid(logits).item()

        return float(prob)