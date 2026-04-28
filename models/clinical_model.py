import torch
import torch.nn as nn


class ClinicalModel(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=32):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, output_dim)
        )

        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)   # 🔥 NO SIGMOID
        return features, out