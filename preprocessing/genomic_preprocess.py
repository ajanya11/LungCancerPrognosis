import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)

    # Drop non-genomic / problematic columns
    drop_cols = [
        'WARNING', 'LABORATORY_BATCH',
        'SMOKING_HISTORY', 'SURGICAL_MARGINS',
        'PATHOLOGIC_N_STAGE', 'PATHOLOGIC_T_STAGE',
        'MONTHS_TO_FIRST_PROGRESSION',
        'MTHS_TO_LAST_CLINICAL_ASSESSMENT'
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Keep only numeric (GENES)
    df = df.select_dtypes(include=[np.number])

    # Remove low variance genes
    var = df.var()
    df = df.loc[:, var > 1e-5]

    # Fill missing values
    df = df.fillna(df.mean())

    print("✅ Cleaned Shape:", df.shape)

    return df


def normalize_data(df):
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    return data, scaler


def save_tensor(data, save_path):
    tensor = torch.tensor(data, dtype=torch.float32)
    torch.save(tensor, save_path)
    print(f"✅ Saved tensor: {save_path}")


if __name__ == "__main__":
    path = "dataset/GENOMIC DATA/clean/complete_dataframe.csv"

    df = load_and_clean(path)
    data, scaler = normalize_data(df)

    save_tensor(data, "genomic_tensor.pt")