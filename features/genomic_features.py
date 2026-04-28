import torch


def extract_genomic_features(model, X):
    device = torch.device("cpu")
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X).to(device)
        features = model.encode(X_tensor)

    return features.cpu().numpy()