import torch
import torch.nn as nn
import numpy as np
import pickle
import os

INPUT_DIM   = 53
LATENT_DIM  = 8
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "autoencoder_real.pt")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

SEUIL_NORMAL = 2.0
SEUIL_ELEVE  = 8.0


class TransformerAutoEncoder(nn.Module):
    def __init__(self, n=INPUT_DIM, lat=LATENT_DIM):
        super().__init__()
        self.attn_w   = nn.Linear(n, n)
        self.enc_proj = nn.Linear(n, 32)
        enc = nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=64,
                                          dropout=0.1, batch_first=True)
        self.enc      = nn.TransformerEncoder(enc, num_layers=2)
        self.to_lat   = nn.Linear(32, lat)
        self.from_lat = nn.Linear(lat, 32)
        dec = nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=64,
                                          dropout=0.1, batch_first=True)
        self.dec      = nn.TransformerEncoder(dec, num_layers=2)
        self.out      = nn.Linear(32, n)

    def forward(self, x):
        a = torch.softmax(self.attn_w(x), dim=-1)
        h = self.enc_proj(x * a).unsqueeze(1)
        h = self.enc(h).squeeze(1)
        h = self.from_lat(self.to_lat(h)).unsqueeze(1)
        h = self.dec(h).squeeze(1)
        return self.out(h), a


def load_model():
    model = TransformerAutoEncoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def compute_anomaly_score(model, scaler, features: np.ndarray):
    features_scaled = scaler.transform(features.reshape(1, -1)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features_scaled)
        reconstruction, attn = model(x)
        error_per_feature = (x - reconstruction).pow(2).squeeze(0).numpy()
        score = float(error_per_feature.mean())
        attn_scores = attn.squeeze(0).numpy()
    return score, attn_scores, error_per_feature


def get_risk_level(score: float) -> str:
    if score < SEUIL_NORMAL:
        return "faible"
    elif score < SEUIL_ELEVE:
        return "modéré"
    else:
        return "élevé"


if __name__ == "__main__":
    print("Test du modèle réel...")
    try:
        model, scaler = load_model()
        test = np.random.randn(INPUT_DIM).astype(np.float32)
        score, attn, err = compute_anomaly_score(model, scaler, test)
        print(f"Score test : {score:.4f}")
        print("Modèle chargé avec succès !")
    except FileNotFoundError:
        print("Modèle non trouvé — lance d'abord train.py")
