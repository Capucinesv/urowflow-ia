"""
train.py — Entraînement du modèle sur les vraies données d'Edouard
Lance avec : python3 train.py

Génère 3 fichiers dans models/ :
  - autoencoder_real.pt  → le modèle entraîné
  - scaler.pkl           → la normalisation
  - feature_cols.pkl     → les noms des 53 features
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from sklearn.preprocessing import StandardScaler

# ── Chemins ───────────────────────────────────────────────────────────────────
CSV_PATH    = "data/final_calibration_sounds_features.csv"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Chargement du CSV ─────────────────────────────────────────────────────────
print("Chargement du CSV...")
df = pd.read_csv(CSV_PATH)
feature_cols = [c for c in df.columns if c not in ['device_id', 'debit']]
print(f"  {len(df)} lignes, {len(feature_cols)} features")

# Sons normaux = débit entre 5 et 25 ml/s
normal   = df[(df['debit'] >= 5) & (df['debit'] <= 25)][feature_cols].values
abnormal = df[df['debit'] < 3][feature_cols].values
print(f"  Sons normaux : {len(normal)}, anormaux : {len(abnormal)}")

# ── Normalisation ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
normal_scaled   = scaler.fit_transform(normal).astype(np.float32)
abnormal_scaled = scaler.transform(abnormal).astype(np.float32)

# ── Modèle ────────────────────────────────────────────────────────────────────
N = len(feature_cols)  # 53

class TransformerAE(nn.Module):
    def __init__(self, n=53, lat=8):
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

model   = TransformerAE(n=N)
opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
X       = torch.tensor(normal_scaled)

# ── Entraînement ──────────────────────────────────────────────────────────────
print("\nEntraînement (150 époques)...")
model.train()
for ep in range(150):
    opt.zero_grad()
    r, _ = model(X)
    l = loss_fn(r, X)
    l.backward()
    opt.step()
    if (ep + 1) % 30 == 0:
        print(f"  Époque {ep+1}/150 — Loss : {l.item():.4f}")

# ── Validation ────────────────────────────────────────────────────────────────
model.eval()
def mean_score(arr):
    scores = []
    with torch.no_grad():
        for x in arr[:30]:
            r, _ = model(torch.tensor(x).unsqueeze(0))
            scores.append(float((torch.tensor(x) - r.squeeze()).pow(2).mean()))
    return np.mean(scores)

sn = mean_score(normal_scaled)
sa = mean_score(abnormal_scaled)
print(f"\nScore sons normaux  : {sn:.4f}")
print(f"Score sons anormaux : {sa:.4f}")
print(f"Ratio discrimination : ×{sa/sn:.1f}")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), f"{MODELS_DIR}/autoencoder_real.pt")
with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{MODELS_DIR}/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print(f"\n✓ 3 fichiers sauvegardés dans {MODELS_DIR}/")
print("  - autoencoder_real.pt")
print("  - scaler.pkl")
print("  - feature_cols.pkl")
print("\nProjet prêt !")
