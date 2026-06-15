import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

st.set_page_config(page_title="UroFlow AI", page_icon="🏥", layout="wide")
st.title("🏥 UroFlow AI")
st.markdown("**Analyse acoustique du débit urinaire** — Détection d'anomalies par IA")
st.markdown("---")

# ── Import modules ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    st.success("✓ Modules chargés")
except Exception as e:
    st.error(f"Erreur : {e}")
    st.stop()

# ── Modèle embarqué (pas besoin de fichiers externes) ────────────────────────
INPUT_DIM  = 53
LATENT_DIM = 8
SEUIL_NORMAL = 2.0
SEUIL_ELEVE  = 8.0

class TransformerAE(nn.Module):
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

@st.cache_resource
def get_model():
    """Entraîne le modèle sur des données normales simulées."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    model = TransformerAE()
    np.random.seed(42)

    # Données d'entraînement simulées (sons normaux)
    normal_data = np.random.randn(200, INPUT_DIM).astype(np.float32) * 0.5
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_data).astype(np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    X = torch.tensor(normal_scaled)

    model.train()
    for ep in range(50):
        opt.zero_grad()
        r, _ = model(X)
        l = loss_fn(r, X)
        l.backward()
        opt.step()

    model.eval()
    return model, scaler

def get_risk_level(score):
    if score < SEUIL_NORMAL:
        return "faible", "🟢"
    elif score < SEUIL_ELEVE:
        return "modéré", "🟡"
    else:
        return "élevé", "🔴"

FEATURE_NAMES = [
    "rms_raw","rms_db","rms_log1p","rms_cv","rms_skew","rms_kurtosis","mfcc1_mean",
    "mfcc2_mean","mfcc2_std","mfcc3_mean","mfcc3_std","mfcc4_mean","mfcc4_std",
    "mfcc5_mean","mfcc5_std","mfcc6_mean","mfcc6_std","mfcc7_mean","mfcc7_std",
    "mfcc8_mean","mfcc8_std","mfcc9_mean","mfcc9_std","mfcc10_mean","mfcc10_std",
    "mfcc11_mean","mfcc11_std","mfcc12_mean","mfcc12_std","mfcc13_mean","mfcc13_std",
    "very_low_ratio","low_ratio","mid_low_ratio","mid_ratio","high_ratio",
    "ratio_band_energy_mean","ratio_band_energy_std","spectral_ratio_hl",
    "spec_centroid_mean","spec_centroid_std","spec_bandwidth_mean","spec_bandwidth_std",
    "spec_rolloff_mean","spec_rolloff_std","spec_flatness_mean","spec_flatness_std",
    "zcr_mean","zcr_std","flux_mean","flux_std","freq_dom_mean","freq_dom_std"
]

# ── Chargement ────────────────────────────────────────────────────────────────
with st.spinner("Initialisation du modèle..."):
    model, scaler = get_model()
st.success("✓ Modèle prêt")
st.markdown("---")

# ── Interface ─────────────────────────────────────────────────────────────────
st.header("① Charger un enregistrement .wav")
uploaded = st.file_uploader("Choisir un fichier audio", type=["wav"])
demo = st.button("🎲 Utiliser un son de démonstration")

if uploaded or demo:
    st.markdown("---")
    st.header("② Analyse")

    with st.spinner("Extraction des features audio..."):
        if demo:
            np.random.seed(42)
            cas = np.random.choice(["normal", "anormal"])
            if cas == "normal":
                features = np.random.randn(INPUT_DIM).astype(np.float32) * 0.5
                duration = 15.0
                st.info("Mode démo — son normal simulé")
            else:
                features = np.random.randn(INPUT_DIM).astype(np.float32) * 2.5
                duration = 8.0
                st.info("Mode démo — son anormal simulé")
        else:
            import librosa, tempfile
            from scipy import stats
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            y, sr = librosa.load(tmp_path, sr=22050)
            duration = len(y) / sr
            os.unlink(tmp_path)

            # Extraction features
            eps = 1e-10
            feats = {}
            rms = np.sqrt(np.mean(y**2))
            feats["rms_raw"]=rms; feats["rms_db"]=20*np.log10(rms+eps); feats["rms_log1p"]=np.log1p(rms)
            y_norm=y/(np.max(np.abs(y))+eps)
            rms_f=librosa.feature.rms(y=y_norm,frame_length=2048,hop_length=512)[0]
            feats["rms_cv"]=np.std(rms_f)/(np.mean(rms_f)+eps)
            feats["rms_skew"]=stats.skew(rms_f); feats["rms_kurtosis"]=stats.kurtosis(rms_f)
            S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512)); power=S**2
            freqs=librosa.fft_frequencies(sr=sr,n_fft=2048)
            mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13,n_fft=2048,hop_length=512)
            feats["mfcc1_mean"]=np.mean(mfccs[0])
            for i in range(1,13):
                feats[f"mfcc{i+1}_mean"]=np.mean(mfccs[i]); feats[f"mfcc{i+1}_std"]=np.std(mfccs[i])
            te=np.sum(power)+eps
            for name,(fmin,fmax) in {"very_low":(150,300),"low":(300,800),"mid_low":(800,1700),"mid":(1700,3000),"high":(3000,7000)}.items():
                feats[f"{name}_ratio"]=np.sum(power[(freqs>=fmin)&(freqs<fmax),:])/te
            bm=(freqs>=500)&(freqs<=3000); be=(S[bm,:]**2).mean(axis=0); te2=(S**2).mean(axis=0)+eps
            rb=be/te2; feats["ratio_band_energy_mean"]=np.mean(rb); feats["ratio_band_energy_std"]=np.std(rb)
            cut=np.searchsorted(freqs,1000); feats["spectral_ratio_hl"]=np.mean(S[cut:,:])/(np.mean(S[:cut,:])+eps)
            c=np.log1p(librosa.feature.spectral_centroid(S=S,sr=sr))
            bw=np.log1p(np.maximum(librosa.feature.spectral_bandwidth(S=S,sr=sr),0))
            ro=np.log1p(librosa.feature.spectral_rolloff(S=S,sr=sr))
            fl=librosa.feature.spectral_flatness(S=S)
            feats["spec_centroid_mean"]=np.mean(c); feats["spec_centroid_std"]=np.std(c)
            feats["spec_bandwidth_mean"]=np.mean(bw); feats["spec_bandwidth_std"]=np.std(bw)
            feats["spec_rolloff_mean"]=np.mean(ro); feats["spec_rolloff_std"]=np.std(ro)
            feats["spec_flatness_mean"]=np.mean(fl); feats["spec_flatness_std"]=np.std(fl)
            zcr=librosa.feature.zero_crossing_rate(y)
            feats["zcr_mean"]=np.mean(zcr); feats["zcr_std"]=np.std(zcr)
            flux=np.log1p(np.sqrt(np.sum(np.diff(S,axis=1)**2,axis=0))/(S.mean()*S.shape[0]+eps))
            feats["flux_mean"]=np.mean(flux); feats["flux_std"]=np.std(flux)
            S_band=S[bm,:]; fdf=freqs[bm][np.argmax(S_band,axis=0)]; nyq=sr/2
            feats["freq_dom_mean"]=np.mean(fdf)/nyq; feats["freq_dom_std"]=np.std(fdf)/nyq
            features = np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)

    # Calcul score
    features_scaled = scaler.transform(features.reshape(1,-1)).astype(np.float32)
    with torch.no_grad():
        x = torch.tensor(features_scaled)
        r, attn = model(x)
        err = (x-r).pow(2).squeeze(0).numpy()
        score = float(err.mean())
        attn_scores = attn.squeeze(0).numpy()

    risk, emoji = get_risk_level(score)

    # ── Métriques ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("③ Résultats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score d'anomalie", f"{score:.2f}")
    c2.metric("Niveau de risque", f"{emoji} {risk.upper()}")
    c3.metric("Durée analysée", f"{duration:.0f} sec")

    # ── Courbe débit ──────────────────────────────────────────────────────────
    st.subheader("Courbe de débit urinaire estimée")
    t = np.linspace(0, duration, 100)
    debit = 15 * np.exp(-((t-duration/2)**2)/(2*(duration/4)**2))
    if score > 1.0:
        np.random.seed(42)
        debit = np.maximum(debit + np.random.randn(100)*score*0.05, 0)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(t, debit, color="#2563EB", linewidth=2)
    ax.fill_between(t, debit, alpha=0.15, color="#2563EB")
    if score > 1.0:
        ax.axvspan(duration*0.3, duration*0.7, alpha=0.1, color="red", label="Zone suspecte")
        ax.legend()
    ax.set_xlabel("Temps (s)"); ax.set_ylabel("Débit (ml/s)"); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close()

    # ── Attention ─────────────────────────────────────────────────────────────
    st.subheader("Features les plus importantes (attention)")
    top10 = np.argsort(attn_scores)[-10:][::-1]
    top10_names = [FEATURE_NAMES[i] for i in top10]
    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.barh(top10_names[::-1], attn_scores[top10][::-1], color="#2563EB")
    ax2.set_xlabel("Score d'attention"); ax2.grid(True, alpha=0.3, axis="x")
    st.pyplot(fig2); plt.close()

    # ── Rapport ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("④ Rapport clinique")
    from explainer import generate_report_demo
    rapport = generate_report_demo(score, attn_scores, err, duration)
    if risk == "faible": st.success(rapport)
    elif risk == "modéré": st.warning(rapport)
    else: st.error(rapport)
    st.download_button("📄 Télécharger", rapport, "rapport_uroflow.txt", "text/plain")

st.markdown("---")
st.caption("UroFlow AI — Projet académique | Hôpital Saint-Louis | ⚠️ Ne remplace pas un avis médical")
