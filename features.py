import numpy as np
import librosa
from scipy import stats

# Liste des features dans le bon ordre (53 features, même ordre que le CSV d'Edouard)
FEATURE_NAMES = [
    "rms_raw","rms_db","rms_log1p","rms_cv","rms_skew","rms_kurtosis",
    "mfcc1_mean",
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

def extract_features(file_path: str, sr: int = 22050) -> np.ndarray:
    """
    Extrait 53 features acoustiques d'un fichier .wav.
    Découpe automatiquement en segments de 0.5s et retourne la moyenne.
    Basée sur la fonction d'Edouard Steiner (Uroflow Meter, 2026).
    """
    y, sr = librosa.load(file_path, sr=sr)
    seg_len = int(0.5 * sr)
    segments = [y[i:i+seg_len] for i in range(0, len(y)-seg_len, seg_len)]
    if not segments:
        segments = [y]
    all_feats = [extract_features_segment(seg, sr) for seg in segments]
    return np.mean(all_feats, axis=0).astype(np.float32)


def extract_features_segment(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extrait 53 features acoustiques d'un segment audio de 0.5s.
    """
    eps = 1e-10
    feats = {}

    # ── RMS ──────────────────────────────────────────────────────────────────
    rms = np.sqrt(np.mean(y**2))
    feats["rms_raw"]   = rms
    feats["rms_db"]    = 20 * np.log10(rms + eps)
    feats["rms_log1p"] = np.log1p(rms)
    y_norm = y / (np.max(np.abs(y)) + eps)
    rms_f  = librosa.feature.rms(y=y_norm, frame_length=2048, hop_length=512)[0]
    feats["rms_cv"]       = np.std(rms_f) / (np.mean(rms_f) + eps)
    feats["rms_skew"]     = stats.skew(rms_f)
    feats["rms_kurtosis"] = stats.kurtosis(rms_f)

    # ── STFT ─────────────────────────────────────────────────────────────────
    S     = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    power = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # ── MFCCs ────────────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    feats["mfcc1_mean"] = np.mean(mfccs[0])
    for i in range(1, 13):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        feats[f"mfcc{i+1}_std"]  = np.std(mfccs[i])

    # ── Bandes d'énergie ─────────────────────────────────────────────────────
    total_e = np.sum(power) + eps
    for name, (fmin, fmax) in {
        "very_low":(150,300), "low":(300,800), "mid_low":(800,1700),
        "mid":(1700,3000),    "high":(3000,7000)
    }.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        feats[f"{name}_ratio"] = np.sum(power[mask, :]) / total_e

    # ── Ratio bande 500-3000 Hz ───────────────────────────────────────────────
    bm  = (freqs >= 500) & (freqs <= 3000)
    be  = (S[bm, :]**2).mean(axis=0)
    te  = (S**2).mean(axis=0) + eps
    rb  = be / te
    feats["ratio_band_energy_mean"] = np.mean(rb)
    feats["ratio_band_energy_std"]  = np.std(rb)

    # ── Ratio haut/bas ────────────────────────────────────────────────────────
    cut = np.searchsorted(freqs, 1000)
    feats["spectral_ratio_hl"] = np.mean(S[cut:, :]) / (np.mean(S[:cut, :]) + eps)

    # ── Features spectrales ───────────────────────────────────────────────────
    centroid  = np.log1p(librosa.feature.spectral_centroid(S=S, sr=sr))
    bandwidth = np.log1p(np.maximum(librosa.feature.spectral_bandwidth(S=S, sr=sr), 0))
    rolloff   = np.log1p(librosa.feature.spectral_rolloff(S=S, sr=sr))
    flatness  = librosa.feature.spectral_flatness(S=S)
    feats["spec_centroid_mean"]  = np.mean(centroid);  feats["spec_centroid_std"]  = np.std(centroid)
    feats["spec_bandwidth_mean"] = np.mean(bandwidth); feats["spec_bandwidth_std"] = np.std(bandwidth)
    feats["spec_rolloff_mean"]   = np.mean(rolloff);   feats["spec_rolloff_std"]   = np.std(rolloff)
    feats["spec_flatness_mean"]  = np.mean(flatness);  feats["spec_flatness_std"]  = np.std(flatness)

    # ── ZCR ──────────────────────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = np.mean(zcr)
    feats["zcr_std"]  = np.std(zcr)

    # ── Flux spectral ─────────────────────────────────────────────────────────
    flux = np.log1p(np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)) / (S.mean()*S.shape[0] + eps))
    feats["flux_mean"] = np.mean(flux)
    feats["flux_std"]  = np.std(flux)

    # ── Fréquence dominante ───────────────────────────────────────────────────
    S_band = S[bm, :]
    fdf    = freqs[bm][np.argmax(S_band, axis=0)]
    nyq    = sr / 2
    feats["freq_dom_mean"] = np.mean(fdf) / nyq
    feats["freq_dom_std"]  = np.std(fdf)  / nyq

    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)


def get_feature_names() -> list:
    return FEATURE_NAMES


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        f = extract_features(sys.argv[1])
        print(f"Features extraites : {len(f)}")
        print(f"Noms : {FEATURE_NAMES[:5]} ...")
    else:
        print(f"Nb features : {len(FEATURE_NAMES)}")
        print("Usage : python3 features_real.py mon_fichier.wav")