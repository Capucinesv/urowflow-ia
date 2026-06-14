import requests
import json
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# MODULE ③ : LLM Explainer
#
# Principe :
#   - Reçoit le score d'anomalie + les features importantes
#   - Envoie ces infos à un LLM (Claude via API Anthropic)
#   - Le LLM génère un rapport clinique lisible par un médecin
#
# Pourquoi un LLM ?
#   L'auto-encodeur donne un chiffre (ex: 7.8). Ce chiffre ne veut rien dire
#   pour un médecin. Le LLM transforme ce chiffre en texte compréhensible.
# ──────────────────────────────────────────────────────────────────────────────

# Noms des 38 features (correspondant à features.py)
FEATURE_NAMES = (
    [f"mfcc_{i+1}_mean" for i in range(13)] +
    [f"mfcc_{i+1}_std"  for i in range(13)] +
    ["rms_mean", "rms_std",
     "zcr_mean", "zcr_std",
     "centroid_mean", "centroid_std",
     "bandwidth_mean", "bandwidth_std",
     "rolloff_mean", "rolloff_std",
     "chroma_mean", "chroma_std"]
)

# Seuils pour interpréter le score d'anomalie
SEUIL_NORMAL  = 1.0   # en dessous → normal
SEUIL_MOYEN   = 3.0   # entre 1 et 3 → à surveiller
                       # au dessus de 3 → anomalie détectée


def get_risk_level(score: float) -> str:
    """Convertit un score numérique en niveau de risque lisible."""
    if score < SEUIL_NORMAL:
        return "faible"
    elif score < SEUIL_MOYEN:
        return "modéré"
    else:
        return "élevé"


def build_prompt(score: float, attn_scores: np.ndarray,
                 error_per_feature: np.ndarray, duration: float = 10.0) -> str:
    """
    Construit le prompt envoyé au LLM.
    On lui donne toutes les infos nécessaires pour générer un rapport utile.
    """
    risk = get_risk_level(score)

    # Top 3 features les plus suspectes (erreur de reconstruction la plus haute)
    top3_idx = np.argsort(error_per_feature)[-3:][::-1]
    top3_names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
                  for i in top3_idx]
    top3_errors = [float(error_per_feature[i]) for i in top3_idx]

    # Top 3 features les plus importantes selon l'attention
    top3_attn_idx = np.argsort(attn_scores)[-3:][::-1]
    top3_attn_names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
                       for i in top3_attn_idx]

    prompt = f"""Tu es un assistant médical spécialisé en urologie. Analyse les données suivantes 
issues d'un système de détection d'anomalies urologiques basé sur l'analyse sonore d'une miction.

DONNÉES D'ANALYSE :
- Score d'anomalie global : {score:.2f} (seuil normal < {SEUIL_NORMAL}, seuil élevé > {SEUIL_MOYEN})
- Niveau de risque estimé : {risk}
- Durée de la miction analysée : {duration:.1f} secondes
- Features les plus anormales (erreur de reconstruction élevée) :
  1. {top3_names[0]} : erreur = {top3_errors[0]:.3f}
  2. {top3_names[1]} : erreur = {top3_errors[1]:.3f}
  3. {top3_names[2]} : erreur = {top3_errors[2]:.3f}
- Features les plus importantes selon le mécanisme d'attention :
  {', '.join(top3_attn_names)}

LEXIQUE pour ton interprétation :
- mfcc : coefficients qui décrivent la forme spectrale du son (timbre)
- rms : énergie acoustique (volume du jet)
- zcr : zero crossing rate (irrégularité du flux)
- centroid : centre de gravité fréquentiel (grave vs aigu)
- bandwidth : étendue des fréquences (jet concentré vs dispersé)

Génère un rapport clinique court (5-7 lignes) en français avec :
1. Une interprétation claire du score pour un médecin non-technicien
2. Ce que les features anormales suggèrent sur le flux urinaire
3. Un niveau de risque clair (faible / modéré / élevé)
4. Une recommandation concrète (surveillance, consultation, urgence)

Sois précis mais accessible. Ne fais pas de diagnostic définitif."""

    return prompt


def generate_report(score: float, attn_scores: np.ndarray,
                    error_per_feature: np.ndarray,
                    api_key: str, duration: float = 10.0) -> str:
    """
    Envoie les données au LLM et retourne le rapport généré.

    Args:
        score             : score d'anomalie calculé par l'auto-encodeur
        attn_scores       : poids d'attention par feature
        error_per_feature : erreur de reconstruction par feature
        api_key           : clé API Anthropic
        duration          : durée de la miction en secondes

    Returns:
        rapport           : texte du rapport clinique généré
    """
    prompt = build_prompt(score, attn_scores, error_per_feature, duration)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    body = {
        "model": "claude-haiku-4-5-20251001",  # modèle rapide et économique
        "max_tokens": 400,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    except requests.exceptions.RequestException as e:
        return f"[Erreur de connexion au LLM : {e}]"
    except (KeyError, IndexError):
        return "[Erreur : réponse inattendue du LLM]"


def generate_report_demo(score: float, attn_scores: np.ndarray,
                         error_per_feature: np.ndarray,
                         duration: float = 10.0) -> str:
    """
    Version de démonstration sans clé API.
    Génère un rapport basé sur des règles simples — utile pour tester l'interface.
    """
    risk = get_risk_level(score)
    top3_idx = np.argsort(error_per_feature)[-3:][::-1]
    top3_names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
                  for i in top3_idx]

    if risk == "faible":
        interpretation = (
            "Le profil acoustique de cette miction est dans les limites normales. "
            "Le flux urinaire présente une régularité et une intensité cohérentes "
            "avec un débit sain."
        )
        recommandation = "Aucune action requise. Poursuite du suivi habituel recommandée."

    elif risk == "modéré":
        interpretation = (
            f"Une légère irrégularité est détectée, principalement sur {top3_names[0]} "
            f"et {top3_names[1]}, suggérant une possible variabilité du flux urinaire. "
            "Cela peut indiquer une légère obstruction ou une tension musculaire."
        )
        recommandation = "Surveillance renforcée conseillée. Consulter un urologue si le phénomène persiste."

    else:
        interpretation = (
            f"Des anomalies significatives sont détectées sur {top3_names[0]}, "
            f"{top3_names[1]} et {top3_names[2]}. Ces irrégularités acoustiques "
            "suggèrent un débit urinaire perturbé, potentiellement lié à une "
            "obstruction prostatique ou une dysfonction du sphincter."
        )
        recommandation = "Consultation urologique recommandée dans les meilleurs délais."

    rapport = f"""RAPPORT D'ANALYSE UROFLOW — MODE DÉMONSTRATION
{'='*50}
Score d'anomalie  : {score:.2f}
Niveau de risque  : {risk.upper()}
Durée analysée    : {duration:.0f} secondes

INTERPRÉTATION :
{interpretation}

RECOMMANDATION :
{recommandation}

---
Note : Ce rapport est généré automatiquement par analyse acoustique.
Il ne remplace pas un avis médical professionnel.
"""
    return rapport


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Test du module LLM Explainer (mode démonstration)...\n")

    # Simule des données venant de l'auto-encodeur
    np.random.seed(0)
    n_features = 38

    # Cas 1 : son normal
    score_n = 0.3
    attn_n  = np.random.dirichlet(np.ones(n_features))
    err_n   = np.random.exponential(0.1, n_features)
    print("── CAS NORMAL ──────────────────────────")
    print(generate_report_demo(score_n, attn_n, err_n, duration=15.0))

    # Cas 2 : anomalie élevée
    score_a = 5.2
    attn_a  = np.random.dirichlet(np.ones(n_features))
    err_a   = np.random.exponential(1.5, n_features)
    print("\n── CAS ANORMAL ─────────────────────────")
    print(generate_report_demo(score_a, attn_a, err_a, duration=8.0))