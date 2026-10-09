import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
app.json.ensure_ascii = False

# Chargement au demarrage, PAS dans la fonction de vue.
try:
    PAQUET = joblib.load(os.environ.get("MODEL_PATH", "model/modele.joblib"))
    PIPELINE = PAQUET["pipeline"]
    SEUIL = float(os.environ.get("SEUIL", PAQUET["seuil"]))
    ERREUR = None
except Exception as exc:
    PAQUET = None
    PIPELINE = None
    SEUIL = None
    ERREUR = f"{type(exc).__name__}: {exc}"

COLONNES = PAQUET["colonnes_attendues"] if PAQUET else []


@app.get("/health")
def health():
    if PIPELINE is None:
        return jsonify({"model_loaded": False, "erreur": ERREUR}), 503
    return jsonify({"model_loaded": True}), 200


@app.get("/model/info")
def model_info():
    if PAQUET is None:
        return jsonify({"error": "modele indisponible", "erreur": ERREUR}), 503
    return jsonify(
        {
            "version": PAQUET["version"],
            "seuil": SEUIL,
            "rappel_cible": PAQUET["rappel_cible"],
            "metriques": PAQUET["metriques"],
            "prevalence_entrainement": PAQUET["prevalence_entrainement"],
            "colonnes_attendues": COLONNES,
        }
    )


@app.post("/predict")
def predict():
    if PIPELINE is None:
        return jsonify({"error": "modele indisponible", "erreur": ERREUR}), 503
    charge = request.get_json(silent=True) or {}
    manquantes = [c for c in COLONNES if c not in charge]
    if manquantes:
        return jsonify({"error": "colonnes manquantes", "manquantes": manquantes[:10]}), 400

    ligne = pd.DataFrame([charge])[COLONNES]
    proba = float(PIPELINE.predict_proba(ligne)[:, 1][0])
    return jsonify(
        {
            "probabilite_maligne": round(proba, 6),
            "seuil": SEUIL,
            "decision": "maligne" if proba >= SEUIL else "bénin",
            "version_modele": PAQUET["version"],
            "rappel_cible": PAQUET["rappel_cible"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
