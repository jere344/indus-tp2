import os
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Colonnes qui trahissent la reponse (renseignees apres / pendant le diagnostic)
FUITE = [
    "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
    "mel_thick_mm", "mel_mitotic_index", "tbp_lv_dnn_lesion_confidence",
    "lesion_id",
]
# Identifiants et metadonnees non cliniques
METADATA = ["isic_id", "attribution", "copyright_license", "image_type"]
GROUPES, CIBLE = "patient_id", "target"
RAPPEL_CIBLE = 0.90

d = pd.read_csv("data/train-metadata.csv", low_memory=False)

y = d[CIBLE]
groupes = d[GROUPES]

a_ecarter = set(FUITE + METADATA + [GROUPES, CIBLE])
features = [c for c in d.columns if c not in a_ecarter]
colonnes_numeriques = [c for c in features if pd.api.types.is_numeric_dtype(d[c])]
colonnes_categorielles = [c for c in features if c not in colonnes_numeriques]
X = d[features]


def etape_2_1():
    decoupe = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train, test = next(decoupe.split(X, y, groups=groupes))

    # La preuve se fait par comptage, pas par confiance.
    assert not (set(groupes[train]) & set(groupes[test])), "fuite de patients"

    for nom, idx in (("train", train), ("test", test)):
        print(
            f"{nom:>5} : {len(idx):>7} lignes | "
            f"{groupes[idx].nunique():>5} patients | "
            f"{int(y[idx].sum()):>4} positifs | "
            f"{y[idx].mean() * 100:.4f} % positifs"
        )
    print("intersection patients :", len(set(groupes[train]) & set(groupes[test])))
    return train, test


def construire_modele():
    pre = ColumnTransformer([
        ("num", Pipeline([("imputation", SimpleImputer(strategy="median")),
                          ("echelle", StandardScaler())]), colonnes_numeriques),
        ("cat", Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                          ("encodage", OneHotEncoder(handle_unknown="ignore"))]),
         colonnes_categorielles),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def etape_2_3(train, test):
    reference = DummyClassifier(strategy="most_frequent").fit(X.iloc[train], y[train])
    modele = construire_modele().fit(X.iloc[train], y[train])

    proba = modele.predict_proba(X.iloc[test])[:, 1]
    proba_ref = reference.predict_proba(X.iloc[test])[:, 1]

    print("Dummy  exactitude", accuracy_score(y[test], reference.predict(X.iloc[test])))
    print("Dummy  ROC-AUC   ", roc_auc_score(y[test], proba_ref))
    print("Dummy  PR-AUC    ", average_precision_score(y[test], proba_ref))
    print("Modele exactitude", accuracy_score(y[test], modele.predict(X.iloc[test])))
    print("Modele ROC-AUC   ", roc_auc_score(y[test], proba))
    print("Modele PR-AUC    ", average_precision_score(y[test], proba))
    print("Hasard PR-AUC    ", y[test].mean())
    return modele, proba


def tableau_seuil(y_true, proba, seuils_vises=(0.50, 0.80, 0.90, 0.99)):
    for s in seuils_vises:
        pred = proba >= s
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        rappel = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        print(
            f"seuil {s:.2f} | rappel {rappel:.3f} | precision {precision:.3f} | "
            f"TP {tp:>3} | FP {fp:>5} | FN {fn:>3} | alertes/1000 {pred.mean() * 1000:.1f}"
        )


def etape_2_4(test, proba):
    y_test = y[test].to_numpy()
    tableau_seuil(y_test, proba)

    precision_c, rappel_c, seuils_c = precision_recall_curve(y_test, proba)
    i = np.where(rappel_c >= RAPPEL_CIBLE)[0][-1]
    seuil_retenu = seuils_c[min(i, len(seuils_c) - 1)]
    pred = proba >= seuil_retenu
    faux_negatifs = int((~pred & (y_test == 1)).sum())

    print("rappel cible                        :", RAPPEL_CIBLE)
    print("seuil retenu                        :", seuil_retenu)
    print("precision a ce seuil                :", precision_c[i])
    print("lesions examinees par maligne trouvee :", round(1 / precision_c[i]))
    print("faux negatifs restants              :", faux_negatifs)

    metriques = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "rappel": float(rappel_c[i]),
        "precision": float(precision_c[i]),
        "faux_negatifs": faux_negatifs,
    }
    return seuil_retenu, metriques


def etape_2_5(modele, train, seuil_retenu, metriques, duree_entrainement):
    paquet = {
        "pipeline": modele,
        "version": "1.0.0",
        "colonnes_attendues": list(features),
        "seuil": float(seuil_retenu),
        "rappel_cible": RAPPEL_CIBLE,
        "metriques": metriques,
        "prevalence_entrainement": float(y[train].mean()),
        "entraine_le": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    chemin = "model/modele.joblib"
    joblib.dump(paquet, chemin)

    print("Mesure 10")
    print("taille du fichier (Ko)       :", round(os.path.getsize(chemin) / 1024, 1))
    print("colonnes attendues declarees :", len(paquet["colonnes_attendues"]))
    print("duree de l'entrainement (s)  :", round(duree_entrainement, 3))
    return paquet


if __name__ == "__main__":
    train, test = etape_2_1()
    debut = time.perf_counter()
    modele, proba = etape_2_3(train, test)
    duree = time.perf_counter() - debut
    seuil_retenu, metriques = etape_2_4(test, proba)
    etape_2_5(modele, train, seuil_retenu, metriques, duree)
