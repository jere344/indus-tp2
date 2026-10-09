# Model Card — DermaScan (agent de tri)

## Usage prévu
Tri de lésions cutanées : à partir des métadonnées d'une lésion, estimer la
probabilité qu'elle soit maligne et **classer** la lésion en « à examiner en
priorité » ou « surveillance ordinaire ». Le modèle est un **auxiliaire de
tri**, jamais un verdict.

## Usage explicitement exclu
**Le diagnostic.** Le service ne dit pas qu'une lésion est un mélanome ; il
dit qu'elle mérite l'attention d'un dermatologue. Est exclu tout usage
diagnostique autonome, toute décision médicale prise sur la seule sortie du
modèle, et tout usage sur d'autres lésions que celles décrites par le schéma
de colonnes attendu.

## Données d'entraînement
- **Origine** : annotations de la compétition Kaggle ISIC 2024 (`SLICE-3D`),
  collectées dans **neuf institutions hospitalières**.
- **Volumétrie** : 401 059 lésions, 1 042 patients, 55 colonnes ; 39 retenues.
- **Prévalence** : 393 lésions malignes, soit **0,098 %** (0,089 % sur le seul
  ensemble d'entraînement).
- **Écart avec la prévalence réelle** : cette prévalence est celle d'un jeu de
  dépistage à grande échelle, pas celle d'une consultation. Une consultation
  spécialisée peut voir **une lésion sur vingt** maligne (5 %), soit ~50 fois
  plus. Le seuil et le couple précision/rappel décrits plus bas sont calibrés
  pour la prévalence d'entraînement et **ne se transposent pas** tels quels.

## Variables utilisées
**39 variables** en deux familles :
- **34 numériques** : taille de la lésion (`clin_size_long_diam_mm`), âge
  (`age_approx`) et mesures colorimétriques / morphologiques de la région
  (`tbp_lv_*` : couleur LAB et ses dérivées, aire, périmètre, excentricité,
  symétries, position 3D…).
- **5 catégorielles** : `sex`, `anatom_site_general`, `tbp_tile_type`,
  `tbp_lv_location`, `tbp_lv_location_simple`.

## Variables écartées, avec le motif
- **Fuites du diagnostic** (encodent directement la réponse) :
  `iddx_full`, `iddx_1`, `iddx_2`, `iddx_3`, `iddx_4`, `iddx_5`
  (libellés diagnostiques), `mel_thick_mm`, `mel_mitotic_index`
  (histologie du mélanome), `tbp_lv_dnn_lesion_confidence` (sortie d'un autre
  modèle), `lesion_id` (identifiant de la lésion).
- **Métadonnées de traçabilité** : `isic_id`, `attribution`,
  `copyright_license`, `image_type`.
- **Clé de regroupement** : `patient_id` (sert au découpage, jamais en
  variable).

## Performance
- **PR-AUC = 0,0381** — valeur du **hasard : 0,00135** (soit la prévalence).
- ROC-AUC = 0,900.
- Exactitude : 0,841 — **non interprétable** seule, vu la rareté de la classe
  positive (un modèle qui prédit toujours « bénin » atteint 0,999
  d'exactitude).

## Seuil et compromis
Seuil retenu : **0,333**, pour un **rappel visé de 0,90**.
- Rappel obtenu : 0,90 (11 faux négatifs sur 110 malignes).
- Précision : 0,0045.
- **223 lésions à examiner pour trouver une maligne** (contre ~1 019 au
  hasard, 1/prévalence).
Le seuil est optimisé pour cette prévalence : dans un contexte plus riche en
cas malins, la précision monterait et le seuil devrait être ré-évalué.

## Populations et angles morts
Angles morts identifiables : les données proviennent de **neuf hôpitaux
occidentaux**, donc sont sous-représentées les **peaux foncées / phototypes
élevés**, les **populations non occidentales** et probablement les **jeunes
patients**. Colonnes d'appui pour l'affirmer : `anatom_site_general`, `sex`
et `age_approx` (la distribution d'entraînement de ces colonnes reflète la
population des neuf hôpitaux, non la population cible). Toute performance est
non garantie pour ces populations.

## Versions
- Modèle (paquet sérialisé) : version **1.0.0** ; Python 3.12.3,
  scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0.
- Image : `dermascaneg08.azurecr.io/dermascan-api:1.0.0`.
- Digest : `sha256:32fccc4d96727e154151ff7bbf30f3a3d98a01d55836d611dd387ff3aa566637`.
