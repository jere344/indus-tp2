**Séance 2 — Du modèle à l'API d'inférence** 

**Master 2 · YNOV Campus Montpellier · Industrialisation de l'IA dans le Cloud · 2026-2027** 

## **En-tête de séance** 

| **Durée**     | 3 h 30 de TP                                                                               |
| ------------- | ------------------------------------------------------------------------------------------ |
| **Modalité**  | Individuel, poste personnel, compte Azure for Students                                     |
| **Marche**    | Pipeline scikit-learn sur données tabulaires, API d'inférence                              |
| **ajoutée**   |                                                                                            |
| **Prérequis** | L'image publiée en séance 1,**votre registre ACR toujours actif**(<br>`az acr show --name` |
|               | `"$ACR_NAME"`doit répondre), Docker en marche, Python 3.10 ou plus                         |
| **Livrable**  | Un dossier<br>`s2-<votre-nom>/`déposé avant la fin de la séance                            |



**Ce que vous saurez faire en sortant.** Caractériser un tableau de données avant de modéliser. Écarter les variables qui trahissent la réponse. Découper un jeu sans fuite quand les lignes ne sont pas indépendantes. Construire un pipeline scikit-learn qui apprend ses prétraitements sur le seul ensemble d'entraînement. **Choisir une métrique et un seuil défendables quand une classe est mille fois plus rare que l'autre.** Emballer le tout dans une API et publier l'artefact. 

**Ce que vous ne ferez pas aujourd'hui.** Vous ne **déployez** rien : aucun service ne tournera dans le cloud à la fin de la séance. Votre image, elle, y part bien — vous la publiez dans votre registre en partie 4, et c'est tout ce qui franchit la frontière. La mise en ligne, c'est la séance 3, et elle ne prendra votre artefact que s'il respecte le contrat d'interface rappelé plus bas. 

## **Contexte métier — le fil rouge du module** 

**DermaScan** est un service de télédermatologie. Des cabines de photographie corporelle totale installées en pharmacie photographient la peau de patients volontaires, découpent automatiquement chaque lésion repérée en une vignette de 15 mm, et en extraient une quarantaine de mesures : taille, couleur, régularité du contour, symétrie, position anatomique. 

Un dermatologue ne peut pas relire les centaines de lésions qu'un seul patient produit. La question posée à l'ingénieur est donc une question de **tri** : quelles lésions méritent un examen humain ? 

C'est une question de santé, et elle a deux coûts asymétriques. Manquer une lésion maligne peut coûter une vie. Signaler à tort immobilise un créneau de dermatologue, inquiète un patient, et coûte un examen. **Il n'existe aucun réglage qui annule les deux.** Votre travail n'est pas de trouver ce réglage inexistant : c'est de choisir le compromis, de le chiffrer, et de l'assumer par écrit. 

**Les données sont réelles.** Elles viennent du jeu ISIC 2024, publié par l'International Skin Imaging Collaboration, et agrègent des lésions de neuf institutions hospitalières. Ce sont des **données de santé** : la partie 5 vous demandera d'en tirer les conséquences. 

## **Convention de lecture de l'énoncé** 

Les blancs `_______` sont à compléter. Ils ne sont jamais décoratifs : chacun porte une décision. 

- **L'énoncé n'affirme aucun chiffre.** Les volumétries, les scores, les seuils et les durées sont **vos** mesures. Deux apprenants n'obtiendront pas exactement les mêmes valeurs, et un chiffre recopié sur le voisin se repère immédiatement. 

- Les relevés sont numérotés `**Mesure k**` , les questions `**Question p.q**` . Le barème du corrigé les désigne par ces numéros. 

- Les commandes se lancent depuis la racine `s2-dermascan/` sauf mention contraire. 

## **Tableau de décisions** 

À compléter **avant** de commencer, et à reporter en tête de `MESURES.md` . 

| **Décision**                                            | **Votre valeur**             |
| ------------------------------------------------------- | ---------------------------- |
| Suffixe personnel, identique à la séance 1              | `EG08`                       |
| Nom du groupe portant le registre, créé en séance 1     | `rg-dermascan-registre-EG08` |
| Nom du registre ACR de la séance 1                      | `dermascaneg08`              |
| Rappel visé, décidé**avant**de voir le moindre résultat | `0.90`                       |
| Date portée par l'étiquette<br>`a_detruire`             | `2027-07-01"`                |



**Question 0.1** — Le rappel visé se décide avant la première mesure. Pourquoi ce n'est pas une coquetterie méthodologique : qu'est-ce qui se passerait si vous le choisissiez après avoir vu la courbe ? 

> Si on choisit le rappel après avoir vu la courbe on risque d'adapter le rappel à nos données ce qui fausserait les résultats.

## **Contrat d'interface à respecter** 

La séance 3 déploiera votre image. Elle attend exactement ceci, et le contrôle est automatique : 

| **Élément**                   | **Valeur imposée**                                               |
| ----------------------------- | ---------------------------------------------------------------- |
| Nom du dépôt dans le registre | `dermascan-api`                                                  |
| Tag                           | `1.0.0`                                                          |
| Port d'écoute                 | lu dans la variable<br>`PORT` , défaut<br>`8000`                 |
| Sonde de santé                | `GET /health` , code 200 si et seulement si le modèle est chargé |
| Carte du modèle               | `GET /model/info`                                                |
| Prédiction                    | `POST /predict` , corps JSON                                     |
| Chemin du modèle dans l'image | lu dans<br>`MODEL_PATH` , défaut<br>`model/modele.joblib`        |



**Partie 1 — Exploration et préparation des données** 

### **Durée indicative : 50 minutes.** 

On ne modélise pas un jeu de données qu'on n'a pas regardé. Cette partie ne produit aucun modèle : elle produit les **chiffres** sans lesquels vous ne saurez pas, en partie 2, si votre modèle est bon ou seulement chanceux. 

## **1.1 — Environnement et obtention des données** 

```
# Dossier de travail de la séance, à côté de celui de la séance 1 et non dedans.
mkdir -p s2-dermascan/{data,model,captures} && cd s2-dermascan
```

```
# Environnement virtuel dédié : l'image, elle, installera ses dépendances seule.
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install pandas scikit-learn joblib flask gunicorn requests kaggle
```

Les données se récupèrent depuis Kaggle. **La procédure complète est en annexe A** : création du jeton, acceptation des règles de la compétition, téléchargement. Faites-la maintenant, elle prend cinq minutes et elle échoue de deux façons prévisibles que l'annexe nomme. 

```
# Produit data/train-metadata.csv. Seul le fichier d'annotation est
# nécessaire aujourd'hui : les images ne servent qu'en séance 4.
ls -lh data/train-metadata.csv
```

**Mesure 1** — versions et volumétrie du fichier. 

| **Relevé**              | **Commande**                                                   | **Votre**<br>**valeur** |
| ----------------------- | -------------------------------------------------------------- | ----------------------- |
| Version de Python       | `python3 --version`                                            | 3.12.3                  |
| Version de scikit-learn | `python3 -c "import sklearn;`<br>`print(sklearn.__version__)"` | 1.9.1                   |
| Version de pandas       | `python3 -c "import pandas; print(pandas.__version__)"`        | 3.0.6                   |
| Poids du fichier        | `ls -lh`                                                       |                         |
| d'annotation            |                                                                |                         |



## **1.2 — Volumétrie et structure** 

Créez `explorer.py` . Ce script ne produit aucun modèle : il produit les chiffres que vous recopierez dans `EXPLORATION.md` . Vous l'écrivez vous-même, à partir de la liste ci-dessous. 

Il doit afficher : le nombre de lignes et de colonnes ; le nombre de **patients distincts** ; le nombre de lésions par patient — minimum, médiane, maximum ; et le type de chaque colonne. 

```
import pandas as pd
d = pd.read_csv("data/train-metadata.csv", low_memory=False)
print("lignes, colonnes :", d.shape)
print("patients distincts :", d["patient_id"].nunique())
tailles = d.groupby("patient_id").size()
print("lesions par patient :", tailles.min(), int(tailles.median()), tailles.max())
```

`low_memory` mérite une seconde d'attention : sur un fichier de 55 colonnes dont plusieurs sont partiellement vides, le réglage par défaut fait deviner les types bloc par bloc et produit des avertissements de type mixte. 

```
python3 explorer.py
```


**Mesure 2** — volumétrie et structure. 

| **Relevé**                                        | **Votre valeur** |
| ------------------------------------------------- | ---------------- |
| Nombre de lignes                                  | 401059           |
| Nombre de colonnes                                | 55               |
| Nombre de patients distincts                      | 1042             |
| Lésions par patient : minimum / médiane / maximum | 1 241 9184       |



**Question 1.1** — Comparez le nombre de lignes au nombre de patients. Une ligne est-elle une observation indépendante des autres ? Votre réponse commande le découpage de la partie 2.1 : dites laquelle des deux grandeurs détermine la taille réelle de votre échantillon. 

> Le nombre de ligne est beaucoup plus grand que le nombre de patient, il y a plusieurs lignes par patient les observations ne sont donc pas indépendantes et peuvent être corréllée. La taille réelle est notre nombre de patient.

## **1.3 — Les colonnes qui trahissent la réponse** 

Un tableau de données issu d'un contexte médical contient presque toujours des colonnes renseignées **après** le diagnostic. Les garder, c'est prédire la réponse avec la réponse : le modèle affiche un score parfait en validation et s'effondre en production, où ces colonnes n'existent pas encore. 

Étendez `explorer.py` : pour chaque colonne suspecte, affichez le nombre de valeurs renseignées, et **parmi les lignes renseignées, combien sont malignes** . 

```
SUSPECTES = ["iddx_full", "iddx_1", "mel_thick_mm", "mel_mitotic_index", "tbp_lv_dnn_lesion_confidence"]
for c in SUSPECTES:
    rempli = d[c].notna().sum()
    malignes = d.loc[d[c].notna(), "target"].sum()
print(f"{c:<22} rempli {rempli:>7}  dont malignes {int(malignes):>4}")
```

Le rapport qui compte n'est pas « combien de valeurs », c'est **« combien de malignes parmi les lignes renseignées »** , rapporté au nombre total de malignes. 

**Mesure 3** — colonnes à écarter pour cause de fuite. 

| **Colonne**         | **Valeurs renseignées** | **Dont malignes** | **Écartée ?** |
| ------------------- | ----------------------- | ----------------- | ------------- |
| `iddx_1`            | 401059                  | 393               | Oui           |
| `iddx_full`         | 401059                  | 393               | Oui           |
| `mel_thick_mm`      | 63                      | 63                | Non           |
| `mel_mitotic_index` | 53                      | 53                | non           |
| `lesion_id`         | 22058                   | 393               | Oui           |



**Question 1.2** — `lesion_id` est un identifiant, pas une mesure clinique. Pourtant il doit être écarté. Regardez votre relevé et dites **pourquoi** : qu'est-ce que sa seule présence révèle, et à quel moment du parcours d'un patient cet identifiant est-il attribué ? 

> Si un identifiant est présent c'est que la lésion a été déjà diagnostiquée, elle va donc être maligne et impacter le modèle.


**Question 1.3** — Un camarade vous dit : « `iddx_1` donne 100 % de bonnes réponses, je le garde. » Formulez en deux phrases ce qui ne va pas, en distinguant ce que le modèle apprend de ce qu'il saura faire le jour où une nouvelle lésion arrive. 

> En production nous n'aurons pas accès à cette colonne, le modèle doit donc être capable de prédire la malignité sans cette information. Le modèle entrainé sur cette colonne sera entrainé sur des données avec déjà le résultats et ne pourra pas généraliser sur de nouvelles données.

## **1.4 — Valeurs manquantes et colonnes sans information** 

```
na = d.isna().sum()
print(na[na > 0].sort_values(ascending=False))
constantes = [c for c in d.columns if d[c].nunique(dropna=True) <= 1]
print("constantes :", constantes)
```

**Mesure 4** — colonnes à valeurs manquantes et colonnes constantes. 

iddx_5                 401058
mel_mitotic_index      401006
mel_thick_mm           400996
iddx_4                 400508
iddx_3                 399994
iddx_2                 399991
lesion_id              379001
sex                     11517
anatom_site_general      5756
age_approx               2798
dtype: int64
constantes : ['image_type', 'iddx_5']

| **Relevé**                                                                         | **Votre valeur**                                              |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Colonnes comportant au moins une valeur manquante, hors colonnes de fuite          | age_approx,se,anatom_site_general,iddx_2,iddx_3,iddx_4,iddx_5 |
| Pour chacune : taux de valeurs manquantes (%)                                      | 0.7, 2.87, 1.44, 99.73, 99.73, 99.87, 100                     |
| Colonnes constantes, c'est-à-dire à une seule modalité                             | image_type, iddx_5                                            |
| Nombre de variables restantes après retrait des fuites, identifiants et constantes | 47                                                            |




**Question 1.4** — Une colonne constante ne gêne aucun algorithme : elle est simplement ignorée. Donnez malgré tout **une** raison de la retirer explicitement, et **une** raison de consigner son existence plutôt que de l'oublier. 

> Raison de le retirer : même si elle ne gène pas la précision de l'algorithme elle l'alourdie et n'a aucun interêt, autant la retirer pour optimiser
> Raison de la consiger : elle peut être utile pour documentation

## **1.5 — Distribution de la cible** 

**Mesure 5** — déséquilibre de la cible. 

| **Relevé**                        | **Votre valeur** |
| --------------------------------- | ---------------- |
| Nombre de lésions<br>`target = 1` | 393              |
| Nombre de lésions<br>`target = 0` | 400666           |
| Taux de positifs, en pourcentage  | 0.0979%          |



Rapport « une maligne pour N bénignes » 

**Question 1.5** — Un classifieur qui répond « bénin » à **toutes** les lésions atteint quelle exactitude sur ce jeu ? Calculez-la à partir de votre mesure 5, et dites ce que cette valeur vous apprend sur l'exactitude comme critère de choix. 

> Un classifieur qui répond "Bénin" tout le temps atteint 97.9% d'exactitude ! L'exactitude n'est pas un critère universel, notre objectif d'exactitude doit être relatif à notre distribution de données. ici 97.9% parait impressionant mais n'est en réalité pas suffisant. 
> Il faut aussi prendre en compte le taux inverse, le taux de faux négatif qui est très important dans notre cas. Ce taux serait de 100% pour ce classifieur, ce qui est inacceptable.

