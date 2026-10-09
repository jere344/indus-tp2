# Séance 2 — Mesures et questions (mesures 6 à 17, questions 2.1 à 6.1)

## Tableau de décisions

| **Décision**                                            | **Votre valeur**             |
| ------------------------------------------------------- | ---------------------------- |
| Suffixe personnel, identique à la séance 1              | `EG08`                       |
| Nom du groupe portant le registre, créé en séance 1     | `rg-dermascan-registre-EG08` |
| Nom du registre ACR de la séance 1                      | `dermascaneg08`              |
| Rappel visé, décidé**avant**de voir le moindre résultat | `0.90`                       |
| Date portée par l'étiquette<br>`a_detruire`             | `2027-07-01"`                |

**Partie 2 — Pipeline scikit-learn : entraînement, évaluation, seuil** 

**Durée indicative : 70 minutes.** 

## **2.1 — Découpage sans fuite** 

Votre réponse à la question 1.1 commande cette partie. Un découpage aléatoire ligne à ligne placerait les lésions d'un **même patient** des deux côtés de la barrière : le modèle serait évalué sur des patients qu'il a déjà vus, et son score serait flatteur. 

```
from sklearn.model_selection import GroupShuffleSplit
decoupe = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train, test = next(decoupe.split(X, y, groups=groupes))
```

**La preuve se fait par comptage, pas par confiance** : l'intersection des patients des deux côtés doit être vide. 

```
assertnot (set(groupes[train]) & set(groupes[test])), "fuite de patients"
```

**Mesure 6** — découpage. 

| **Relevé**         | **Entraînement** | **Test** |
| ------------------ | ---------------- | -------- |
| Nombre de lignes   | 319 514          | 81 545   |
| Nombre de patients | 833              | 209      |
| Nombre de positifs | 283              | 110      |
| Taux de positifs   | 0,0886 %         | 0,1349 % |



**Question 2.1** — Vos deux taux de positifs sont-ils identiques ? `GroupShuffleSplit` ne stratifie pas. Dites ce que cela implique quand la classe rare compte quelques centaines de cas, et nommez la conséquence sur la **stabilité** de vos scores d'une graine à l'autre. 

> Les deux taux de positifs ne sont pas identiques. `GroupShuffleSplit` ne stratifie pas, le taux de positifs n'est pas ajusté entre les deux groupes, ce qui implique des variations assez larges sur des petits jeux de données (ici seulement 393 malignes) et donc des résultats mesurés entre graines instables.

## **2.2 — Construction du pipeline** 

Les prétraitements doivent **apprendre leurs statistiques sur l'entraînement seul** . Une imputation calculée sur le tableau complet fait entrer l'information du test dans le modèle. 

```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```

```
pre = ColumnTransformer([
    ("num", Pipeline([("imputation", SimpleImputer(strategy="median")),
                      ("echelle", StandardScaler())]), colonnes_numeriques),
    ("cat", Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                      ("encodage", OneHotEncoder(handle_unknown="ignore"))]),
colonnes_categorielles),
])
```

**Question 2.2** — `handle_unknown` : que se passe-t-il, en production, si une cabine remonte une position anatomique jamais vue à l'entraînement ? Comparez les deux comportements possibles et dites lequel vous retenez pour un service de tri médical. 

> Si une cabine remonte une position anatomique jamais vue à l'entraînement, le modèle peut soit retourner une erreur, soit reprendre la prédiction en ignorant la variable, comme si elle n'était pas transmise. Pour un service médical, il est important de garantir la stabilité du système, `ignore` est donc la valeur voulue. Un filet additionnel pourrait être prévu pour signaler ce genre de cas et éviter un faux négatif à cause d'une prédiction dégradée.

**Question 2.3** — Pourquoi l'imputation est-elle une **étape du pipeline** et non un traitement appliqué au tableau avant le découpage ? Une phrase, qui nomme ce qui entrerait dans le modèle dans le second cas. 

> Imputer avec des données non groupée fait calculer la médiane par exemple sur les données du groupe test ET du groupe entraînement : il y aurait une fuite des données du groupe de test qui fausserait donc les choix du modèle.

## **2.3 — Entraînement et le piège de l'exactitude** 

Entraînez d'abord une **référence qui n'apprend rien** : `DummyClassifier(strategy="most_frequent")` . Puis votre modèle : une régression logistique, avec `class_weight="balanced"` . 

```
from sklearn.linear_model import LogisticRegression
modele = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
```

```
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
```

```
reference = DummyClassifier(strategy="most_frequent").fit(X.iloc[train], y[train])
proba = modele.predict_proba(X.iloc[test])[:, 1]
print("exactitude", accuracy_score(y[test], modele.predict(X.iloc[test])))
print("PR-AUC   ", average_precision_score(y[test], proba))
print("hasard   ", y[test].mean())
```

**Mesure 7** — comparaison des deux modèles, sur l'ensemble de test. 

| **Modèle**                      | **Exactitude** | **ROC-AUC** | **PR-AUC** |
| ------------------------------- | -------------- | ----------- | ---------- |
| `DummyClassifier`               | 0,9987         | 0,500       | 0,00135    |
| Régression logistique           | 0,8407         | 0,900       | 0,0381     |
| Valeur du hasard pour la PR-AUC | —              | —           | 0,00135    |



**Question 2.4** — Regardez la colonne « Exactitude ». Lequel des deux modèles est le meilleur selon ce seul critère ? Lequel l'est réellement ? **Cette contradiction est le cœur de la séance** : expliquez-la en nommant ce que l'exactitude compte et ce qu'elle ne compte pas. 

> Selon l'exactitude le dummy est le meilleurs modèle. En réalité, c'est celui entrainé, car l'exactitude compte chaque ligne de la même manière, les faux positif et les faux négatifs sont à égalité. Pour une utilisation réelle, il faut prendre en compte la rareté de la classe positive : seulement 110 maligne pour 81 435 bénin.

**Question 2.5** — La PR-AUC du hasard vaut exactement le taux de positifs. Vérifiez-le sur vos chiffres, puis dites pourquoi la PR-AUC est plus informative que la ROC-AUC quand la classe positive est très rare. 

> La PR-AUC du hasard (0,001349) est égale au taux de positifs du test (110/81 545 = 0,001349). Quand la classe positive est très rare, la PR-AUC est plus informative car ainsi chaque faux positif comptes, avec ROC-AUC ils sont noyés par tout les bénins négatifs. Le hasard vaut le taux de positif (0.0013) au lieu de 0.5, donc les gains sont plus significatifs.

## **2.4 — Choix du seuil** 

Un modèle de classification ne décide pas : il attribue une probabilité. **C'est vous qui décidez** , en posant un seuil. Et ce seuil n'est pas un détail technique : c'est un arbitrage clinique, qui se justifie devant un médecin. 

Produisez le tableau du compromis sur votre ensemble de test. 

**Mesure 8** — compromis rappel / précision. 

| **Seuil** | **Rappel** | **Précision** | **Vrais positifs** | **Faux positifs** | **Faux négatifs** | **Alertes pour 1000 lésions** |
| --------- | ---------- | ------------- | ------------------ | ----------------- | ----------------- | ----------------------------- |
| 0,50      | 0,836      | 0,007         | 92                 | 12 969            | 18                | 160,2                         |
| 0,80      | 0,591      | 0,014         | 65                 | 4 548             | 45                | 56,6                          |
| 0,90      | 0,436      | 0,019         | 48                 | 2 414             | 62                | 30,2                          |
| 0,99      | 0,145      | 0,035         | 16                 | 443               | 94                | 5,6                           |



Puis déterminez le seuil qui atteint le **rappel que vous avez inscrit au tableau de décisions** , avant d'avoir vu quoi que ce soit. 

```
from sklearn.metrics import precision_recall_curve
precision, rappel, seuils = precision_recall_curve(y_test, proba)
# precision_recall_curve renvoie un seuil de MOINS que de points.
# Plus petit seuil garantissant au moins le rappel vise :
i = np.where(rappel >= rappel_cible)[0][-1]
seuil_retenu = seuils[min(i, len(seuils) - 1)]
# Cout du compromis, exprime comme un clinicien le comprend :
print("lesions examinees par maligne trouvee :", round(1 / precision[i]))
```

**Mesure 9** — seuil retenu et son coût. 

| **Relevé**                                          | **Votre valeur** |
| --------------------------------------------------- | ---------------- |
| Rappel visé, décidé en début de séance              | 0,90             |
| Seuil correspondant                                 | 0,333            |
| Précision à ce seuil                                | 0,0045           |
| **Nombre de lésions examinées par maligne trouvée** | 223              |
| Faux négatifs restants                              | 11               |



**Question 2.6** — Votre dernière ligne : combien de lésions un dermatologue doit-il examiner pour trouver une maligne, à votre seuil ? Présentez ce chiffre comme vous le feriez devant le médecin-chef de DermaScan, et dites ce que vous lui répondez s'il juge le nombre inacceptable. 

> Avec un seuil de 90%, il faut examiner 223 lésions pour en trouver une maligne. Ce nombre peut paraitre élevé, mais c'est une avancée majeure par rapport aux 1019 évaluations qu'il faut faire actuellement. Cet algorithme à pour effet de manquer 10% des patient avec lésions malignes, mais permet de diviser par plus de 4 le nombre de rendez vous nécéssaire, ce qui pourrait permettre d'accueillir et d'aider en absolut plus de patient.

**Question 2.7** — Les faux négatifs ne tombent jamais à zéro. Formulez en une phrase, **sans adverbe** , ce que votre service promet et ce qu'il ne promet pas. 

> Le service garantit un rappel de 90 %, il ne garantit une détection totale.

## **2.5 — Sérialisation avec métadonnées** 

Le paquet sérialisé doit permettre à un tiers de servir votre modèle **sans vous** . Un pipeline seul n'y suffit pas. 

```
paquet = {
"pipeline": modele,
"version": "1.0.0",
"colonnes_attendues": list(features),
"seuil": float(seuil_retenu),
"rappel_cible": RAPPEL_CIBLE,
"metriques": {...},
"prevalence_entrainement": float(y[train].mean()),
"entraine_le": datetime.now(timezone.utc).isoformat(),
}
joblib.dump(paquet, "model/modele.joblib")
```

**Mesure 10** — paquet produit. 

| **Relevé**                             | **Votre valeur** |
| -------------------------------------- | ---------------- |
| Taille du fichier, en Ko               | 9,2              |
| Nombre de colonnes attendues déclarées | 39               |
| Durée de l'entraînement, en secondes   | 22,1             |



**Question 2.8** — Pourquoi `prevalence_entrainement` figure-t-elle dans le paquet ? Elle ne sert à aucun calcul au moment de la prédiction. À quoi servira-t-elle en séance 5, quand il faudra décider si le flux reçu ressemble encore à celui de l'entraînement ? 

> prevalence_entrainement ne sert à rien aujourd'hui au moment de la prédiction. Mais cette valeur nous permettra en séance 5 de calculer le drift de notre donnée. Si cette valeur diminue, alors notre modèle n'est plus aligné sur nos données réelle et la qualité des résultats va chuter, c'est là qu'il faudra un nouvel entrainement.

**Question 2.9** — Citez deux conséquences concrètes, pour l'API de la partie 3, de l'absence de `colonnes_attendues` dans le paquet. 

> Sans colonnes_attendues dans le paquet nous ne pouvons pas filtrer les données, le transform reçoit des données inattendue et peut avoir un comportement inattendue, une erreur ou une prédiction silencieusement fausse.
> Sans ces données notre api ne pourra pas fournir de d'information sur le schéma attendu (/model/info)

# **Partie 3 — API d'inférence** 

**Durée indicative : 50 minutes.** 

## **3.1 — Chargement du modèle** 

Le paquet se charge **une fois** , au démarrage du processus, et non à chaque requête. 

```
# Chargement au demarrage, PAS dans la fonction de vue.
try:
    PAQUET = joblib.load(os.environ.get("MODEL_PATH", "model/modele.joblib"))
    PIPELINE = PAQUET["pipeline"]
    SEUIL = float(os.environ.get("SEUIL", PAQUET["seuil"]))
except Exception as exc:
    PIPELINE = None
    ERREUR = f"{type(exc).__name__}: {exc}"
```

Un modèle absent ne doit **pas** empêcher le processus de démarrer : c'est la sonde qui signalera l'indisponibilité, pas un plantage. 

**Question 3.1** — Chiffrez l'enjeu à partir de votre mesure 10 : quelle serait la latence ajoutée à chaque appel si le modèle était rechargé à chaque requête ? 

> Le chiffre de la mesure 10 est celui de l'entrainement du modele, pas de son chargement. Son chargement prend ~1s mesuré. Le recharger à chaque requête entrainerai 1 seconde de latence additionnelle à chaque appel. (ps: en réalité il y a du cache intégré et les rechargements ultérieurs ne prennent que 1ms mais c'est vraiment une spécificité du système utilisé)

## **3.2 — Les trois endpoints** 

| **Endpoint**    | **Comportement attendu**                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `GET /health`   | 200 si le modèle est chargé,**503 sinon**— le processus vivant et le service en état de servir ne |
|                 | sont pas la même chose                                                                            |
| `GET`           | Version, seuil, rappel visé, métriques, prévalence d'entraînement, colonnes attendues             |
| `/model/info`   |                                                                                                   |
| `POST /predict` | Probabilité, seuil appliqué,**décision**, version du modèle                                       |



**Validez avant de prédire.** Une charge utile incomplète doit produire un `400` nommant ce qui manque, jamais un `500` . 

```
manquantes = [c for c in COLONNES if c notin charge]
if manquantes:
return jsonify({"error": "colonnes manquantes", "manquantes": manquantes[:10]}), 400
```

```
return jsonify({
"probabilite_maligne": round(proba, 6),
"seuil": SEUIL,
"decision": "maligne"if proba >= SEUIL else"bénin",
"version_modele": PAQUET["version"],
```

```
"rappel_cible": PAQUET["rappel_cible"],
})
```

**Question 3.2** — Votre `/predict` renvoie `decision` **et** `seuil` **et** `rappel_cible` . Pourquoi renvoyer les trois plutôt que la seule décision ? Pensez au dermatologue qui recevra cette réponse dans six mois, quand le seuil aura changé. 

> Une décision seule manque de contexte : si le seuil bouge ou qu'un nouvel entraînement a lieu, le médecin ne le saura pas forcément, et les prédictions précédentes seront impossibles à interpréter.

## **3.3 — Tests locaux** 

```
PORT=8000 python3 app.py &
curl -s -i http://localhost:8000/health | head -1
curl -s http://localhost:8000/model/info
# Prediction sur une ligne reelle extraite de votre jeu de test.
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d @charge.json
```

**Mesure 11** — réponses de l'API. 

| **Appel** |                                | **Code HTTP** | **Élément relevé**                                    |
| --------- | ------------------------------ | ------------- | ----------------------------------------------------- |
| `GET`     | `/health`                      | 200           | champ<br>`model_loaded` : `true`                      |
| `GET`     | `/model/info`                  | 200           | 39 colonnes attendues                                 |
| `POST`    | `/predict` , charge valide     | 200           | `probabilite_maligne` 0,866314 / `decision` `maligne` |
| `POST`    | `/predict` , charge incomplète | 400           | 3 variables manquantes signalées                      |
| `POST`    | `/predict` , corps vide        | 400           | 10 variables signalées (liste tronquée)               |



**Mesure 12** — latence de dix appels successifs sur charge valide : minimum **6,7 ms**, médiane **7,6 ms**, maximum **8,6 ms**. **Question 3.3** — Retirez temporairement `model/modele.joblib` et relancez l'API. Quel code renvoie 

`/health` ? Pourquoi est-ce **préférable** à un plantage au démarrage, du point de vue de la plateforme qui hébergera ce service en séance 3 ? 

> Sans modèle, `/health` renvoie 503. Si le service ne renvoyait rien, la plateforme risquerait de le considérer comme mort et de le redémarrer en boucle. Avec un 503, la plateforme agit en conséquence : le processus reste ouvert et l'erreur est observable dans les logs.

**Partie 4 — Conteneurisation et publication** 

### **Durée indicative : 40 minutes.** 

Vous ne réécrivez pas votre Dockerfile de la séance 1. Vous l'étendez : trois lignes de dépendances supplémentaires et un fichier de modèle à copier. 

```
# Les dependances d'abord, le code ensuite : l'ordre de la seance 1 tient toujours.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Le modele est un artefact, pas du code : il a sa propre ligne.
COPY model/modele.joblib ./model/
COPY app.py ./
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
```

**Question 4.1** — Votre `requirements.txt` passe de une à cinq dépendances directes. À partir de votre mesure de la séance 1, estimez l'écart de taille d'image attendu, puis vérifiez. 

> 467 / 133 ~= 3.5
> L'écart de taille d'image est de 350% malgré le peu de code ajouté. C'est à cause des dépendances ajoutées qui ont chacune leurs dépendances transitives.

**Mesure 13** — image de la séance 2, comparée à celle de la séance 1. 

| **Relevé**                             | **Séance 1** | **Séance 2** |
| -------------------------------------- | ------------ | ------------ |
| Taille de l'image                      | 133 Mo       | 467 Mo       |
| Nombre de couches                      | 9            | 10           |
| Durée de construction à froid          | 6,362 s      | 56,6 s       |
| Nombre de paquets dans<br>`pip freeze` | 7            | 18           |



**Question 4.2** — Le modèle pèse quelques kilo-octets, les dépendances plusieurs centaines de mégaoctets. Qu'est-ce que cela vous dit sur ce qui coûte cher dans un artefact d'inférence, et sur ce qu'il faudrait optimiser en premier si la taille devenait un problème ? 

> Le modèle pèse 9ko et l'image 467Mo. Si la taille devient un problème, pour optimiser il faudrait limiter les paquets inutiles ou trouver des dépendances équivalentes plus légères plutôt que d'essayer de travailler sur notre petit algorithme.

## **4.3 — L'artefact est lié à la version qui l'a produit** 

Lancez votre conteneur et appelez `/predict` . **Il est probable que vous obteniez une erreur** , alors que le même appel fonctionnait hors conteneur quelques minutes plus tôt. 

```
docker run -d --name dermascan-api -p 8000:8000 dermascan-api:1.0.0
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d @charge.json
```

Un pipeline scikit-learn sérialisé n'est **pas** un format d'échange : c'est un graphe d'objets Python, et il ne se recharge pas dans une version de bibliothèque différente de celle qui l'a produit. Votre environnement virtuel et votre image n'ont aucune raison d'avoir les mêmes versions — sauf si vous les y forcez. 

**Mesure 15** — cohérence des versions entre l'entraînement et l'image. 

**Relevé Votre valeur** 

| **Relevé**                                                                      | **Votre valeur**            |
| ------------------------------------------------------------------------------- | --------------------------- |
| Version de scikit-learn ayant entraîné le modèle — champ<br>`versions`du paquet | 1.9.1                       |
| Version de scikit-learn épinglée dans<br>`requirements.txt`                     | 1.9.1                       |
| Sont-elles identiques ?                                                         | oui                         |
| Message d'erreur exact, s'il y en a un                                          | aucun (versions identiques) |



Corrigez en épinglant dans `requirements.txt` **les versions relevées à la mesure 1** , puis reconstruisez. **Question 4.3** — Vous auriez pu ne pas épingler du tout et laisser `pip` installer la dernière version. Donnez les deux moments où cette image aurait cessé de fonctionner, et dites lequel des deux est le plus dangereux pour un service déjà en production. 

> Sans épingler la version des paquets il y a un risque qu'un rebuild futur casse l'image au démarrage en production.
> Ne pas épingler la version des paquets c'est accepter que l'environnement n'est pas stable, reconstruire le même tag donne une nouvelle image, impossible de reproduire la situation prévue au développement.
> Le danger en production est en cas de nouveau déploiement, une image fonctionnelle sera remplacée par une image cassée et on ne le découvrira qu'au démarrage de la nouvelle image.

## **4.4 — Publication** 

```
# Reconstruction pour la cible du cloud, comme en seance 1.
docker build --platform linux/amd64 --provenance=false -t "$IMAGE_FULL" .
docker push "$IMAGE_FULL"
az acr repository show-tags --name "$ACR_NAME" --repository dermascan-api --output table
```

**Mesure 16** — publication : durée du push, digest obtenu, tags présents dans le registre. 

> Durée du push : **5 min 51 s**.
> Digest : `sha256:32fccc4d96727e154151ff7bbf30f3a3d98a01d55836d611dd387ff3aa566637`.
> Tags présents dans `dermascan-api` : `1.0.0`.

**Partie 5 — Gouvernance : model card** 

### **Durée indicative : 25 minutes.** 

Les données que vous venez d'exploiter sont des **données de santé** , collectées dans neuf institutions hospitalières. Une model card n'est pas un exercice de style : c'est ce qui permet à quelqu'un d'autre de décider s'il peut utiliser votre modèle, et pour quoi. 

Rédigez `MODEL_CARD.md` avec ces rubriques : 

| **Rubrique**                      | **Ce qu'on y attend**                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| Usage prévu                       | Le tri, et rien d'autre                                                                |
| **Usage explicitement exclu**     | Nommez-le                                                                              |
| Données d'entraînement            | Origine, volumétrie, prévalence,**et le fait qu'elle diffère de la prévalence réelle** |
| Variables utilisées               | Nombre, et les familles                                                                |
| Variables écartées, avec le motif | Les fuites, nommément                                                                  |
| Performance                       | PR-AUC, et la valeur du hasard en regard                                               |
| Seuil et compromis                | Le chiffre de votre mesure 9                                                           |
| Populations et angles morts       | Qui est sous-représenté dans neuf hôpitaux occidentaux ?                               |
| Versions                          | Modèle, image, digest                                                                  |



**Question 5.1** — Votre modèle a été entraîné sur une prévalence de l'ordre de 0,1 %. Si DermaScan l'installe dans une consultation spécialisée où une lésion sur vingt est maligne, votre seuil reste-t-il valable ? Justifiez, et dites ce qu'il faudrait refaire. 

> Le seuil choisi n'est plus valable à la nouvelle situation. Ici les lésions sont les mêmes, on pourra probablement garder le même entrainement en recalculant le seuil. Mais si les lésions de ces patients sont légèrements différentes alors il pourrait être nécessaire de faire un nouvel entrainnement avec les nouvelles données.

**Question 5.2** — Citez une population pour laquelle vous **refuseriez** de garantir les performances annoncées, et dites sur quelle colonne de votre jeu vous vous appuyez pour l'affirmer. 

> Pour les populations peu représentés dans le modèle de donnée je refuserais de garantir les performances du modèle. Pour les adolescents par exemple nous n'avons presque aucune données, 0.16% < 18 ans (colonne age_approx)

# **Nettoyage obligatoire** 

Le nettoyage n'est pas la destruction aveugle de ce que vous avez produit : c'est la **décision explicite** de ce qui survit et de ce qui disparaît, avec sa raison et sa date. 

**Pourquoi cette section décide de la suite du module.** Votre abonnement _Azure for Students_ n'est pas seulement une réserve d'argent : c'est lui qui vous donne accès à `Microsoft.ContainerRegistry` et `Microsoft.ContainerInstance` . **Un crédit épuisé ne vous laisse pas un abonnement pauvre : il le fait basculer sur l'offre** **_Starter_ , qui interdit purement et simplement ces deux fournisseurs.** Vous n'y perdez pas la possibilité de payer, vous y perdez votre registre — donc l'artefact que les séances suivantes consomment. Épuiser son crédit en séance 3, c'est ne plus pouvoir faire la séance 4, la séance 5, ni le projet fil rouge. Le nettoyage n'est pas une bonne pratique de plus : c'est ce qui vous permet de terminer le module. 

Comme en séance 1, vous laissez volontairement une ressource active — la même, votre registre. Raison de plus pour relever votre crédit : comparez-le au relevé de la séance 1, c'est l' **écart** entre deux mesures qui révèle ce qui tourne à votre insu. 

```
# Conteneurs de la seance, puis images locales : l'image publiee, elle, reste.
docker rm -f dermascan-api 2>/dev/null
docker system prune -f
deactivate
```

**Décision sur les ressources Azure.** Cette séance **ne crée aucun groupe de ressources** : elle publie dans le registre créé en séance 1, qui est conservé **jusqu'à la fin du module** — la séance 4 et le projet fil rouge y publient encore. Vous ne détruisez donc rien côté Azure, et vous contrôlez que ce que vous laissez actif est bien ce que vous croyez — une ressource, et une date d'échéance encore valide. 

```
az resource list --resource-group "$RG" --output table
az group show --name "$RG" --query "tags" --output json
```

**Mesure 17** — contrôles de nettoyage et coût de ce que vous laissez actif. 

| **Contrôle de fin de séance**                     | **Votre observation**    |
| ------------------------------------------------- | ------------------------ |
| Espace disque libéré par<br>`docker system prune` | 1,421 Go                 |
| Espace libéré en séance 1, pour comparaison       | 76,91 Go                 |
| Nombre de ressources laissées actives             | 1 (le registre)          |
| Coût journalier estimé du SKU<br>`Basic`          | $0,167/jour (~$5/mois)   |
| Date portée par l'étiquette<br>`a_detruire`       | 2027-07-01               |
| Crédit restant, relevé dans<br>`Cost Management`  | $88 (aucune utilisation) |
| Écart avec le relevé de la séance précédente      | 0 ($88 → $88)            |



**Question 6.1** — En séance 1 vous avez rédigé la phrase de ticket qui justifie de laisser ce registre actif. Relisez-la : est-elle encore exacte ? Le registre ne contient plus la même chose qu'il y a une semaine, et sa date d'échéance a peut-être été écrite avant que vous sachiez combien de séances en dépendraient. Corrigez-la, et dites en une ligne ce qui vous a fait la corriger — ou pourquoi elle tient toujours. 

> J'avais indiqué "jusqu'à la fin du module", elle est donc toujours exacte.

## **Avant de quitter la salle — extinction** 

**Toute ressource active est une ressource que vous payez.** Vous payez la _réservation_ , pas l'usage : un service que personne n'appelle, la nuit et le week-end, coûte exactement le même prix qu'un service sollicité. C'est cette règle, et elle seule, qui décide si vous finirez le module avec du crédit. 

Trois gestes, dans cet ordre. Aucun n'est facultatif, et ils se font **avant** de fermer votre session, pas « plus tard depuis chez vous » — un oubli du vendredi se paie jusqu'au lundi. 

1. **Côté Azure. rien n'a été créé aujourd'hui, et rien ne doit l'être.** Cette séance n'a fait que publier une image dans le registre de la séance 1. Contrôlez qu'aucun groupe ne s'est ajouté à votre insu. 

2. **Côté poste — arrêtez ce qui tourne encore.** Conteneurs de la séance supprimés, Docker Desktop **quitté** (un démon laissé actif relance vos conteneurs au prochain démarrage et consomme mémoire et batterie), environnement virtuel désactivé. 

3. **Éteignez votre machine.** Éteinte, pas en veille. 

```
docker ps -a                 # plus aucun conteneur de la séance
az group list --output table # toujours un seul groupe, celui du registre
```

| **Contrôle d'extinction**                                         | **Fait** |
| ----------------------------------------------------------------- | -------- |
| `az group list`ne montre toujours qu'un groupe, celui du registre | yes      |
| Aucun conteneur de la séance dans<br>`docker ps -a`               | yes      |
| Docker Desktop quitté                                             | yes      |
| Machine éteinte                                                   | yes      |



