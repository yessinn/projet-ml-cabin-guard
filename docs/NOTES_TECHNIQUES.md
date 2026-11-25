# CabinGuard AI

Projet de Machine Learning LSI3, Faculté des Sciences de Sfax. La première expérience compare KNN et SVM linéaire pour détecter un téléphone visible sur une image d’habitacle.

## Contenu

- `scripts/evaluer.py` exécute la comparaison Mendeley ou l’évaluation sur un corpus indépendant.
- `scripts/preparer_mendeley_mobile.py` prépare les données, les graphiques Seaborn et les descripteurs HOG à partir du registre local.
- `reports/resultats_mendeley.json` contient les scores par pli, les paramètres et les matrices de confusion confirmés le 25 novembre 2026.
- `reports/mendeley-mobile/` contient les graphiques de couverture et de qualité.
- `docs/CabinGuard_Phase_2_et_collecte.md` documente l’expérience et le protocole de collecte.
- `manifest_collecte.csv` contient les colonnes du manifeste à remplir à partir de photos consenties.

## Reproduire les résultats

Utiliser Python 3.14 et installer les dépendances avec `python -m pip install -r requirements.txt`. Depuis le dossier extrait, lancer :

```powershell
python -B scripts/evaluer.py mendeley "data/processed/mendeley-mobile" --sortie reports/reproduction.json
```

Le ZIP contient le manifeste et les descripteurs HOG nécessaires à la reproduction. Les images originales restent dans le dossier de travail du projet. Les empreintes des données préparées sont consignées dans le rapport et les résultats JSON.

Source des images de l’expérience · Arafat Sahin Afridi, Arafath Kafy, Nazmun Nessa Moon et Md. Shahriar Shakil (2024), *Driver Behavior Image Dataset*, CC BY 4.0. [Mendeley Data, version 2](https://doi.org/10.17632/6y3g6vs2k4.2)

## Tester sur des participants réservés

Placer un manifeste complété à la racine du corpus. La colonne `chemin_relatif` est relative au dossier du manifeste. Les labels possibles sont `0`, `1` et `?`. L’ensemble doit valoir `developpement` ou `test`. Attribuer des identifiants de session et de séquence uniques dans le corpus, par exemple `P01-S01` et `P01-S01-V01`.

```powershell
python -B scripts/evaluer.py independant "<corpus>/manifest.csv" --controler --sortie reports/controle_collecte.json
python -B scripts/evaluer.py independant "<corpus>/manifest.csv" --sortie reports/test_independant.json
```

Le contrôle vérifie entre autres les empreintes, la séparation des participants, les rapprochements dHash inter-ensembles et la couverture des classes dans les plis. Examiner les rapprochements signalés avant de poursuivre. Lancer le deuxième appel après avoir figé le corpus et les grilles de réglage. Le script optimise les modèles uniquement sur le développement puis calcule les métriques sur le test. Les chemins de sortie doivent être nouveaux.

## Périmètre des résultats

La comparaison publiée est exploratoire. Le même découpage technique sert à régler et comparer les modèles Mendeley. La collecte de volontaires et l’évaluation indépendante restent à réaliser. Une étude de la non-réactivité d’un conducteur nécessiterait un corpus vidéo et des labels adaptés.

