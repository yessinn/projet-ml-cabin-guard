# CabinGuard AI — Un téléphone est-il visible dans l’habitacle ?

Projet de Machine Learning · Licence LSI3 · Faculté des Sciences de Sfax · 2026–2027
Enseignant : Mohamed Hammami
Étudiants : **Yassine Hariz** · **Yusuf Gharbi**

**Présentation (10 min) :** https://claude.ai/artifact/K3zEvQhr8dk1G7dvksSgKz

---

## Le projet

À partir d’une photo de l’intérieur d’un véhicule, le modèle prédit si un téléphone est visible :
`1` = téléphone visible, `0` = pas de téléphone. C’est une **classification binaire supervisée** sur des **images**.

Chaque image est transformée en vecteur de caractéristiques (descripteur **HOG**), ce qui permet d’appliquer la démarche du cours : sélection de caractéristiques, deux modèles (**KNN** et **SVM linéaire**), réglage par **GridSearchCV** avec une **validation croisée à 4 plis**, matrices de confusion et tableau récapitulatif.

## Correspondance avec le cahier des charges

| Étape demandée | Où la trouver |
|---|---|
| Choix du jeu de données | Driver Behavior Image Dataset, Mendeley Data v2 (CC BY 4.0) — 57 images retenues (31 sans, 26 avec téléphone) |
| Afficher et interpréter les données | `data/processed/mendeley-mobile/manifest.json`, `reports/mendeley-mobile/couverture.png` |
| Visualisation avec Seaborn | `scripts/preparer_mendeley_mobile.py` (fonction `explore` : `countplot`, `boxplot`, `stripplot`) |
| Valeurs aberrantes | Règle de Tukey (1,5 × IQR) dans `explore` ; résultat : `reports/mendeley-mobile/qualite_images.png` |
| Prétraitement | `scripts/descripteurs_adms.py` (gris, orientation EXIF, 128 × 128, HOG → 8 100 valeurs) + `StandardScaler` |
| Sélection des caractéristiques | `VarianceThreshold` + `SelectKBest(f_classif)`, k ∈ {64, 256, toutes} dans `scripts/pipelines_mendeley_mobile.py` |
| Au moins deux modèles | KNN et SVM linéaire (`build_searches`) |
| Paramètres optimaux avec CV = 4 | `GridSearchCV` sur 4 plis prédéfinis (un groupe de véhicule par pli) |
| Matrices de confusion | `reports/reproduction.json`, `reports/figures_presentation/07_confusion.png` |
| Tableaux récapitulatifs | ci-dessous et dans la présentation |

## Résultats

| Indicateur | KNN | SVM linéaire |
|---|---|---|
| Réglages retenus | k = 5, distance euclidienne, 256 variables | C = 0,1, 256 variables |
| F1 macro moyen (4 plis) | **0,597** ± 0,262 | 0,429 ± 0,112 |
| F1 macro sur les 57 images | **0,524** | 0,486 |
| Téléphones trouvés (rappel classe 1) | 8 / 26 (31 %) | **11 / 26 (42 %)** |
| Fausses alertes | **7** | 14 |
| Matrice de confusion `[[VN, FP], [FN, VP]]` | `[[24, 7], [18, 8]]` | `[[17, 14], [15, 11]]` |

Le KNN obtient le meilleur F1 macro ; le SVM détecte plus de téléphones mais déclenche deux fois plus de fausses alertes.
Le score de 1,00 du KNN sur le pli 3 porte sur 3 images seulement. Avec 57 images, ces résultats sont **exploratoires**.

## Reproduire les résultats

Testé avec Miniconda, Python 3.13 et scikit-learn 1.9.1 (Windows 11).

```bash
conda create -n cabinguard python=3.13 -y
conda activate cabinguard
pip install numpy pandas scikit-learn seaborn matplotlib pillow scipy pyyaml
python -B scripts/evaluer.py mendeley "data/processed/mendeley-mobile" --sortie reports/ma_reproduction.json
```

Le fichier de sortie doit être nouveau. Il contient les réglages retenus, les scores par pli et les matrices de confusion ; ils sont identiques à `reports/resultats_mendeley.json`.
Les descripteurs HOG déjà calculés (`hog.npy`) sont fournis : les images originales ne sont pas nécessaires pour relancer l’évaluation.

Pour régénérer les schémas de la présentation :

```bash
python scripts/figures_presentation.py reports/reproduction.json reports/figures_presentation
```

## Organisation du dépôt

```
scripts/
  descripteurs_adms.py          descripteur HOG (8 100 valeurs par image)
  preparer_mendeley_mobile.py   préparation, graphiques Seaborn, valeurs aberrantes, plis
  pipelines_mendeley_mobile.py  Pipeline + GridSearchCV pour KNN et SVM
  evaluer.py                    lance la comparaison et écrit les résultats
  adms_checks.py, adms_layout.py  audit du jeu ADMS (étape préliminaire)
  figures_presentation.py       schémas de la présentation
data/processed/mendeley-mobile/ manifeste des 57 images et matrice HOG
reports/                        résultats JSON, graphiques Seaborn, schémas
docs/                           protocole, notes techniques, captures d’écran
```

## Source des données

Arafat Sahin Afridi, Arafath Kafy, Nazmun Nessa Moon et Md. Shahriar Shakil (2024), *Driver Behavior Image Dataset*, Mendeley Data, version 2, licence CC BY 4.0. https://doi.org/10.17632/6y3g6vs2k4.2
