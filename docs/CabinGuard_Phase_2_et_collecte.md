# CabinGuard AI · Phase 2 et collecte

25 novembre 2026 · Machine Learning LSI3

## Résultats sur Mendeley v2

La tâche est de repérer la présence d’un téléphone visible dans une photo d’habitacle. KNN et SVM linéaire ont été réglés et comparés sur les mêmes quatre plis prédéfinis. La validation couvre 57 images, dont 26 avec un téléphone visible et 31 sans téléphone visible. Chaque pipeline ajuste le filtre de variance, la standardisation et la sélection ANOVA sur les seules images d’apprentissage du pli.

| Modèle | Réglages retenus | Pli 1 (7) | Pli 2 (33) | Pli 3 (3) | Pli 4 (14) | F1 macro moyen ± écart-type | F1 macro sur les 57 prédictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KNN | 5 voisins, distance euclidienne, 256 variables | 0,286 | 0,476 | 1,000 | 0,626 | 0,597 ± 0,262 | 0,524 |
| SVM linéaire | C = 0,1, 256 variables | 0,533 | 0,515 | 0,250 | 0,417 | 0,429 ± 0,112 | 0,486 |

### Matrices de confusion

| Modèle | État observé | Prédit absent | Prédit présent |
| --- | --- | ---: | ---: |
| KNN | Absent | 24 | 7 |
| KNN | Présent | 18 | 8 |
| SVM linéaire | Absent | 17 | 14 |
| SVM linéaire | Présent | 15 | 11 |

Le KNN obtient le meilleur F1 macro moyen. Le SVM repère 11 téléphones sur 26, contre 8 pour le KNN, avec 14 fausses alertes contre 7. Le pli parfait du KNN ne contient que trois images. Le réglage et la comparaison utilisent les mêmes plis et l’identité des conducteurs du corpus public reste inconnue. Ces résultats décrivent une validation exploratoire et devront être complétés par un test indépendant.

Exécution avec scikit-learn 1.9.1. Le script de référence est `scripts/pipelines_mendeley_mobile.py`. Les deux grilles ont évalué 24 configurations KNN et 9 configurations SVM, avec le F1 macro comme critère. Les prédictions de validation ont été recalculées avec les réglages retenus et vérifiées contre les scores des quatre plis.

| Fichier de référence | SHA-256 |
| --- | --- |
| `data/processed/mendeley-mobile/manifest.json` | `2f447ab86a993ada219cfdc4abb1dc29d875ae303c54cc988efeb058b54addec` |
| `data/processed/mendeley-mobile/hog.npy` | `afe132a8b56a0eb94eab9c2fc423273c52f92276f25061cb0a0b0299f54cff63` |

## Collecte du corpus indépendant

- Recruter dix volontaires consentants et attribuer un identifiant par personne. Avant la première séance, réserver P07 à P10 pour le test. Affecter P01 à P06 au développement. Garder le registre d’accord à part des images et du manifeste.
- Prévoir deux courtes séances par personne. Photographier le haut du corps, le volant et la console dans un véhicule immobilisé, moteur coupé et frein de stationnement serré. Vérifier qu’aucune plaque, personne extérieure ou donnée personnelle affichée à l’écran n’entre dans le champ.
- Viser 10 à 15 images exploitables par classe et par personne au total, soit 200 à 300 images. Alterner les deux états dans chaque séance. Varier les positions du téléphone, les postures et les objets ressemblants dans les deux classes. Écarter les rafales de vues quasi identiques.
- Annoter chaque image à partir des pixels. Utiliser `1` pour un téléphone identifiable, qu’il soit tenu, posé ou sur un support. Utiliser `0` pour une image exploitable où aucun téléphone n’est visible. Utiliser `?` si un objet, le flou ou l’occlusion empêche de trancher. Documenter ce dernier cas et l’écarter des scores binaires.
- Garder les images et labels du test à part pendant les réglages. Contrôler la séparation par participant, session et séquence ainsi que les doublons entre les ensembles. Figer quatre plis de développement par participant et vérifier que chaque pli et son complément contiennent les deux classes. Une fois les réglages fixés, produire sur le test les mêmes métriques et matrices pour les deux modèles.

### Modèle de manifeste CSV

Utiliser une ligne par photo. Identifier les sessions et séquences avec le préfixe du participant. Calculer le SHA-256 sur le fichier original. Le registre reste local avec les photos.

```csv
chemin_relatif,sha256,participant,session,sequence,habitacle,ensemble,mobile,motif
```

| Champ | Valeur attendue |
| --- | --- |
| `chemin_relatif` | Chemin de la photo depuis la racine du corpus |
| `sha256` | Empreinte du fichier sur 64 caractères hexadécimaux |
| `participant`, `session`, `sequence`, `habitacle` | Identifiants pseudonymes |
| `ensemble` | `developpement` ou `test`, selon le participant |
| `mobile` | `0`, `1` ou `?` |
| `motif` | Indice visible ou raison de l’incertitude |

## Entrée pour PROJECT_STATUS.md

> Phase 2 exploratoire Mendeley v2 exécutée le 25 novembre 2026. KNN et SVM linéaire comparés sur 57 images et quatre plis techniques. F1 macro moyen de 0,597 pour KNN et 0,429 pour SVM. Collecte du corpus indépendant et test sur des participants réservés en attente.

