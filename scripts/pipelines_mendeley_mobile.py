"""Construire les recherches CV et les sorties d'évaluation Mendeley à blanc."""

from collections import defaultdict
import hashlib
import json
import os

import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             make_scorer, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preparer_mendeley_mobile import FOLD_COUNT, OUT, ROOT, STATES


def load_cv_data(directory):
    payload = (directory / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    if manifest["evaluation"] != "exploratoire" or manifest["test_independant"]:
        raise ValueError("Ce module attend une évaluation exploratoire Mendeley.")
    rows = [row for row in manifest["rows"] if row["pli_validation"] is not None]
    if any(row["exclusion"] or row["mobile"] not in STATES for row in rows):
        raise ValueError("Une image affectée au développement est exclue ou sans label.")
    folds = defaultdict(set)
    for row in rows:
        folds[row["groupe"]].add(row["pli_validation"])
    if any(len(group_folds) > 1 for group_folds in folds.values()):
        raise ValueError("Un groupe traverse plusieurs plis.")
    assignments = np.asarray([row["pli_validation"] for row in rows], dtype=int)
    if set(assignments) != set(range(1, FOLD_COUNT + 1)):
        raise ValueError("Il faut exactement quatre plis prédéfinis.")
    labels = np.asarray([row["mobile"] for row in rows], dtype=int)
    vectors = np.load(directory / "hog.npy", allow_pickle=False)
    expected_shape = (len(manifest["rows"]), 8100)
    if vectors.shape != expected_shape or sorted(row["index_hog"] for row in manifest["rows"]) != list(range(len(vectors))):
        raise ValueError("La matrice HOG et le manifeste ne correspondent pas.")
    features = vectors[[row["index_hog"] for row in rows]]
    if not np.isfinite(features).all():
        raise ValueError("Les descripteurs HOG doivent être finis.")
    cv = PredefinedSplit(assignments - 1)
    for train, valid in cv.split():
        if set(labels[train]) != set(STATES) or set(labels[valid]) != set(STATES):
            raise ValueError("Les deux classes doivent couvrir chaque pli et son complément.")
    return features, labels, cv, hashlib.sha256(payload).hexdigest()


def prepare_searches(directory=OUT):
    """Instancier deux GridSearchCV sans ajuster d'estimateur ni de transformation."""
    features, labels, cv, version = load_cv_data(directory)
    return features, labels, build_searches(features, cv), version


def build_searches(features, cv):
    train_splits = [train for train, _ in cv.split()]
    min_train = min(map(len, train_splits))
    min_nonconstant = min(np.count_nonzero(np.ptp(features[train], axis=0)) for train in train_splits)
    if min_nonconstant == 0:
        raise ValueError("Un complément d'apprentissage ne présente aucune variation HOG.")
    feature_counts = [size for size in (64, 256) if size < min_nonconstant] + ["all"]
    neighbor_counts = [size for size in (1, 3, 5, 9) if size <= min_train]
    models = {
        "KNN": (KNeighborsClassifier(algorithm="brute"), {
            "modele__n_neighbors": neighbor_counts,
            "modele__metric": ["euclidean", "manhattan"],
        }),
        "SVM": (SVC(kernel="linear"), {"modele__C": [0.1, 1, 10]}),
    }
    searches = {}
    for name, (model, model_grid) in models.items():
        pipeline = Pipeline([
            ("variance", VarianceThreshold()),
            ("standardisation", StandardScaler()),
            ("selection", SelectKBest(f_classif)),
            ("modele", model),
        ])
        grid = {"selection__k": feature_counts, **model_grid}
        scorer = make_scorer(f1_score, average="macro", labels=STATES, zero_division=0)
        searches[name] = GridSearchCV(pipeline, grid, scoring=scorer, cv=cv, refit=False,
                                      n_jobs=1, error_score="raise")
    return searches


def evaluation_summary(y_true, y_pred):
    """Calculer les sorties à partir de prédictions futures sur les plis de validation."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or not len(y_true):
        raise ValueError("Les deux vecteurs doivent avoir la même longueur non nulle.")
    if not set(y_true).issubset(STATES) or not set(y_pred).issubset(STATES):
        raise ValueError("Les métriques binaires demandent uniquement 0 et 1.")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=STATES, zero_division=0)
    return {
        "matrice_confusion": confusion_matrix(y_true, y_pred, labels=STATES).tolist(),
        "f1_macro": float(np.mean(f1)),
        "par_classe": {str(state): {"precision": float(precision[i]), "rappel": float(recall[i]),
                                  "f1": float(f1[i]), "effectif": int(support[i])}
                       for i, state in enumerate(STATES)},
    }


def draw_confusion(summary, model_name):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    display = ConfusionMatrixDisplay(np.asarray(summary["matrice_confusion"]),
                                     display_labels=["0 · absence", "1 · présence"])
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    display.ax_.set(title=f"{model_name} · validation exploratoire",
                    xlabel="État prédit", ylabel="État observé")
    return display.figure_


if __name__ == "__main__":
    X, y, configurations, digest = prepare_searches()
    print("Préparation à blanc", X.shape, f"SHA-256 manifeste {digest}")
    for name, search in configurations.items():
        print(name, f"{search.cv.get_n_splits()} plis", search.param_grid)
