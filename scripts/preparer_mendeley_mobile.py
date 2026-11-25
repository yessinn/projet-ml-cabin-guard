"""Préparer le corpus Mendeley exploratoire à partir d'une revue des pixels."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from adms_checks import difference_hash
from descripteurs_adms import hog_image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVUE = DATA / "processed/choix-donnees/mendeley-v2-revue-mobile.json"
OUT = DATA / "processed/mendeley-mobile"
FIGURES = ROOT / "reports/figures/mendeley-mobile"
HASH_DISTANCE = 8
FOLD_COUNT = 4
STATES = (0, 1)


def read_image(row):
    path = (ROOT / row["path"]).resolve()
    if not path.is_relative_to(DATA / "raw"):
        raise ValueError(f"Chemin hors des originaux du projet : {path}")
    with path.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != row["size"] or sha256 != row["sha256"]:
        raise ValueError(f"Intégrité invalide : {row['id']}")

    with Image.open(path) as source, ImageOps.exif_transpose(source) as image:
        width, height = image.size
        if (width, height) != (row["width"], row["height"]):
            raise ValueError(f"Dimensions incohérentes : {row['id']}")
        fingerprint = difference_hash(image)
        gray = ImageOps.contain(image.convert("L"), (256, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32) / 255
    dy, dx = np.gradient(pixels)
    measurements = {
        "megapixels": width * height / 1_000_000,
        "aspect": width / height,
        "luminance": float(pixels.mean()),
        "gradient_moyen": float(np.hypot(dx, dy).mean()),
    }
    descriptor = hog_image(path)
    if descriptor.shape != (8100,) or not np.isfinite(descriptor).all():
        raise ValueError(f"HOG invalide : {row['id']}")
    return fingerprint, measurements, descriptor


def group_similar(rows, fingerprints):
    groups = {row["id"]: row["groupe_visuel"] or row["id"] for row in rows}
    candidate_pairs = []
    for i, j in combinations(range(len(rows)), 2):
        distance = (fingerprints[i] ^ fingerprints[j]).bit_count()
        if distance > HASH_DISTANCE:
            continue
        a, b = rows[i]["id"], rows[j]["id"]
        candidate_pairs.append({"a": a, "b": b, "distance_dhash": distance})
        first, second = sorted((groups[a], groups[b]))
        if first != second:
            groups = {key: first if group == second else group for key, group in groups.items()}
    return groups, candidate_pairs


def counts(labels):
    return {str(label): int(np.sum(labels == label)) for label in STATES}


def choose_folds(labels, groups):
    """Fixer la partition avec les effectifs de labels et les groupes, sans modèle."""
    best = None
    for seed in range(128):
        cv = StratifiedGroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(labels)), labels, groups))
        if any(set(labels[train]) != set(STATES) or set(labels[valid]) != set(STATES)
               for train, valid in splits):
            continue
        by_label = np.asarray([[np.sum(labels[valid] == label) for label in STATES]
                               for _, valid in splits])
        imbalance = np.square(by_label - by_label.mean(axis=0)).sum()
        if best is None or imbalance < best[0]:
            best = imbalance, seed, splits
    return best


def explore(rows, metric_names, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = pd.DataFrame(rows)
    known = frame[frame["pli_validation"].notna()]
    summary = {}
    for name in metric_names:
        # Ce contrôle descriptif est limité aux images affectées au développement.
        q1, q3 = known[name].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        flags = known.loc[(known[name] < low) | (known[name] > high), "id"].tolist()
        summary[name] = {"q1": float(q1), "q3": float(q3), "borne_basse": low,
                         "borne_haute": high, "alertes": flags}
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    sns.countplot(data=frame, x="etat", order=["0", "1", "?"], ax=axes[0], color="#337e88")
    axes[0].set(xlabel="Téléphone visible", ylabel="Images", title="Revue du sous-ensemble")
    sns.countplot(data=known, x="pli_validation", hue="etat", hue_order=["0", "1"],
                  ax=axes[1], palette=["#526aa3", "#a9523b"])
    axes[1].set(xlabel="Pli de validation", ylabel="Images", title="Couverture par pli")
    if axes[1].legend_ is not None:
        axes[1].legend(title="Téléphone visible")
    fig.savefig(figures / "couverture.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    captions = ["Résolution (Mpx)", "Rapport largeur / hauteur", "Luminance moyenne",
                "Gradient moyen à 256 px"]
    for ax, name, caption in zip(axes.flat, metric_names, captions):
        sns.boxplot(data=known, x="etat", y=name, order=["0", "1"], ax=ax, color="#bdd9cf")
        sns.stripplot(data=known, x="etat", y=name, order=["0", "1"], ax=ax,
                      color="#27343b", size=3, jitter=False)
        ax.set(xlabel="Téléphone visible", ylabel=caption)
    fig.savefig(figures / "qualite_images.png", dpi=160)
    plt.close(fig)
    return summary


def prepare(destination, figures):
    source = json.loads(REVUE.read_text(encoding="utf-8"))
    rows = source["rows"]
    for field in ("id", "path", "sha256"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Une ligne ou un original est dupliqué : {field}")
    results = [read_image(row) for row in rows]
    groups, pairs = group_similar(rows, [result[0] for result in results])
    features = np.stack([result[2] for result in results])
    for index, (row, (_, measurements, _)) in enumerate(zip(rows, results)):
        row.update(measurements, index_hog=index, etat=str(row["mobile"]),
                   groupe=groups[row["id"]], pli_validation=None)
    eligible = [row for row in rows if row["mobile"] in STATES and not row["exclusion"]]
    labels = np.asarray([row["mobile"] for row in eligible])
    cv_groups = np.asarray([row["groupe"] for row in eligible])
    selection = choose_folds(labels, cv_groups) if len(set(cv_groups)) >= FOLD_COUNT else None
    if selection is None:
        raise ValueError("Moins de quatre validations groupées couvrent les deux états.")
    _, seed, splits = selection
    coverage = []
    for fold, (train, valid) in enumerate(splits, 1):
        assert set(cv_groups[train]).isdisjoint(cv_groups[valid])
        for i in valid:
            eligible[i]["pli_validation"] = fold
        coverage.append({"pli": fold, "validation": counts(labels[valid]),
                         "apprentissage": counts(labels[train]),
                         "groupes_validation": sorted(set(cv_groups[valid])),
                         "groupes_apprentissage": sorted(set(cv_groups[train]))})
    folds = {row["id"]: row["pli_validation"] for row in rows if row["pli_validation"] is not None}
    assert all(folds[pair["a"]] == folds[pair["b"]] for pair in pairs
               if pair["a"] in folds and pair["b"] in folds)
    destination.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    quality = explore(rows, list(results[0][1]), figures)
    manifest = {
        "corpus": source["doi"], "evaluation": "exploratoire",
        "test_independant": False, "graine_cv": seed, "plis": coverage,
        "nombre_par_etat": dict(Counter(row["etat"] for row in rows)),
        "distance_dhash_max": HASH_DISTANCE, "paires_dhash": pairs,
        "indicateurs_qualite": quality, "rows": rows,
    }
    np.save(destination / "hog.npy", features, allow_pickle=False)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sortie", type=Path, default=OUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    paths = [args.sortie.resolve(), args.figures.resolve()]
    if any(not path.is_relative_to(ROOT) or path.is_relative_to(DATA / "raw") for path in paths):
        parser.error("Les sorties doivent rester dans le projet, hors des données brutes.")
    if paths[0] == paths[1] or any(path.exists() for path in paths):
        parser.error("Indiquer deux dossiers de sortie distincts qui n'existent pas encore.")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    result = prepare(*paths)
    print(result["nombre_par_etat"], len(result["paires_dhash"]), "paires dHash")
