"""Reproduire l'expérience Mendeley ou évaluer un corpus indépendant annoté."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import clone
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold, cross_val_predict

from adms_checks import difference_hash
from descripteurs_adms import hog_image
from pipelines_mendeley_mobile import build_searches, evaluation_summary, prepare_searches

FIELDS = {"chemin_relatif", "sha256", "participant", "session", "sequence",
          "habitacle", "ensemble", "mobile", "motif"}
STATES = {0, 1}
SEED = 2026


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_independent(manifest):
    root = manifest.parent.resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if set(reader.fieldnames or ()) != FIELDS or len(reader.fieldnames or ()) != len(FIELDS):
            raise ValueError("Colonnes du manifeste invalides.")
        rows = list(reader)
    partitions = {"developpement": [], "test": []}
    owners = {}
    hashes = set()
    for number, row in enumerate(rows, 2):
        if None in row or any(not value or not value.strip() for value in row.values()):
            raise ValueError(f"Ligne {number} incomplète.")
        split = row["ensemble"]
        label = row["mobile"]
        if split not in partitions or label not in {"0", "1", "?"}:
            raise ValueError(f"Ligne {number} : ensemble ou classe invalide.")
        for field in ("participant", "session", "sequence"):
            key = field, row[field]
            if owners.setdefault(key, split) != split:
                raise ValueError(f"{field} partagé entre développement et test : {row[field]}")
        path = (root / row["chemin_relatif"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Ligne {number} : chemin hors du corpus.")
        digest = file_digest(path)
        if digest != row["sha256"].lower() or digest in hashes:
            raise ValueError(f"Ligne {number} : empreinte invalide ou doublon exact.")
        hashes.add(digest)
        with Image.open(path) as source, ImageOps.exif_transpose(source) as image:
            fingerprint = difference_hash(image)
        partitions[split].append((path, row["participant"], label, fingerprint))

    return partitions


def as_arrays(rows):
    known = [row for row in rows if row[2] != "?"]
    if {int(row[2]) for row in known} != STATES:
        raise ValueError("Les deux classes doivent être présentes dans chaque ensemble.")
    return (np.stack([hog_image(row[0]) for row in known]),
            np.array([int(row[2]) for row in known]),
            np.array([row[1] for row in known]))


def train_results(features, labels, searches, holdout=None):
    results = {}
    for name, search in searches.items():
        search.fit(features, labels)
        tuned = clone(search.estimator).set_params(**search.best_params_)
        best = search.best_index_
        scores = [float(search.cv_results_[f"split{fold}_test_score"][best])
                  for fold in range(search.cv.get_n_splits())]
        validation = cross_val_predict(tuned, features, labels, cv=search.cv, n_jobs=1)
        for (_, valid), score in zip(search.cv.split(), scores, strict=True):
            if not np.isclose(evaluation_summary(labels[valid], validation[valid])["f1_macro"], score):
                raise RuntimeError(f"Les prédictions et le score de validation divergent pour {name}.")
        result = {
            "parametres": search.best_params_,
            "candidats": len(search.cv_results_["params"]),
            "f1_macro_par_pli": scores,
            "f1_macro_moyen": float(search.best_score_),
            "ecart_type": float(np.std(scores)),
            "validation_interne": evaluation_summary(labels, validation),
        }
        if holdout is not None:
            test_features, test_labels = holdout
            tuned.fit(features, labels)
            result["test_independant"] = evaluation_summary(test_labels, tuned.predict(test_features))
        results[name] = result
    return results


def independent(args):
    partitions = load_independent(args.source)
    pairs = [
        [str(dev[0].relative_to(args.source.parent.resolve())),
         str(test[0].relative_to(args.source.parent.resolve()))]
        for dev in partitions["developpement"] for test in partitions["test"]
        if (dev[3] ^ test[3]).bit_count() <= 8
    ]
    coverage = {
        split: {"participants": len({r[1] for r in rows if r[2] != "?"}),
                "classes": dict(Counter(r[2] for r in rows))}
        for split, rows in partitions.items()
    }
    result = {"evaluation": "independante", "manifeste_sha256": file_digest(args.source),
              "couverture": coverage, "paires_visuellement_proches": pairs}
    if pairs:
        raise ValueError(f"{len(pairs)} paires proches traversent la séparation. "
                         f"Examiner les rapprochements avant le test. Exemples : {pairs[:5]}")
    if coverage["developpement"]["participants"] < 4:
        raise ValueError("Quatre participants annotés sont nécessaires pour la CV de développement.")
    features, labels, participants = as_arrays(partitions["developpement"])
    split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    assignment = np.full(len(labels), -1)
    for fold, (train, valid) in enumerate(split.split(features, labels, participants)):
        if set(labels[train]) != STATES or set(labels[valid]) != STATES:
            raise ValueError(f"Les deux classes doivent couvrir le pli {fold + 1} et son complément.")
        assignment[valid] = fold
    result["participants_par_pli"] = [sorted(set(participants[assignment == fold]))
                                      for fold in range(4)]
    cv = PredefinedSplit(assignment)
    if not args.controler:
        test_features, test_labels, _ = as_arrays(partitions["test"])
        result["modeles"] = train_results(features, labels, build_searches(features, cv),
                                          (test_features, test_labels))
    elif set(coverage["test"]["classes"]) - {"?"} != {"0", "1"}:
        raise ValueError("Les deux classes doivent être présentes dans le test.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    for name in ("mendeley", "independant"):
        command = subcommands.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("--sortie", type=Path, required=True)
        if name == "independant":
            command.add_argument("--controler", action="store_true",
                                 help="Vérifier le manifeste et les plis sans ajuster les modèles.")
    args = parser.parse_args()
    if args.sortie.exists():
        parser.error(f"La sortie existe déjà : {args.sortie}")
    try:
        if args.mode == "mendeley":
            features, labels, searches, digest = prepare_searches(args.source)
            result = {"evaluation": "exploratoire", "images": len(labels),
                      "manifeste_sha256": digest, "hog_sha256": file_digest(args.source / "hog.npy"),
                      "modeles": train_results(features, labels, searches)}
        else:
            result = independent(args)
        with args.sortie.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2)
            target.write("\n")
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    print(args.sortie)


if __name__ == "__main__":
    main()

