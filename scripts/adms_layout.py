"""Inventaire et résolution des chemins de la distribution YOLO."""

import os
import re
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCE = "https://www.kaggle.com/datasets/smartcity12/adms-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif", ".avif", ".dng", ".mpo", ".pfm"}
SPLIT_ALIASES = {
    "train": "train", "training": "train", "trainset": "train",
    "val": "val", "valid": "val", "validation": "val", "valset": "val",
    "test": "test", "testing": "test", "testset": "test",
}
SPLITS = ("train", "val", "test")
IMAGE_DIRS = {"images", "image", "imgs", "jpegimages"}
LABEL_DIRS = ("labels", "annotations", "label", "annotation")
TAXONOMY_FILES = {"classes.txt", "obj.names", "classes.names"}


def relative(path, base=ROOT):
    return path.relative_to(base).as_posix()


def local_path(path, within=ROOT):
    """Resolve a path and reject access through links outside its scope."""
    path = path.resolve()
    if not path.is_relative_to(within.resolve()):
        raise ValueError("chemin hors du périmètre autorisé")
    return path


def issue(issues, level, code, path=None, detail="", line=None):
    issues.append({"niveau": level, "code": code, "fichier": relative(path) if path else "",
                   "ligne": line, "detail": str(detail)})


def walk(root, issues):
    files, directories = [], []

    def report_error(error):
        issue(issues, "bloquant", "inventaire_incomplet", Path(error.filename), error.strerror)

    for folder, subdirs, names in os.walk(root, onerror=report_error):
        directory = Path(folder)
        directories.append(directory)
        for name in sorted(subdirs + names):
            path = directory / name
            if path.is_symlink() or path.is_junction():
                issue(issues, "avertissement", "lien_ignore", path)
                if name in subdirs:
                    subdirs.remove(name)
            elif name in names:
                files.append(path)
        subdirs.sort()
    return files, directories


def scope_for(raw_dirs):
    candidates = [path for path in raw_dirs
                  if re.sub(r"[^a-z0-9]", "", path.name.casefold()) == "yoloannotateddataset"]
    if len(candidates) > 1:
        choices = ", ".join(relative(path) for path in candidates)
        raise ValueError(f"Plusieurs dossiers YOLO. Préciser --dataset parmi {choices}")
    return candidates[0] if candidates else raw_dirs[0]


class DatasetLoader(yaml.SafeLoader):
    """Reject duplicate keys that would hide a split or a class."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, int, float, bool, type(None))) or key in keys:
                raise yaml.YAMLError("clé YAML en double ou non scalaire")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def read_config(scope, files, explicit, issues):
    records = {}
    for path in [explicit] if explicit else files:
        if path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        standard_name = path.stem.lower() in {"data", "dataset"}
        try:
            data = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=DatasetLoader)
        except (OSError, ValueError, yaml.YAMLError) as error:
            if explicit or standard_name:
                issue(issues, "bloquant", "yaml_illisible", path, str(error).splitlines()[0])
            continue
        if isinstance(data, dict) and (set(data) & (set(SPLIT_ALIASES) | {"names", "nc"})):
            records[path] = data
        elif explicit or standard_name:
            issue(issues, "bloquant", "yaml_non_yolo", path)
    if not records:
        issue(issues, "avertissement", "yaml_absent", scope)
        return None, {}, []

    def rank(path):
        return len(path.relative_to(scope).parts), path.stem.lower() not in {"data", "dataset"}

    best_rank = min(map(rank, records))
    preferred = [path for path in records if rank(path) == best_rank]
    if len(preferred) > 1:
        issue(issues, "bloquant", "yaml_ambigu", scope,
              "Préciser --config parmi " + ", ".join(relative(path) for path in preferred))
        return None, {}, list(records)
    path = preferred[0]
    if len(records) > 1:
        issue(issues, "avertissement", "plusieurs_yaml_yolo", path,
              f"Configuration retenue parmi {len(records)} candidates.")
    return path, records[path], list(records)


def read_classes(config, config_path, files, issues):
    names = config.get("names")
    source = config_path
    if names is None:
        candidates = [path for path in files if path.name.lower() in TAXONOMY_FILES]
        if len(candidates) == 1:
            source = candidates[0]
            try:
                names = source.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as error:
                issue(issues, "bloquant", "classes_illisibles", source, error)
        elif len(candidates) > 1:
            issue(issues, "bloquant", "sources_classes_ambigues", detail=", ".join(map(relative, candidates)))
    if isinstance(names, list):
        names = dict(enumerate(names))
    classes = {}
    if isinstance(names, dict):
        for key, name in names.items():
            if not re.fullmatch(r"[0-9]{1,12}", str(key)) or not isinstance(name, str) or not name.strip():
                issue(issues, "bloquant", "definition_classe_invalide", source, repr((key, name)))
                continue
            class_id = int(key)
            if class_id in classes:
                issue(issues, "bloquant", "indice_classe_ambigu", source, str(key))
                continue
            classes[class_id] = name.strip()
    if not classes:
        issue(issues, "bloquant", "classes_non_verifiees", source)
    elif sorted(classes) != list(range(len(classes))):
        issue(issues, "bloquant", "indices_classes_non_contigus", source)
    if len(set(classes.values())) != len(classes):
        issue(issues, "anomalie", "noms_classes_repetes", source)
    nc = config.get("nc")
    if nc is not None and (type(nc) is not int or nc != len(classes)):
        issue(issues, "bloquant", "nc_incoherent", config_path, f"nc={nc}, classes lues={len(classes)}")
    return classes, source if classes else None


def path_candidates(value, bases, scope):
    path = Path(value.replace("\\", "/"))
    candidates = []
    for base in bases:
        try:
            candidate = local_path(base / path, scope)
        except (OSError, ValueError):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def declared_paths(config):
    """Keep declared values serializable even for malformed YAML entries."""
    def plain(value):
        if isinstance(value, list):
            return [plain(item) if not isinstance(item, list) else repr(item) for item in value]
        return value if value is None or type(value) in {str, int, bool} else repr(value)

    return {key: plain(config[key]) for key in config if key in {*SPLIT_ALIASES, "path", "nc"}}


def read_splits(scope, images, files, directories, config_path, config, issues):
    memberships = defaultdict(set)
    declared, lists, resolved = set(), set(), defaultdict(list)
    root_value = config.get("path")
    base = config_path.parent if config_path else scope
    file_set, dir_set = set(files), set(directories)
    images_by_dir = defaultdict(list)
    for image in images:
        for parent in image.parents:
            if parent not in dir_set:
                break
            images_by_dir[parent].append(image)
    if root_value is not None:
        roots = path_candidates(root_value, [base, scope, ROOT], scope) if isinstance(root_value, str) else []
        matched = next((path for path in roots if path in dir_set), None)
        if matched is None:
            issue(issues, "avertissement", "racine_yaml_introuvable", config_path,
                  f"path={root_value!r}. Résolution depuis le dossier YAML.")
        else:
            base = matched

    for key, value in config.items():
        if key not in SPLIT_ALIASES or value is None:
            continue
        split = SPLIT_ALIASES[key]
        declared.add(split)
        entries = value if isinstance(value, list) else [value]
        if not entries:
            issue(issues, "anomalie", "split_declare_vide", config_path, str(key))
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                issue(issues, "bloquant", "chemin_split_invalide", config_path, f"{key}={entry!r}")
                continue
            candidates = path_candidates(entry, [base], scope)
            target = next((p for p in candidates if p in file_set or p in dir_set), None)
            if target is None:
                issue(issues, "bloquant", "chemin_split_introuvable", config_path, f"{key}={entry!r}")
                continue
            resolved[split].append(relative(target))
            found = images_by_dir.get(target, [])
            if target in images:
                found = [target]
            elif target.suffix.lower() == ".txt" and target in file_set:
                lists.add(target)
                found = []
                found_set = set()
                try:
                    entries_in_list = target.read_text(encoding="utf-8-sig").splitlines()
                except (OSError, UnicodeError) as error:
                    issue(issues, "bloquant", "liste_images_illisible", target, error)
                    continue
                for line_no, row in enumerate(entries_in_list, 1):
                    row = row.strip()
                    if not row or row.startswith("#"):
                        continue
                    bases = [target.parent] if row.startswith(("./", ".\\")) else [base, target.parent]
                    matches = [p for p in path_candidates(row, bases, scope) if p in images]
                    if len(matches) != 1:
                        issue(issues, "bloquant", "image_liste_introuvable_ou_ambigue", target, row, line_no)
                    else:
                        if matches[0] in found_set:
                            issue(issues, "avertissement", "image_repetee_dans_liste", target, row, line_no)
                        found.append(matches[0])
                        found_set.add(matches[0])
            if not found:
                issue(issues, "anomalie", "source_split_sans_image", target, split)
            for path in found:
                memberships[path].add(split)

    for path in images:
        if not memberships[path]:
            guessed = next((SPLIT_ALIASES[part.lower()] for part in reversed(path.relative_to(scope).parts[:-1])
                            if part.lower() in SPLIT_ALIASES), None)
            if guessed and guessed not in declared:
                memberships[path].add(guessed)
        if len(memberships[path]) > 1:
            issue(issues, "bloquant", "meme_fichier_plusieurs_splits", path, ", ".join(sorted(memberships[path])))
    for split in SPLITS:
        if not any(split in membership for membership in memberships.values()):
            level = "avertissement" if split == "test" else "bloquant"
            issue(issues, level, "split_sans_image", scope, split)
    outside = sum(not membership for membership in memberships.values())
    if outside:
        issue(issues, "avertissement", "images_hors_split", scope, outside)
    return memberships, lists, dict(resolved), sorted(declared)


def matching_labels(scope, files, images, excluded, issues):
    texts = {path for path in files if path.suffix.lower() == ".txt" and path not in excluded
             and path.name.lower() not in TAXONOMY_FILES and not path.stem.lower().startswith(("readme", "license", "licence"))}
    image_dirs = {image.parent for image in images}
    matched = {}
    used = defaultdict(list)
    annotation_dirs = set(LABEL_DIRS) | IMAGE_DIRS | set(SPLIT_ALIASES)
    recognized = {path for path in texts if path.parent in image_dirs or
                  annotation_dirs.intersection(part.lower() for part in path.relative_to(scope).parts[:-1])}

    for image in sorted(images):
        parts = image.relative_to(scope).parts
        filename = image.with_suffix(".txt").name
        options = {image.with_suffix(".txt")}
        for index, part in enumerate(parts[:-1]):
            if part.lower() in IMAGE_DIRS:
                options.update(scope.joinpath(*parts[:index], label_dir, *parts[index + 1:-1], filename)
                               for label_dir in LABEL_DIRS)
        options.update(image.parent / label_dir / filename for label_dir in LABEL_DIRS)
        candidates = sorted(options & texts)
        if len(candidates) == 1:
            matched[image] = candidates[0]
        elif candidates:
            issue(issues, "anomalie", "annotation_ambigue", image, ", ".join(map(relative, candidates)))
            matched[image] = None
        for path in candidates:
            used[path].append(image)
        recognized.update(candidates)

    for label, paths in used.items():
        if len(paths) > 1:
            issue(issues, "anomalie", "annotation_partagee", label, ", ".join(map(relative, paths)))
            for image in paths:
                matched[image] = None
    for label in sorted(recognized - used.keys()):
        issue(issues, "anomalie", "annotation_orpheline", label)
    return matched, recognized
