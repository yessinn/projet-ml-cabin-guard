"""Contrôles d'images, d'annotations et de séparation des données."""

import csv
import hashlib
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from adms_layout import issue, path_candidates, relative

EPSILON = 1e-6
VISUAL_DISTANCE = 4
GROUP_FIELDS = {
    "driver_id": ("driver_id", "driver", "conducteur_id", "subject_id", "participant_id"),
    "session_id": ("session_id", "session", "recording_id"),
    "video_id": ("video_id", "video", "video_file", "video_path", "source_video"),
}
IMAGE_FIELDS = ("image", "image_path", "image_file", "filename", "file_name", "frame_path", "file")


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float


@dataclass
class Labels:
    status: str
    boxes: list
    class_ids: set
    invalid_lines: int = 0
    repeated_lines: int = 0


@dataclass
class Frame:
    path: Path
    splits: set
    label: Path | None
    labels: Labels
    sha256: str = ""
    dhash: int | None = None
    format: str = ""
    width: int = 0
    height: int = 0

    @property
    def valid(self):
        return self.width > 0 and self.height > 0

    def eligible(self, taxonomy_verified):
        return (taxonomy_verified and self.valid and bool(self.sha256) and len(self.splits) == 1
                and self.labels.status in {"vide", "valide"})


def parse_box(line, classes):
    columns = line.split()
    class_id = int(columns[0]) if columns and re.fullmatch(r"[0-9]{1,12}", columns[0]) else None
    if len(columns) != 5:
        return None, class_id, "colonnes_yolo", f"{len(columns)} champs au lieu de 5"
    if class_id is None or (classes and class_id not in classes):
        return None, class_id, "indice_yolo_invalide", columns[0]
    try:
        x, y, width, height = map(float, columns[1:])
    except ValueError:
        return None, class_id, "coordonnees_non_numeriques", " ".join(columns[1:])
    values = x, y, width, height
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        return None, class_id, "coordonnees_hors_intervalle", values
    if width <= 0 or height <= 0:
        return None, class_id, "surface_nulle", values
    if min(x - width / 2, y - height / 2) < -EPSILON or max(x + width / 2, y + height / 2) > 1 + EPSILON:
        return None, class_id, "boite_hors_image", values
    return Box(class_id, x, y, width, height), class_id, None, ""


def read_label(path, classes, issues):
    try:
        rows = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        issue(issues, "anomalie", "annotation_illisible", path, error)
        return Labels("illisible", [], set())
    boxes, observed, seen = [], set(), set()
    errors = repetitions = 0
    for line_no, line in enumerate(rows, 1):
        if not line.strip():
            continue
        box, class_id, code, detail = parse_box(line, classes)
        if class_id is not None:
            observed.add(class_id)
        if code:
            errors += 1
            issue(issues, "anomalie", code, path, detail, line_no)
        else:
            if box in seen:
                repetitions += 1
                issue(issues, "anomalie", "boite_repetee", path, line=line_no)
            else:
                boxes.append(box)
                seen.add(box)
    status = "invalide" if errors or repetitions else "valide" if boxes else "vide"
    return Labels(status, boxes, observed, errors, repetitions)


def difference_hash(image):
    with image.convert("L") as gray, gray.resize((9, 8), Image.Resampling.LANCZOS) as small:
        pixels = small.tobytes()
    fingerprint = 0
    for row in range(8):
        for column in range(8):
            index = row * 9 + column
            fingerprint = (fingerprint << 1) | (pixels[index] > pixels[index + 1])
    return fingerprint


def inspect_frame(path, splits, label, labels, issues, visual=True):
    frame = Frame(path, splits, label, labels)
    try:
        with path.open("rb") as source:
            frame.sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as error:
        issue(issues, "anomalie", "hachage_image_impossible", path, error)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                frame.format = source.format
                source.verify()
            with Image.open(path) as source, ImageOps.exif_transpose(source) as image:
                image.load()
                frame.width, frame.height = image.size
                if getattr(source, "n_frames", 1) > 1:
                    issue(issues, "avertissement", "image_multi_images", path, "Première image contrôlée")
                if visual:
                    frame.dhash = difference_hash(image)
    except (OSError, ValueError, SyntaxError, EOFError, TypeError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        frame.width = frame.height = 0
        issue(issues, "anomalie", "image_illisible_par_pillow", path, error)
    return frame


def exact_duplicates(frames, issues):
    buckets = defaultdict(list)
    for frame in frames:
        if frame.sha256:
            buckets[frame.sha256].append(frame)
    duplicates = []
    for digest, members in buckets.items():
        if len(members) < 2:
            continue
        splits = {split for frame in members for split in frame.splits}
        label_signatures = {(frame.labels.status, frozenset(frame.labels.boxes)) for frame in members}
        inter_split = len(splits) > 1 and sum(bool(frame.splits) for frame in members) > 1
        different = len(label_signatures) > 1
        if inter_split:
            issue(issues, "bloquant", "doublon_exact_inter_split", members[0].path, digest)
        if different:
            issue(issues, "anomalie", "doublon_exact_annotations_differentes", members[0].path, digest)
        duplicates.append((digest, members, splits, inter_split, different))
    return duplicates


def hash_bands(value):
    # At most four differing bits guarantee one shared band among these five.
    shift = 0
    for size in (13, 13, 13, 13, 12):
        yield shift, (value >> shift) & ((1 << size) - 1)
        shift += size


def visual_signatures(frames):
    signatures = defaultdict(list)
    for frame in frames:
        if frame.dhash is not None:
            signatures[frame.dhash].append(frame)
    return signatures


def visual_pairs(signatures):
    """Yield one candidate per fingerprint pair, with a witness in each split."""
    by_band = defaultdict(list)
    representatives = {}
    for fingerprint, frames in signatures.items():
        by_split = defaultdict(dict)
        for frame in frames:
            for split in frame.splits:
                if len(by_split[split]) < 2:
                    by_split[split].setdefault(frame.sha256 or relative(frame.path), frame)
        representatives[fingerprint] = [(split, digest, frame) for split, members in sorted(by_split.items())
                                        for digest, frame in members.items()]

    for fingerprint, members in representatives.items():
        related = {fingerprint}
        for band in hash_bands(fingerprint):
            related.update(by_band[band])
            by_band[band].append(fingerprint)
        for other in sorted(related):
            distance = (fingerprint ^ other).bit_count()
            if distance > VISUAL_DISTANCE:
                continue
            pair = next(((first, second, left, right)
                         for left, first_sha, first in members
                         for right, second_sha, second in representatives[other]
                         if left != right and first_sha != second_sha), None)
            if pair:
                yield fingerprint, other, distance, *pair


def group_overlaps(scope, csv_paths, frames, issues, explicit):
    frame_paths = {frame.path: frame for frame in frames}
    names = defaultdict(list)
    for path in frame_paths:
        names[path.name.casefold()].append(path)
    grouped = defaultdict(set)
    coverage = defaultdict(set)
    sources = []
    for source in [explicit] if explicit else csv_paths:
        try:
            with source.open(encoding="utf-8-sig", newline="") as stream:
                head = stream.read(8192)
                stream.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(head, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(stream, dialect=dialect)
                headers = {name.strip().lower().replace(" ", "_"): name for name in reader.fieldnames or []}
                image_key = next((headers[key] for key in IMAGE_FIELDS if key in headers), None)
                group_keys = {field: next(headers[key] for key in aliases if key in headers)
                              for field, aliases in GROUP_FIELDS.items() if any(key in headers for key in aliases)}
                if not image_key or not group_keys:
                    if explicit:
                        issue(issues, "bloquant", "colonnes_groupes_absentes", source)
                    continue
                sources.append(relative(source))
                for row in reader:
                    image_key_value = (row.get(image_key) or "").strip()
                    if not image_key_value:
                        continue
                    matches = [path for path in path_candidates(image_key_value, [scope, source.parent], scope)
                               if path in frame_paths]
                    if not matches and "/" not in image_key_value and "\\" not in image_key_value:
                        matches = names[image_key_value.casefold()]
                    if len(matches) != 1:
                        issue(issues, "avertissement", "image_metadonnees_introuvable_ou_ambigue",
                              source, image_key_value, reader.line_num)
                        continue
                    path = matches[0]
                    for field, key in group_keys.items():
                        value = (row.get(key) or "").strip()
                        if value.casefold() not in {"", "unknown", "none", "null", "nan", "n/a", "non vérifié"}:
                            grouped[field, value].add(path)
                            coverage[field].add(path)
        except (OSError, UnicodeError, csv.Error) as error:
            issue(issues, "bloquant" if explicit else "avertissement", "metadonnees_csv_illisibles", source, error)
    overlaps = []
    for (field, value), paths in sorted(grouped.items()):
        splits = {split for path in paths for split in frame_paths[path].splits}
        if len(splits) > 1:
            overlaps.append((field, value, sorted(paths), sorted(splits)))
    if overlaps:
        issue(issues, "anomalie", "groupes_inter_split", detail=f"{len(overlaps)} identifiants à contrôler")
    return sources, {field: len(paths) for field, paths in coverage.items()}, overlaps
