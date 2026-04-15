from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, TypeVar

T = TypeVar("T")

DEFAULT_MANIFEST_NAME = "video_split_manifest.json"

FRAME_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

_STEM_RE = re.compile(r"^(.+)_frame_\d+$")


def video_stem_from_path(path: Path) -> str:
    m = _STEM_RE.match(path.stem)
    return m.group(1) if m else path.stem


def iter_frame_image_files(frames_root: Path):
    """Yield frame image paths under ``frames_root/<class>/``."""
    if not frames_root.is_dir():
        return
    for class_dir in sorted(frames_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in FRAME_IMAGE_EXTENSIONS:
                yield p


def count_frame_images(frames_root: Path) -> int:
    """Number of frame image files (recurses only one level: class folders)."""
    return sum(1 for _ in iter_frame_image_files(frames_root))


def frames_directory_has_images(frames_root: Path) -> bool:
    """True if there is at least one frame image (used to decide download vs skip)."""
    return count_frame_images(frames_root) > 0


def discover_videos_by_class(frames_root: Path) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    if not frames_root.is_dir():
        return out
    for class_dir in sorted(frames_root.iterdir()):
        if not class_dir.is_dir():
            continue
        stems: Set[str] = set()
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in FRAME_IMAGE_EXTENSIONS:
                stems.add(video_stem_from_path(p))
        if stems:
            out[class_dir.name] = stems
    return out


def _stratify_ok(labels: List[int]) -> bool:
    cnt = Counter(labels)
    return len(cnt) >= 2 and all(v >= 2 for v in cnt.values())


def _n_test_items(n: int, test_fraction: float) -> int:
    if n <= 0:
        return 0
    nt = max(0, min(int(round(n * test_fraction)), n))
    if n >= 2 and nt >= n:
        nt = n - 1
    return nt


def _split_random(items: List[T], test_fraction: float, seed: int) -> Tuple[List[T], List[T]]:
    rng = random.Random(seed)
    buf = list(items)
    rng.shuffle(buf)
    n_test = _n_test_items(len(buf), test_fraction)
    return buf[n_test:], buf[:n_test]


def _split_stratified(
    items: List[T], labels: List[int], test_fraction: float, seed: int,
) -> Tuple[List[T], List[T]]:
    rng = random.Random(seed)
    train_out: List[T] = []
    test_out: List[T] = []
    by_label: Dict[int, List[T]] = defaultdict(list)
    for it, lab in zip(items, labels):
        by_label[lab].append(it)
    for lab in sorted(by_label.keys()):
        group = list(by_label[lab])
        rng.shuffle(group)
        n_test = _n_test_items(len(group), test_fraction)
        test_out.extend(group[:n_test])
        train_out.extend(group[n_test:])
    return train_out, test_out


def build_train_val_test_keys(
    by_class: Dict[str, Set[str]],
    train_ratio: float, val_ratio: float, test_ratio: float, seed: int,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-3:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0 "
            f"(got {train_ratio + val_ratio + test_ratio:.6f})"
        )

    keys: List[Tuple[str, str]] = []
    labels: List[int] = []
    class_list = sorted(by_class.keys())
    class_to_i = {c: i for i, c in enumerate(class_list)}
    for c in class_list:
        for stem in sorted(by_class[c]):
            keys.append((c, stem))
            labels.append(class_to_i[c])

    if len(keys) < 3:
        raise RuntimeError(f"Need at least 3 distinct source videos; found {len(keys)}.")

    strat = _stratify_ok(labels)
    if strat:
        tv_keys, te_keys = _split_stratified(keys, labels, test_ratio, seed)
    else:
        tv_keys, te_keys = _split_random(keys, test_ratio, seed)

    tv_labels = [class_to_i[k[0]] for k in tv_keys]
    rel_val = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0.15
    if _stratify_ok(tv_labels):
        tr_keys, va_keys = _split_stratified(tv_keys, tv_labels, rel_val, seed + 1)
    else:
        tr_keys, va_keys = _split_random(tv_keys, rel_val, seed + 1)

    return set(tr_keys), set(va_keys), set(te_keys)


def manifest_path_for_frames_dir(frames_root: Path) -> Path:
    return frames_root.resolve().parent / DEFAULT_MANIFEST_NAME


def write_split_manifest(
    frames_root: Path, manifest_path: Path,
    train_ratio: float, val_ratio: float, test_ratio: float, seed: int,
) -> Dict:
    by_class = discover_videos_by_class(frames_root)
    if not by_class:
        raise RuntimeError(f"No frame images found under {frames_root}")

    train_k, val_k, test_k = build_train_val_test_keys(
        by_class, train_ratio, val_ratio, test_ratio, seed
    )

    def _sorted_keys(s: Set[Tuple[str, str]]) -> List[Dict[str, str]]:
        return [{"class": c, "stem": t} for c, t in sorted(s)]

    doc = {
        "version": 1,
        "frames_root": str(frames_root.resolve()),
        "seed": seed,
        "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio,
        "splits": {
            "train": _sorted_keys(train_k),
            "val": _sorted_keys(val_k),
            "test": _sorted_keys(test_k),
        },
        "counts": {
            "train_videos": len(train_k),
            "val_videos": len(val_k),
            "test_videos": len(test_k),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc


def load_split_manifest(manifest_path: Path) -> Dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def keys_from_manifest_split(manifest: Dict, split_name: str) -> Set[Tuple[str, str]]:
    return {(e["class"], e["stem"]) for e in manifest["splits"][split_name]}


def _all_keys_from_manifest(manifest: Dict) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for s in ("train", "val", "test"):
        out |= keys_from_manifest_split(manifest, s)
    return out


def sync_manifest_with_frames_dir(
    frames_root: Path,
    manifest_path: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> bool:
    """Rewrite the manifest if frame files on disk do not match manifest entries.

    Returns True if the manifest was rewritten.
    """
    by_class = discover_videos_by_class(frames_root)
    discovered: Set[Tuple[str, str]] = {(c, stem) for c, stems in by_class.items() for stem in stems}

    try:
        manifest = load_split_manifest(manifest_path)
        known = _all_keys_from_manifest(manifest)
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        print(f"[Split] Manifest missing or unreadable; rebuilding: {manifest_path}")
        write_split_manifest(frames_root, manifest_path, train_ratio, val_ratio, test_ratio, seed)
        return True

    if discovered == known:
        return False

    n_new = len(discovered - known)
    n_stale = len(known - discovered)
    print(
        f"[Split] Manifest out of sync with frames on disk "
        f"({n_new} new video(s), {n_stale} only in manifest); rebuilding…"
    )
    write_split_manifest(frames_root, manifest_path, train_ratio, val_ratio, test_ratio, seed)
    return True


def ensure_split_manifest(
    frames_root: Path, *,
    train_ratio: float, val_ratio: float, test_ratio: float, seed: int,
    manifest_path: Path | None = None, regenerate: bool = False,
) -> Path:
    path = manifest_path or manifest_path_for_frames_dir(frames_root)
    if regenerate:
        write_split_manifest(frames_root, path, train_ratio, val_ratio, test_ratio, seed)
        print(f"[Split] Wrote train/val/test manifest: {path}")
        return path
    if not path.exists():
        write_split_manifest(frames_root, path, train_ratio, val_ratio, test_ratio, seed)
        print(f"[Split] Wrote train/val/test manifest: {path}")
        return path
    if sync_manifest_with_frames_dir(frames_root, path, train_ratio, val_ratio, test_ratio, seed):
        print(f"[Split] Wrote train/val/test manifest: {path}")
    return path


def warn_if_new_videos_not_in_manifest(frames_root: Path, manifest: Dict) -> None:
    by_class = discover_videos_by_class(frames_root)
    known: Set[Tuple[str, str]] = set()
    for s in ("train", "val", "test"):
        known |= keys_from_manifest_split(manifest, s)

    discovered = {(c, stem) for c, stems in by_class.items() for stem in stems}
    missing = discovered - known
    if missing:
        print(
            f"\n[WARN] {len(missing)} source video(s) on disk are not in "
            f"the split manifest. Re-run with --regenerate-splits to include them."
        )
