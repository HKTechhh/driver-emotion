"""KMU-FED subject-level split (separate from src/data.py: a different split unit -
subjects, not files - a different class count, and a flat, filename-encoded layout instead
of FER2013's one-folder-per-class structure, so contorting src/data.py's pipeline to fit
it would do more harm than good).

KMU-FED ships as 1106 images sitting flat in one folder (no class subfolders at all), named
like "01_AN_mr_001.jpg": <numeric subject alias>_<class code>_<subject initials>_<index>.

- The numeric alias (field 0) has a real bug in the shipped filenames: subject "mr" appears
  as both "01" (120 files) and a malformed "1" (6 files, a dropped leading zero) - confirmed
  by cross-referencing against field 2 (see docs/experiment_log.md for the full inspection
  trail). `parse_filename` always int()-casts this field, which fixes it: int("01") ==
  int("1") == 1, so both groups land on the same subject.
- Field 1 is the class code: AN, DI, FE, HA, SA, SU - no "neutral".
- Field 2 is the subject's initials (e.g. "mr", "s01") - a clean, already-unique 12-value
  identifier confirmed to be a 1:1 alias of the (corrected) numeric field.
- Field 3 is a per-subject-per-class running index, unused here.

Only 7 of the 12 subjects contributed any disgust images at all (5 have zero), so
`select_subject_split` constrains its seeded random draw to keep at least one disgust-covered
subject in both val and test - an unconstrained split could easily leave disgust with no
val/test examples, making its macro-F1 undefined for that split.

Run as a module to (re)compute and save the split:
    python -m src.kmu_fed_data
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from config import KMU_FED, RESULTS_DIR, SEED
from src.utils import ensure_dirs, set_seed

# Subject initials, for human-readable reporting only - the numeric ID (post int() fix) is
# what every split and every downstream script actually keys on.
SUBJECT_INITIALS: Dict[int, str] = {
    1: "mr", 2: "s01", 3: "s02", 4: "s03", 5: "uj", 6: "dy",
    7: "sw", 8: "nh", 9: "sj", 10: "ko", 11: "ej", 12: "gu",
}

# This project's own verified inspection of the real dataset (docs/experiment_log.md has the
# full trail: directory listing, filename-field breakdown, and the field2-vs-field0 cross-tab
# that exposed and confirmed the "01"/"1" bug). Used as a fallback wherever the raw KMU-FED
# files aren't present locally - the same pattern app_collect.py's fer2013_train_class_counts()
# uses for FER2013, since the raw images only ever live on Kaggle, not on every machine that
# imports this module.
PUBLISHED_SUBJECT_CLASS_COUNTS: Dict[int, Dict[str, int]] = {
    1: {"AN": 26, "DI": 20, "FE": 20, "HA": 20, "SA": 20, "SU": 20},
    2: {"AN": 10, "DI": 0, "FE": 20, "HA": 10, "SA": 0, "SU": 20},
    3: {"AN": 10, "DI": 0, "FE": 20, "HA": 20, "SA": 20, "SU": 20},
    4: {"AN": 0, "DI": 0, "FE": 20, "HA": 20, "SA": 10, "SU": 10},
    5: {"AN": 20, "DI": 20, "FE": 20, "HA": 20, "SA": 20, "SU": 20},
    6: {"AN": 20, "DI": 20, "FE": 20, "HA": 20, "SA": 20, "SU": 20},
    7: {"AN": 20, "DI": 0, "FE": 20, "HA": 20, "SA": 20, "SU": 20},
    8: {"AN": 20, "DI": 20, "FE": 10, "HA": 20, "SA": 20, "SU": 20},
    9: {"AN": 20, "DI": 0, "FE": 10, "HA": 20, "SA": 10, "SU": 10},
    10: {"AN": 10, "DI": 10, "FE": 20, "HA": 20, "SA": 0, "SU": 20},
    11: {"AN": 20, "DI": 20, "FE": 20, "HA": 0, "SA": 20, "SU": 20},
    12: {"AN": 20, "DI": 10, "FE": 0, "HA": 20, "SA": 20, "SU": 0},
}


class SubjectSplit(NamedTuple):
    """The result of `select_subject_split`: the seed that satisfied the constraint, and the
    resulting subject-ID assignment (each a sorted list of ints, disjoint, covering everyone).
    """
    seed_used: int
    train: List[int]
    val: List[int]
    test: List[int]


def parse_filename(filename: str) -> Tuple[int, str, str, str]:
    """Parse one KMU-FED filename ("01_AN_mr_001.jpg") into (subject_id, class_code,
    initials, index). `subject_id` is int()-cast, which is what fixes the "01" vs "1" bug -
    both parse to the same integer. Every one of the real dataset's 1106 files matches this
    exact 4-field, underscore-separated shape (verified by notebooks/kmu_fed_inspect_cell.py).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Filename {filename!r} doesn't match KMU-FED's 4-field "
            f"'<subject>_<class>_<initials>_<index>' pattern (got {len(parts)} fields)"
        )
    subject_str, cls, initials, index = parts
    return int(subject_str), cls, initials, index


def discover_subject_class_counts(kmu_dir: Path = None) -> Dict[int, Dict[str, int]]:
    """Scan the real KMU-FED files and count images per (subject, class).

    Falls back to PUBLISHED_SUBJECT_CLASS_COUNTS if `kmu_dir` doesn't exist locally (the raw
    dataset lives on Kaggle, not on every machine that imports this module) - not an error.
    """
    kmu_dir = Path(kmu_dir) if kmu_dir is not None else Path(KMU_FED["dir"])
    if not kmu_dir.exists():
        return {sid: dict(counts) for sid, counts in PUBLISHED_SUBJECT_CLASS_COUNTS.items()}

    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {c: 0 for c in KMU_FED["class_names"]})
    for f in sorted(kmu_dir.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        subject_id, cls, _initials, _index = parse_filename(f.name)
        counts[subject_id][cls] += 1
    return dict(counts)


def zero_coverage_report(subject_class_counts: Dict[int, Dict[str, int]]) -> Dict[str, List[int]]:
    """For each class, which subject IDs contributed zero images of it."""
    classes = KMU_FED["class_names"]
    return {
        cls: sorted(sid for sid, counts in subject_class_counts.items() if counts.get(cls, 0) == 0)
        for cls in classes
    }


def select_subject_split(
    subject_class_counts: Dict[int, Dict[str, int]],
    seed: Optional[int] = None,
    n_train: int = KMU_FED["n_train_subjects"],
    n_val: int = KMU_FED["n_val_subjects"],
    n_test: int = KMU_FED["n_test_subjects"],
    require_class: str = "DI",
) -> SubjectSplit:
    """Seeded, constrained subject-level split: `n_train`/`n_val`/`n_test` subjects, such that
    both val and test each contain at least one subject with a `require_class` (default:
    disgust) image.

    Only 7 of KMU-FED's 12 subjects contributed any disgust images (see module docstring); an
    unconstrained random split could easily leave disgust with no val/test examples at all,
    making its macro-F1 undefined for that split. Tries `seed` (default: KMU_FED["split_seed"]),
    then seed+1, seed+2, ... deterministically until a split satisfies the constraint.
    """
    if seed is None:
        seed = KMU_FED["split_seed"]
    subjects = sorted(subject_class_counts)
    n_total = n_train + n_val + n_test
    if len(subjects) != n_total:
        raise ValueError(f"select_subject_split expects exactly {n_total} subjects, got {len(subjects)}")

    covered = {sid for sid, counts in subject_class_counts.items() if counts.get(require_class, 0) > 0}

    tried_seed = seed
    while True:
        rng = random.Random(tried_seed)
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        train, val, test = shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]
        if (set(val) & covered) and (set(test) & covered):
            return SubjectSplit(tried_seed, sorted(train), sorted(val), sorted(test))
        tried_seed += 1


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for (re)computing and saving the KMU-FED subject split."""
    parser = argparse.ArgumentParser(description="Discover KMU-FED's subject/class breakdown and the seeded subject split.")
    parser.add_argument("--seed", type=int, default=None, help="Starting seed to try (default: KMU_FED['split_seed']).")
    return parser


def main() -> None:
    """Print the zero-coverage table and the resolved subject split, and save both to
    results/kmu_fed_subject_split.json.
    """
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    counts = discover_subject_class_counts()
    zero_coverage = zero_coverage_report(counts)
    split = select_subject_split(counts, seed=args.seed)

    print("Per-subject, per-class image counts:")
    header = "subject".ljust(14) + "".join(c.ljust(6) for c in KMU_FED["class_names"])
    print(header)
    for sid in sorted(counts):
        row = f"{sid} ({SUBJECT_INITIALS[sid]})".ljust(14) + "".join(str(counts[sid][c]).ljust(6) for c in KMU_FED["class_names"])
        print(row)

    print("\nSubjects with ZERO images, per class:")
    for cls, subs in zero_coverage.items():
        print(f"  {cls}: {len(subs)} subject(s) -> {[f'{s} ({SUBJECT_INITIALS[s]})' for s in subs]}")

    def fmt(ids: List[int]) -> str:
        return ", ".join(f"{i} ({SUBJECT_INITIALS[i]})" for i in ids)

    print(f"\nSeed used: {split.seed_used} (started from {args.seed if args.seed is not None else KMU_FED['split_seed']})")
    print(f"TRAIN ({len(split.train)}): {fmt(split.train)}")
    print(f"VAL   ({len(split.val)}): {fmt(split.val)}")
    print(f"TEST  ({len(split.test)}): {fmt(split.test)}")

    out_path = RESULTS_DIR / "kmu_fed_subject_split.json"
    with open(out_path, "w") as f:
        json.dump({
            "seed_used": split.seed_used,
            "train_subjects": split.train,
            "val_subjects": split.val,
            "test_subjects": split.test,
            "subject_initials": SUBJECT_INITIALS,
            "subject_class_counts": counts,
            "zero_coverage": zero_coverage,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
