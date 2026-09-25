"""Checks for src/kmu_fed_data.py's filename parsing and subject-level split logic."""
import pytest

from src.kmu_fed_data import (
    PUBLISHED_SUBJECT_CLASS_COUNTS,
    discover_subject_class_counts,
    parse_filename,
    select_subject_split,
    zero_coverage_report,
)


def test_parse_filename_fixes_the_leading_zero_bug() -> None:
    """subject "mr" ships as both "01_..." and a malformed "1_..." - both must parse to the
    same integer subject ID, or a real person would be silently split into two "subjects"."""
    assert parse_filename("01_AN_mr_001.jpg")[0] == 1
    assert parse_filename("1_AN_mr_001.jpg")[0] == 1


def test_parse_filename_extracts_all_fields() -> None:
    subject_id, cls, initials, index = parse_filename("10_DI_ko_007.jpg")
    assert (subject_id, cls, initials, index) == (10, "DI", "ko", "007")


def test_parse_filename_rejects_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        parse_filename("not_the_right_shape.jpg")


def test_discover_subject_class_counts_falls_back_when_dataset_missing(tmp_path) -> None:
    """The raw KMU-FED files only exist on Kaggle; a missing local copy is expected, not an
    error, so this should fall back to the project's own verified published counts."""
    counts = discover_subject_class_counts(tmp_path / "does_not_exist")
    assert counts == PUBLISHED_SUBJECT_CLASS_COUNTS


def test_discover_subject_class_counts_matches_real_files(tmp_path) -> None:
    (tmp_path / "01_AN_mr_001.jpg").touch()
    (tmp_path / "1_AN_mr_002.jpg").touch()   # same subject as "01", must be counted together
    (tmp_path / "02_DI_s01_001.jpg").touch()
    (tmp_path / "not_an_image.txt").touch()  # must be ignored, not crash

    counts = discover_subject_class_counts(tmp_path)
    assert counts[1]["AN"] == 2
    assert counts[2]["DI"] == 1


def test_zero_coverage_report_matches_the_real_dataset() -> None:
    report = zero_coverage_report(PUBLISHED_SUBJECT_CLASS_COUNTS)
    assert report["DI"] == [2, 3, 4, 7, 9]         # 5 of 12 subjects: the disgust gap
    assert report["AN"] == [4]
    assert report["FE"] == [12]
    assert report["HA"] == [11]
    assert report["SA"] == [2, 10]
    assert report["SU"] == [12]


def test_select_subject_split_partitions_every_subject_exactly_once() -> None:
    split = select_subject_split(PUBLISHED_SUBJECT_CLASS_COUNTS, seed=42)
    all_subjects = split.train + split.val + split.test
    assert sorted(all_subjects) == list(range(1, 13))
    assert len(set(all_subjects)) == 12
    assert len(split.train) == 8 and len(split.val) == 2 and len(split.test) == 2


def test_select_subject_split_keeps_disgust_covered_in_val_and_test() -> None:
    disgust_covered = {1, 5, 6, 8, 10, 11, 12}
    for seed in range(10):  # several starting seeds, not just the one that happened to work
        split = select_subject_split(PUBLISHED_SUBJECT_CLASS_COUNTS, seed=seed)
        assert set(split.val) & disgust_covered, f"seed {seed}: val {split.val} has no disgust coverage"
        assert set(split.test) & disgust_covered, f"seed {seed}: test {split.test} has no disgust coverage"


def test_select_subject_split_is_reproducible() -> None:
    a = select_subject_split(PUBLISHED_SUBJECT_CLASS_COUNTS, seed=42)
    b = select_subject_split(PUBLISHED_SUBJECT_CLASS_COUNTS, seed=42)
    assert a == b


def test_select_subject_split_matches_the_decided_assignment() -> None:
    """Locks in the actual split this project uses (config.py's KMU_FED dict), so a future
    change to the selection algorithm can't silently swap which real people are held out."""
    split = select_subject_split(PUBLISHED_SUBJECT_CLASS_COUNTS, seed=42)
    assert split.seed_used == 42
    assert split.train == [3, 4, 6, 7, 8, 9, 10, 12]
    assert split.val == [1, 5]
    assert split.test == [2, 11]
