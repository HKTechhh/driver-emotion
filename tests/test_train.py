"""train.py housekeeping: a re-used run name must not stack runs in one history CSV."""
from src.train import _remove_stale_history


def test_removes_stale_history_for_the_same_run_name(tmp_path) -> None:
    stale = tmp_path / "vgg16_v1_history.csv"
    stale.write_text("epoch,accuracy\n0,0.3\n")
    assert _remove_stale_history("vgg16_v1", results_dir=tmp_path) is True
    assert not stale.exists()


def test_leaves_other_runs_and_missing_files_alone(tmp_path) -> None:
    other = tmp_path / "cnn_v1_history.csv"
    other.write_text("epoch,accuracy\n0,0.3\n")
    assert _remove_stale_history("vgg16_v1", results_dir=tmp_path) is False
    assert other.exists()
