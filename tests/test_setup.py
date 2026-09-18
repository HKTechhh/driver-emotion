"""Phase 0 sanity checks: config is importable and directory scaffolding works."""
import config
from src.utils import ensure_dirs


def test_num_classes() -> None:
    assert config.NUM_CLASSES == 7


def test_ensure_dirs_creates_all_dirs() -> None:
    ensure_dirs()
    for directory in config.ALL_DIRS:
        assert directory.exists() and directory.is_dir()
