"""Portable path defaults for experiment and paper-generation scripts."""

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


PROJECT_ROOT = _env_path("DYNA_PROJECT_ROOT", Path(__file__).resolve().parents[1])
DATASETS_ROOT = _env_path("DYNA_DATASETS_ROOT", PROJECT_ROOT.parent)
PHOTO_SLAM_ROOT = _env_path("PHOTO_SLAM_ROOT", PROJECT_ROOT.parent / "Photo-SLAM")
EXPERIMENTS_ROOT = _env_path("DYNA_EXPERIMENTS_ROOT", PROJECT_ROOT / "experiments")
PAPER2ANY_ROOT = _env_path("PAPER2ANY_ROOT", PROJECT_ROOT.parent / "Paperasset2Paper")
