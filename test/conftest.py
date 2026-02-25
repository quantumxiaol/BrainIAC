from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Avoid matplotlib cache permission issues in constrained environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def checkpoints_dir(repo_root: Path) -> Path:
    return repo_root / "src" / "checkpoints"


@pytest.fixture(scope="session")
def brainiac_ckpt(checkpoints_dir: Path) -> Path:
    return checkpoints_dir / "BrainIAC.ckpt"


@pytest.fixture(scope="session")
def segmentation_ckpt(checkpoints_dir: Path) -> Path:
    return checkpoints_dir / "segmentation.ckpt"
