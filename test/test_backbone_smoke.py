from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac


@pytest.mark.integration
def test_brainiac_backbone_forward_smoke(repo_root: Path, brainiac_ckpt: Path) -> None:
    if not brainiac_ckpt.exists():
        pytest.skip(f"Missing checkpoint: {brainiac_ckpt}")

    sample_csv = repo_root / "src" / "data" / "csvs" / "sample.csv"
    sample_root = repo_root / "src" / "data" / "sample" / "processed"
    if not sample_csv.exists() or not sample_root.exists():
        pytest.skip("Missing sample data for smoke test.")

    ds = BrainAgeDataset(
        csv_path=str(sample_csv),
        root_dir=str(sample_root),
        transform=get_validation_transform(),
    )
    sample = ds[0]
    x = sample["image"].unsqueeze(0)

    model = load_brainiac(str(brainiac_ckpt), device="cpu")
    model.eval()
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 768)
    assert torch.isfinite(y).all()
