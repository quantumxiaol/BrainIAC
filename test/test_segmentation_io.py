from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from generate_segmentation import save_segmentation


def test_save_segmentation_writes_nifti_with_expected_shape(tmp_path: Path) -> None:
    output_path = tmp_path / "pred.nii.gz"
    seg = torch.ones((1, 16, 16, 16), dtype=torch.float32)
    meta = {"affine": np.eye(4, dtype=np.float32)}

    save_segmentation(seg, meta, str(output_path))

    assert output_path.exists()
    nii = nib.load(str(output_path))
    assert nii.shape == (16, 16, 16)
    assert np.count_nonzero(nii.get_fdata()) == 16 * 16 * 16


def test_save_segmentation_falls_back_to_identity_affine(tmp_path: Path) -> None:
    output_path = tmp_path / "pred_identity.nii.gz"
    seg = np.zeros((8, 8, 8), dtype=np.uint8)
    seg[1:3, 1:3, 1:3] = 1

    save_segmentation(seg, {}, str(output_path))

    nii = nib.load(str(output_path))
    assert np.allclose(nii.affine, np.eye(4))
