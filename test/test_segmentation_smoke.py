from __future__ import annotations

from pathlib import Path

import nibabel as nib
import pytest
import torch

from generate_segmentation import (
    generate_segmentation,
    load_model_for_inference,
    preprocess_image,
    save_segmentation,
)


@pytest.mark.integration
@pytest.mark.gpu
def test_segmentation_inference_smoke(
    repo_root: Path,
    brainiac_ckpt: Path,
    segmentation_ckpt: Path,
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the current segmentation inference code path.")
    if not brainiac_ckpt.exists() or not segmentation_ckpt.exists():
        pytest.skip("Missing segmentation checkpoints for smoke test.")

    image_path = repo_root / "src" / "data" / "sample" / "processed" / "I10307487_0000.nii.gz"
    if not image_path.exists():
        pytest.skip(f"Missing sample image: {image_path}")

    checkpoint = torch.load(str(segmentation_ckpt), map_location="cpu")
    config = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]
    config["pretrain"]["simclr_checkpoint_path"] = str(brainiac_ckpt)

    model = load_model_for_inference(config, state_dict)
    image_tensor, meta_dict = preprocess_image(str(image_path), config)
    pred = generate_segmentation(model, image_tensor, config)

    output_path = tmp_path / "sample_seg.nii.gz"
    save_segmentation(pred.cpu(), meta_dict, str(output_path))

    assert output_path.exists()
    assert nib.load(str(output_path)).shape == tuple(config["model"]["img_size"])
