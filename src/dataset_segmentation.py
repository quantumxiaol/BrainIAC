import pandas as pd
import numpy as np
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    NormalizeIntensityd,
    RandFlipd,
    RandRotated,
    Rand3DElasticd,
    RandBiasFieldd,
    RandGaussianNoised,
    ConcatItemsd,
    DeleteItemsd,
    ToTensord,
)
import monai.transforms.transform as monai_transform
import monai.utils.misc as monai_misc


def _patch_monai_max_seed_for_numpy2() -> None:
    """
    MONAI<=1.3.x may use MAX_SEED=2**32, which overflows with newer NumPy
    in random state plumbing. Clamp to uint32 max to keep behavior stable.
    """
    max_uint32 = int(np.iinfo(np.uint32).max)  # 4294967295
    if getattr(monai_transform, "MAX_SEED", max_uint32) > max_uint32:
        monai_transform.MAX_SEED = max_uint32
    if getattr(monai_misc, "MAX_SEED", max_uint32) > max_uint32:
        monai_misc.MAX_SEED = max_uint32


_patch_monai_max_seed_for_numpy2()

def get_segmentation_dataloader(csv_file, img_size, batch_size, num_workers, is_train=True):
    df = pd.read_csv(csv_file)
    has_second_image = "image_path2" in df.columns and df["image_path2"].fillna("").str.strip().ne("").any()

    items = []
    for _, row in df.iterrows():
        if has_second_image:
            items.append(
                {
                    "image_dwi": row["image_path"],
                    "image_adc": row["image_path2"],
                    "label": row["mask_path"],
                }
            )
        else:
            items.append({"image": row["image_path"], "label": row["mask_path"]})

    if has_second_image:
        image_keys = ["image_dwi", "image_adc"]
    else:
        image_keys = ["image"]
    spatial_keys = image_keys + ["label"]
    resize_modes = tuple(["trilinear"] * len(image_keys) + ["nearest"])
    deform_modes = tuple(["bilinear"] * len(image_keys) + ["nearest"])

    if is_train:
        transform_list = [
            LoadImaged(keys=spatial_keys, image_only=False),
            EnsureChannelFirstd(keys=spatial_keys),
            Resized(keys=spatial_keys, spatial_size=img_size, mode=resize_modes),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=0),
            RandRotated(keys=spatial_keys, range_x=0.1, prob=0.5, mode=deform_modes),
            Rand3DElasticd(
                keys=spatial_keys,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                prob=0.2,
                mode=deform_modes,
            ),
            RandBiasFieldd(keys=image_keys, prob=0.3),
            RandGaussianNoised(keys=image_keys, prob=0.2),
        ]
    else:
        transform_list = [
            LoadImaged(keys=spatial_keys, image_only=False),
            EnsureChannelFirstd(keys=spatial_keys),
            Resized(keys=spatial_keys, spatial_size=img_size, mode=resize_modes),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        ]

    if has_second_image:
        transform_list.extend(
            [
                ConcatItemsd(keys=image_keys, name="image", dim=0),
                DeleteItemsd(keys=image_keys),
            ]
        )

    transform_list.extend([EnsureTyped(keys=["image", "label"]), ToTensord(keys=["image", "label"])])
    transforms = Compose(transform_list)

    ds = CacheDataset(data=items, transform=transforms, cache_rate=0.1, num_workers=num_workers)
    return ds 
