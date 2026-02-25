import pandas as pd
import numpy as np
from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Resized, NormalizeIntensityd, RandFlipd, RandRotated, Rand3DElasticd, RandBiasFieldd, RandGaussianNoised, ToTensord
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
    items = [
        {'image': row['image_path'], 'label': row['mask_path']} for _, row in df.iterrows()
    ]
    if is_train:
        transforms = Compose([
            LoadImaged(keys=['image', 'label'], image_only=False),
            EnsureChannelFirstd(keys=['image', 'label']),
            Resized(keys=['image', 'label'], spatial_size=img_size, mode=('trilinear', 'nearest')),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandRotated(keys=['image', 'label'], range_x=0.1, prob=0.5, mode='bilinear'),
            Rand3DElasticd(keys=['image', 'label'], sigma_range=(5,8), magnitude_range=(100,200), prob=0.2),
            RandBiasFieldd(keys=['image'], prob=0.3),
            RandGaussianNoised(keys=['image'], prob=0.2),
            EnsureTyped(keys=['image', 'label']),
            ToTensord(keys=['image', 'label'])
        ])
    else:
        transforms = Compose([
            LoadImaged(keys=['image', 'label'], image_only=False),
            EnsureChannelFirstd(keys=['image', 'label']),
            Resized(keys=['image', 'label'], spatial_size=img_size, mode=('trilinear', 'nearest')),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            EnsureTyped(keys=['image', 'label']),
            ToTensord(keys=['image', 'label'])
        ])
    ds = CacheDataset(data=items, transform=transforms, cache_rate=0.1, num_workers=num_workers)
    return ds 
