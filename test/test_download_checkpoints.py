from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zipfile import ZipInfo

from download_checkpoints import common_zip_root, safe_zip_relpath, to_direct_download_url


def test_to_direct_download_url_sets_dropbox_dl_flag() -> None:
    url = "https://www.dropbox.com/scl/fo/abc/def?rlkey=xyz&st=123&dl=0"
    direct = to_direct_download_url(url)
    query = parse_qs(urlparse(direct).query)
    assert query["dl"] == ["1"]
    assert query["rlkey"] == ["xyz"]


def test_to_direct_download_url_keeps_non_dropbox_url() -> None:
    url = "https://example.com/file.zip?dl=0"
    assert to_direct_download_url(url) == url


def test_common_zip_root_detects_single_root_folder() -> None:
    members = [ZipInfo("checkpoints/BrainIAC.ckpt"), ZipInfo("checkpoints/segmentation.ckpt")]
    assert common_zip_root(members) == "checkpoints"


def test_common_zip_root_none_for_mixed_roots() -> None:
    members = [ZipInfo("BrainIAC.ckpt"), ZipInfo("checkpoints/segmentation.ckpt")]
    assert common_zip_root(members) is None


def test_safe_zip_relpath_strips_root_and_blocks_unsafe_paths() -> None:
    assert safe_zip_relpath("checkpoints/BrainIAC.ckpt", "checkpoints") == Path("BrainIAC.ckpt")
    assert safe_zip_relpath("/abs/path.ckpt", None) is None
    assert safe_zip_relpath("../escape.ckpt", None) is None
