#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


DEFAULT_DROPBOX_URL = (
    "https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/"
    "AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&st=b9cnvwh8&dl=0"
)
DEFAULT_OUTPUT_DIR = Path("src/checkpoints")
CHUNK_SIZE = 1024 * 1024


def to_direct_download_url(url: str) -> str:
    parsed = urlparse(url)
    if "dropbox.com" not in parsed.netloc:
        return url
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def guess_download_name(response, fallback_url: str) -> str:
    filename = response.headers.get_filename()
    if filename:
        return Path(filename).name
    redirected_name = Path(urlparse(response.geturl()).path).name
    if redirected_name:
        return redirected_name
    fallback_name = Path(urlparse(fallback_url).path).name
    return fallback_name or "brainiac_checkpoints.zip"


def download_file(url: str, work_dir: Path, timeout: int = 60) -> Path:
    direct_url = to_direct_download_url(url)
    req = Request(direct_url, headers={"User-Agent": "BrainIAC-checkpoint-downloader/1.0"})

    with urlopen(req, timeout=timeout) as response:
        file_name = guess_download_name(response, direct_url)
        output_path = work_dir / file_name
        total_raw = response.headers.get("Content-Length")
        total = int(total_raw) if total_raw and total_raw.isdigit() else None
        downloaded = 0

        print(f"Downloading from: {direct_url}")
        with output_path.open("wb") as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    sys.stdout.write(
                        f"\rDownloaded {human_size(downloaded)} / {human_size(total)} ({percent:5.1f}%)"
                    )
                    sys.stdout.flush()
                elif downloaded % (20 * CHUNK_SIZE) < CHUNK_SIZE:
                    sys.stdout.write(f"\rDownloaded {human_size(downloaded)}")
                    sys.stdout.flush()
        sys.stdout.write("\n")

    return output_path


def looks_like_html(path: Path) -> bool:
    sample = path.read_bytes()[:2048].lower()
    return b"<html" in sample or b"<!doctype html" in sample


def common_zip_root(members: Iterable[zipfile.ZipInfo]) -> str | None:
    file_parts = []
    for member in members:
        if member.is_dir() or not member.filename or member.filename.startswith("__MACOSX/"):
            continue
        parts = PurePosixPath(member.filename).parts
        if not parts:
            continue
        file_parts.append(parts)

    if not file_parts:
        return None

    root_names = {parts[0] for parts in file_parts}
    if len(root_names) != 1:
        return None

    # Strip the common root only when all files are really inside that folder.
    if all(len(parts) >= 2 for parts in file_parts):
        return next(iter(root_names))
    return None


def safe_zip_relpath(member_name: str, strip_root: str | None) -> Path | None:
    raw = PurePosixPath(member_name)
    if raw.is_absolute() or ".." in raw.parts:
        return None

    rel = raw.as_posix()
    if strip_root:
        root_prefix = f"{strip_root}/"
        if rel.startswith(root_prefix):
            rel = rel[len(root_prefix) :]

    if not rel or rel.endswith("/"):
        return None
    return Path(rel)


def extract_zip(archive_path: Path, output_dir: Path, force: bool) -> tuple[int, int]:
    extracted = 0
    skipped = 0

    with zipfile.ZipFile(archive_path) as zf:
        members = zf.infolist()
        strip_root = common_zip_root(members)
        if strip_root:
            print(f"Archive has common top-level folder '{strip_root}', flattening it into {output_dir}")

        for member in members:
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            rel = safe_zip_relpath(member.filename, strip_root)
            if rel is None:
                continue
            target = output_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists() and not force:
                skipped += 1
                continue

            with zf.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    return extracted, skipped


def install_single_file(file_path: Path, output_dir: Path, force: bool) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / file_path.name
    if target.exists() and not force:
        return 0, 1
    shutil.copy2(file_path, target)
    return 1, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BrainIAC checkpoints and place them under ./src/checkpoints/"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_DROPBOX_URL,
        help="Dropbox shared URL. Ignored if --archive is provided.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use a local archive/file instead of downloading from URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for extracted checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist in the destination.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep a copy of the downloaded archive in output-dir.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds for opening the download URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="brainiac_ckpt_") as tmp:
        work_dir = Path(tmp)
        downloaded = False

        if args.archive is not None:
            archive_path = args.archive.expanduser().resolve()
            if not archive_path.exists():
                print(f"Error: local archive not found: {archive_path}", file=sys.stderr)
                return 1
        else:
            archive_path = download_file(args.url, work_dir, timeout=args.timeout)
            downloaded = True

        if downloaded and args.keep_archive:
            kept_path = output_dir / archive_path.name
            if kept_path.exists() and not args.force:
                print(f"Archive copy skipped (already exists): {kept_path}")
            else:
                shutil.copy2(archive_path, kept_path)
                print(f"Saved archive copy to: {kept_path}")

        if zipfile.is_zipfile(archive_path):
            extracted, skipped = extract_zip(archive_path, output_dir, args.force)
        else:
            if looks_like_html(archive_path):
                print(
                    "Error: downloaded content looks like HTML, not checkpoint files. "
                    "The Dropbox link may be invalid/expired or requires manual access.",
                    file=sys.stderr,
                )
                return 1
            extracted, skipped = install_single_file(archive_path, output_dir, args.force)

    print(f"Done. Extracted: {extracted}, skipped(existing): {skipped}")
    print(f"Checkpoints directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
