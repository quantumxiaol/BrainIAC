#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    session_id: str
    image_path: Path
    mask_path: Path


def _find_flair(subj_dir: Path, case_id: str, session_id: str) -> Path | None:
    anat_dir = subj_dir / session_id / "anat"
    exact = anat_dir / f"{case_id}_{session_id}_FLAIR.nii.gz"
    if exact.exists():
        return exact
    candidates = sorted(anat_dir.glob(f"{case_id}_{session_id}_*FLAIR*.nii.gz"))
    return candidates[0] if candidates else None


def _find_mask(derivatives_root: Path, case_id: str, session_id: str) -> Path | None:
    ses_dir = derivatives_root / case_id / session_id
    exact = ses_dir / f"{case_id}_{session_id}_msk.nii.gz"
    if exact.exists():
        return exact
    candidates = sorted(ses_dir.glob(f"{case_id}_{session_id}_*msk*.nii.gz"))
    if not candidates:
        candidates = sorted(ses_dir.glob("*_msk.nii.gz"))
    return candidates[0] if candidates else None


def discover_isles_cases(isles_root: Path) -> tuple[list[CaseRecord], list[str]]:
    cases: list[CaseRecord] = []
    missing: list[str] = []
    derivatives_root = isles_root / "derivatives"

    for subj_dir in sorted(isles_root.glob("sub-strokecase*")):
        if not subj_dir.is_dir():
            continue
        case_id = subj_dir.name
        ses_dirs = sorted(p for p in subj_dir.glob("ses-*") if p.is_dir())
        for ses_dir in ses_dirs:
            session_id = ses_dir.name
            flair = _find_flair(subj_dir, case_id, session_id)
            mask = _find_mask(derivatives_root, case_id, session_id)
            if flair and mask and flair.exists() and mask.exists():
                cases.append(
                    CaseRecord(
                        case_id=case_id,
                        session_id=session_id,
                        image_path=flair.resolve(),
                        mask_path=mask.resolve(),
                    )
                )
            else:
                missing.append(f"{case_id}/{session_id}")
    return cases, missing


def _safe_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(f"Need at least 3 valid cases for train/val/test split, got {total}.")
    train_n = max(1, int(round(total * train_ratio)))
    val_n = max(1, int(round(total * val_ratio)))
    if train_n + val_n >= total:
        overflow = train_n + val_n - (total - 1)
        train_n = max(1, train_n - overflow)
    test_n = total - train_n - val_n
    if test_n < 1:
        test_n = 1
        train_n = max(1, total - val_n - test_n)
    return train_n, val_n, test_n


def split_cases(
    cases: list[CaseRecord], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[CaseRecord], list[CaseRecord], list[CaseRecord]]:
    shuffled = cases[:]
    random.Random(seed).shuffle(shuffled)
    train_n, val_n, _ = _safe_counts(len(shuffled), train_ratio, val_ratio)
    train_cases = shuffled[:train_n]
    val_cases = shuffled[train_n : train_n + val_n]
    test_cases = shuffled[train_n + val_n :]
    return train_cases, val_cases, test_cases


def _to_str(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return str(path)
    return str(path.relative_to(base_dir))


def write_csv(path: Path, cases: list[CaseRecord], relative_to: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "session_id", "image_path", "mask_path"],
        )
        writer.writeheader()
        for c in cases:
            writer.writerow(
                {
                    "case_id": c.case_id,
                    "session_id": c.session_id,
                    "image_path": _to_str(c.image_path, relative_to),
                    "mask_path": _to_str(c.mask_path, relative_to),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test CSVs for ISLES-2022 lesion segmentation.")
    parser.add_argument(
        "--isles-root",
        type=Path,
        required=True,
        help="Path to ISLES-2022 root, e.g. /data/datasets/ISLES-2022",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/data/csvs/isles2022"),
        help="Directory to save generated CSVs.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store paths relative to current working directory instead of absolute paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    isles_root = args.isles_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not isles_root.exists():
        raise FileNotFoundError(f"ISLES root does not exist: {isles_root}")

    if not (0.0 < args.train_ratio < 1.0 and 0.0 < args.val_ratio < 1.0):
        raise ValueError("train-ratio and val-ratio must be in (0, 1).")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be < 1.0")

    cases, missing = discover_isles_cases(isles_root)
    if not cases:
        raise RuntimeError(
            f"No valid ISLES cases found under: {isles_root}. "
            "Expected FLAIR at sub-*/ses-*/anat/*_FLAIR.nii.gz and mask at derivatives/sub-*/ses-*/*_msk.nii.gz"
        )

    train_cases, val_cases, test_cases = split_cases(
        cases, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    rel_base = Path.cwd().resolve() if args.relative_paths else None
    train_csv = output_dir / "isles2022_train.csv"
    val_csv = output_dir / "isles2022_val.csv"
    test_csv = output_dir / "isles2022_test.csv"
    write_csv(train_csv, train_cases, relative_to=rel_base)
    write_csv(val_csv, val_cases, relative_to=rel_base)
    write_csv(test_csv, test_cases, relative_to=rel_base)

    print(f"Discovered valid cases: {len(cases)}")
    print(f"Missing FLAIR/mask cases skipped: {len(missing)}")
    if missing:
        print("First 10 skipped:", ", ".join(missing[:10]))
    print(f"Train/Val/Test: {len(train_cases)}/{len(val_cases)}/{len(test_cases)}")
    print(f"train_csv: {train_csv}")
    print(f"val_csv:   {val_csv}")
    print(f"test_csv:  {test_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
