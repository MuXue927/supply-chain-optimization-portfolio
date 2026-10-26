#!/usr/bin/env python
"""Derive tight-capacity dataset variants from archived CSVs.

This script intentionally does *not* call ``generate_data.py``.  The archived
CSV files are the authoritative instances; the generator has drifted from them
and re-running it changes the structure of the instances.

Only ``plant_size_upper_limit.csv`` is rewritten in each variant.  Every other
file is copied byte-for-byte from the archived dataset.  The capacity for each
plant is ``ceil(k * initial_inventory(plant))``: M65 proved that the
earliest-shipping lower bound on peak inventory equals the initial inventory
on the audited datasets.

Usage::

    python tools/make_tight_capacity_variants.py
    python tools/make_tight_capacity_variants.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO / "multi-period-delivery" / "datasets" / "multiple-periods"
BASES = (
    ("medium", "dataset_13"),
    ("large", "dataset_3"),
)
KS = (1.10, 1.25, 1.50, 2.00)
CAP_FILE = "plant_size_upper_limit.csv"
UNCHANGED_FILES = (
    "order.csv",
    "product_size.csv",
    "vehicle.csv",
    "warehouse_production.csv",
    "warehouse_storage.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_initial_inventory(dataset: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    with (dataset / "warehouse_storage.csv").open(
        newline="", encoding="utf-8-sig"
    ) as fp:
        for row in csv.DictReader(fp):
            plant = str(row["plant_code"])
            totals[plant] = totals.get(plant, 0) + int(float(row["volume"]))
    return totals


def read_capacity_rows(dataset: Path) -> list[dict[str, str]]:
    with (dataset / CAP_FILE).open(newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def variant_name(base: str, k: float) -> str:
    # Keep the dataset name free of ``.`` so existing probe parsers can still
    # split ``dataset_<id>`` components without special cases.
    return f"{base}_tight_k{round(k * 100):03d}"


def make_variant(
    base_dir: Path,
    output_dir: Path,
    k: float,
    check: bool,
    force: bool,
) -> dict:
    initial = read_initial_inventory(base_dir)
    rows = read_capacity_rows(base_dir)
    caps = {
        str(row["plant_code"]): math.ceil(k * float(initial[str(row["plant_code"])]))
        for row in rows
    }

    if check:
        if output_dir.exists():
            raise SystemExit(f"variant already exists: {output_dir}")
        return {
            "path": str(output_dir.relative_to(REPO)),
            "k": k,
            "initial": initial,
            "capacity": caps,
            "would_create": True,
        }

    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"variant already exists: {output_dir} (pass --force to replace it)"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for filename in UNCHANGED_FILES:
        shutil.copyfile(base_dir / filename, output_dir / filename)

    with (output_dir / CAP_FILE).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["plant_code", "max_volume"])
        for row in rows:
            plant = str(row["plant_code"])
            writer.writerow([plant, caps[plant]])

    return {
        "path": str(output_dir.relative_to(REPO)),
        "k": k,
        "initial": initial,
        "capacity": caps,
        "unchanged_sha256": {
            name: sha256(output_dir / name) for name in UNCHANGED_FILES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would be written without creating or changing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing variant directories (archived bases are never touched)",
    )
    args = parser.parse_args()

    manifest = {"bases": {}, "variants": []}
    for scale, base in BASES:
        base_dir = DATASET_ROOT / scale / base
        if not base_dir.is_dir():
            raise SystemExit(f"missing archived dataset: {base_dir}")
        manifest["bases"][f"{scale}/{base}"] = {
            name: sha256(base_dir / name)
            for name in (CAP_FILE, *UNCHANGED_FILES)
        }
        initial = read_initial_inventory(base_dir)
        for k in KS:
            out = DATASET_ROOT / scale / variant_name(base, k)
            row = make_variant(base_dir, out, k, args.check, args.force)
            row["base"] = f"{scale}/{base}"
            row["initial"] = initial
            manifest["variants"].append(row)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
