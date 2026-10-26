"""Regression checks for the tight-capacity variant family."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from multi_period_delivery.alns.InputDataALNS import DataALNS

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "multiple-periods"
CASES = (
    ("medium", "dataset_13", 110),
    ("medium", "dataset_13", 125),
    ("medium", "dataset_13", 150),
    ("medium", "dataset_13", 200),
    ("large", "dataset_3", 110),
    ("large", "dataset_3", 125),
    ("large", "dataset_3", 150),
    ("large", "dataset_3", 200),
)
UNCHANGED = (
    "order.csv",
    "product_size.csv",
    "vehicle.csv",
    "warehouse_production.csv",
    "warehouse_storage.csv",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


@pytest.mark.parametrize("scale,base,k", CASES)
def test_variant_differs_only_in_capacity(
    scale: str, base: str, k: int
) -> None:
    base_dir = ROOT / scale / base
    variant_dir = ROOT / scale / f"{base}_tight_k{k:03d}"
    assert variant_dir.is_dir()

    for filename in UNCHANGED:
        assert (variant_dir / filename).read_bytes() == (
            base_dir / filename
        ).read_bytes()

    initial: dict[str, int] = {}
    for row in _rows(base_dir / "warehouse_storage.csv"):
        plant = row["plant_code"]
        initial[plant] = initial.get(plant, 0) + int(float(row["volume"]))

    caps = {
        row["plant_code"]: float(row["max_volume"])
        for row in _rows(variant_dir / "plant_size_upper_limit.csv")
    }
    assert set(caps) == set(initial)
    assert all(caps[plant] >= quantity for plant, quantity in initial.items())


@pytest.mark.parametrize("scale,base,k", CASES)
def test_variant_loads_with_the_project_loader(
    scale: str, base: str, k: int
) -> None:
    variant = f"{base}_tight_k{k:03d}"
    data = DataALNS(
        str(ROOT / scale),
        str(ROOT / scale / variant / "tmp"),
        variant,
    )
    data.load()
    assert data.plant_inv_limit
    assert data.demands
    assert data.sku_prod_each_day
