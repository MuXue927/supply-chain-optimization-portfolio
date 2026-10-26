# Shared Dataset Generator

`generate_data.py` is the shared generator for both optimization projects.
It is intentionally stored at the repository root because the single-period
and multi-period generators share the same instance-generation logic.

Using one generator keeps the generated data format consistent and avoids
maintaining two near-identical copies.

## Archived Data Is Authoritative

The checked-in benchmark CSV files are the authoritative instances. The
generator has drifted from them: regenerating the documented medium seed
`20260408` produces a different vehicle-type set and different row counts.
Do not regenerate a benchmark dataset and expect the historical results to
remain comparable.

Tight-capacity variants are derived from the archived CSV files with
`tools/make_tight_capacity_variants.py`. That script copies every non-capacity
file byte-for-byte and changes only `plant_size_upper_limit.csv`.

## Generate Single-period Data

```powershell
python tools/generate_data.py `
  --problem single `
  --scale small `
  --count 3 `
  --seed 20260408
```

This writes:

```text
single-period-delivery/datasets/single-period/small/
```

## Generate Multi-period Data

```powershell
python tools/generate_data.py `
  --problem multi `
  --periods 5 `
  --scale small `
  --count 3 `
  --seed 20260408
```

This writes:

```text
multi-period-delivery/datasets/multiple-periods/small/
```

## Options

- `--problem`: required; use `single` or `multi`.
- `--scale`: `small`, `medium`, or `large`.
- `--periods`: planning horizon for multi-period data; ignored for single-period data.
- `--count`: number of `dataset_N` directories to create.
- `--seed`: optional random seed for reproducible generation.
- `--output-dir`: optional exact output directory that overrides the default project path.
