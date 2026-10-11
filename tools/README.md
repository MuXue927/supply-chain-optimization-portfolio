# Shared Dataset Generator

`generate_data.py` is the shared generator for both optimization projects.
It is intentionally stored at the repository root because the single-period
and multi-period generators share the same instance-generation logic.

Using one generator keeps the generated data format consistent and avoids
maintaining two near-identical copies.

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
