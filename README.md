# Anonymous Reproducibility Repository

This repository contains a cleaned, script-based reproduction path for the
sketch-based recommender experiments. It is organized so reviewers can inspect the
implemented algorithms, run lightweight checks locally, and reproduce reported
best configurations on appropriate GPU hardware.

## Current status

Implemented in this scaffold:

- shared dataset/split/evaluation utilities converted from the original utilities notebook
- YouTube-style baseline model and space-tracked experiment wrapper
- AutoRec baseline model and space-tracked experiment wrapper
- LEAF baseline model and space-tracked experiment wrapper
- UniSketchMF / JL-RACE-MF side-info model and space-tracked experiment wrapper
- config-driven best-run execution for YouTube, AutoRec, LEAF, and UniSketchMF
- CPU smoke-test configs for YouTube, AutoRec, LEAF, and UniSketchMF on a small MovieLens-100K subset
- local JSON/CSV outputs
- W&B disabled by default

All four paper model families are now represented in the script-based reproduction path.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data

Place the parquet datasets under `datasets/`:

```text
datasets/100K_ratings.parquet
datasets/1M_ratings.parquet
datasets/10M_ratings.parquet
datasets/Beauty-25-10_ratings.parquet
```

Each parquet should contain `user_id`, `item_id`, and `rating`.

## Lightweight local checks, no GPU required

First validate imports, config files, registry wiring, and dataset schemas:

```bash
python scripts/check_setup.py
```

If you only want to check Python/package imports before copying datasets:

```bash
python scripts/check_setup.py --skip-data
```

A tiny CPU smoke test is also provided. It uses a small subset and one epoch only,
so it is **not** expected to reproduce paper metrics; it only checks that the full
experiment wiring runs end to end.

```bash
python scripts/run_single.py \
  --model youtube \
  --dataset 100K \
  --config config/debug/youtube_smoke.yaml \
  --device cpu

python scripts/run_single.py \
  --model autorec \
  --dataset 100K \
  --config config/debug/autorec_smoke.yaml \
  --device cpu

python scripts/run_single.py \
  --model leaf \
  --dataset 100K \
  --config config/debug/leaf_smoke.yaml \
  --device cpu

python scripts/run_single.py \
  --model unisketchmf \
  --dataset 100K \
  --config config/debug/unisketchmf_smoke.yaml \
  --device cpu
```

Outputs are written to:

```text
outputs/<model>_smoke_100K/
  metrics.json
  results.json
  artifacts.json
  config_used.json
```

## Full best-run reproduction, GPU recommended

The reported best-run configs are intended for GPU execution. The original runs
were performed on high-end GPU hardware; CPU execution of full configurations may
be very slow.

Run a configured best run, for example YouTube on MovieLens 1M:

```bash
python scripts/run_single.py --model youtube --dataset 1M --device cuda
```

or explicitly pass the config:

```bash
python scripts/run_single.py \
  --model youtube \
  --dataset 1M \
  --config config/best_runs/youtube.yaml \
  --device cuda
```

Outputs are written to:

```text
outputs/youtube_1M/
  metrics.json
  results.json
  artifacts.json
  config_used.json
```

## Reproduce all currently configured best runs

```bash
python scripts/reproduce_best_runs.py --models youtube autorec leaf unisketchmf --device cuda
python scripts/aggregate_results.py
```

This produces:

```text
outputs/summary.csv
```

## W&B

W&B is disabled by default through `WANDB_MODE=disabled`. To enable it manually:

```bash
python scripts/run_single.py --model youtube --dataset 1M --enable-wandb
```

## Repository organization

- `src/recsys_edge/models/`: model and experiment implementations.
- `config/best_runs/`: exact best-run configs intended for GPU reproduction.
- `config/debug/`: tiny CPU smoke-test configs, not used for reported metrics.
- `scripts/run_single.py`: run one configured model/dataset.
- `scripts/reproduce_best_runs.py`: run all configured best runs.
- `scripts/aggregate_results.py`: aggregate JSON outputs into `outputs/summary.csv`.

## Notes

The code intentionally preserves the original experiment logic as much as possible.
Changes made during conversion are limited to import paths, optional W&B handling,
repository-relative dataset loading, and optional debug-only data reduction for
smoke tests. The debug configs are not used for reported results.
