#!/usr/bin/env python
"""Lightweight environment and repository sanity check.

This script does not train models. It validates imports, config files, dataset
presence/schema, and model registry wiring so reviewers can check the repository
without GPU access.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REQUIRED_COLUMNS = {"user_id", "item_id", "rating"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _check_imports() -> None:
    modules = [
        "numpy",
        "pandas",
        "torch",
        "sklearn",
        "scipy",
        "mmh3",
        "yaml",
        "recsys_edge.core",
        "recsys_edge.experiments.registry",
        "recsys_edge.models.youtube",
        "recsys_edge.models.autorec",
        "recsys_edge.models.leaf",
        "recsys_edge.models.unisketchmf",
    ]
    print("Checking imports...")
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"  OK {name}")
        except ModuleNotFoundError as exc:
            missing.append((name, str(exc)))
            print(f"  MISSING {name}: {exc}")
    if missing:
        raise ModuleNotFoundError(
            "Missing required dependencies. Run `pip install -r requirements.txt` "
            "and then `pip install -e .` before running checks."
        )


def _check_registry() -> None:
    from recsys_edge.experiments.registry import available_models, get_experiment_class

    print("Checking model registry...")
    models = available_models()
    if not models:
        raise RuntimeError("No models registered.")
    for model in models:
        cls = get_experiment_class(model)
        print(f"  OK {model}: {cls.__name__}")


def _check_datasets(data_dir: Path, max_rows: int = 5) -> None:
    cfg_path = REPO_ROOT / "config" / "datasets.yaml"
    cfg = _load_yaml(cfg_path)
    datasets = cfg.get("datasets", {}) or {}
    if not datasets:
        raise RuntimeError(f"No datasets listed in {cfg_path}")

    print("Checking datasets...")
    for dataset_name, rel_path in datasets.items():
        # config paths are repo-relative; --data-dir lets users override the base folder.
        filename = Path(rel_path).name
        data_path = data_dir / filename
        if not data_path.exists():
            raise FileNotFoundError(
                f"Missing dataset '{dataset_name}': expected {data_path}. "
                "Place parquet files under datasets/ or pass --data-dir."
            )

        df = pd.read_parquet(data_path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset '{dataset_name}' is missing columns: {sorted(missing)}")
        preview = df.head(max_rows)
        print(
            f"  OK {dataset_name}: rows={len(df)} "
            f"users={df['user_id'].nunique()} items={df['item_id'].nunique()} "
            f"path={data_path}"
        )
        if max_rows > 0:
            print(preview.to_string(index=False))


def _check_configs() -> None:
    print("Checking configs...")
    config_files = sorted((REPO_ROOT / "config").glob("**/*.yaml"))
    if not config_files:
        raise RuntimeError("No YAML config files found under config/.")
    for path in config_files:
        cfg = _load_yaml(path)
        if path.parts[-2] in {"best_runs", "debug"}:
            if "best_runs" not in cfg:
                raise ValueError(f"Config {path} does not contain a best_runs section.")
        print(f"  OK {path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--skip-data", action="store_true", help="Skip parquet dataset validation")
    parser.add_argument("--preview-rows", type=int, default=0)
    args = parser.parse_args()

    _check_imports()
    _check_registry()
    _check_configs()
    if not args.skip_data:
        _check_datasets(Path(args.data_dir), max_rows=args.preview_rows)

    print("\nSetup check completed successfully.")


if __name__ == "__main__":
    main()
