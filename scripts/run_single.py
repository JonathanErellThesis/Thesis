#!/usr/bin/env python
"""Run one configured best-run or debug experiment.

Examples:
    python scripts/run_single.py --model youtube --dataset 1M
    python scripts/run_single.py --model youtube --dataset 100K --config config/debug/youtube_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys_edge.core import ExperimentParams, build_datasets_slim, load_dataset  # noqa: E402
from recsys_edge.experiments.registry import get_experiment_class  # noqa: E402


def _as_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _as_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _select_run(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    runs = cfg.get("best_runs", {}) or {}
    if dataset not in runs:
        available = ", ".join(sorted(runs))
        raise KeyError(f"Dataset '{dataset}' not found in config. Available: {available}")
    return runs[dataset]


def _apply_debug_subset(raw_df: pd.DataFrame, debug_cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Apply optional debug-only dataset reductions.

    This is intentionally controlled by config/debug/*.yaml and is not used by the
    best-run configs. It lets reviewers validate wiring on CPU without altering the
    full reproduction protocol.
    """
    if not debug_cfg:
        return raw_df

    out = raw_df.copy()

    min_interactions = debug_cfg.get("min_interactions_per_user")
    if min_interactions is not None:
        min_interactions = int(min_interactions)
        counts = out.groupby("user_id", sort=False).size()
        keep_users = counts[counts >= min_interactions].index
        out = out[out["user_id"].isin(keep_users)].copy()

    max_users = debug_cfg.get("max_users")
    if max_users is not None:
        max_users = int(max_users)
        users = pd.Index(out["user_id"].drop_duplicates())
        if len(users) > max_users:
            sampled_users = users.to_series().sample(n=max_users, random_state=seed).to_numpy()
            out = out[out["user_id"].isin(sampled_users)].copy()

    max_rows = debug_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if len(out) > max_rows:
            out = out.sample(n=max_rows, random_state=seed).copy()

    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError("Debug subset is empty. Relax debug filters in the config.")

    print(
        "[debug subset] "
        f"rows={len(out)} users={out['user_id'].nunique()} items={out['item_id'].nunique()}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key, e.g. youtube")
    parser.add_argument("--dataset", required=True, help="Dataset key, e.g. 1M")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to model YAML. Defaults to config/best_runs/<model>.yaml",
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--device", default=None, help="Override configured device")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    if not args.enable_wandb:
        os.environ.setdefault("WANDB_MODE", "disabled")

    config_path = Path(args.config) if args.config else REPO_ROOT / "config" / "best_runs" / f"{args.model}.yaml"
    cfg = _load_yaml(config_path)
    run_cfg = _select_run(cfg, args.dataset)

    seed = int(run_cfg.get("seed", 42))
    device = args.device or run_cfg.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    raw_df = load_dataset(args.dataset, data_dir=args.data_dir)
    raw_df = _apply_debug_subset(raw_df, run_cfg.get("debug", {}) or {}, seed=seed)
    datasets = build_datasets_slim(raw_df, seed=seed)

    eval_cfg = run_cfg.get("eval", {}) or {}
    params_cfg = run_cfg.get("params", {}) or {}

    params = ExperimentParams(
        dataset=args.dataset,
        model_name=cfg.get("model_name", args.model),
        seed=seed,
        device=device,
        eval_top_ks=eval_cfg.get("eval_top_ks"),
        relevance_thresholds=eval_cfg.get("relevance_thresholds"),
        wandb_project=run_cfg.get("wandb_project", "anonymous-reproduction"),
        model_init=run_cfg.get("model_init", {}) or {},
        model_hps=run_cfg.get("model_hps", {}) or {},
        incremental_hps=run_cfg.get("incremental_hps", {}) or {},
        sketch=params_cfg.get("sketch", {}) or {},
        use_clip_rmse=bool(eval_cfg.get("use_clip_rmse", True)),
        rmse_clip_bounds=tuple(eval_cfg.get("rmse_clip_bounds", [1.0, 5.0])),
        online_inference_pred_type=params_cfg.get("online_inference_pred_type", "per_user"),
        train_on_user_residuals=bool(params_cfg.get("train_on_user_residuals", False)),
    )

    exp_cls = get_experiment_class(args.model)
    exp = exp_cls(datasets=datasets, params=params)
    results, artifacts = exp.run()

    run_name = run_cfg.get("run_name") or f"{args.model}_{args.dataset}"
    out_dir = Path(args.output_dir) / str(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "offline_rmse": results.offline.rmse,
        "incremental_rmse": results.incremental.rmse,
        "online_rmse": results.online.rmse,
        "offline_post_incremental_rmse": results.offline_post_incremental.rmse,
    }
    (out_dir / "metrics.json").write_text(json.dumps(_as_jsonable(metrics), indent=2), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(_as_jsonable(results), indent=2), encoding="utf-8")
    (out_dir / "artifacts.json").write_text(json.dumps(_as_jsonable(artifacts), indent=2), encoding="utf-8")
    (out_dir / "config_used.json").write_text(json.dumps(_as_jsonable(run_cfg), indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
