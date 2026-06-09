"""
Core utilities for the anonymous RecSys edge-deployment reproduction package.

This module is generated from the original experiment utilities notebook. The goal is
not to change the experiment logic, but to make it importable from scripts.

Notes:
- W&B is optional and disabled by default for anonymous reproducibility.
- A hard-coded W&B login call from the notebook is intentionally removed.
- Dataset loading is made relative to the repository root by default.
"""
from __future__ import annotations

import abc
import datetime
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

try:
    import tensorflow as tf  # type: ignore
    tf.debugging.set_log_device_placement(False)
except Exception:  # TensorFlow is not required by the reproduction scripts.
    tf = None  # type: ignore

pd.options.mode.copy_on_write = True

gpu_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")

RATING_BOUNDS: Tuple[float, float] = (1.0, 5.0)
GLOBAL_SEED = 42

# Keep W&B disabled unless the runner explicitly enables it.
os.environ.setdefault("WANDB_MODE", "disabled")

class _NoOpWandbConfig(dict):
    def update(self, *args, **kwargs):
        return None

class _NoOpWandb:
    class Settings:
        def __init__(self, *args, **kwargs):
            pass

    def __init__(self):
        self.run = None
        self.config = _NoOpWandbConfig()
        self.summary = _NoOpWandbConfig()

    def init(self, *args, **kwargs):
        self.run = object()
        return self.run

    def finish(self, *args, **kwargs):
        self.run = None
        return None

    def log(self, *args, **kwargs):
        return None

    def define_metric(self, *args, **kwargs):
        return None

try:
    import wandb as _wandb  # type: ignore
    wandb = _wandb
except Exception:
    wandb = _NoOpWandb()  # type: ignore


def _repo_root() -> Path:
    # src/recsys_edge/core.py -> repo root is three parents up
    return Path(__file__).resolve().parents[2]


def load_dataset(dataset: str, data_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load a ratings parquet by dataset name.

    Expected filename pattern: ``<dataset>_ratings.parquet``.
    By default, files are read from ``<repo>/datasets``.
    """
    base = Path(data_dir) if data_dir is not None else _repo_root() / "datasets"
    data_path = base / f"{dataset}_ratings.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset parquet not found: {data_path}. "
            "Place the ratings parquet files under the repository datasets/ folder."
        )
    return pd.read_parquet(data_path)

Reduce = Literal['last', 'sum', 'mean', 'max']

def _infer_shape(datasets: Iterable[pd.DataFrame], n_users=None, n_items=None) -> Tuple[int, int]:
    if n_users is None:
        n_users = max(int(df['user_id'].max()) if len(df) else -1 for df in datasets) + 1
    if n_items is None:
        n_items = max(int(df['item_id'].max()) if len(df) else -1 for df in datasets) + 1
    return n_users, n_items

def convert_to_sparse_arr(
    datasets: List[pd.DataFrame],
    n_users: int | None = None,
    n_items: int | None = None,
    rating_col: str = 'rating',
    device: str = 'cpu',
    as_dense: bool = False,
    reduce: Reduce = 'last',
):
    """
    Returns a list of tensors, one per dataset:
      - sparse (default): torch.sparse_coo_tensor of shape (n_users, n_items)
      - dense:            torch.FloatTensor of shape (n_users, n_items)

    Assumes user_id/item_id are already 0-based contiguous (your reindexing does that).
    Handles duplicates via `reduce`: 'last' | 'sum' | 'mean' | 'max'.
    """
    out = []
    n_users, n_items = _infer_shape(datasets, n_users, n_items)

    for df in datasets:
        if df.empty:
            T = torch.sparse_coo_tensor(
                torch.empty((2,0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                (n_users, n_items),
            )
            out.append(T.to_dense() if as_dense else T.coalesce())
            continue

        u_np = df['user_id'].to_numpy(copy=True)
        i_np = df['item_id'].to_numpy(copy=True)

        u = torch.as_tensor(u_np, dtype=torch.long, device=device)
        i = torch.as_tensor(i_np, dtype=torch.long, device=device)
        r = torch.as_tensor(df[rating_col].to_numpy(), dtype=torch.float32, device=device)

        # Unique (u,i) pairs via 1D key; then reduce duplicates
        key = u * n_items + i
        uniq, inv = torch.unique(key, return_inverse=True)
        if reduce == 'last':
            # keep last: stable overwrite by counting occurrences
            # build map index -> last position
            last_pos = torch.zeros(uniq.numel(), dtype=torch.long, device=device)
            last_pos.index_copy_(0, inv, torch.arange(inv.numel(), device=device))
            # gather last values
            r_last = r[last_pos]
            uu = (uniq // n_items).to(torch.long)
            ii = (uniq % n_items).to(torch.long)
            idx = torch.stack([uu, ii], dim=0)
            vals = r_last
        else:
            # segment reductions
            counts = torch.bincount(inv, minlength=uniq.numel()).to(r.dtype)
            sums   = torch.zeros(uniq.numel(), dtype=r.dtype, device=device)
            sums.index_add_(0, inv, r)
            if reduce == 'sum':
                vals = sums
            elif reduce == 'mean':
                vals = sums / counts.clamp_min(1.0)
            elif reduce == 'max':
                # accumulate max
                vals = torch.full((uniq.numel(),), float('-inf'), dtype=r.dtype, device=device)
                # walk once; O(N) without scatter_max in base torch
                # (vectorized fallback)
                order = torch.argsort(inv)
                inv_sorted = inv[order]
                r_sorted   = r[order]
                # find segment starts
                change = torch.ones_like(inv_sorted, dtype=torch.bool)
                change[1:] = inv_sorted[1:] != inv_sorted[:-1]
                seg_idx = torch.cumsum(change, dim=0) - 1  # 0..num_segments-1
                seg_max = torch.zeros_like(vals)
                seg_max.index_put_((inv_sorted,), r_sorted, accumulate=False)  # first per segment
                # do real max with scatter
                vals = torch.zeros_like(vals).index_reduce_(0, inv, r, 'amax')
            else:
                raise ValueError(f"Unknown reduce='{reduce}'")

            uu = (uniq // n_items).to(torch.long)
            ii = (uniq % n_items).to(torch.long)
            idx = torch.stack([uu, ii], dim=0)

        sparse = torch.sparse_coo_tensor(idx, vals, (n_users, n_items))
        out.append(sparse.to_dense() if as_dense else sparse.coalesce())

    return out



def _global_item_reindex(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
    df = df.copy()
    items = np.sort(df['item_id'].unique())
    item_map = {iid: i for i, iid in enumerate(items)}
    df['item_id'] = df['item_id'].map(item_map)
    return df, item_map

def _split_users_three_way(users: np.ndarray, frac_inc: float, frac_online: float, seed: int):
    rng = np.random.default_rng(seed)
    users = users.copy(); rng.shuffle(users)
    n = len(users)
    n_inc = int(round(n * frac_inc))
    n_online = int(round(n * frac_online))
    n_off = n - n_inc - n_online
    off = users[:n_off]; inc = users[n_off:n_off+n_inc]; on = users[n_off+n_inc:]
    return np.sort(off), np.sort(inc), np.sort(on)

def _reindex_users_with_offsets(off_df, inc_df, on_df):
    off_df = off_df.copy(); inc_df = inc_df.copy(); on_df = on_df.copy()

    off_users = np.sort(off_df['user_id'].unique())
    off_map = {u: i for i, u in enumerate(off_users)}
    off_df['user_id'] = off_df['user_id'].map(off_map)
    n_off = len(off_map)

    inc_users = np.sort(inc_df['user_id'].unique())
    inc_map = {u: (n_off + i) for i, u in enumerate(inc_users)}
    inc_df['user_id'] = inc_df['user_id'].map(inc_map)
    n_inc = len(inc_map)

    on_users = np.sort(on_df['user_id'].unique())
    on_map = {u: (n_off + n_inc + i) for i, u in enumerate(on_users)}
    on_df['user_id'] = on_df['user_id'].map(on_map)
    n_on = len(on_map)

    return off_df, inc_df, on_df, {'offline_user_map': off_map, 'incremental_user_map': inc_map, 'online_user_map': on_map}

def _per_user_split(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int,
                    require_test: bool, require_val: bool):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    trains, vals, tests = [], [], []
    for _, g in df.groupby('user_id', sort=False):
        idx = np.arange(len(g)); rng.shuffle(idx); n = len(idx)
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        n_test  = n - n_train - n_val
        if n_test < 0: n_test, n_val = 0, max(0, n - n_train)
        if require_test and n >= 2 and n_test == 0:
            n_test, n_train = 1, max(0, n - 1 - n_val)
        if require_val and n >= 3 and n_val == 0:
            n_val, n_train = 1, max(0, n - n_test - 1)
        cut1, cut2 = n_train, n_train + n_val
        trains.append(g.iloc[idx[:cut1]])
        if n_val  > 0: vals.append(g.iloc[idx[cut1:cut2]])
        if n_test > 0: tests.append(g.iloc[idx[cut2:]])
    def _finish(parts):
        if parts:
            dfc = pd.concat(parts, ignore_index=True)
            return dfc.sort_values(['user_id','item_id']).reset_index(drop=True)
        return df.iloc[[]].copy()
    return _finish(trains), _finish(vals), _finish(tests)


def build_datasets_slim(
    data: pd.DataFrame,
    frac_incremental_users: float = 0.2,
    frac_online_users: float = 0.2,
    seed: int = 42,
    offline_splits = (0.8, 0.1, 0.1),      # train/val/test
    incremental_splits = (0.8, 0.1, 0.1),  # train/val/test
    online_splits = (0.8, 0.2),            # train/test
):
    df = data[['user_id','item_id','rating']].copy()
    df['group'] = data['group'].values if 'group' in data.columns else 0

    # items fixed globally
    df, item_map = _global_item_reindex(df)
    n_items = len(item_map)

    # disjoint user pools
    users = df['user_id'].unique()
    off_users, inc_users, on_users = _split_users_three_way(users, frac_incremental_users, frac_online_users, seed)
    offline_df = df[df['user_id'].isin(off_users)].copy()
    incremental_df = df[df['user_id'].isin(inc_users)].copy()
    online_df = df[df['user_id'].isin(on_users)].copy()

    # contiguous user ids with offsets
    offline_df, incremental_df, online_df, user_maps = _reindex_users_with_offsets(offline_df, incremental_df, online_df)
    n_off, n_inc, n_on = len(user_maps['offline_user_map']), len(user_maps['incremental_user_map']), len(user_maps['online_user_map'])

    # splits
    ot, ov, os = offline_splits
    off_train, off_val, off_test = _per_user_split(offline_df, ot, ov, os, seed, require_test=True, require_val=True)

    it, iv, is_ = incremental_splits
    inc_train, inc_val, inc_test = _per_user_split(incremental_df, it, iv, is_, seed, require_test=True, require_val=True)

    rt, rs = online_splits
    on_train, _, on_test = _per_user_split(online_df, rt, 0.0, rs, seed, require_test=True, require_val=False)

    datasets = {
        "offline": {
            "full": offline_df,
            "train": off_train, "val": off_val, "test": off_test,
        },
        "incremental": {
            "full": incremental_df,
            "train": inc_train, "val": inc_val, "test": inc_test,
        },
        "online": {
            "full": online_df,
            "train": on_train, "test": on_test,
        },
        "meta": {
            "n_items": n_items,
            "n_offline_users": n_off,
            "n_incremental_users": n_inc,
            "n_online_users": n_on,
            **user_maps,
            "item_map": item_map,
            "splits": {
                "offline": offline_splits,
                "incremental": incremental_splits,
                "online": online_splits,
            },
        },
    }

    print(f"[offline] users={n_off} rows: train={len(off_train)}, val={len(off_val)}, test={len(off_test)}")
    print(f"[incremental] users={n_inc} rows: train={len(inc_train)}, val={len(inc_val)}, test={len(inc_test)}")
    print(f"[online] users={n_on} rows: train={len(on_train)}, test={len(on_test)}")
    return datasets

import abc
import datetime
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch


# ========= configs / results =========
@dataclass
class ExperimentEarlyStopRule:
    phase: Literal["offline", "incremental", "offline_post_incremental", "online"]
    metric: Literal["rmse"] = "rmse"
    op: Literal[">=", ">", "<=", "<"] = ">="
    threshold: float = float("inf")
    fill_rmse: float = 1e9
    fill_ranking: float = -1.0
    enabled: bool = True


@dataclass
class ExperimentParams:
    dataset: str
    model_name: str
    seed: int = GLOBAL_SEED
    device: Optional[str] = None  # "cuda", "cpu", or None -> auto

    # Ranking evaluation only
    eval_top_k: Optional[int] = None
    eval_top_ks: Optional[list[int]] = None

    # Backward-compatible single-threshold alias.
    # If relevance_thresholds is None and this is set, that single threshold is used.
    relevance_threshold: Optional[float] = None

    # Preferred multi-threshold ranking config.
    # If both are None -> defaults to [4.0, 5.0]
    relevance_thresholds: Optional[list[float]] = None

    wandb_project: str = "test-run_2"

    # Hyperparams
    model_init: Dict[str, Any] = field(default_factory=dict)
    model_hps: Dict[str, Any] = field(default_factory=dict)
    incremental_hps: Dict[str, Any] = field(default_factory=dict)
    sketch: Dict[str, Any] = field(default_factory=dict)

    # Eval behaviour
    use_clip_rmse: bool = True
    rmse_clip_bounds: Optional[Tuple[float, float]] = RATING_BOUNDS
    online_inference_pred_type: Literal["per_user", "batch"] = "batch"

    # Future algorithm option
    train_on_user_residuals: bool = False
    experiment_early_stop: Optional[ExperimentEarlyStopRule] = None


@dataclass
class PhaseEvaluation:
    summary: Dict[str, Any] = field(default_factory=dict)
    ranking_by_threshold: Dict[float, Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    per_rating_rmse: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def rmse(self) -> Optional[float]:
        v = self.summary.get("rmse")
        return None if v is None else float(v)


@dataclass
class ExperimentEvaluation:
    offline: PhaseEvaluation = field(default_factory=PhaseEvaluation)
    incremental: PhaseEvaluation = field(default_factory=PhaseEvaluation)
    online: PhaseEvaluation = field(default_factory=PhaseEvaluation)
    offline_post_incremental: PhaseEvaluation = field(default_factory=PhaseEvaluation)
    use_clip_rmse: bool = True
    clip_bounds: Optional[Tuple[float, float]] = None

    @property
    def offline_rmse(self) -> Optional[float]:
        return self.offline.rmse

    @property
    def incremental_rmse(self) -> Optional[float]:
        return self.incremental.rmse

    @property
    def online_rmse(self) -> Optional[float]:
        return self.online.rmse

    @property
    def offline_post_incremental_rmse(self) -> Optional[float]:
        return self.offline_post_incremental.rmse


@dataclass
class ExperimentArtifacts:
    logs: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


# ========= BaseExperiment =========

class BaseExperiment(abc.ABC):
    """
    Orchestrates:
      1) fit_offline
      2) fit_incremental
      3) predict/evaluate online
      4) evaluate each phase with:
         - RMSE
         - per-rating RMSE over true ratings
         - NDCG-based ranking metrics only:
             * binary ndcg over multiple relevance thresholds
             * graded/raw-rating ndcg over multiple evaluation Ks

    Clear distinction:
      - params.eval_top_k / params.eval_top_ks:
            used ONLY for ranking evaluation metrics
      - params.model_hps["neighbor_top_k"]:
            used ONLY by the algorithm itself for neighbor retrieval / candidate generation

    Subclasses implement:
      - _init_models
      - _fit_offline
      - _fit_incremental
      - _predict_df

    Important:
      - _predict_df must return predictions in the ORIGINAL rating space.
      - If a subclass trains on user residuals, it should restore predictions back
        to rating space before returning from _predict_df, using the helpers below.
      - RMSE clipping is applied only for RMSE / per-rating RMSE, never for ranking metrics.
    """

    PHASES = ("offline", "incremental", "offline_post_incremental", "online")
    DEFAULT_EVAL_TOP_KS = [2, 3, 5, 7, 10, 15]
    DEFAULT_RELEVANCE_THRESHOLDS = [4.0, 5.0]
    DEFAULT_NEIGHBOR_TOP_K = 20

    def __init__(self, datasets: Dict[str, Dict[str, pd.DataFrame]], params: ExperimentParams):
        self.ds = datasets
        self.params = params

        # Make a defensive copy so we can safely set defaults.
        self.params.model_hps = dict(self.params.model_hps or {})
        self.params.model_hps.setdefault("neighbor_top_k", self.DEFAULT_NEIGHBOR_TOP_K)

        self.eval_ks = self._resolve_eval_ks()
        self.eval_thresholds = self._resolve_eval_thresholds()

        dev_str = params.device if params.device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(dev_str)

        self.artifacts = ExperimentArtifacts()
        self._set_seed(self.params.seed)
        self.run_id = self._make_run_id()

        self.artifacts.logs.update({
            "seed": self.params.seed,
            "device": str(self.device),
            "eval.top_ks": list(self.eval_ks),
            "eval.relevance_thresholds": list(self.eval_thresholds),
            "model.neighbor_top_k": int(self.neighbor_top_k),
            "eval.use_clip_rmse": bool(self.params.use_clip_rmse),
            "eval.rmse_clip_bounds": self._rmse_clip_bounds(),
            "train_on_user_residuals": bool(self.params.train_on_user_residuals),
            "splits.rows": {
                "offline": {
                    k: len(v) for k, v in self.ds["offline"].items()
                    if isinstance(v, pd.DataFrame)
                },
                "incremental": {
                    k: len(v) for k, v in self.ds["incremental"].items()
                    if isinstance(v, pd.DataFrame)
                },
                "online": {
                    k: len(v) for k, v in self.ds["online"].items()
                    if isinstance(v, pd.DataFrame)
                },
            },
        })

        self.artifacts.logs["config.flat"] = self._flat_config()
        self.artifacts.logs["config.model_init"] = dict(self.params.model_init or {})
        self.artifacts.logs["config.model_hps"] = dict(self.params.model_hps or {})
        self.artifacts.logs["config.incremental_hps"] = dict(self.params.incremental_hps or {})
        self.artifacts.logs["config.sketch"] = dict(self.params.sketch or {})

        meta_splits = (self.ds.get("meta") or {}).get("splits", {})
        self.artifacts.logs["splits.ratios.requested"] = meta_splits
        self.artifacts.logs["splits.ratios.realized"] = {
            "offline": self._split_ratios(self.ds["offline"], ("train", "val", "test")),
            "incremental": self._split_ratios(self.ds["incremental"], ("train", "val", "test")),
            "online": self._split_ratios(self.ds["online"], ("train", "test")),
        }

        with self._time("model_init"):
            self._init_models()

    # --- utils ---

    @property
    def neighbor_top_k(self) -> int:
        return int(self.params.model_hps["neighbor_top_k"])

    def _resolve_eval_ks(self) -> list[int]:
        if self.params.eval_top_ks is not None:
            raw = self.params.eval_top_ks
        elif self.params.eval_top_k is not None:
            raw = [self.params.eval_top_k]
        else:
            raw = list(self.DEFAULT_EVAL_TOP_KS)

        ks = sorted({int(k) for k in raw if k is not None})
        if not ks:
            raise ValueError("At least one evaluation K must be provided via eval_top_k or eval_top_ks.")
        if any(k <= 0 for k in ks):
            raise ValueError(f"All evaluation K values must be positive. Got: {ks}")
        return ks

    def _resolve_eval_thresholds(self) -> list[float]:
        if self.params.relevance_thresholds is not None:
            raw = self.params.relevance_thresholds
        elif self.params.relevance_threshold is not None:
            raw = [self.params.relevance_threshold]
        else:
            raw = list(self.DEFAULT_RELEVANCE_THRESHOLDS)

        thresholds = sorted({float(t) for t in raw if t is not None})
        if not thresholds:
            raise ValueError("At least one relevance threshold must be provided.")
        if any(not np.isfinite(t) for t in thresholds):
            raise ValueError(f"All relevance thresholds must be finite. Got: {thresholds}")
        return thresholds

    @staticmethod
    def _threshold_label(threshold: float) -> str:
        t = float(threshold)
        if t.is_integer():
            return f"{t:.1f}"
        return f"{t:g}"

    def _threshold_log_key(self, threshold: float) -> str:
        return f"threshold_{self._threshold_label(threshold).replace('.', '_').replace('-', 'neg_')}"

    @staticmethod
    def _rating_log_key(rating_value: int) -> str:
        return f"rating_{int(rating_value)}"

    def _flat_config(self) -> dict:
        p = asdict(self.params)

        out = {
            "dataset": p.get("dataset"),
            "model_name": p.get("model_name"),
            "seed": p.get("seed"),
            "device": str(self.device),
            "eval_top_ks": list(self.eval_ks),
            "relevance_thresholds": list(self.eval_thresholds),
            "neighbor_top_k": int(self.neighbor_top_k),
            "use_clip_rmse": p.get("use_clip_rmse"),
            "rmse_clip_bounds": list(self._rmse_clip_bounds()) if self._rmse_clip_bounds() is not None else None,
            "train_on_user_residuals": p.get("train_on_user_residuals"),
        }

        def _flatten(prefix: str, value) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    child_prefix = f"{prefix}.{k}" if prefix else str(k)
                    _flatten(child_prefix, v)
                return

            if isinstance(value, tuple):
                value = list(value)

            if isinstance(value, list):
                normalized = []
                for item in value:
                    if isinstance(item, tuple):
                        normalized.append(list(item))
                    elif isinstance(item, dict):
                        normalized.append(str(item))
                    elif isinstance(item, (int, float, str, bool)) or item is None:
                        normalized.append(item)
                    else:
                        normalized.append(str(item))
                out[prefix] = normalized
                return

            if isinstance(value, (int, float, str, bool)) or value is None:
                out[prefix] = value
                return

            out[prefix] = str(value)

        for group in ("model_init", "model_hps", "incremental_hps", "sketch"):
            _flatten(group, p.get(group) or {})

        return out

    def _wandb_define_metrics(self):
        wandb.define_metric("offline/epoch")
        wandb.define_metric("incremental/epoch")

        wandb.define_metric("offline/*", step_metric="offline/epoch")
        wandb.define_metric("incremental/*", step_metric="incremental/epoch")

        pinnables = [
            "offline/rmse",
            "incremental/rmse",
            "online/rmse",
            "offline_post_incremental/rmse",
            "time/model_init_sec",
            "time/offline_train_sec",
            "time/incremental_fit_sec",
            "time/online_predict_sec",
            "time/total_sec",
        ]
        for k in pinnables:
            wandb.define_metric(k, summary="last")

    def _log_dataset_meta_to_wandb(self):
        m = (self.ds or {}).get("meta", {})
        if not m:
            return

        rec = {
            "data/n_items": int(m.get("n_items", 0)),
            "data/offline/n_users": int(m.get("n_offline_users", 0)),
            "data/incremental/n_users": int(m.get("n_incremental_users", 0)),
            "data/online/n_users": int(m.get("n_online_users", 0)),
        }

        splits = (m.get("splits") or {})

        for phase in ("offline", "incremental"):
            vals = splits.get(phase)
            if isinstance(vals, (list, tuple)) and len(vals) == 3:
                rec[f"splits/requested/{phase}/train"] = float(vals[0])
                rec[f"splits/requested/{phase}/val"] = float(vals[1])
                rec[f"splits/requested/{phase}/test"] = float(vals[2])

        vals = splits.get("online")
        if isinstance(vals, (list, tuple)) and len(vals) == 2:
            rec["splits/requested/online/train"] = float(vals[0])
            rec["splits/requested/online/test"] = float(vals[1])

        self.artifacts.logs["data.meta_for_wandb"] = rec
        try:
            if wandb.run is not None:
                wandb.config.update(rec, allow_val_change=True)
        except Exception:
            pass

    def _log_epoch(self, phase: str, epoch: int, **metrics):
        rec = {f"{phase}/epoch": epoch}
        rec.update({f"{phase}/{k}": v for k, v in metrics.items()})
        wandb.log(rec)

    def _log_metrics(self, prefix: str, **metrics):
        wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()})

    def _make_run_id(self) -> str:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        return f"{ts}_{uuid.uuid4().hex[:8]}"

    def _split_ratios(self, split_dict: Dict[str, pd.DataFrame], keys: tuple) -> Dict[str, float]:
        total = sum(len(split_dict.get(k, pd.DataFrame())) for k in keys)
        if total == 0:
            return {k: float("nan") for k in keys}
        return {k: len(split_dict.get(k, pd.DataFrame())) / total for k in keys}

    def _now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    class _Timer:
        def __init__(self, log_dict: Dict[str, Any], key: str):
            self.log_dict = log_dict
            self.key = key

        def __enter__(self):
            self.t0 = time.time()

        def __exit__(self, exc_type, exc, tb):
            self.log_dict[f"time.{self.key}_sec"] = time.time() - self.t0
            return False

    def _time(self, key: str):
        return self._Timer(self.artifacts.logs, key)

    def _rmse_clip_bounds(self) -> Optional[Tuple[float, float]]:
        if not self.params.use_clip_rmse:
            return None
        bounds = self.params.rmse_clip_bounds
        if bounds is None:
            return None
        lo, hi = bounds
        lo = float(lo)
        hi = float(hi)
        if lo >= hi:
            raise ValueError(f"Invalid rmse_clip_bounds: {(lo, hi)}")
        return lo, hi

    def _clip_preds_for_rmse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        bounds = self._rmse_clip_bounds()
        if bounds is None:
            return x

        lo, hi = bounds
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, lo, hi)
        return np.clip(x, lo, hi)

    def _prediction_to_numpy(
        self,
        pred: Union[np.ndarray, torch.Tensor, Sequence[float]],
    ) -> np.ndarray:
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().float().cpu().numpy()
        elif not isinstance(pred, np.ndarray):
            pred = np.asarray(pred, dtype=np.float32)

        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.reshape(-1)
        if pred.ndim != 1:
            raise ValueError(f"Predictions must be rank-1 or shape (N,1). Got shape={pred.shape}.")
        return pred

    def _validate_prediction_source_df(self, df: pd.DataFrame) -> None:
        required = {"user_id", "item_id", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Prediction/eval df is missing required columns: {sorted(missing)}")

    def _make_eval_prediction_df(self, df: pd.DataFrame, *, phase: str) -> pd.DataFrame:
        self._validate_prediction_source_df(df)

        pred = self._predict_df(df=df, phase=phase)
        pred = self._prediction_to_numpy(pred)

        if pred.shape[0] != len(df):
            raise ValueError(
                f"Prediction length mismatch for phase='{phase}': "
                f"got {pred.shape[0]}, expected {len(df)}"
            )

        out = df.loc[:, ["user_id", "item_id", "rating"]].copy()
        out["prediction"] = pred
        return out

    def _reference_train_df_for_phase(self, phase: str) -> Optional[pd.DataFrame]:
        """
        Override this if a subclass needs a different reference frame for
        residual-based training / restoration.
        """
        if phase == "offline":
            return self.ds["offline"].get("train")
        if phase == "incremental":
            return self.ds["incremental"].get("train")
        if phase == "offline_post_incremental":
            return self.ds["offline"].get("train")
        if phase == "online":
            return self.ds["online"].get("train")
        raise ValueError(f"Unknown phase: {phase}")

    def _eval_df_for_phase(self, phase: str) -> pd.DataFrame:
        if phase == "offline":
            return self.ds["offline"]["test"]
        if phase == "incremental":
            return self.ds["incremental"]["test"]
        if phase == "offline_post_incremental":
            return self.ds["offline"]["test"]
        if phase == "online":
            return self.ds["online"]["test"]
        raise ValueError(f"Unknown phase: {phase}")

    # --- helpers for optional residual-based training ---

    def _user_mean_array(
        self,
        df: pd.DataFrame,
        *,
        reference_df: pd.DataFrame,
        fill_with_global_mean: bool = True,
    ) -> np.ndarray:
        if reference_df is None or len(reference_df) == 0:
            raise ValueError("reference_df must be provided and non-empty.")

        if not {"user_id", "rating"}.issubset(reference_df.columns):
            raise ValueError("reference_df must contain ['user_id', 'rating'].")

        user_means = reference_df.groupby("user_id", sort=False)["rating"].mean()
        arr = df["user_id"].map(user_means).to_numpy(dtype=np.float32, copy=False)

        if fill_with_global_mean:
            global_mean = float(reference_df["rating"].mean())
            arr = np.where(np.isnan(arr), np.float32(global_mean), arr)

        return arr.astype(np.float32, copy=False)

    def _training_targets(
        self,
        df: pd.DataFrame,
        *,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        y = df["rating"].to_numpy(dtype=np.float32, copy=False)
        if not self.params.train_on_user_residuals:
            return y

        ref = reference_df if reference_df is not None else df
        base = self._user_mean_array(df, reference_df=ref, fill_with_global_mean=True)
        return (y - base).astype(np.float32, copy=False)

    def _restore_rating_predictions(
        self,
        df: pd.DataFrame,
        pred: Union[np.ndarray, torch.Tensor, Sequence[float]],
        *,
        reference_df: pd.DataFrame,
    ) -> np.ndarray:
        pred_np = self._prediction_to_numpy(pred)
        if not self.params.train_on_user_residuals:
            return pred_np

        base = self._user_mean_array(df, reference_df=reference_df, fill_with_global_mean=True)
        return (pred_np + base).astype(np.float32, copy=False)

    # --- evaluation helpers ---

    @staticmethod
    def _eval_validate_df(df: pd.DataFrame) -> None:
        required = {"user_id", "item_id", "rating", "prediction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df is missing required columns: {sorted(missing)}")

    @staticmethod
    def _eval_group_starts_counts(group_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if group_codes.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, group_codes[1:] != group_codes[:-1]]).astype(np.int64, copy=False)
        counts = np.diff(np.r_[starts, group_codes.size]).astype(np.int64, copy=False)
        return starts, counts

    @staticmethod
    def _eval_clip_predictions(
        preds: np.ndarray,
        clip_bounds: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        if clip_bounds is None:
            return preds
        lo, hi = clip_bounds
        return np.clip(preds, lo, hi)

    def _eval_prepare_binary_desc(
        self,
        df: pd.DataFrame,
        relevance_threshold: float,
    ) -> dict:
        self._eval_validate_df(df)

        user_codes, _ = pd.factorize(df["user_id"], sort=False)
        ratings = df["rating"].to_numpy(dtype=np.float64, copy=False)
        preds = df["prediction"].to_numpy(dtype=np.float64, copy=False)
        y = (ratings >= relevance_threshold).astype(np.int8, copy=False)

        order = np.lexsort((-preds, user_codes))
        u = user_codes[order]
        y = y[order]

        starts, counts = self._eval_group_starts_counts(u)
        row_pos = np.arange(len(u), dtype=np.int64) - np.repeat(starts, counts)
        pos_counts = np.add.reduceat(y.astype(np.int64, copy=False), starts)

        return {
            "starts": starts,
            "counts": counts,
            "row_pos": row_pos,
            "y_sorted": y,
            "pos_counts": pos_counts,
            "n_users": int(len(starts)),
        }

    def _eval_prepare_graded_ndcg(
        self,
        df: pd.DataFrame,
        *,
        k: int,
    ) -> dict:
        self._eval_validate_df(df)

        user_codes, _ = pd.factorize(df["user_id"], sort=False)
        ratings = df["rating"].to_numpy(dtype=np.float64, copy=False)
        preds = df["prediction"].to_numpy(dtype=np.float64, copy=False)

        order_pred = np.lexsort((-preds, user_codes))
        u_pred = user_codes[order_pred]
        gains_pred = ratings[order_pred]

        starts, counts = self._eval_group_starts_counts(u_pred)
        row_pos_pred = np.arange(len(u_pred), dtype=np.int64) - np.repeat(starts, counts)

        max_k = int(min(k, counts.max())) if counts.size else 0
        if max_k == 0:
            return {
                "graded_ndcg_vals": np.full(len(starts), np.nan, dtype=np.float64),
                "valid_graded_ndcg": np.zeros(len(starts), dtype=bool),
            }

        discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))

        dcg_weights = np.zeros(len(u_pred), dtype=np.float64)
        in_top_pred = row_pos_pred < max_k
        dcg_weights[in_top_pred] = discounts[row_pos_pred[in_top_pred]]
        dcg = np.add.reduceat(gains_pred * dcg_weights, starts)

        order_ideal = np.lexsort((-ratings, user_codes))
        u_ideal = user_codes[order_ideal]
        gains_ideal = ratings[order_ideal]
        starts_ideal, counts_ideal = self._eval_group_starts_counts(u_ideal)
        row_pos_ideal = np.arange(len(u_ideal), dtype=np.int64) - np.repeat(starts_ideal, counts_ideal)

        idcg_weights = np.zeros(len(u_ideal), dtype=np.float64)
        in_top_ideal = row_pos_ideal < max_k
        idcg_weights[in_top_ideal] = discounts[row_pos_ideal[in_top_ideal]]
        idcg = np.add.reduceat(gains_ideal * idcg_weights, starts_ideal)

        valid = idcg > 0
        graded_ndcg_vals = np.full(len(starts), np.nan, dtype=np.float64)
        graded_ndcg_vals[valid] = dcg[valid] / idcg[valid]

        return {
            "graded_ndcg_vals": graded_ndcg_vals,
            "valid_graded_ndcg": valid,
        }

    def _eval_summary(
        self,
        df: pd.DataFrame,
        *,
        clip_bounds: Optional[Tuple[float, float]],
    ) -> Dict[str, Any]:
        self._eval_validate_df(df)

        if df.empty:
            return {
                "clip_bounds": clip_bounds,
                "n_rows": 0,
                "n_users_total": 0,
                "n_prediction_nans": 0,
                "rmse": np.nan,
            }

        pred_raw = df["prediction"].to_numpy(dtype=np.float64, copy=False)
        rating_arr = df["rating"].to_numpy(dtype=np.float64, copy=False)
        pred_for_rmse = self._eval_clip_predictions(pred_raw, clip_bounds)

        err = rating_arr - pred_for_rmse
        rmse = float(np.sqrt(np.mean(err ** 2)))

        return {
            "clip_bounds": clip_bounds,
            "n_rows": int(len(df)),
            "n_users_total": int(df["user_id"].nunique()),
            "n_prediction_nans": int(np.isnan(pred_raw).sum()),
            "rmse": rmse,
        }

    def _eval_per_rating_rmse(
        self,
        df: pd.DataFrame,
        *,
        clip_bounds: Optional[Tuple[float, float]],
    ) -> Dict[int, Dict[str, Any]]:
        self._eval_validate_df(df)

        if df.empty:
            return {}

        pred_raw = df["prediction"].to_numpy(dtype=np.float64, copy=False)
        rating_arr = df["rating"].to_numpy(dtype=np.float64, copy=False)
        pred_for_rmse = self._eval_clip_predictions(pred_raw, clip_bounds)

        # Assumes true ratings are integer-valued in [1, 5] (possibly stored as float).
        rating_int = np.rint(rating_arr).astype(np.int64, copy=False)

        out: Dict[int, Dict[str, Any]] = {}
        for r in sorted(np.unique(rating_int).tolist()):
            mask = (rating_int == r)
            total = int(mask.sum())
            if total == 0:
                continue

            err = rating_arr[mask] - pred_for_rmse[mask]
            out[int(r)] = {
                "total": total,
                "rmse": float(np.sqrt(np.mean(err ** 2))),
            }

        return out

    def _eval_one_split(
        self,
        df: pd.DataFrame,
        *,
        k: int,
        relevance_threshold: float,
    ) -> dict:
        self._eval_validate_df(df)

        if df.empty:
            return {
                "relevance_threshold": float(relevance_threshold),
                "k": int(k),
                "n_users_used_binary": 0,
                "n_users_used_graded_ndcg": 0,
                "ndcg": np.nan,
                "ndcg_raw_rating": np.nan,
            }

        binary = self._eval_prepare_binary_desc(df, relevance_threshold=relevance_threshold)

        starts = binary["starts"]
        counts = binary["counts"]
        row_pos = binary["row_pos"]
        y = binary["y_sorted"]
        pos_counts = binary["pos_counts"]
        n_users = binary["n_users"]

        valid_binary = pos_counts > 0
        ndcg_vals = np.full(n_users, np.nan, dtype=np.float64)

        max_k = int(min(k, counts.max())) if counts.size else 0
        if max_k > 0:
            discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))

            dcg_weights = np.zeros(len(y), dtype=np.float64)
            in_top = row_pos < max_k
            dcg_weights[in_top] = discounts[row_pos[in_top]]
            dcg = np.add.reduceat(y.astype(np.float64, copy=False) * dcg_weights, starts)

            cum_discounts = np.r_[0.0, np.cumsum(discounts)]
            idcg = cum_discounts[np.minimum(pos_counts, max_k)]

            ndcg_vals[valid_binary] = dcg[valid_binary] / idcg[valid_binary]

        graded = self._eval_prepare_graded_ndcg(df, k=k)
        graded_ndcg_vals = graded["graded_ndcg_vals"]
        valid_graded_ndcg = graded["valid_graded_ndcg"]

        return {
            "relevance_threshold": float(relevance_threshold),
            "k": int(k),
            "n_users_used_binary": int(valid_binary.sum()),
            "n_users_used_graded_ndcg": int(valid_graded_ndcg.sum()),
            "ndcg": float(np.nanmean(ndcg_vals)) if np.isfinite(ndcg_vals).any() else np.nan,
            "ndcg_raw_rating": float(np.nanmean(graded_ndcg_vals)) if np.isfinite(graded_ndcg_vals).any() else np.nan,
        }

    def _metric_compare(self, value: float, op: str, threshold: float) -> bool:
        if op == ">=":
            return value >= threshold
        if op == ">":
            return value > threshold
        if op == "<=":
            return value <= threshold
        if op == "<":
            return value < threshold
        raise ValueError(f"Unsupported early-stop op: {op}")

    def _maybe_trigger_experiment_early_stop(
        self,
        *,
        phase: str,
        phase_eval: PhaseEvaluation,
    ) -> Optional[Dict[str, Any]]:
        rule = self.params.experiment_early_stop
        if rule is None or not rule.enabled:
            return None
        if rule.phase != phase:
            return None
        if rule.metric != "rmse":
            raise ValueError(f"Unsupported early-stop metric: {rule.metric}")

        value = phase_eval.summary.get("rmse")
        if value is None or not np.isfinite(value):
            return None

        value = float(value)
        if not self._metric_compare(value, rule.op, float(rule.threshold)):
            return None

        return {
            "phase": phase,
            "metric": rule.metric,
            "op": rule.op,
            "threshold": float(rule.threshold),
            "value": value,
            "fill_rmse": float(rule.fill_rmse),
            "fill_ranking": float(rule.fill_ranking),
        }

    def _fill_remaining_phases_as_irrelevant(
        self,
        results: ExperimentEvaluation,
        *,
        stop_info: Dict[str, Any],
    ) -> None:
        phase_order = list(self.PHASES)
        stopped_after = stop_info["phase"]
        start_idx = phase_order.index(stopped_after) + 1

        self.artifacts.logs["experiment_early_stop"] = {
            "triggered": True,
            "phase": stopped_after,
            "metric": stop_info["metric"],
            "op": stop_info["op"],
            "threshold": stop_info["threshold"],
            "value": stop_info["value"],
        }

        try:
            wandb.log({
                "experiment_early_stop/triggered": 1,
                "experiment_early_stop/phase": stopped_after,
                "experiment_early_stop/threshold": stop_info["threshold"],
                "experiment_early_stop/value": stop_info["value"],
            })
        except Exception:
            pass

        for phase in phase_order[start_idx:]:
            phase_eval = self._make_irrelevant_phase_evaluation(
                fill_rmse=stop_info["fill_rmse"],
                fill_ranking=stop_info["fill_ranking"],
                stopped_after=stopped_after,
                reason=f"experiment early-stopped after {stopped_after}",
                trigger_metric=stop_info["metric"],
                trigger_value=stop_info["value"],
                trigger_threshold=stop_info["threshold"],
            )
            setattr(results, phase, phase_eval)
            self._store_phase_evaluation(phase, phase_eval)

    def _empty_phase_evaluation(self) -> PhaseEvaluation:
        return PhaseEvaluation(
            summary={
                "clip_bounds": self._rmse_clip_bounds(),
                "n_rows": 0,
                "n_users_total": 0,
                "n_prediction_nans": 0,
                "rmse": np.nan,
            },
            ranking_by_threshold={
                float(th): {
                    int(k): {
                        "n_users_used_binary": 0,
                        "n_users_used_graded_ndcg": 0,
                        "ndcg": np.nan,
                        "ndcg_raw_rating": np.nan,
                    }
                    for k in self.eval_ks
                }
                for th in self.eval_thresholds
            },
            per_rating_rmse={},
        )

    def _evaluate_phase(self, phase: str) -> PhaseEvaluation:
        df = self._eval_df_for_phase(phase)
        if df is None or len(df) == 0:
            return self._empty_phase_evaluation()

        eval_df = self._make_eval_prediction_df(df, phase=phase)

        clip_bounds = self._rmse_clip_bounds()
        summary = self._eval_summary(eval_df, clip_bounds=clip_bounds)
        per_rating_rmse = self._eval_per_rating_rmse(eval_df, clip_bounds=clip_bounds)

        ranking_by_threshold: Dict[float, Dict[int, Dict[str, Any]]] = {}

        for threshold in self.eval_thresholds:
            threshold_metrics: Dict[int, Dict[str, Any]] = {}

            for k in self.eval_ks:
                split_metrics = self._eval_one_split(
                    eval_df,
                    k=int(k),
                    relevance_threshold=float(threshold),
                )

                threshold_metrics[int(k)] = {
                    "n_users_used_binary": int(split_metrics["n_users_used_binary"]),
                    "n_users_used_graded_ndcg": int(split_metrics["n_users_used_graded_ndcg"]),
                    "ndcg": float(split_metrics["ndcg"]) if split_metrics["ndcg"] is not None else np.nan,
                    "ndcg_raw_rating": (
                        float(split_metrics["ndcg_raw_rating"])
                        if split_metrics["ndcg_raw_rating"] is not None
                        else np.nan
                    ),
                }

            ranking_by_threshold[float(threshold)] = threshold_metrics

        return PhaseEvaluation(
            summary=summary,
            ranking_by_threshold=ranking_by_threshold,
            per_rating_rmse=per_rating_rmse,
        )

    def _wandb_phase_record(self, phase: str, phase_eval: PhaseEvaluation) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}

        for key, value in phase_eval.summary.items():
            if key == "clip_bounds":
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float, bool)) or value is None:
                rec[f"{phase}/{key}"] = value

        for rating_value, metrics in phase_eval.per_rating_rmse.items():
            rating_key = self._rating_log_key(int(rating_value))
            for key, value in metrics.items():
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (int, float, bool)) or value is None:
                    rec[f"{phase}/per_rating_rmse/{rating_key}/{key}"] = value

        for threshold, by_k in phase_eval.ranking_by_threshold.items():
            th_key = self._threshold_log_key(float(threshold))
            for k, metrics in by_k.items():
                for key, value in metrics.items():
                    if isinstance(value, np.generic):
                        value = value.item()
                    if isinstance(value, (int, float, bool)) or value is None:
                        rec[f"{phase}/{th_key}/top_{int(k)}/{key}"] = value

        return rec

    def _store_phase_evaluation(self, phase: str, phase_eval: PhaseEvaluation) -> None:
        self.artifacts.logs.setdefault("evaluation", {})[phase] = asdict(phase_eval)
        self.artifacts.logs[f"rmse.{phase}"] = phase_eval.summary.get("rmse")

        rec = self._wandb_phase_record(phase, phase_eval)
        if rec:
            wandb.log(rec)

    def _write_final_summary(self, results: ExperimentEvaluation):
        rec = {}
        for phase in self.PHASES:
            rec.update(self._wandb_phase_record(phase, getattr(results, phase)))

        rec.update({
            "time/model_init_sec": self.artifacts.logs.get("time.model_init_sec"),
            "time/offline_train_sec": self.artifacts.logs.get("time.offline_train_sec"),
            "time/incremental_fit_sec": self.artifacts.logs.get("time/incremental_fit_sec"),
            "time/online_predict_sec": self.artifacts.logs.get("time.online_predict_sec"),
            "time/total_sec": self.artifacts.logs.get("time.total_sec"),
        })
        wandb.summary.update(rec)

    # --- main run ---

    def _make_irrelevant_phase_evaluation(
        self,
        *,
        fill_rmse: float,
        fill_ranking: float,
        stopped_after: str,
        reason: str,
        trigger_metric: str,
        trigger_value: float,
        trigger_threshold: float,
    ) -> PhaseEvaluation:
        return PhaseEvaluation(
            summary={
                "clip_bounds": self._rmse_clip_bounds(),
                "n_rows": -1,
                "n_users_total": -1,
                "n_prediction_nans": -1,
                "rmse": float(fill_rmse),
                "status": "skipped_irrelevant",
                "reason": reason,
                "stopped_after": stopped_after,
                "trigger_metric": trigger_metric,
                "trigger_value": float(trigger_value),
                "trigger_threshold": float(trigger_threshold),
            },
            ranking_by_threshold={
                float(th): {
                    int(k): {
                        "n_users_used_binary": 0,
                        "n_users_used_graded_ndcg": 0,
                        "ndcg": float(fill_ranking),
                        "ndcg_raw_rating": float(fill_ranking),
                    }
                    for k in self.eval_ks
                }
                for th in self.eval_thresholds
            },
            per_rating_rmse={},
        )

    def run(self) -> Tuple[ExperimentEvaluation, ExperimentArtifacts]:
        wandb.init(
            project=self.params.wandb_project,
            config=self._flat_config(),
            name=f"{self.params.model_name}_{self.run_id}",
            settings=wandb.Settings(
                silent=True,
                console="off",
            ),
        )
        self._wandb_define_metrics()
        self._log_dataset_meta_to_wandb()

        self.artifacts.started_at = self._now_iso()

        results = ExperimentEvaluation(
            use_clip_rmse=self.params.use_clip_rmse,
            clip_bounds=self._rmse_clip_bounds(),
        )

        stop_info = None

        with self._time("total"):
            results.offline = self.fit_offline_and_eval()
            stop_info = self._maybe_trigger_experiment_early_stop(
                phase="offline",
                phase_eval=results.offline,
            )

            if stop_info is None:
                results.incremental, results.offline_post_incremental = self.fit_incremental_and_eval()

                stop_info = self._maybe_trigger_experiment_early_stop(
                    phase="incremental",
                    phase_eval=results.incremental,
                )

                if stop_info is None:
                    stop_info = self._maybe_trigger_experiment_early_stop(
                        phase="offline_post_incremental",
                        phase_eval=results.offline_post_incremental,
                    )

            if stop_info is None:
                results.online = self.predict_online_and_eval()

                stop_info = self._maybe_trigger_experiment_early_stop(
                    phase="online",
                    phase_eval=results.online,
                )

            if stop_info is not None:
                self._fill_remaining_phases_as_irrelevant(
                    results,
                    stop_info=stop_info,
                )

        self.artifacts.ended_at = self._now_iso()

        wandb.log({
            "time/model_init_sec": self.artifacts.logs.get("time.model_init_sec"),
            "time/offline_train_sec": self.artifacts.logs.get("time.offline_train_sec"),
            "time/incremental_fit_sec": self.artifacts.logs.get("time.incremental_fit_sec"),
            "time/online_predict_sec": self.artifacts.logs.get("time.online_predict_sec"),
            "time/total_sec": self.artifacts.logs.get("time.total_sec"),
        })

        self._write_final_summary(results)
        wandb.finish()

        return results, self.artifacts

    # --- phases ---

    def fit_offline_and_eval(self) -> PhaseEvaluation:
        with self._time("offline_train"):
            self._fit_offline()

        phase_eval = self.evaluate_offline()
        self._store_phase_evaluation("offline", phase_eval)
        return phase_eval

    def fit_incremental_and_eval(self) -> Tuple[PhaseEvaluation, PhaseEvaluation]:
        with self._time("incremental_fit"):
            self._fit_incremental()

        inc_eval = self.evaluate_incremental()
        self._store_phase_evaluation("incremental", inc_eval)

        off_post_eval = self.evaluate_offline_post_incremental()
        self._store_phase_evaluation("offline_post_incremental", off_post_eval)

        return inc_eval, off_post_eval

    def predict_online_and_eval(self) -> PhaseEvaluation:
        with self._time("online_predict"):
            phase_eval = self.evaluate_online()

        self._store_phase_evaluation("online", phase_eval)
        return phase_eval

    # --- abstract hooks implemented by subclasses ---

    @abc.abstractmethod
    def _init_models(self) -> None:
        ...

    @abc.abstractmethod
    def _fit_offline(self) -> None:
        ...

    @abc.abstractmethod
    def _fit_incremental(self) -> None:
        ...

    @abc.abstractmethod
    def _predict_df(
        self,
        df: pd.DataFrame,
        *,
        phase: str,
    ) -> Union[np.ndarray, torch.Tensor, Sequence[float]]:
        """
        Return one prediction per row in df, in the same order as df.

        Expected phases:
          - "offline"
          - "incremental"
          - "offline_post_incremental"
          - "online"

        IMPORTANT:
        Return predictions in original rating space.
        If the model internally predicts user-residuals, convert them back with:
            self._restore_rating_predictions(
                df,
                pred,
                reference_df=self._reference_train_df_for_phase(phase),
            )
        """
        ...

    # --- default evaluators ---

    def evaluate_offline(self) -> PhaseEvaluation:
        return self._evaluate_phase("offline")

    def evaluate_incremental(self) -> PhaseEvaluation:
        return self._evaluate_phase("incremental")

    def evaluate_offline_post_incremental(self) -> PhaseEvaluation:
        return self._evaluate_phase("offline_post_incremental")

    def evaluate_online(self) -> PhaseEvaluation:
        return self._evaluate_phase("online")

    # --- reproducibility ---

    def _set_seed(self, seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch


def _torch_dtype_to_key(dt: torch.dtype) -> str:
    if dt == torch.float16:
        return "float16"
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.float32:
        return "float32"
    if dt == torch.float64:
        return "float64"
    if dt == torch.int64:
        return "int64"
    if dt == torch.int32:
        return "int32"
    if dt == torch.int8:
        return "int8"
    if dt == torch.bool:
        return "bool"
    return str(dt).replace("torch.", "")


# ----------------------------
# Phases
# ----------------------------
class Phase(str, Enum):
    OFFLINE = "offline"
    INCREMENTAL = "incremental"
    ONLINE = "online"


# ----------------------------
# Byte helpers
# ----------------------------
_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


def _bytes_dense(shape: Tuple[int, ...], dtype: str) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * _DTYPE_BYTES[dtype]


# ----------------------------
# Components / Snapshots
# ----------------------------
@dataclass(frozen=True)
class SpaceComponent:
    """
    key:
      Stable id for "the same persisted object" across phases.

    Delta logic:
      - If a key is new in phase P: its full bytes count toward delta(P).
      - If a key existed previously and its bytes change: delta(P) includes the difference.
      - If a key disappears: delta(P) includes negative bytes (rare; but supported).
    """
    key: str
    name: str
    bytes: int
    phase: Phase

    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    formula: Optional[str] = None
    note: Optional[str] = None


@dataclass
class SpaceSnapshot:
    """
    Snapshot = the full persisted state you assume exists AFTER the given phase.
    """
    components: List[SpaceComponent] = field(default_factory=list)

    def total_bytes(self) -> int:
        return sum(c.bytes for c in self.components)

    def by_key(self) -> Dict[str, SpaceComponent]:
        out: Dict[str, SpaceComponent] = {}
        for c in self.components:
            if c.key in out:
                raise ValueError(f"Duplicate component key in snapshot ({c.phase.value}): {c.key}")
            out[c.key] = c
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.total_bytes(),
            "components": [
                {
                    "key": c.key,
                    "name": c.name,
                    "bytes": c.bytes,
                    "phase": c.phase.value,
                    "shape": c.shape,
                    "dtype": c.dtype,
                    "formula": c.formula,
                    "note": c.note,
                }
                for c in self.components
            ],
        }


# ----------------------------
# Report: per-phase total + delta
# ----------------------------
@dataclass(frozen=True)
class PhaseSpaceTotals:
    total_bytes: int
    delta_bytes: int
    per_key_delta_bytes: Dict[str, int]


def build_phase_space_totals(space_by_phase: Dict[Phase, SpaceSnapshot]) -> Dict[Phase, PhaseSpaceTotals]:
    """
    Returns for each phase:
      - total_bytes: total persisted memory after phase
      - delta_bytes: (total_bytes - previous_phase_total_bytes)
      - per_key_delta_bytes: diff per component key (sums to delta_bytes)
    """
    order = [Phase.OFFLINE, Phase.INCREMENTAL, Phase.ONLINE]
    out: Dict[Phase, PhaseSpaceTotals] = {}

    prev_map: Dict[str, int] = {}
    prev_total = 0

    for p in order:
        snap = space_by_phase.get(p)
        if snap is None:
            out[p] = PhaseSpaceTotals(
                total_bytes=prev_total,
                delta_bytes=0,
                per_key_delta_bytes={},
            )
            continue

        cur_map = {k: c.bytes for k, c in snap.by_key().items()}
        total = sum(cur_map.values())

        per_key_delta: Dict[str, int] = {}
        all_keys = set(prev_map) | set(cur_map)
        for k in all_keys:
            per_key_delta[k] = cur_map.get(k, 0) - prev_map.get(k, 0)

        delta = total - prev_total

        out[p] = PhaseSpaceTotals(
            total_bytes=total,
            delta_bytes=delta,
            per_key_delta_bytes=per_key_delta,
        )

        prev_map = cur_map
        prev_total = total

    return out


# ----------------------------
# Base experiment wrapper
# ----------------------------
class SpaceTrackedExperimentBase(BaseExperiment, abc.ABC):
    """
    Extends BaseExperiment with persisted-space accounting.

    Logs into artifacts.logs:
      - space.offline
      - space.incremental
      - space.online
      - space.phase_totals

    Logs into wandb.summary:
      - space/offline/bytes_total
      - space/offline/bytes_delta
      - space/incremental/bytes_total
      - space/incremental/bytes_delta
      - space/online/bytes_total
      - space/online/bytes_delta
    """

    def __init__(self, datasets, params, *, count_completed_artifacts: bool = False):
        self.count_completed_artifacts = count_completed_artifacts
        self.space_by_phase: Dict[Phase, SpaceSnapshot] = {}
        super().__init__(datasets, params)

    @abc.abstractmethod
    def _build_space_snapshot(self, phase: Phase) -> SpaceSnapshot:
        ...

    def fit_offline_and_eval(self) -> PhaseEvaluation:
        phase_eval = super().fit_offline_and_eval()

        snap = self._build_space_snapshot(Phase.OFFLINE)
        self.space_by_phase[Phase.OFFLINE] = snap
        self.artifacts.logs["space.offline"] = snap.as_dict()

        return phase_eval

    def fit_incremental_and_eval(self) -> Tuple[PhaseEvaluation, PhaseEvaluation]:
        inc_eval, off_post_eval = super().fit_incremental_and_eval()

        snap = self._build_space_snapshot(Phase.INCREMENTAL)
        self.space_by_phase[Phase.INCREMENTAL] = snap
        self.artifacts.logs["space.incremental"] = snap.as_dict()

        return inc_eval, off_post_eval

    def predict_online_and_eval(self) -> PhaseEvaluation:
        phase_eval = super().predict_online_and_eval()

        snap = self._build_space_snapshot(Phase.ONLINE)
        self.space_by_phase[Phase.ONLINE] = snap
        self.artifacts.logs["space.online"] = snap.as_dict()

        return phase_eval

    def _write_final_summary(self, results: ExperimentEvaluation):
        super()._write_final_summary(results)
        self.artifacts.logs["space.early_stop"] = self.artifacts.logs.get("experiment_early_stop")

        totals = build_phase_space_totals(self.space_by_phase)
        payload = {
            p.value: {
                "total_bytes": t.total_bytes,
                "delta_bytes": t.delta_bytes,
                "per_key_delta_bytes": t.per_key_delta_bytes,
            }
            for p, t in totals.items()
        }
        self.artifacts.logs["space.phase_totals"] = payload

        try:
            wandb.summary.update({
                "space/offline/bytes_total": totals[Phase.OFFLINE].total_bytes,
                "space/offline/bytes_delta": totals[Phase.OFFLINE].delta_bytes,
                "space/incremental/bytes_total": totals[Phase.INCREMENTAL].total_bytes,
                "space/incremental/bytes_delta": totals[Phase.INCREMENTAL].delta_bytes,
                "space/online/bytes_total": totals[Phase.ONLINE].total_bytes,
                "space/online/bytes_delta": totals[Phase.ONLINE].delta_bytes,
            })
        except Exception:
            pass


_PHASE_NAMES = ("offline", "incremental", "offline_post_incremental", "online")


def _round_numeric_df(df: pd.DataFrame, round_digits: Optional[int]) -> pd.DataFrame:
    if round_digits is None or df.empty:
        return df
    num_cols = df.select_dtypes(include=[np.number]).columns
    df = df.copy()
    df[num_cols] = df[num_cols].round(round_digits)
    return df


def _sorted_threshold_items(
    ranking_by_threshold: Dict[float, Dict[int, Dict[str, Any]]]
) -> list[tuple[float, Dict[int, Dict[str, Any]]]]:
    if not ranking_by_threshold:
        return []
    return sorted(
        [(float(th), by_k or {}) for th, by_k in ranking_by_threshold.items()],
        key=lambda x: x[0],
    )


def _match_threshold_key(
    ranking_by_threshold: Dict[float, Dict[int, Dict[str, Any]]],
    threshold: float,
) -> Optional[float]:
    if threshold in ranking_by_threshold:
        return threshold
    for th in ranking_by_threshold:
        if np.isclose(float(th), float(threshold)):
            return th
    return None


def _resolve_summary_threshold(
    ranking_by_threshold: Dict[float, Dict[int, Dict[str, Any]]],
    pick_threshold: Optional[float],
) -> Optional[float]:
    if not ranking_by_threshold:
        return None

    if pick_threshold is not None:
        return _match_threshold_key(ranking_by_threshold, float(pick_threshold))

    available = sorted(float(th) for th in ranking_by_threshold.keys())

    # Backward-friendly default: prefer 4.0 if present, else smallest threshold.
    for candidate in available:
        if np.isclose(candidate, 4.0):
            return candidate
    return available[0]


def build_phase_results_tables(
    results,
    *,
    round_digits: Optional[int] = 6,
    include_counts: bool = True,
    thresholds: Optional[Iterable[float]] = None,
    ks: Optional[Iterable[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build one result table per phase from:
        results, artifacts = exp.run()

    Expected `results` type:
        ExperimentEvaluation

    Returns:
        {
            "offline": pd.DataFrame,
            "incremental": pd.DataFrame,
            "offline_post_incremental": pd.DataFrame,
            "online": pd.DataFrame,
        }

    Each phase table contains:
      - one summary row
      - one row per (threshold, K)
    """
    out: Dict[str, pd.DataFrame] = {}

    threshold_filter = None if thresholds is None else [float(t) for t in thresholds]
    k_filter = None if ks is None else {int(k) for k in ks}

    for phase in _PHASE_NAMES:
        phase_eval = getattr(results, phase, None)
        if phase_eval is None:
            out[phase] = pd.DataFrame()
            continue

        summary = getattr(phase_eval, "summary", {}) or {}
        ranking_by_threshold = getattr(phase_eval, "ranking_by_threshold", {}) or {}

        rows = []

        # summary row
        summary_row = {
            "phase": phase,
            "row_type": "summary",
            "threshold": np.nan,
            "k": np.nan,
            "rmse": summary.get("rmse", np.nan),
        }

        if include_counts:
            summary_row.update({
                "n_rows": summary.get("n_rows", np.nan),
                "n_users_total": summary.get("n_users_total", np.nan),
                "n_prediction_nans": summary.get("n_prediction_nans", np.nan),
            })

        rows.append(summary_row)

        # ranking rows
        threshold_items = _sorted_threshold_items(ranking_by_threshold)

        if threshold_filter is not None:
            filtered_items = []
            seen = set()
            for requested in threshold_filter:
                matched = _match_threshold_key(ranking_by_threshold, requested)
                if matched is not None and float(matched) not in seen:
                    filtered_items.append((float(matched), ranking_by_threshold[matched] or {}))
                    seen.add(float(matched))
            threshold_items = filtered_items

        for threshold, by_k in threshold_items:
            for k in sorted(by_k):
                if k_filter is not None and int(k) not in k_filter:
                    continue

                m = by_k[k] or {}

                row = {
                    "phase": phase,
                    "row_type": "ranking",
                    "threshold": float(threshold),
                    "k": int(k),
                    "rmse": summary.get("rmse", np.nan),
                    "precision": m.get("precision", np.nan),
                    "recall": m.get("recall", np.nan),
                    "map": m.get("map", np.nan),
                    "ndcg": m.get("ndcg", np.nan),
                    "ndcg_raw_rating": m.get("ndcg_raw_rating", np.nan),
                    "auc": m.get("auc", np.nan),
                }

                if include_counts:
                    row.update({
                        "n_rows": summary.get("n_rows", np.nan),
                        "n_users_total": summary.get("n_users_total", np.nan),
                        "n_prediction_nans": summary.get("n_prediction_nans", np.nan),
                        "n_users_used_binary": m.get("n_users_used_binary", np.nan),
                        "n_users_used_auc": m.get("n_users_used_auc", np.nan),
                        "n_users_used_graded_ndcg": m.get("n_users_used_graded_ndcg", np.nan),
                    })

                rows.append(row)

        df = pd.DataFrame(rows)

        preferred_cols = [
            "phase",
            "row_type",
            "threshold",
            "k",
            "rmse",
            "precision",
            "recall",
            "map",
            "ndcg",
            "ndcg_raw_rating",
            "auc",
            "n_rows",
            "n_users_total",
            "n_prediction_nans",
            "n_users_used_binary",
            "n_users_used_auc",
            "n_users_used_graded_ndcg",
        ]
        df = df[[c for c in preferred_cols if c in df.columns]]
        df = _round_numeric_df(df, round_digits)
        out[phase] = df

    return out


def build_combined_results_table(
    results,
    *,
    round_digits: Optional[int] = 6,
    include_counts: bool = True,
    thresholds: Optional[Iterable[float]] = None,
    ks: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Stack all phase tables into one dataframe.
    """
    phase_tables = build_phase_results_tables(
        results,
        round_digits=round_digits,
        include_counts=include_counts,
        thresholds=thresholds,
        ks=ks,
    )
    non_empty = [df for df in phase_tables.values() if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def print_phase_results_tables(
    results,
    *,
    round_digits: Optional[int] = 6,
    include_counts: bool = True,
    thresholds: Optional[Iterable[float]] = None,
    ks: Optional[Iterable[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience printer for notebook usage.
    """
    phase_tables = build_phase_results_tables(
        results,
        round_digits=round_digits,
        include_counts=include_counts,
        thresholds=thresholds,
        ks=ks,
    )

    for phase, df in phase_tables.items():
        print(f"\n=== {phase.upper()} ===")
        if df.empty:
            print("(empty)")
        else:
            print(df.to_string(index=False))

    return phase_tables


def build_phase_summary_table(
    results,
    *,
    pick_threshold: Optional[float] = None,
    pick_k: Optional[int] = None,
    round_digits: Optional[int] = 6,
) -> pd.DataFrame:
    """
    One row per phase.

    If pick_k is given, also pulls ranking metrics for one threshold/K pair.
    If pick_threshold is None and ranking metrics are requested, the function:
      - prefers threshold 4.0 if present
      - otherwise uses the smallest available threshold
    """
    rows = []

    for phase in _PHASE_NAMES:
        phase_eval = getattr(results, phase, None)
        if phase_eval is None:
            continue

        summary = getattr(phase_eval, "summary", {}) or {}
        ranking_by_threshold = getattr(phase_eval, "ranking_by_threshold", {}) or {}

        row = {
            "phase": phase,
            "rmse": summary.get("rmse", np.nan),
            "n_rows": summary.get("n_rows", np.nan),
            "n_users_total": summary.get("n_users_total", np.nan),
            "n_prediction_nans": summary.get("n_prediction_nans", np.nan),
        }

        if pick_k is not None:
            resolved_threshold = _resolve_summary_threshold(ranking_by_threshold, pick_threshold)
            if resolved_threshold is not None:
                threshold_key = _match_threshold_key(ranking_by_threshold, resolved_threshold)
                by_k = ranking_by_threshold.get(threshold_key, {}) or {}

                row["threshold"] = float(resolved_threshold)
                row["k"] = int(pick_k)

                if int(pick_k) in by_k:
                    m = by_k[int(pick_k)] or {}
                    row.update({
                        "precision": m.get("precision", np.nan),
                        "recall": m.get("recall", np.nan),
                        "map": m.get("map", np.nan),
                        "ndcg": m.get("ndcg", np.nan),
                        "ndcg_raw_rating": m.get("ndcg_raw_rating", np.nan),
                        "auc": m.get("auc", np.nan),
                        "n_users_used_binary": m.get("n_users_used_binary", np.nan),
                        "n_users_used_auc": m.get("n_users_used_auc", np.nan),
                        "n_users_used_graded_ndcg": m.get("n_users_used_graded_ndcg", np.nan),
                    })
                else:
                    row.update({
                        "precision": np.nan,
                        "recall": np.nan,
                        "map": np.nan,
                        "ndcg": np.nan,
                        "ndcg_raw_rating": np.nan,
                        "auc": np.nan,
                        "n_users_used_binary": np.nan,
                        "n_users_used_auc": np.nan,
                        "n_users_used_graded_ndcg": np.nan,
                    })

        rows.append(row)

    df = pd.DataFrame(rows)
    return _round_numeric_df(df, round_digits)
