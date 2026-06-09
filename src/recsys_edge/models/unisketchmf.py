"""
UniSketchMF / JL-RACE-MF side-info model and lifecycle experiment.

Generated from the original UniSketchMF notebook with import paths adjusted for
this anonymous reproduction package. Class/function bodies are preserved as much
as possible so the reported runs can be reproduced from the original logic.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import mmh3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader

from recsys_edge.core import *  # noqa: F401,F403 - preserves notebook-style globals
from recsys_edge.core import _bytes_dense, _torch_dtype_to_key

try:
    pd.options.mode.copy_on_write = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass



# ===== Notebook cell 7 =====
# Side Info Schema

_ALLOWED_REFERENCE_MODES = {"offline_frozen", "appended_live"}

_ALLOWED_REPRESENTATION_TYPES = {
    "group_id",
    "scalar",
    "d_vector",
    "dr_matrix",
    "engineered_features",
}

_ALLOWED_REPRESENTATION_SOURCES = {
    "scalar_freq",
    "row_mom",
    "raw_dr",
    "engineered",
}

_ALLOWED_TRANSFORM_KINDS = {
    "none",
    "log1p",
    "standardize",
    "log1p_standardize",
    "minmax",
    "log1p_minmax",
}

_ALLOWED_STATS_MODES = {
    "offline_reference",
    "current_reference",
}

_ALLOWED_MATRIX_ENCODER_INPUT_MODES = {
    "flatten",
    "row_shared",
}

_ALLOWED_PREDICTIVE_MODES = {
    "none",
    "group_bias",
    "additive_scalar",
    "embedding_residual",
}

_ALLOWED_REGULARIZATION_MODES = {
    "none",
    "user_l2",
    "anchor_l2",
}

_ALLOWED_REGULARIZATION_TARGETS = {
    "user_embedding",
}

_ALLOWED_REGULARIZATION_MAPPINGS = {
    "fixed_rule",
    "mlp",
}

_ALLOWED_ENGINEERED_FEATURE_SETS = {
    "compact_v1",
    "debug_raw_concat",
}

_ALLOWED_ANCHORS = {
    "zero",
    "group_centroid",
    "neighbor_mean",
    "offline_global_mean",
}

_ALLOWED_INCREMENTAL_BRANCH_POLICIES = {
    "freeze",
    "train",
}


def _si_require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass
class SideInfoTransformConfig:
    kind: str = "none"
    stats_mode: str = "offline_reference"
    clip_value: Optional[float] = None

    def validate(self) -> None:
        _si_require(
            self.kind in _ALLOWED_TRANSFORM_KINDS,
            f"Unsupported transform.kind='{self.kind}'. Allowed={sorted(_ALLOWED_TRANSFORM_KINDS)}"
        )
        _si_require(
            self.stats_mode in _ALLOWED_STATS_MODES,
            f"Unsupported transform.stats_mode='{self.stats_mode}'. Allowed={sorted(_ALLOWED_STATS_MODES)}"
        )
        if self.clip_value is not None:
            _si_require(self.clip_value > 0.0, "clip_value must be > 0 when provided.")


@dataclass
class SideInfoGroupingConfig:
    binner: str = "log_uniform"
    n_bins: int = 32

    def validate(self) -> None:
        _si_require(self.binner == "log_uniform", "For now, binner is locked to 'log_uniform'.")
        _si_require(self.n_bins >= 2, "n_bins must be >= 2.")


@dataclass
class SideInfoEngineeredConfig:
    feature_set: str = "compact_v1"
    include_scalar_freq: bool = True
    include_row_mom: bool = False
    include_raw_dr_flat: bool = False

    def validate(self) -> None:
        _si_require(
            self.feature_set in _ALLOWED_ENGINEERED_FEATURE_SETS,
            f"Unsupported engineered.feature_set='{self.feature_set}'. "
            f"Allowed={sorted(_ALLOWED_ENGINEERED_FEATURE_SETS)}"
        )

        if self.feature_set == "debug_raw_concat":
            _si_require(
                self.include_scalar_freq or self.include_row_mom or self.include_raw_dr_flat,
                "For engineered.feature_set='debug_raw_concat', at least one raw source must be enabled."
            )


@dataclass
class SideInfoRepresentationConfig:
    type: str = "group_id"
    source: str = "scalar_freq"
    transform: SideInfoTransformConfig = field(default_factory=SideInfoTransformConfig)
    grouping: SideInfoGroupingConfig = field(default_factory=SideInfoGroupingConfig)
    engineered: SideInfoEngineeredConfig = field(default_factory=SideInfoEngineeredConfig)
    matrix_encoder_input: dict[str, Any] = field(default_factory=lambda: {"mode": "flatten"})

    def validate(self) -> None:
        _si_require(
            self.type in _ALLOWED_REPRESENTATION_TYPES,
            f"Unsupported representation.type='{self.type}'. Allowed={sorted(_ALLOWED_REPRESENTATION_TYPES)}"
        )
        _si_require(
            self.source in _ALLOWED_REPRESENTATION_SOURCES,
            f"Unsupported representation.source='{self.source}'. Allowed={sorted(_ALLOWED_REPRESENTATION_SOURCES)}"
        )
        self.transform.validate()
        self.grouping.validate()
        self.engineered.validate()

        mode = str(self.matrix_encoder_input.get("mode", "flatten")).lower()
        _si_require(
            mode in _ALLOWED_MATRIX_ENCODER_INPUT_MODES,
            f"Unsupported matrix_encoder_input.mode='{mode}'. Allowed={sorted(_ALLOWED_MATRIX_ENCODER_INPUT_MODES)}"
        )

        if self.type == "group_id":
            _si_require(self.source == "scalar_freq", "group_id currently requires source='scalar_freq'.")
        elif self.type == "scalar":
            _si_require(self.source == "scalar_freq", "scalar currently requires source='scalar_freq'.")
        elif self.type == "d_vector":
            _si_require(self.source == "row_mom", "d_vector currently requires source='row_mom'.")
        elif self.type == "dr_matrix":
            _si_require(self.source == "raw_dr", "dr_matrix currently requires source='raw_dr'.")
        elif self.type == "engineered_features":
            _si_require(self.source == "engineered", "engineered_features currently requires source='engineered'.")


@dataclass
class SideInfoPredictiveConfig:
    mode: str = "group_bias"
    hidden_dims: list[int] = field(default_factory=lambda: [32])
    dropout: float = 0.0
    activation: str = "relu"

    def validate(self) -> None:
        _si_require(
            self.mode in _ALLOWED_PREDICTIVE_MODES,
            f"Unsupported predictive.mode='{self.mode}'. Allowed={sorted(_ALLOWED_PREDICTIVE_MODES)}"
        )
        _si_require(self.dropout >= 0.0, "predictive.dropout must be >= 0.")
        _si_require(all(int(h) > 0 for h in self.hidden_dims), "All predictive.hidden_dims must be > 0.")


@dataclass
class SideInfoRegularizationConfig:
    mode: str = "none"
    target: str = "user_embedding"
    lambda_min: float = 0.0
    lambda_max: float = 1e-2
    mapping: str = "mlp"
    anchor: str = "zero"
    apply_in_offline: bool = False
    apply_in_incremental: bool = True

    def validate(self) -> None:
        _si_require(
            self.mode in _ALLOWED_REGULARIZATION_MODES,
            f"Unsupported regularization.mode='{self.mode}'. Allowed={sorted(_ALLOWED_REGULARIZATION_MODES)}"
        )
        _si_require(
            self.target in _ALLOWED_REGULARIZATION_TARGETS,
            f"Unsupported regularization.target='{self.target}'. Allowed={sorted(_ALLOWED_REGULARIZATION_TARGETS)}"
        )
        _si_require(
            self.mapping in _ALLOWED_REGULARIZATION_MAPPINGS,
            f"Unsupported regularization.mapping='{self.mapping}'. Allowed={sorted(_ALLOWED_REGULARIZATION_MAPPINGS)}"
        )
        _si_require(
            self.anchor in _ALLOWED_ANCHORS,
            f"Unsupported regularization.anchor='{self.anchor}'. Allowed={sorted(_ALLOWED_ANCHORS)}"
        )
        _si_require(self.lambda_min >= 0.0, "lambda_min must be >= 0.")
        _si_require(self.lambda_max >= self.lambda_min, "lambda_max must be >= lambda_min.")
        _si_require(
            self.apply_in_offline or self.apply_in_incremental or self.mode == "none",
            "regularization must be enabled in at least one phase when mode != 'none'."
        )


@dataclass
class SideInfoIncrementalBranchPolicyConfig:
    predictive: str = "freeze"
    regularization: str = "freeze"

    def validate(self) -> None:
        _si_require(
            self.predictive in _ALLOWED_INCREMENTAL_BRANCH_POLICIES,
            f"Unsupported incremental_branch_policy.predictive='{self.predictive}'. "
            f"Allowed={sorted(_ALLOWED_INCREMENTAL_BRANCH_POLICIES)}"
        )
        _si_require(
            self.regularization in _ALLOWED_INCREMENTAL_BRANCH_POLICIES,
            f"Unsupported incremental_branch_policy.regularization='{self.regularization}'. "
            f"Allowed={sorted(_ALLOWED_INCREMENTAL_BRANCH_POLICIES)}"
        )


@dataclass
class SideInfoUsageConfig:
    predictive: SideInfoPredictiveConfig = field(default_factory=SideInfoPredictiveConfig)
    regularization: SideInfoRegularizationConfig = field(default_factory=SideInfoRegularizationConfig)
    incremental_branch_policy: SideInfoIncrementalBranchPolicyConfig = field(
        default_factory=SideInfoIncrementalBranchPolicyConfig
    )

    def validate(self) -> None:
        self.predictive.validate()
        self.regularization.validate()
        self.incremental_branch_policy.validate()


@dataclass
class SideInfoConfig:
    enabled: bool = True
    reference_mode: str = "offline_frozen"
    representation: SideInfoRepresentationConfig = field(default_factory=SideInfoRepresentationConfig)
    usage: SideInfoUsageConfig = field(default_factory=SideInfoUsageConfig)

    def validate(self) -> None:
        _si_require(
            self.reference_mode in _ALLOWED_REFERENCE_MODES,
            f"Unsupported reference_mode='{self.reference_mode}'. Allowed={sorted(_ALLOWED_REFERENCE_MODES)}"
        )
        self.representation.validate()
        self.usage.validate()

        rep_type = self.representation.type
        pred_mode = self.usage.predictive.mode
        reg_cfg = self.usage.regularization

        if pred_mode == "group_bias":
            _si_require(
                rep_type == "group_id",
                "predictive.mode='group_bias' requires representation.type='group_id'."
            )

        if pred_mode == "embedding_residual":
            _si_require(
                rep_type in {"scalar", "d_vector", "dr_matrix", "engineered_features"},
                "predictive.mode='embedding_residual' requires a continuous representation."
            )

        if reg_cfg.anchor == "group_centroid":
            _si_require(
                rep_type == "group_id",
                "regularization.anchor='group_centroid' currently requires representation.type='group_id'."
            )

        if reg_cfg.mode == "anchor_l2" and reg_cfg.apply_in_offline and reg_cfg.anchor != "zero":
            raise ValueError(
                "For now, apply_in_offline=True with anchor_l2 only supports anchor='zero'. "
                "Use non-zero anchors in incremental first."
            )

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> "SideInfoConfig":
        raw = dict(raw or {})

        rep_raw = dict(raw.get("representation", {}) or {})
        transform_raw = dict(rep_raw.get("transform", {}) or {})
        grouping_raw = dict(rep_raw.get("grouping", {}) or {})
        engineered_raw = dict(rep_raw.get("engineered", {}) or {})
        matrix_encoder_raw = dict(rep_raw.get("matrix_encoder_input", {}) or {})

        usage_raw = dict(raw.get("usage", {}) or {})
        predictive_raw = dict(usage_raw.get("predictive", {}) or {})
        regularization_raw = dict(usage_raw.get("regularization", {}) or {})
        incremental_branch_policy_raw = dict(usage_raw.get("incremental_branch_policy", {}) or {})

        cfg = cls(
            enabled=bool(raw.get("enabled", True)),
            reference_mode=str(raw.get("reference_mode", "offline_frozen")).lower(),
            representation=SideInfoRepresentationConfig(
                type=str(rep_raw.get("type", "group_id")).lower(),
                source=str(rep_raw.get("source", "scalar_freq")).lower(),
                transform=SideInfoTransformConfig(
                    kind=str(transform_raw.get("kind", "none")).lower(),
                    stats_mode=str(transform_raw.get("stats_mode", "offline_reference")).lower(),
                    clip_value=transform_raw.get("clip_value", None),
                ),
                grouping=SideInfoGroupingConfig(
                    binner=str(grouping_raw.get("binner", "log_uniform")).lower(),
                    n_bins=int(grouping_raw.get("n_bins", 32)),
                ),
                engineered=SideInfoEngineeredConfig(
                    feature_set=str(engineered_raw.get("feature_set", "compact_v1")).lower(),
                    include_scalar_freq=bool(engineered_raw.get("include_scalar_freq", True)),
                    include_row_mom=bool(engineered_raw.get("include_row_mom", False)),
                    include_raw_dr_flat=bool(engineered_raw.get("include_raw_dr_flat", False)),
                ),
                matrix_encoder_input={
                    "mode": str(matrix_encoder_raw.get("mode", "flatten")).lower(),
                },
            ),
            usage=SideInfoUsageConfig(
                predictive=SideInfoPredictiveConfig(
                    mode=str(predictive_raw.get("mode", "group_bias")).lower(),
                    hidden_dims=[int(x) for x in predictive_raw.get("hidden_dims", [32])],
                    dropout=float(predictive_raw.get("dropout", 0.0)),
                    activation=str(predictive_raw.get("activation", "relu")).lower(),
                ),
                regularization=SideInfoRegularizationConfig(
                    mode=str(regularization_raw.get("mode", "none")).lower(),
                    target=str(regularization_raw.get("target", "user_embedding")).lower(),
                    lambda_min=float(regularization_raw.get("lambda_min", 0.0)),
                    lambda_max=float(regularization_raw.get("lambda_max", 1e-2)),
                    mapping=str(regularization_raw.get("mapping", "mlp")).lower(),
                    anchor=str(regularization_raw.get("anchor", "zero")).lower(),
                    apply_in_offline=bool(regularization_raw.get("apply_in_offline", False)),
                    apply_in_incremental=bool(regularization_raw.get("apply_in_incremental", True)),
                ),
                incremental_branch_policy=SideInfoIncrementalBranchPolicyConfig(
                    predictive=str(incremental_branch_policy_raw.get("predictive", "freeze")).lower(),
                    regularization=str(incremental_branch_policy_raw.get("regularization", "freeze")).lower(),
                ),
            ),
        )
        cfg.validate()
        return cfg



# ===== Notebook cell 9 =====
# Side Info Utilities


@dataclass
class TensorPack:
    u: torch.Tensor
    i: torch.Tensor
    f: torch.Tensor
    y: torch.Tensor


@dataclass
class RawSketchOutputs:
    scalar_freq: np.ndarray   # [n]
    row_mom: np.ndarray       # [n, d]
    raw_dr: np.ndarray        # [n, d, R]


@dataclass
class SideInfoBatch:
    features: Optional[np.ndarray]   # continuous features; shape depends on representation
    group_ids: Optional[np.ndarray]  # [n] for group-based usage
    raw: RawSketchOutputs


class LogUniformGroupAssigner:
    """
    Always groups on log1p(raw_scalar_freq).
    """

    def __init__(self, n_bins: int):
        self.n_bins = int(n_bins)
        self.edges_: Optional[np.ndarray] = None
        self.majority_group_: int = 0

    def fit(self, raw_scalar_freq: np.ndarray) -> "LogUniformGroupAssigner":
        x = np.log1p(np.maximum(np.asarray(raw_scalar_freq, dtype=np.float32), 0.0))
        lo = float(x.min()) if x.size else 0.0
        hi = float(x.max()) if x.size else 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0

        edges = np.linspace(lo, hi, self.n_bins + 1, dtype=np.float32)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.edges_ = edges

        groups = self.transform(raw_scalar_freq)
        counts = np.bincount(groups, minlength=self.n_bins)
        self.majority_group_ = int(np.argmax(counts)) if counts.size else 0
        return self

    def transform(self, raw_scalar_freq: np.ndarray) -> np.ndarray:
        if self.edges_ is None:
            raise RuntimeError("LogUniformGroupAssigner.fit must be called before transform().")

        x = np.log1p(np.maximum(np.asarray(raw_scalar_freq, dtype=np.float32), 0.0))
        return np.digitize(x, self.edges_[1:-1], right=True).astype(np.int64)

    @property
    def num_bins(self) -> int:
        return self.n_bins


class FeatureTransform:
    """
    Fits feature statistics on [n, ...] arrays and preserves the original shape.
    """

    def __init__(self, kind: str = "none"):
        self.kind = str(kind).lower()
        self._shape: Optional[tuple[int, ...]] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        self._shape = tuple(x.shape[1:])
        return x.reshape(x.shape[0], -1)

    def _restore(self, x_flat: np.ndarray) -> np.ndarray:
        if self._shape is None:
            raise RuntimeError("FeatureTransform must be fit before transform().")
        out = x_flat.reshape((x_flat.shape[0],) + self._shape)
        if len(self._shape) == 1 and self._shape[0] == 1:
            return out.reshape(-1, 1)
        return out

    def _pre(self, x_flat: np.ndarray) -> np.ndarray:
        if self.kind in {"log1p", "log1p_standardize", "log1p_minmax"}:
            return np.log1p(np.maximum(x_flat, 0.0))
        return x_flat

    def fit(self, x: np.ndarray) -> "FeatureTransform":
        x_flat = self._flatten(x)
        x_pre = self._pre(x_flat)

        if self.kind in {"standardize", "log1p_standardize"}:
            self.mean_ = x_pre.mean(axis=0, dtype=np.float64).astype(np.float32)
            self.std_ = x_pre.std(axis=0, dtype=np.float64).astype(np.float32)
            self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

        if self.kind in {"minmax", "log1p_minmax"}:
            self.min_ = x_pre.min(axis=0).astype(np.float32)
            self.max_ = x_pre.max(axis=0).astype(np.float32)
            span = self.max_ - self.min_
            self.max_ = self.min_ + np.where(span < 1e-8, 1.0, span)

        return self

    def transform(self, x: np.ndarray, clip_value: Optional[float] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]

        x_flat = x.reshape(x.shape[0], -1)
        x_pre = self._pre(x_flat)

        if self.kind in {"standardize", "log1p_standardize"}:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("FeatureTransform.fit must be called before standardize transform.")
            x_pre = (x_pre - self.mean_) / self.std_

        if self.kind in {"minmax", "log1p_minmax"}:
            if self.min_ is None or self.max_ is None:
                raise RuntimeError("FeatureTransform.fit must be called before minmax transform.")
            x_pre = (x_pre - self.min_) / (self.max_ - self.min_)

        if clip_value is not None:
            x_pre = np.clip(x_pre, -float(clip_value), float(clip_value))

        return self._restore(x_pre.astype(np.float32, copy=False))


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    act_name = str(activation).lower()
    act_cls = nn.ReLU if act_name == "relu" else nn.GELU

    layers: list[nn.Module] = []
    prev = int(input_dim)

    for h in hidden_dims:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = int(h)

    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)



# ===== Notebook cell 11 =====
def _infer_shape(dfs: Sequence[pd.DataFrame]) -> tuple[int, int]:
    max_user = -1
    max_item = -1
    for df in dfs:
        if df is None or df.empty:
            continue
        if "user_id" in df.columns:
            max_user = max(max_user, int(df["user_id"].max()))
        if "item_id" in df.columns:
            max_item = max(max_item, int(df["item_id"].max()))
    if max_user < 0 or max_item < 0:
        raise ValueError("Could not infer matrix shape from empty dataframes.")
    return max_user + 1, max_item + 1


def _drop_last_duplicates(df: pd.DataFrame, rating_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["user_id", "item_id", rating_col])
    return (
        df.loc[:, ["user_id", "item_id", rating_col]]
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .copy()
    )


def _series_or_array_user_means(
    user_ids: np.ndarray,
    user_means: Optional[pd.Series | np.ndarray],
    *,
    default_value: float,
) -> np.ndarray:
    if user_means is None:
        return np.full(user_ids.shape[0], float(default_value), dtype=np.float32)

    if isinstance(user_means, pd.Series):
        out = pd.Series(user_ids).map(user_means).to_numpy(dtype=np.float32, copy=False)
        out = np.where(np.isnan(out), np.float32(default_value), out)
        return out.astype(np.float32, copy=False)

    arr = np.asarray(user_means, dtype=np.float32)
    out = np.full(user_ids.shape[0], float(default_value), dtype=np.float32)
    valid = (user_ids >= 0) & (user_ids < arr.shape[0])
    out[valid] = arr[user_ids[valid]]
    return out


def csr_from_df(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
    rating_col: str = "rating",
    *,
    center_by_user_mean: bool = False,
    user_means: Optional[pd.Series | np.ndarray] = None,
) -> sparse.csr_matrix:
    """
    CSR with 'last' semantics on duplicate (user_id, item_id).

    If center_by_user_mean=True, values become:
        rating - mean_user_rating
    where means are computed from the deduped dataframe unless user_means is provided.
    """
    if df is None or df.empty:
        return sparse.csr_matrix((n_users, n_items), dtype=np.float32)

    work = _drop_last_duplicates(df, rating_col=rating_col)

    rows = work["user_id"].to_numpy(dtype=np.int64, copy=False)
    cols = work["item_id"].to_numpy(dtype=np.int64, copy=False)

    if center_by_user_mean:
        if user_means is None:
            means = work.groupby("user_id", sort=False)[rating_col].mean()
            default_mean = float(work[rating_col].mean()) if not work.empty else 0.0
            mu = _series_or_array_user_means(rows, means, default_value=default_mean)
        else:
            default_mean = float(work[rating_col].mean()) if not work.empty else 0.0
            mu = _series_or_array_user_means(rows, user_means, default_value=default_mean)

        vals = work[rating_col].to_numpy(dtype=np.float32, copy=False) - mu
    else:
        vals = work[rating_col].to_numpy(dtype=np.float32, copy=False)

    return sparse.coo_matrix(
        (vals.astype(np.float32, copy=False), (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()


def dense_from_df(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
    rating_col: str = "rating",
    *,
    center_by_user_mean: bool = False,
    user_means: Optional[pd.Series | np.ndarray] = None,
) -> np.ndarray:
    return csr_from_df(
        df=df,
        n_users=n_users,
        n_items=n_items,
        rating_col=rating_col,
        center_by_user_mean=center_by_user_mean,
        user_means=user_means,
    ).toarray().astype(np.float32, copy=False)


def normalize_nonneg(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= eps:
        if w.size == 0:
            return w
        return np.full_like(w, 1.0 / w.size)
    return w / s


def normalize_rows_nonneg(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    M = np.maximum(M, 0.0)
    if M.ndim != 2:
        raise ValueError("normalize_rows_nonneg expects a rank-2 array.")

    row_sums = M.sum(axis=1, keepdims=True)
    zero_rows = row_sums <= eps
    if np.any(zero_rows):
        M = M.copy()
        M[zero_rows[:, 0]] = 1.0
        row_sums = M.sum(axis=1, keepdims=True)
    return M / row_sums



# ===== Notebook cell 12 =====
class CustomJL:
    def __init__(
        self,
        n_components: int,
        train_matrix,
        *,
        device: str | torch.device | None = None,
    ):
        self.n_components = int(n_components)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        X_fit = self._to_sklearn_input(train_matrix)
        self.transformer = SparseRandomProjection(n_components=self.n_components)
        self.transformer.fit(X_fit)

        self.train_projected = self.project(X_fit, return_torch=True, device=self.device)

    def _to_sklearn_input(self, matrix):
        if isinstance(matrix, torch.Tensor):
            return matrix.detach().cpu().numpy()
        if sparse.issparse(matrix):
            return matrix
        return np.asarray(matrix, dtype=np.float32)

    def project(
        self,
        matrix,
        *,
        return_torch: bool = True,
        device: str | torch.device | None = None,
    ):
        X = self._to_sklearn_input(matrix)
        out = self.transformer.transform(X)

        if sparse.issparse(out):
            out = out.toarray()

        out = np.asarray(out, dtype=np.float32, order="C")

        if not return_torch:
            return out

        tgt_device = self.device if device is None else torch.device(device)
        return torch.from_numpy(out).to(tgt_device)



# ===== Notebook cell 15 =====
def perm_MoM(arr, n_blocks, perm=1):
    arr = np.asarray(arr, dtype=np.float32)
    L = arr.shape[-1]
    n_blocks = max(1, min(n_blocks, L))

    # Fast path 1: one block = plain mean
    if n_blocks == 1:
        return arr.mean(axis=-1, dtype=np.float32)

    # Fast path 2: for your common setup (R=4, n_blocks=2),
    # MoM is exactly equal to the plain mean over the last axis.
    if L == 4 and n_blocks == 2:
        return arr.mean(axis=-1, dtype=np.float32)

    # General fallback
    sizes = np.full(n_blocks, L // n_blocks, dtype=int)
    sizes[: L % n_blocks] += 1
    starts = np.concatenate(([0], np.cumsum(sizes[:-1])))

    meds = np.empty(arr.shape[:-1] + (perm,), dtype=np.float32)

    for p in range(perm):
        idx = np.random.permutation(L)
        part = np.take(arr, idx, axis=-1)

        block_means = np.empty(arr.shape[:-1] + (n_blocks,), dtype=np.float32)
        for b, (s, sz) in enumerate(zip(starts, sizes)):
            block_means[..., b] = part[..., s:s+sz].mean(axis=-1, dtype=np.float32)

        meds[..., p] = np.median(block_means, axis=-1).astype(np.float32, copy=False)

    return meds.mean(axis=-1, dtype=np.float32)



def indexed_projection_original(original_data_tensor, all_projections_tensor, indices_i_to_hij):
    # Shapes:
    # all_projections_tensor: (d, w, R, k, m)
    # original_data_tensor: (n, m)
    # indices_i_to_hij: (n, d)

    # Get dimensions
    n, m = original_data_tensor.shape
    d, w, R, k, _ = all_projections_tensor.shape

    # Prepare indices for data points (n) and hash tables (i)
    data_indices = torch.arange(n).unsqueeze(1).expand(n, d)  # Shape: (n, d)
    hash_table_indices = torch.arange(d).unsqueeze(0).expand(n, d)  # Shape: (n, d)

    # Get bin indices from indices_i_to_hij
    bin_indices = indices_i_to_hij  # Shape: (n, d)

    # Expand data indices to match R and k dimensions
    data_indices_expanded = data_indices.unsqueeze(2).unsqueeze(3)  # Shape: (n, d, 1, 1)
    hash_table_indices_expanded = hash_table_indices.unsqueeze(2).unsqueeze(3)  # Shape: (n, d, 1, 1)
    bin_indices_expanded = bin_indices.unsqueeze(2).unsqueeze(3)  # Shape: (n, d, 1, 1)

    # Expand to match R and k
    data_indices_expanded = data_indices_expanded.expand(-1, -1, R, k)  # Shape: (n, d, R, k)
    hash_table_indices_expanded = hash_table_indices_expanded.expand(-1, -1, R, k)  # Shape: (n, d, R, k)
    bin_indices_expanded = bin_indices_expanded.expand(-1, -1, R, k)  # Shape: (n, d, R, k)

    # Now we need to extract the corresponding slices from all_projections_tensor
    # Prepare indices for R and k
    R_indices = torch.arange(R).view(1, 1, R, 1).expand(n, d, R, k)
    k_indices = torch.arange(k).view(1, 1, 1, k).expand(n, d, R, k)

    # Now, gather the projection vectors
    # First, ensure all indices are of the correct data type and on the correct device
    data_indices_expanded = data_indices_expanded.to(all_projections_tensor.device)
    hash_table_indices_expanded = hash_table_indices_expanded.to(all_projections_tensor.device)
    bin_indices_expanded = bin_indices_expanded.to(all_projections_tensor.device)
    R_indices = R_indices.to(all_projections_tensor.device)
    k_indices = k_indices.to(all_projections_tensor.device)

    # Now, extract the required projection vectors
    # Shape of projection_vectors: (n, d, R, k, m)
    projection_vectors = all_projections_tensor[
        hash_table_indices_expanded,
        bin_indices_expanded,
        R_indices,
        k_indices,
        :
    ]  # Using advanced indexing

    # Now, for each data point n, we need to compute the dot product between original_data_tensor[n, :]
    # and projection_vectors[n, d, R, k, :]
    # We need to expand original_data_tensor to match the dimensions
    # original_data_tensor_expanded: (n, 1, 1, 1, m)
    original_data_tensor_expanded = original_data_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # Compute the dot products
    # Resulting shape: (n, d, R, k)
    dot_products = (projection_vectors * original_data_tensor_expanded).sum(dim=-1)

    return dot_products


def indexed_projection(original_data_tensor, all_projections_tensor, indices_i_to_hij,
                       data_batch_size=1000, hash_table_batch_size=5):
    n, m = original_data_tensor.shape
    d, w, R, k, _ = all_projections_tensor.shape

    # Initialize result tensor
    # Shape: (n, d, R, k)
    projections = torch.zeros(n, d, R, k, device=original_data_tensor.device)

    for data_batch_start in range(0, n, data_batch_size):
        data_batch_end = min(data_batch_start + data_batch_size, n)
        data_batch_indices = slice(data_batch_start, data_batch_end)
        data_batch_size_actual = data_batch_end - data_batch_start

        # Extract batch data
        original_data_batch = original_data_tensor[data_batch_indices, :]  # Shape: (data_batch_size_actual, m)
        bin_indices_batch = indices_i_to_hij[data_batch_indices, :]       # Shape: (data_batch_size_actual, d)

        for hash_batch_start in range(0, d, hash_table_batch_size):
            hash_batch_end = min(hash_batch_start + hash_table_batch_size, d)
            hash_batch_size_actual = hash_batch_end - hash_batch_start

            # Extract hash table indices for this batch
            hash_table_indices = torch.arange(hash_batch_start, hash_batch_end, device=original_data_tensor.device)
            hash_table_indices = hash_table_indices.unsqueeze(0).expand(data_batch_size_actual, -1)  # Shape: (data_batch_size_actual, hash_batch_size_actual)

            # Get corresponding bin indices
            bin_indices = bin_indices_batch[:, hash_batch_start:hash_batch_end]  # Shape: (data_batch_size_actual, hash_batch_size_actual)

            # Prepare indices
            data_indices = torch.arange(data_batch_size_actual, device=original_data_tensor.device).unsqueeze(1).expand(-1, hash_batch_size_actual)  # Shape: (data_batch_size_actual, hash_batch_size_actual)

            # Expand dimensions
            data_indices_expanded = data_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, R, k)
            hash_table_indices_expanded = hash_table_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, R, k)
            bin_indices_expanded = bin_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, R, k)

            R_indices = torch.arange(R, device=original_data_tensor.device).view(1, 1, R, 1).expand(data_batch_size_actual, hash_batch_size_actual, R, k)
            k_indices = torch.arange(k, device=original_data_tensor.device).view(1, 1, 1, k).expand(data_batch_size_actual, hash_batch_size_actual, R, k)

            # Create a mask for valid bin indices
            valid_mask = bin_indices_expanded >= 0  # Adjust based on how invalid indices are represented

            # Flatten valid indices
            valid_data_indices = data_indices_expanded[valid_mask]
            valid_hash_table_indices = hash_table_indices_expanded[valid_mask]
            valid_bin_indices = bin_indices_expanded[valid_mask]
            valid_R_indices = R_indices[valid_mask]
            valid_k_indices = k_indices[valid_mask]

            # Extract the required projection vectors
            projection_vectors = all_projections_tensor[
                valid_hash_table_indices,
                valid_bin_indices,
                valid_R_indices,
                valid_k_indices,
                :
            ]  # Shape: (num_valid_entries, m)

            # Extract corresponding data vectors
            data_vectors = original_data_batch[valid_data_indices, :]  # Shape: (num_valid_entries, m)

            # Compute dot products
            dot_products = (projection_vectors * data_vectors).sum(dim=-1)  # Shape: (num_valid_entries,)

            # Place the results into the projections tensor
            global_data_indices = data_batch_start + valid_data_indices
            global_hash_table_indices = valid_hash_table_indices

            # Since indices are flattened, we need to map them back to 4D indices
            projections[global_data_indices, global_hash_table_indices, valid_R_indices, valid_k_indices] = dot_products

            # Clean up variables to free memory
            del (data_indices_expanded, hash_table_indices_expanded, bin_indices_expanded,
                 valid_mask, valid_data_indices, valid_hash_table_indices, valid_bin_indices,
                 valid_R_indices, valid_k_indices, projection_vectors, data_vectors, dot_products)

            # Empty CUDA cache
            torch.cuda.empty_cache()

        # Also delete variables from the data batch loop if they won't be used again
        del (original_data_batch, bin_indices_batch)

        # Empty CUDA cache
        torch.cuda.empty_cache()

    return projections  # Shape: (n, d, R, k)



# ===== Notebook cell 17 =====
class VectorizedRaceLSH:
    """
    RACE + Count-Min (via MoM reduction) with incremental-friendly APIs.
    """

    def __init__(
        self,
        k,
        d,
        w,
        R,
        data,
        n_blocks=2,
        n_perms=5,
        batch_size=1024,
        query_batch_size: int | None = None,
        seed: int = GLOBAL_SEED,
        build_initial: bool = True,
        use_gpu: bool | None = True,
    ):
        self.k, self.d, self.w, self.R = k, d, w, R

        self._d_idx_np = np.arange(self.d, dtype=np.int64)
        self._r_idx_np = np.arange(self.R, dtype=np.int64)
        self._w_idx_np = np.arange(self.w, dtype=np.int64)

        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.gpu_device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.cpu_device = torch.device("cpu")

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        self.original_data = data.copy()
        self.js, self.features_size = data.shape

        self.rng = np.random.default_rng(seed)

        self.all_projections = self._generate_projections().astype(np.float32)
        self._all_proj_t = torch.tensor(
            self.all_projections,
            device=self.gpu_device,
            dtype=torch.float32,
        )

        self.indices_i_to_hij = self._hash_indices_for_users(start_j=0, count=self.js)

        self.sketch_A = np.zeros((self.d, self.w, self.R, 2 ** self.k), dtype=np.int32)

        self.bucket_dtype = (
            np.uint32 if (2 ** self.k - 1) > np.iinfo(np.int16).max else np.int16
        )
        if np.issubdtype(self.bucket_dtype, np.signedinteger):
            self.bucket_empty = -1
        else:
            self.bucket_empty = np.uint32(2 ** self.k)

        self.user_buckets = np.full(
            (self.js, self.d, self.R),
            fill_value=self.bucket_empty,
            dtype=self.bucket_dtype,
        )
        self.offline_freqs = np.zeros(self.js, dtype=np.float32)

        self.batch_size = int(batch_size)
        self.query_batch_size = int(query_batch_size or batch_size)
        self.n_blocks = int(n_blocks)
        self.n_perms = int(n_perms)

        self.one_pass_indices = None

        self._bit_shifts = torch.arange(
            self.k - 1,
            -1,
            -1,
            device=self.gpu_device,
            dtype=torch.int64,
        )

        if build_initial:
            self.one_pass_sketch_A(
                user_ids=None,
                data=None,
                rebuild=True,
                return_df=False,
            )
            self.fetch_frequencies(user_ids=None)

    def generate_iid_hash_functions(self):
        return self.indices_i_to_hij, self.all_projections

    def _generate_projections(self):
        return self.rng.normal(size=(self.d, self.w, self.R, self.k, self.features_size))

    def _hash_indices_for_users(self, start_j: int, count: int) -> np.ndarray:
        out = np.empty((count, self.d), dtype=np.int32)
        js = np.arange(start_j, start_j + count, dtype=np.int64)
        for i in range(self.d):
            out[:, i] = [mmh3.hash(str(int(j)), seed=int(i)) % self.w for j in js]
        return out

    def _resolve_candidate_cols(self, candidate_user_ids=None) -> np.ndarray:
        if candidate_user_ids is None:
            return self.indices_i_to_hij.T.astype(np.int64, copy=False)

        cand = np.asarray(candidate_user_ids, dtype=np.int64)
        if cand.size > 0 and (cand.min() < 0 or cand.max() >= self.js):
            raise ValueError("candidate_user_ids out of range")
        return self.indices_i_to_hij[cand].T.astype(np.int64, copy=False)

    def extend_indices_for_new_users(self, count: int, start_j: int | None = None) -> np.ndarray:
        if start_j is None:
            start_j = self.js

        new_idx = self._hash_indices_for_users(start_j=start_j, count=count)
        self.indices_i_to_hij = np.vstack([self.indices_i_to_hij, new_idx])

        new_buckets = np.full(
            (count, self.d, self.R),
            fill_value=self.bucket_empty,
            dtype=self.bucket_dtype,
        )
        self.user_buckets = np.vstack([self.user_buckets, new_buckets])
        self.offline_freqs = np.concatenate(
            [self.offline_freqs, np.zeros(count, dtype=np.float32)]
        )

        self.js += count
        return new_idx

    def one_pass_sketch_A(
        self,
        user_ids=None,
        data=None,
        rebuild: bool = False,
        overwrite: bool = False,
        return_df: bool = False,
    ):
        if user_ids is None:
            batch_ids = np.arange(self.js, dtype=np.int64)
        else:
            batch_ids = np.asarray(user_ids, dtype=np.int64)

        if rebuild:
            self.sketch_A.fill(0)

        if data is None:
            batch_data = self.original_data[batch_ids]
        else:
            if isinstance(data, torch.Tensor):
                batch_data = data.detach().cpu().numpy()
            else:
                batch_data = np.asarray(data, dtype=np.float32)
            assert batch_data.shape[1] == self.features_size, "data has wrong #features"

        batch_hash_cols = self.indices_i_to_hij[batch_ids]

        buckets = self.all_projections_over_data(
            batch_data,
            offline_phase=True,
            indices_override=batch_hash_cols,
        )

        B = buckets.shape[0]

        if overwrite:
            old = self.user_buckets[batch_ids]
            i_old = np.broadcast_to(self._d_idx_np[None, :, None], old.shape)
            r_old = np.broadcast_to(self._r_idx_np[None, None, :], old.shape)
            c_old = np.broadcast_to(batch_hash_cols[:, :, None], old.shape)
            np.add.at(
                self.sketch_A,
                (i_old.ravel(), c_old.ravel(), r_old.ravel(), old.ravel()),
                -1,
            )

        self.user_buckets[batch_ids] = buckets.astype(self.bucket_dtype, copy=False)

        i_idx = np.broadcast_to(self._d_idx_np[None, :, None], (B, self.d, self.R))
        r_idx = np.broadcast_to(self._r_idx_np[None, None, :], (B, self.d, self.R))
        c_idx = np.broadcast_to(batch_hash_cols[:, :, None], (B, self.d, self.R))
        np.add.at(
            self.sketch_A,
            (i_idx.ravel(), c_idx.ravel(), r_idx.ravel(), buckets.ravel()),
            1,
        )

        if not return_df:
            return None, self.sketch_A

        j_idx = np.broadcast_to(batch_ids[:, None, None], (B, self.d, self.R))
        one_pass_indices = pd.DataFrame(
            {
                "j_ind": j_idx.ravel(),
                "row": i_idx.ravel(),
                "hashed_col": c_idx.ravel(),
                "r": r_idx.ravel(),
                "inc_bucket": buckets.ravel(),
            }
        )
        return one_pass_indices, self.sketch_A

    def fetch_frequencies(self, freqs_multiplier=None, user_ids=None):
        if user_ids is None:
            user_ids = np.arange(self.js, dtype=np.int64)
        else:
            user_ids = np.asarray(user_ids, dtype=np.int64)

        buckets = self.user_buckets[user_ids].astype(np.int64, copy=False)
        cols = self.indices_i_to_hij[user_ids].astype(np.int64, copy=False)
        B, d, R = buckets.shape

        if (buckets == self.bucket_empty).any():
            raise ValueError(
                "fetch_frequencies: some users have empty buckets; run one_pass_sketch_A first."
            )

        i_idx = np.broadcast_to(self._d_idx_np[None, :, None], (B, d, R))
        r_idx = np.broadcast_to(self._r_idx_np[None, None, :], (B, d, R))
        c_idx = np.broadcast_to(cols[:, :, None], (B, d, R))

        fetched = self.sketch_A[i_idx, c_idx, r_idx, buckets]

        eff_blocks = max(1, min(self.n_blocks, fetched.shape[-1]))
        if fetched.shape[-1] == 4 and eff_blocks == 2:
            cms = fetched.mean(axis=-1, dtype=np.float32)
        else:
            cms = perm_MoM(fetched, n_blocks=eff_blocks, perm=self.n_perms)

        freqs = np.min(cms, axis=1)

        if freqs_multiplier is not None:
            freqs = freqs * freqs_multiplier

        self.offline_freqs[user_ids] = freqs.astype(np.float32, copy=False)
        return freqs

    def querying_algorithm(
        self,
        query_vectors,
        threshold=0,
        candidate_user_ids=None,
        candidate_cols: np.ndarray | None = None,
    ):
        fetched_buckets = self.all_projections_over_data(
            query_vectors,
            offline_phase=False,
        )
        Q, d, w, R = fetched_buckets.shape

        i_idx = np.broadcast_to(self._d_idx_np[None, :, None, None], (Q, d, w, R))
        w_idx = np.broadcast_to(self._w_idx_np[None, None, :, None], (Q, d, w, R))
        r_idx = np.broadcast_to(self._r_idx_np[None, None, None, :], (Q, d, w, R))

        counters = self.sketch_A[i_idx, w_idx, r_idx, fetched_buckets]

        eff_blocks = max(1, min(self.n_blocks, counters.shape[-1]))
        if counters.shape[-1] == 4 and eff_blocks == 2:
            full_cms = counters.mean(axis=-1, dtype=np.float32)
        else:
            full_cms = perm_MoM(counters, n_blocks=eff_blocks, perm=self.n_perms)

        cols = candidate_cols if candidate_cols is not None else self._resolve_candidate_cols(candidate_user_ids)

        C = cols.shape[1]
        if C == 0:
            kernels_estimations = np.empty((Q, 0), dtype=np.float32)
            return full_cms, kernels_estimations

        kernels_estimations = full_cms[:, 0, cols[0]].astype(np.float32, copy=True)
        for i in range(1, self.d):
            np.minimum(kernels_estimations, full_cms[:, i, cols[i]], out=kernels_estimations)

        return full_cms, kernels_estimations

    def find_closest_users(
        self,
        query_vectors,
        top_k,
        return_kernels: bool = False,
        threshold: int = 0,
        candidate_user_ids=None,
    ):
        cand = None
        if candidate_user_ids is not None:
            cand = np.asarray(candidate_user_ids, dtype=np.int64)

        if isinstance(query_vectors, torch.Tensor):
            Q_total = int(query_vectors.shape[0])
        else:
            query_vectors = np.asarray(query_vectors, dtype=np.float32)
            Q_total = int(query_vectors.shape[0])

        qb = int(self.query_batch_size)
        if qb <= 0:
            qb = Q_total

        candidate_cols = self._resolve_candidate_cols(candidate_user_ids)

        all_top_vals = []
        all_neighbors = []
        all_kernels = [] if return_kernels else None

        for start in range(0, Q_total, qb):
            end = min(start + qb, Q_total)
            q_batch = query_vectors[start:end]

            _, kernels_b = self.querying_algorithm(
                q_batch,
                threshold=threshold,
                candidate_user_ids=None,
                candidate_cols=candidate_cols,
            )

            Qb, C = kernels_b.shape
            k = min(int(top_k), int(C))

            if k <= 0:
                top_vals_b = np.zeros((Qb, 0), dtype=kernels_b.dtype)
                neighbors_b = np.zeros((Qb, 0), dtype=np.int64)
                if return_kernels:
                    all_kernels.append(kernels_b)
                all_top_vals.append(top_vals_b)
                all_neighbors.append(neighbors_b)
                continue

            part = np.argpartition(-kernels_b, kth=k - 1, axis=1)[:, :k]
            row = np.arange(Qb)[:, None]
            sel = kernels_b[row, part]
            order = np.argsort(-sel, axis=1)
            top_idx = part[row, order]
            top_vals_b = sel[row, order]

            if cand is not None:
                neighbors_b = cand[top_idx]
            else:
                neighbors_b = top_idx.astype(np.int64, copy=False)

            all_top_vals.append(top_vals_b)
            all_neighbors.append(neighbors_b)

            if return_kernels:
                all_kernels.append(kernels_b)

        top_vals = (
            np.concatenate(all_top_vals, axis=0)
            if len(all_top_vals) > 1
            else all_top_vals[0]
        )
        neighbors = (
            np.concatenate(all_neighbors, axis=0)
            if len(all_neighbors) > 1
            else all_neighbors[0]
        )

        if return_kernels:
            kernels = (
                np.concatenate(all_kernels, axis=0)
                if len(all_kernels) > 1
                else all_kernels[0]
            )
            return top_vals, neighbors, kernels

        return top_vals, neighbors

    def all_projections_over_data(self, data, offline_phase: bool, indices_override=None):
        if isinstance(data, np.ndarray):
            x = torch.as_tensor(data, device=self.gpu_device, dtype=torch.float32)
        else:
            x = data.to(self.gpu_device, dtype=torch.float32)

        if offline_phase:
            idx = self.indices_i_to_hij if indices_override is None else indices_override
            idx_t = torch.as_tensor(idx, device=self.gpu_device, dtype=torch.long)

            projected = indexed_projection(
                x,
                self._all_proj_t,
                idx_t,
                data_batch_size=self.batch_size,
            )

            binary = (projected > 0).to(torch.int64)
            buckets = torch.sum(binary << self._bit_shifts, dim=-1).to(torch.int64)

        else:
            mprod = torch.matmul(self._all_proj_t, x.T).permute(4, 0, 1, 2, 3)
            binary = (mprod > 0).to(torch.int64)
            buckets = torch.sum(binary << self._bit_shifts, dim=-1)

        return buckets.to(self.cpu_device).numpy()



# ===== Notebook cell 19 =====
# Reference Sketch Readout + Side Info Processor

class ReferenceSketchReadout:
    """
    Uses your existing CustomJL + VectorizedRaceLSH as a frozen or live reference,
    and extracts raw sketch outputs for arbitrary user_ids without changing the
    downstream representation yet.

    This class does not replace your sketch. It just reads from it.
    """

    def __init__(self, jl: CustomJL, race: VectorizedRaceLSH, device: torch.device):
        self.jl = jl
        self.race = race
        self.device = device

    def hash_cols_for_user_ids(self, user_ids: np.ndarray) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        out = np.empty((user_ids.shape[0], self.race.d), dtype=np.int64)
        for i in range(self.race.d):
            out[:, i] = [mmh3.hash(str(int(uid)), seed=int(i)) % self.race.w for uid in user_ids]
        return out

    def project_dense_matrix(self, dense_matrix: np.ndarray) -> torch.Tensor:
        return self.jl.project(dense_matrix, return_torch=True, device=self.device)

    def raw_outputs_from_projected(
        self,
        projected_vectors: torch.Tensor | np.ndarray,
        user_ids: np.ndarray,
    ) -> RawSketchOutputs:
        """
        Returns:
            raw_dr:   [n, d, R]
            row_mom:  [n, d]
            scalar:   [n]
        """
        user_ids = np.asarray(user_ids, dtype=np.int64)
        fetched_buckets = self.race.all_projections_over_data(projected_vectors, offline_phase=False)
        # fetched_buckets shape: [n, d, w, R]

        n, d, w, R = fetched_buckets.shape

        i_idx = np.broadcast_to(np.arange(d, dtype=np.int64)[None, :, None, None], (n, d, w, R))
        w_idx = np.broadcast_to(np.arange(w, dtype=np.int64)[None, None, :, None], (n, d, w, R))
        r_idx = np.broadcast_to(np.arange(R, dtype=np.int64)[None, None, None, :], (n, d, w, R))

        counters = self.race.sketch_A[i_idx, w_idx, r_idx, fetched_buckets]   # [n, d, w, R]

        hash_cols = self.hash_cols_for_user_ids(user_ids)  # [n, d]
        cols_exp = np.repeat(hash_cols[:, :, None, None], repeats=R, axis=3)   # [n, d, 1, R]
        raw_dr = np.take_along_axis(counters, cols_exp, axis=2).squeeze(2).astype(np.float32, copy=False)  # [n, d, R]

        eff_blocks = max(1, min(self.race.n_blocks, raw_dr.shape[-1]))
        if raw_dr.shape[-1] == 4 and eff_blocks == 2:
            row_mom = raw_dr.mean(axis=-1, dtype=np.float32)
        else:
            row_mom = perm_MoM(raw_dr, n_blocks=eff_blocks, perm=self.race.n_perms).astype(np.float32, copy=False)

        scalar_freq = np.min(row_mom, axis=1).astype(np.float32, copy=False)

        return RawSketchOutputs(
            scalar_freq=scalar_freq,
            row_mom=row_mom,
            raw_dr=raw_dr,
        )

class SideInfoProcessor:
    """
    Converts raw sketch outputs into the final representation the model will consume.

    Supports:
    - group_id
    - scalar
    - d_vector
    - dr_matrix
    - engineered_features
    """

    def __init__(self, cfg: SideInfoConfig):
        self.cfg = cfg
        self.group_assigner: Optional[LogUniformGroupAssigner] = None
        self.feature_transform: Optional[FeatureTransform] = None
        self._feature_shape: Optional[tuple[int, ...]] = None

    def _fit_grouping(self, raw: RawSketchOutputs) -> None:
        grouping_cfg = self.cfg.representation.grouping
        self.group_assigner = LogUniformGroupAssigner(n_bins=int(grouping_cfg.n_bins)).fit(raw.scalar_freq)

    @staticmethod
    def _safe_divide(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        num = np.asarray(num, dtype=np.float32)
        den = np.asarray(den, dtype=np.float32)
        return (num / np.maximum(den, eps)).astype(np.float32, copy=False)

    def _build_engineered_features(self, raw: RawSketchOutputs) -> np.ndarray:
        cfg = self.cfg.representation.engineered
        feature_set = str(cfg.feature_set).lower()

        if feature_set == "debug_raw_concat":
            parts = []
            if cfg.include_scalar_freq:
                parts.append(raw.scalar_freq.reshape(-1, 1).astype(np.float32, copy=False))
            if cfg.include_row_mom:
                parts.append(raw.row_mom.astype(np.float32, copy=False))
            if cfg.include_raw_dr_flat:
                parts.append(raw.raw_dr.reshape(raw.raw_dr.shape[0], -1).astype(np.float32, copy=False))

            if not parts:
                raise ValueError("No engineered feature sources were enabled.")

            two_d_parts = [p if p.ndim == 2 else p.reshape(p.shape[0], -1) for p in parts]
            return np.concatenate(two_d_parts, axis=1).astype(np.float32, copy=False)

        if feature_set != "compact_v1":
            raise ValueError(f"Unsupported engineered feature_set='{feature_set}'")

        scalar = raw.scalar_freq.astype(np.float32, copy=False)
        row = raw.row_mom.astype(np.float32, copy=False)
        dr = raw.raw_dr.astype(np.float32, copy=False)

        n, d = row.shape
        _ = n, d

        row_mean = row.mean(axis=1, dtype=np.float32)
        row_std = row.std(axis=1, dtype=np.float32)
        row_min = row.min(axis=1)
        row_max = row.max(axis=1)
        row_med = np.median(row, axis=1).astype(np.float32, copy=False)
        row_q25 = np.quantile(row, 0.25, axis=1).astype(np.float32, copy=False)
        row_q75 = np.quantile(row, 0.75, axis=1).astype(np.float32, copy=False)

        row_span = (row_max - row_min).astype(np.float32, copy=False)
        min_over_mean = self._safe_divide(row_min, row_mean)
        min_over_max = self._safe_divide(row_min, row_max)

        d_denom = float(max(row.shape[1] - 1, 1))
        argmin_norm = (row.argmin(axis=1).astype(np.float32) / d_denom).astype(np.float32, copy=False)
        argmax_norm = (row.argmax(axis=1).astype(np.float32) / d_denom).astype(np.float32, copy=False)

        row_std_over_r = dr.std(axis=2, dtype=np.float32)
        row_range_over_r = (dr.max(axis=2) - dr.min(axis=2)).astype(np.float32, copy=False)

        dr_row_std_mean = row_std_over_r.mean(axis=1, dtype=np.float32)
        dr_row_std_max = row_std_over_r.max(axis=1)
        dr_row_range_mean = row_range_over_r.mean(axis=1, dtype=np.float32)
        dr_global_std = dr.reshape(dr.shape[0], -1).std(axis=1, dtype=np.float32)

        feats = np.column_stack(
            [
                scalar,
                np.log1p(np.maximum(scalar, 0.0)).astype(np.float32, copy=False),
                row_mean,
                row_std,
                row_min,
                row_max,
                row_med,
                row_q25,
                row_q75,
                row_span,
                min_over_mean,
                min_over_max,
                argmin_norm,
                argmax_norm,
                dr_row_std_mean,
                dr_row_std_max,
                dr_row_range_mean,
                dr_global_std,
            ]
        ).astype(np.float32, copy=False)

        return feats

    def _select_raw_feature_array(self, raw: RawSketchOutputs) -> np.ndarray:
        rep = self.cfg.representation

        if rep.type == "group_id":
            raise RuntimeError("group_id does not use continuous feature arrays.")

        if rep.type == "scalar":
            return raw.scalar_freq.reshape(-1, 1).astype(np.float32, copy=False)

        if rep.type == "d_vector":
            return raw.row_mom.astype(np.float32, copy=False)

        if rep.type == "dr_matrix":
            return raw.raw_dr.astype(np.float32, copy=False)

        if rep.type == "engineered_features":
            return self._build_engineered_features(raw)

        raise ValueError(f"Unsupported representation.type='{rep.type}'")

    def fit_offline_reference(self, raw: RawSketchOutputs) -> None:
        rep = self.cfg.representation

        if rep.type == "group_id":
            self._fit_grouping(raw)
            self._feature_shape = None
            return

        base = self._select_raw_feature_array(raw)
        self.feature_transform = FeatureTransform(rep.transform.kind).fit(base)
        transformed = self.feature_transform.transform(base, clip_value=rep.transform.clip_value)
        self._feature_shape = tuple(transformed.shape[1:])

    def transform(
        self,
        raw: RawSketchOutputs,
        *,
        refit_if_current_reference: bool = False,
    ) -> SideInfoBatch:
        rep = self.cfg.representation

        if rep.type == "group_id":
            if self.group_assigner is None:
                raise RuntimeError("SideInfoProcessor.fit_offline_reference must be called before group_id transform().")
            group_ids = self.group_assigner.transform(raw.scalar_freq)
            return SideInfoBatch(features=None, group_ids=group_ids, raw=raw)

        base = self._select_raw_feature_array(raw)

        if rep.transform.stats_mode == "current_reference" and refit_if_current_reference:
            self.feature_transform = FeatureTransform(rep.transform.kind).fit(base)

        if self.feature_transform is None:
            raise RuntimeError("SideInfoProcessor.fit_offline_reference must be called before transform().")

        features = self.feature_transform.transform(
            base,
            clip_value=rep.transform.clip_value,
        ).astype(np.float32, copy=False)

        return SideInfoBatch(features=features, group_ids=None, raw=raw)

    @property
    def num_groups(self) -> Optional[int]:
        if self.group_assigner is None:
            return None
        return int(self.group_assigner.num_bins)

    @property
    def feature_shape(self) -> Optional[tuple[int, ...]]:
        return self._feature_shape



# ===== Notebook cell 22 =====
# Side-Info-Aware MF Backbone

class SideInfoMFBackbone(nn.Module):
    """
    Pure MF backbone without the old freq_bias embedding.
    Side-info usage is added on top by the trainer wrapper.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_size: int,
        *,
        use_user_bias: bool = True,
        use_global_bias: bool = True,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(int(num_users), int(embedding_size))
        self.item_embedding = nn.Embedding(int(num_items), int(embedding_size))
        self.item_bias = nn.Embedding(int(num_items), 1)

        self.use_user_bias = bool(use_user_bias)
        self.use_global_bias = bool(use_global_bias)

        self.user_bias = nn.Embedding(int(num_users), 1) if self.use_user_bias else None
        self.global_bias = nn.Parameter(torch.zeros((), dtype=torch.float32)) if self.use_global_bias else None

    def score_from_user_embedding(
        self,
        user_emb: torch.Tensor,
        item_ids: torch.Tensor,
        user_ids_for_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        item_bias = self.item_bias(item_ids).squeeze(-1)
        out = (user_emb * item_emb).sum(dim=-1) + item_bias

        if self.user_bias is not None:
            if user_ids_for_bias is None:
                raise ValueError("user_ids_for_bias is required when user_bias is enabled.")
            out = out + self.user_bias(user_ids_for_bias).squeeze(-1)

        if self.global_bias is not None:
            out = out + self.global_bias

        return out

    def forward_base(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        return self.score_from_user_embedding(user_emb=user_emb, item_ids=item_ids, user_ids_for_bias=user_ids)



# ===== Notebook cell 24 =====
# Side-Info-Aware MF Trainer

class RowSharedMatrixEncoder(nn.Module):
    """
    Encodes an input of shape [batch, d, R] by applying the same row encoder to each
    row of length R, then averaging across d.
    """

    def __init__(
        self,
        row_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.row_mlp = build_mlp(
            input_dim=int(row_dim),
            hidden_dims=hidden_dims,
            output_dim=int(output_dim),
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, d, R]
        b, d, r = x.shape
        x2 = x.reshape(b * d, r)
        z = self.row_mlp(x2)          # [b*d, out]
        z = z.reshape(b, d, -1).mean(dim=1)
        return z


class SideInfoAwareMF:
    """
    Replaces the old CustomMF when you want side info to be pluggable.

    Supported predictive modes:
    - group_bias
    - additive_scalar
    - embedding_residual

    Supported regularization modes:
    - none
    - user_l2
    - anchor_l2   (currently same as shrink-to-zero for user embedding)
    """

    def __init__(
        self,
        model_init_params: dict,
        model_hps: dict,
        side_info_cfg: SideInfoConfig,
        *,
        num_groups: Optional[int] = None,
        feature_shape: Optional[tuple[int, ...]] = None,
    ):
        self.model_init_params = dict(model_init_params)
        self.model_hps = dict(model_hps)
        self.side_info_cfg = side_info_cfg
        self.side_info_cfg.validate()

        self.default_patience = int(
            self.model_init_params.get("patience", self.model_hps.get("patience", 5))
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16

        self.loss_type = str(self.model_init_params.get("loss_type", "mse")).lower()
        if self.loss_type not in {"mse", "wmse", "residual_mse"}:
            raise ValueError("loss_type must be one of {'mse', 'wmse', 'residual_mse'}")

        self.residual_target = self.loss_type == "residual_mse"
        self.use_user_bias = not self.residual_target
        self.use_global_bias = not self.residual_target

        self.backbone = SideInfoMFBackbone(
            num_users=int(self.model_init_params["num_users"]),
            num_items=int(self.model_init_params["num_items"]),
            embedding_size=int(self.model_hps["embedding_size"]),
            use_user_bias=self.use_user_bias,
            use_global_bias=self.use_global_bias,
        ).to(self.device)

        self.embedding_size = int(self.model_hps["embedding_size"])
        self._num_groups = None if num_groups is None else int(num_groups)
        self._feature_shape = feature_shape
        self._flat_feature_dim = None if feature_shape is None else int(np.prod(feature_shape))

        self.side_features_per_user: Optional[torch.Tensor] = None
        self.group_ids_per_user: Optional[torch.Tensor] = None
        self.user_mean_per_user: Optional[torch.Tensor] = None
        self.anchor_vectors_per_user: Optional[torch.Tensor] = None

        self.predictive_head = None
        self.group_bias = None
        self.reg_head = None

        self._build_side_heads()

        self.optimizer_type = str(self.model_init_params["optimizer_type"]).lower()
        self.optimizer = self._make_optimizer(
            lr=float(self.model_hps["lr"]),
            reg=float(self.model_hps["reg_rate"]),
            optimizer_type=self.optimizer_type,
        )

        self._loss_rating_values: Optional[torch.Tensor] = None
        self._loss_rating_weights: Optional[torch.Tensor] = None

        self.history: list[dict] = []
        self._best_state: Optional[dict] = None
        self._regularization_phase: str = "disabled"
        self.scheduler = None
        self._best_optimizer_state = None
        self._best_scheduler_state = None
        self.side_reg_scale = float(self.model_hps.get("side_reg_scale", 1.0))

    # ---------- model building ----------
    def _debug_predictive_stats(
        self,
        user_ids: torch.Tensor,
        base_user_emb: torch.Tensor,
    ) -> dict[str, float]:
        mode = self.side_info_cfg.usage.predictive.mode
        out = {}

        if mode == "none":
            return out

        if mode == "group_bias":
            gids = self._get_group_ids_for_users(user_ids)
            if gids is not None and self.group_bias is not None:
                vals = self.group_bias(gids).squeeze(-1).detach().float()
                out["predictive.group_bias.mean"] = float(vals.mean().item())
                out["predictive.group_bias.std"] = float(vals.std().item()) if vals.numel() > 1 else 0.0
            return out

        feats = self._get_features_for_users(user_ids)
        if feats is None or self.predictive_head is None:
            return out

        if mode == "additive_scalar":
            if feats.ndim == 3 and self.side_info_cfg.representation.matrix_encoder_input["mode"] == "row_shared":
                vals = self.predictive_head(feats).squeeze(-1).detach().float()
            else:
                vals = self.predictive_head(self._flatten_features(feats)).squeeze(-1).detach().float()

            out["predictive.additive.mean"] = float(vals.mean().item())
            out["predictive.additive.std"] = float(vals.std().item()) if vals.numel() > 1 else 0.0
            return out

        if mode == "embedding_residual":
            if feats.ndim == 3 and self.side_info_cfg.representation.matrix_encoder_input["mode"] == "row_shared":
                delta = self.predictive_head(feats).detach().float()
            else:
                delta = self.predictive_head(self._flatten_features(feats)).detach().float()

            norms = torch.linalg.norm(delta, dim=1)
            out["predictive.embedding_residual.norm_mean"] = float(norms.mean().item())
            out["predictive.embedding_residual.norm_std"] = float(norms.std().item()) if norms.numel() > 1 else 0.0
            return out

        return out

    def set_regularization_phase(self, phase: str) -> None:
        phase = str(phase).lower()
        if phase not in {"offline", "incremental", "disabled"}:
            raise ValueError("phase must be one of {'offline', 'incremental', 'disabled'}")
        self._regularization_phase = phase


    def _regularization_is_active(self) -> bool:
        reg_cfg = self.side_info_cfg.usage.regularization
        if reg_cfg.mode == "none":
            return False

        if self._regularization_phase == "offline":
            return bool(reg_cfg.apply_in_offline)

        if self._regularization_phase == "incremental":
            return bool(reg_cfg.apply_in_incremental)

        return False


    def _get_anchor_vectors_for_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        reg_cfg = self.side_info_cfg.usage.regularization

        if reg_cfg.anchor == "zero":
            return torch.zeros(
                (user_ids.shape[0], self.embedding_size),
                dtype=torch.float32,
                device=self.device,
            )

        if self.anchor_vectors_per_user is None:
            raise RuntimeError(
                f"anchor_vectors_per_user is not initialized for anchor='{reg_cfg.anchor}'."
            )

        return self.anchor_vectors_per_user.index_select(0, user_ids)

    def _debug_regularization_stats(self, user_ids: torch.Tensor) -> dict[str, float]:
        out = {}
        if not self._regularization_is_active():
            return out

        uniq = torch.unique(user_ids)
        lam = self._lambda_from_side_info(uniq).detach().float()

        out["regularization.lambda.mean"] = float(lam.mean().item())
        out["regularization.lambda.std"] = float(lam.std().item()) if lam.numel() > 1 else 0.0
        out["regularization.lambda.min"] = float(lam.min().item())
        out["regularization.lambda.max"] = float(lam.max().item())

        reg_cfg = self.side_info_cfg.usage.regularization
        if reg_cfg.mode == "anchor_l2":
            emb = self.backbone.user_embedding(uniq).detach().float()
            anchor = self._get_anchor_vectors_for_users(uniq).detach().float()
            dist = torch.linalg.norm(emb - anchor, dim=1)
            out["regularization.anchor_distance.mean"] = float(dist.mean().item())
            out["regularization.anchor_distance.std"] = float(dist.std().item()) if dist.numel() > 1 else 0.0

        return out


    def _build_side_heads(self) -> None:
        pred_cfg = self.side_info_cfg.usage.predictive
        reg_cfg = self.side_info_cfg.usage.regularization
        rep_cfg = self.side_info_cfg.representation

        if pred_cfg.mode == "group_bias":
            if self._num_groups is None:
                raise ValueError("num_groups is required for predictive.mode='group_bias'.")
            self.group_bias = nn.Embedding(int(self._num_groups), 1).to(self.device)

        elif pred_cfg.mode in {"additive_scalar", "embedding_residual"}:
            out_dim = 1 if pred_cfg.mode == "additive_scalar" else self.embedding_size
            self.predictive_head = self._build_feature_head(
                feature_shape=self._feature_shape,
                output_dim=out_dim,
                hidden_dims=pred_cfg.hidden_dims,
                activation=pred_cfg.activation,
                dropout=pred_cfg.dropout,
                matrix_mode=rep_cfg.matrix_encoder_input["mode"],
            ).to(self.device)

        if reg_cfg.mode != "none":
            if reg_cfg.mapping == "mlp":
                self.reg_head = self._build_feature_head(
                    feature_shape=self._feature_shape if self.side_info_cfg.representation.type != "group_id" else (1,),
                    output_dim=1,
                    hidden_dims=[32],
                    activation="relu",
                    dropout=0.0,
                    matrix_mode=rep_cfg.matrix_encoder_input["mode"],
                ).to(self.device)
            elif reg_cfg.mapping == "fixed_rule":
                self.reg_head = None

    def _build_feature_head(
        self,
        feature_shape: Optional[tuple[int, ...]],
        output_dim: int,
        hidden_dims: list[int],
        activation: str,
        dropout: float,
        matrix_mode: str,
    ) -> nn.Module:
        if feature_shape is None:
            raise ValueError("Continuous feature_shape is required for this head.")

        if len(feature_shape) == 1:
            return build_mlp(
                input_dim=int(feature_shape[0]),
                hidden_dims=hidden_dims,
                output_dim=int(output_dim),
                activation=activation,
                dropout=dropout,
            )

        if len(feature_shape) == 2:
            d, r = feature_shape
            if matrix_mode == "flatten":
                return build_mlp(
                    input_dim=int(d * r),
                    hidden_dims=hidden_dims,
                    output_dim=int(output_dim),
                    activation=activation,
                    dropout=dropout,
                )
            if matrix_mode == "row_shared":
                return RowSharedMatrixEncoder(
                    row_dim=int(r),
                    output_dim=int(output_dim),
                    hidden_dims=hidden_dims,
                    activation=activation,
                    dropout=dropout,
                )

        raise ValueError(f"Unsupported feature_shape={feature_shape} for feature head.")

    # ---------- side info storage ----------

    def set_all_side_info(
        self,
        *,
        features: Optional[np.ndarray],
        group_ids: Optional[np.ndarray],
        user_means: Optional[np.ndarray] = None,
        anchor_vectors: Optional[np.ndarray] = None,
    ) -> None:
        n_users = int(self.backbone.user_embedding.num_embeddings)

        if features is not None:
            ft = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            if int(ft.shape[0]) != n_users:
                raise ValueError(f"features first dim must equal n_users={n_users}. Got {ft.shape[0]}.")
            self.side_features_per_user = ft

        if group_ids is not None:
            gid = torch.as_tensor(group_ids, dtype=torch.long, device=self.device)
            if int(gid.shape[0]) != n_users:
                raise ValueError(f"group_ids first dim must equal n_users={n_users}. Got {gid.shape[0]}.")
            self.group_ids_per_user = gid

        if self.residual_target:
            if user_means is None:
                raise ValueError("user_means must be provided in residual_mse mode.")
            mu = torch.as_tensor(user_means, dtype=torch.float32, device=self.device)
            if int(mu.shape[0]) != n_users:
                raise ValueError(f"user_means first dim must equal n_users={n_users}. Got {mu.shape[0]}.")
            self.user_mean_per_user = mu

        if anchor_vectors is not None:
            av = torch.as_tensor(anchor_vectors, dtype=torch.float32, device=self.device)
            if tuple(av.shape) != (n_users, self.embedding_size):
                raise ValueError(
                    f"anchor_vectors must have shape {(n_users, self.embedding_size)}. "
                    f"Got {tuple(av.shape)}."
                )
            self.anchor_vectors_per_user = av

    def add_users(
        self,
        n_new: int,
        *,
        new_features: Optional[np.ndarray] = None,
        new_group_ids: Optional[np.ndarray] = None,
        new_user_means: Optional[np.ndarray] = None,
        new_anchor_vectors: Optional[np.ndarray] = None,
    ) -> None:
        n_new = int(n_new)
        if n_new <= 0:
            return

        old_n = int(self.backbone.user_embedding.num_embeddings)
        new_n = old_n + n_new

        self.backbone.user_embedding = self._extend_embedding(
            self.backbone.user_embedding,
            new_n,
            init_std=0.01,
            is_bias=False,
        )
        if self.backbone.user_bias is not None:
            self.backbone.user_bias = self._extend_embedding(
                self.backbone.user_bias,
                new_n,
                init_std=0.0,
                is_bias=True,
            )

        if self.side_features_per_user is not None:
            if new_features is None:
                raise ValueError("new_features must be provided when continuous side features are enabled.")
            ft = torch.as_tensor(new_features, dtype=torch.float32, device=self.device)
            self.side_features_per_user = torch.cat([self.side_features_per_user, ft], dim=0)

        if self.group_ids_per_user is not None:
            if new_group_ids is None:
                raise ValueError("new_group_ids must be provided when group_ids are enabled.")
            gid = torch.as_tensor(new_group_ids, dtype=torch.long, device=self.device)
            self.group_ids_per_user = torch.cat([self.group_ids_per_user, gid], dim=0)

        if self.residual_target:
            if self.user_mean_per_user is None:
                raise RuntimeError("user_mean_per_user must exist in residual mode before add_users.")
            if new_user_means is None:
                raise ValueError("new_user_means must be provided in residual_mse mode.")
            mu = torch.as_tensor(new_user_means, dtype=torch.float32, device=self.device)
            self.user_mean_per_user = torch.cat([self.user_mean_per_user, mu], dim=0)

        if new_anchor_vectors is not None:
            av = torch.as_tensor(new_anchor_vectors, dtype=torch.float32, device=self.device)
            if tuple(av.shape) != (n_new, self.embedding_size):
                raise ValueError(
                    f"new_anchor_vectors must have shape {(n_new, self.embedding_size)}. "
                    f"Got {tuple(av.shape)}."
                )

            if self.anchor_vectors_per_user is None:
                old_anchor = torch.zeros(
                    (old_n, self.embedding_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.anchor_vectors_per_user = torch.cat([old_anchor, av], dim=0)
            else:
                self.anchor_vectors_per_user = torch.cat([self.anchor_vectors_per_user, av], dim=0)

        elif self.anchor_vectors_per_user is not None:
            zeros = torch.zeros(
                (n_new, self.embedding_size),
                dtype=torch.float32,
                device=self.device,
            )
            self.anchor_vectors_per_user = torch.cat([self.anchor_vectors_per_user, zeros], dim=0)

        self._rebuild_optimizer()

    # ---------- helpers ----------

    def _extend_embedding(
        self,
        emb: nn.Embedding,
        new_num: int,
        init_std: float,
        is_bias: bool,
    ) -> nn.Embedding:
        old_num, dim = emb.num_embeddings, emb.embedding_dim
        if int(new_num) <= int(old_num):
            return emb

        new_emb = nn.Embedding(int(new_num), int(dim), dtype=emb.weight.dtype, device=emb.weight.device)
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(emb.weight)
            if is_bias:
                new_emb.weight[old_num:].zero_()
            else:
                new_emb.weight[old_num:].normal_(0.0, float(init_std))
        return new_emb

    def _make_optimizer(self, *, lr: float, reg: float, optimizer_type: str, params=None):
        def _dedupe(ps):
            out, seen = [], set()
            for p in ps:
                if p is None or not isinstance(p, torch.nn.Parameter):
                    continue
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(p)
            return out

        if params is None:
            ps = list(self.backbone.parameters())
            if self.group_bias is not None:
                ps += list(self.group_bias.parameters())
            if self.predictive_head is not None:
                ps += list(self.predictive_head.parameters())
            if self.reg_head is not None:
                ps += list(self.reg_head.parameters())
            opt_params = _dedupe(ps)
        else:
            opt_params = _dedupe(list(params))

        # return optim.Adam(opt_params, lr=float(lr), weight_decay=float(reg))
        if optimizer_type == "adaw":
            return optim.AdamW(opt_params, lr=float(lr), weight_decay=float(reg))
        elif optimizer_type == "adam":
            return optim.Adam(opt_params, lr=float(lr), weight_decay=float(reg))

        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _rebuild_optimizer(self) -> None:
        self.optimizer = self._make_optimizer(
            lr=float(self.model_hps["lr"]),
            reg=float(self.model_hps["reg_rate"]),
            optimizer_type=self.optimizer_type,
        )

    def _get_user_means(self, user_ids: torch.Tensor) -> torch.Tensor:
        if not self.residual_target:
            return torch.zeros_like(user_ids, dtype=torch.float32, device=user_ids.device)
        if self.user_mean_per_user is None:
            raise RuntimeError("user_mean_per_user is not initialized.")
        return self.user_mean_per_user.index_select(0, user_ids)

    def _target_from_raw(self, y_raw: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        y_raw = y_raw.float()
        if self.residual_target:
            return y_raw - self._get_user_means(user_ids)
        return y_raw

    def _model_output_to_raw(self, pred_model: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        pred_model = pred_model.float()
        if self.residual_target:
            return pred_model + self._get_user_means(user_ids)
        return pred_model

    def _flatten_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(-1)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    def _get_features_for_users(self, user_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.side_features_per_user is None:
            return None
        return self.side_features_per_user.index_select(0, user_ids)

    def _get_group_ids_for_users(self, user_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.group_ids_per_user is None:
            return None
        return self.group_ids_per_user.index_select(0, user_ids)

    def _predictive_side_contrib(
        self,
        user_ids: torch.Tensor,
        base_user_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            user_emb_after_side, additive_scalar_contrib
        """
        pred_cfg = self.side_info_cfg.usage.predictive
        mode = pred_cfg.mode

        if mode == "none":
            zeros = torch.zeros(user_ids.shape[0], dtype=torch.float32, device=self.device)
            return base_user_emb, zeros

        if mode == "group_bias":
            gids = self._get_group_ids_for_users(user_ids)
            if gids is None or self.group_bias is None:
                raise RuntimeError("group_bias mode requires group ids and group_bias embedding.")
            add = self.group_bias(gids).squeeze(-1)
            return base_user_emb, add

        feats = self._get_features_for_users(user_ids)
        if feats is None or self.predictive_head is None:
            raise RuntimeError(f"{mode} requires continuous side features and predictive_head.")

        if mode == "additive_scalar":
            if feats.ndim == 3 and self.side_info_cfg.representation.matrix_encoder_input["mode"] == "row_shared":
                add = self.predictive_head(feats).squeeze(-1)
            else:
                add = self.predictive_head(self._flatten_features(feats)).squeeze(-1)
            return base_user_emb, add

        if mode == "embedding_residual":
            if feats.ndim == 3 and self.side_info_cfg.representation.matrix_encoder_input["mode"] == "row_shared":
                delta = self.predictive_head(feats)
            else:
                delta = self.predictive_head(self._flatten_features(feats))
            return base_user_emb + delta, torch.zeros(user_ids.shape[0], dtype=torch.float32, device=self.device)

        raise ValueError(f"Unsupported predictive mode='{mode}'")


    def _lambda_from_side_info(self, user_ids: torch.Tensor) -> torch.Tensor:
        reg_cfg = self.side_info_cfg.usage.regularization
        if reg_cfg.mode == "none":
            return torch.zeros(user_ids.shape[0], dtype=torch.float32, device=self.device)

        lo, hi = float(reg_cfg.lambda_min), float(reg_cfg.lambda_max)

        if self.side_info_cfg.representation.type == "group_id":
            gids = self._get_group_ids_for_users(user_ids)
            if gids is None:
                raise RuntimeError("Group ids are required for group-based regularization.")
            x = gids.float().unsqueeze(-1)
        else:
            feats = self._get_features_for_users(user_ids)
            if feats is None:
                raise RuntimeError("Continuous features are required for regularization.")
            if feats.ndim == 3 and self.side_info_cfg.representation.matrix_encoder_input["mode"] == "row_shared":
                x = feats
            else:
                x = self._flatten_features(feats)

        if reg_cfg.mapping == "fixed_rule":
            if x.ndim == 3:
                score = x.mean(dim=(1, 2))
            else:
                score = x.mean(dim=1)
            score = torch.sigmoid(score)
        else:
            if self.reg_head is None:
                raise RuntimeError("reg_head is required for regularization.mapping='mlp'.")
            if x.ndim == 3 and self.side_info_cfg.representation.matrix_encoder_input["mode"] == "row_shared":
                score = torch.sigmoid(self.reg_head(x).squeeze(-1))
            else:
                score = torch.sigmoid(self.reg_head(self._flatten_features(x)).squeeze(-1))

        lam = lo + (hi - lo) * score
        return lam

    def _batch_regularization_penalty(self, user_ids: torch.Tensor) -> torch.Tensor:
        reg_cfg = self.side_info_cfg.usage.regularization
        if not self._regularization_is_active():
            return torch.zeros((), dtype=torch.float32, device=self.device)

        uniq = torch.unique(user_ids)
        lam = self._lambda_from_side_info(uniq)
        emb = self.backbone.user_embedding(uniq)

        if reg_cfg.mode == "user_l2":
            diff = emb
        elif reg_cfg.mode == "anchor_l2":
            anchor = self._get_anchor_vectors_for_users(uniq)
            diff = emb - anchor
        else:
            return torch.zeros((), dtype=torch.float32, device=self.device)

        pen = (lam.unsqueeze(-1) * diff.square()).sum(dim=1).mean()
        return pen


    def forward_model_space(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        base_user_emb = self.backbone.user_embedding(user_ids)
        user_emb_after_side, add = self._predictive_side_contrib(user_ids, base_user_emb)
        out = self.backbone.score_from_user_embedding(
            user_emb=user_emb_after_side,
            item_ids=item_ids,
            user_ids_for_bias=user_ids,
        )
        return out + add

    # ---------- loss ----------

    def _set_loss_from_train_df(self, df: pd.DataFrame) -> None:
        if self.loss_type != "wmse":
            self._loss_rating_values = None
            self._loss_rating_weights = None
            return

        rating_col = self.model_init_params["rating_col"]
        vc = df[rating_col].value_counts().sort_index()

        rating_values = vc.index.to_numpy(dtype=np.float32, copy=True)
        rating_counts = vc.to_numpy(dtype=np.float32, copy=True)

        alpha = float(self.model_init_params.get("wmse_alpha", 0.5))
        weights = 1.0 / np.power(rating_counts, alpha)

        cap = self.model_init_params.get("wmse_cap", None)
        if cap is not None:
            weights = np.minimum(weights, float(cap))

        weights = weights / weights.mean()

        self._loss_rating_values = torch.as_tensor(rating_values, dtype=torch.float32, device=self.device)
        self._loss_rating_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

    def _get_sample_weights(self, y_true_raw: torch.Tensor) -> torch.Tensor:
        if self.loss_type != "wmse":
            return torch.ones_like(y_true_raw, dtype=torch.float32, device=y_true_raw.device)

        if self._loss_rating_values is None or self._loss_rating_weights is None:
            raise RuntimeError("WMSE is enabled, but weights were not initialized.")

        weights = torch.ones_like(y_true_raw, dtype=torch.float32, device=y_true_raw.device)
        vals = self._loss_rating_values.to(y_true_raw.device)
        wts = self._loss_rating_weights.to(y_true_raw.device)

        for rv, rw in zip(vals, wts):
            weights = torch.where(y_true_raw == rv, rw, weights)
        return weights

    def _compute_total_loss(
        self,
        pred_model: torch.Tensor,
        y_raw: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_target = self._target_from_raw(y_raw, user_ids)

        if self.loss_type == "wmse":
            diff = pred_model.float() - y_target.float()
            rating_loss = (diff.square() * self._get_sample_weights(y_raw)).mean()
        else:
            rating_loss = torch.mean((pred_model.float() - y_target.float()) ** 2)

        reg_pen = self._batch_regularization_penalty(user_ids)
        total = rating_loss + self.side_reg_scale * reg_pen
        return total, reg_pen

    def _make_scheduler(self, cfg: Optional[dict]):
        if not cfg:
            return None

        name = str(cfg.get("name", "none")).lower()
        if name == "none":
            return None

        if name == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(cfg.get("factor", 0.5)),
                patience=int(cfg.get("patience", 3)),
                threshold=float(cfg.get("threshold", 1e-4)),
                min_lr=float(cfg.get("min_lr", 1e-6)),
            )

        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(cfg["t_max"]),
                eta_min=float(cfg.get("min_lr", 1e-6)),
            )

        raise ValueError(f"Unsupported scheduler: {name}")

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _current_weight_decay(self) -> float:
        return float(self.optimizer.param_groups[0].get("weight_decay", 0.0))


    def set_optimizer_hparams(
        self,
        *,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> None:
        for group in self.optimizer.param_groups:
            if lr is not None:
                group["lr"] = float(lr)
            if weight_decay is not None:
                group["weight_decay"] = float(weight_decay)

    def set_side_reg_scale(self, value: float) -> None:
        self.side_reg_scale = float(value)


    # ---------- pack helpers ----------

    def _df_to_tensorpack(self, df: pd.DataFrame) -> TensorPack:
        rating_col = self.model_init_params["rating_col"]

        u = torch.as_tensor(df["user_id"].to_numpy(dtype=np.int64, copy=True), device="cpu")
        i = torch.as_tensor(df["item_id"].to_numpy(dtype=np.int64, copy=True), device="cpu")
        y = torch.as_tensor(df[rating_col].to_numpy(dtype=np.float32, copy=True), device="cpu")

        return TensorPack(
            u=u.to(self.device, non_blocking=True),
            i=i.to(self.device, non_blocking=True),
            f=torch.empty((len(df),), dtype=torch.long, device=self.device),   # unused placeholder
            y=y.to(self.device, non_blocking=True),
        )

    def _iter_minibatches(self, pack: TensorPack, batch_size: int, shuffle: bool):
        n = int(pack.y.numel())
        if n == 0:
            return
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            yield pack.u[b], pack.i[b], pack.y[b]

    # ---------- training loops ----------
    def _train_epoch_pack(self, tr_pack: TensorPack) -> dict:
        self.backbone.train()
        if self.group_bias is not None:
            self.group_bias.train()
        if self.predictive_head is not None:
            self.predictive_head.train()
        if self.reg_head is not None:
            self.reg_head.train()

        bs = int(self.model_hps["batch_size"])
        sse = 0.0
        n_obs = 0
        reg_sum = 0.0
        n_batches = 0

        debug_sums: dict[str, float] = {}
        debug_counts: dict[str, int] = {}

        for u, i, y in self._iter_minibatches(tr_pack, bs, shuffle=True):
            self.optimizer.zero_grad(set_to_none=True)

            pred_model = self.forward_model_space(u, i)
            loss, reg_pen = self._compute_total_loss(pred_model, y, u)

            loss.backward()
            self.optimizer.step()

            pred_raw = self._model_output_to_raw(pred_model.detach(), u)
            diff_raw = pred_raw.float() - y.float()

            sse += float(diff_raw.square().sum().item())
            n_obs += int(y.numel())
            reg_sum += float(reg_pen.detach().item())
            n_batches += 1

            # -----------------------------
            # Debug / observability logging
            # -----------------------------
            with torch.no_grad():
                base_user_emb = self.backbone.user_embedding(u)

                pred_stats = {}
                reg_stats = {}

                if hasattr(self, "_debug_predictive_stats"):
                    pred_stats = self._debug_predictive_stats(u, base_user_emb) or {}

                if hasattr(self, "_debug_regularization_stats"):
                    reg_stats = self._debug_regularization_stats(u) or {}

                for stats_dict in (pred_stats, reg_stats):
                    for k, v in stats_dict.items():
                        debug_sums[k] = debug_sums.get(k, 0.0) + float(v)
                        debug_counts[k] = debug_counts.get(k, 0) + 1

        out = {
            "train_step_mse": sse / max(n_obs, 1),
            "avg_reg_penalty": reg_sum / max(n_batches, 1),
        }

        for k, total in debug_sums.items():
            out[k] = total / max(debug_counts.get(k, 1), 1)

        return out

    @torch.no_grad()
    def _evaluate_pack(self, pack: TensorPack) -> float:
        self.backbone.eval()
        if self.group_bias is not None:
            self.group_bias.eval()
        if self.predictive_head is not None:
            self.predictive_head.eval()
        if self.reg_head is not None:
            self.reg_head.eval()

        if int(pack.y.numel()) == 0:
            return 0.0

        bs = int(self.model_hps["batch_size"])
        sse = 0.0
        n_obs = 0

        for u, i, y in self._iter_minibatches(pack, bs, shuffle=False):
            pred_model = self.forward_model_space(u, i)
            pred_raw = self._model_output_to_raw(pred_model, u)
            diff = pred_raw.float() - y.float()
            sse += float(diff.square().sum().item())
            n_obs += int(y.numel())

        return sse / max(n_obs, 1)

    def _reset_best(self) -> None:
        self._best_state = None

    def _save_best(self) -> None:
        state = {
            "backbone": {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()},
            "group_bias": None if self.group_bias is None else {k: v.detach().cpu() for k, v in self.group_bias.state_dict().items()},
            "predictive_head": None if self.predictive_head is None else {k: v.detach().cpu() for k, v in self.predictive_head.state_dict().items()},
            "reg_head": None if self.reg_head is None else {k: v.detach().cpu() for k, v in self.reg_head.state_dict().items()},
        }
        self._best_state = state
        self._best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self._best_scheduler_state = None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict())

    def _load_best(self) -> None:
        if self._best_state is None:
            return
        self.backbone.load_state_dict({k: v.to(self.device) for k, v in self._best_state["backbone"].items()})
        if self.group_bias is not None and self._best_state["group_bias"] is not None:
            self.group_bias.load_state_dict({k: v.to(self.device) for k, v in self._best_state["group_bias"].items()})
        if self.predictive_head is not None and self._best_state["predictive_head"] is not None:
            self.predictive_head.load_state_dict({k: v.to(self.device) for k, v in self._best_state["predictive_head"].items()})
        if self.reg_head is not None and self._best_state["reg_head"] is not None:
            self.reg_head.load_state_dict({k: v.to(self.device) for k, v in self._best_state["reg_head"].items()})
        if self._best_optimizer_state is not None:
            self.optimizer.load_state_dict(self._best_optimizer_state)
        if self.scheduler is not None and self._best_scheduler_state is not None:
            self.scheduler.load_state_dict(self._best_scheduler_state)


    # ---------- public train / finetune ----------
    def fit_offline(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        use_early_stopping: bool = True,
        min_delta: float = 0.0,
        patience: Optional[int] = None,
        verbose: bool = True,
    ) -> list[dict]:
        scheduler_cfg = self.model_hps.get("offline_scheduler", None)
        self.scheduler = self._make_scheduler(scheduler_cfg)
        epochs = int(self.model_init_params["num_epochs"])
        patience = self.default_patience if patience is None else int(patience)
        self._set_loss_from_train_df(train_df)

        if not self.residual_target:
            rating_col = self.model_init_params["rating_col"]
            init_global = float(train_df[rating_col].mean()) if train_df is not None and not train_df.empty else 0.0
            if self.backbone.global_bias is not None:
                with torch.no_grad():
                    self.backbone.global_bias.fill_(float(init_global))

        tr_pack = self._df_to_tensorpack(train_df)
        vl_pack = self._df_to_tensorpack(val_df) if val_df is not None and not val_df.empty else None
        te_pack = self._df_to_tensorpack(test_df) if test_df is not None and not test_df.empty else None

        best_val = float("inf")
        bad = 0
        self.history.clear()
        self._reset_best()
        self.set_regularization_phase("offline")

        try:
            for ep in range(1, epochs + 1):
                train_stats = self._train_epoch_pack(tr_pack)
                train_eval_mse = float(self._evaluate_pack(tr_pack))
                val_mse = train_eval_mse if vl_pack is None else float(self._evaluate_pack(vl_pack))
                test_rmse = 0.0 if te_pack is None else float(self._evaluate_pack(te_pack)) ** 0.5

                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_mse)
                    else:
                        self.scheduler.step()

                row = {
                    "epoch": ep,
                    "train_mse": train_eval_mse,
                    "val_mse": val_mse,
                    "train_step_mse": float(train_stats["train_step_mse"]),
                    "avg_reg_penalty": float(train_stats["avg_reg_penalty"]),
                    "lr": self._current_lr(),
                    "weight_decay": self._current_weight_decay(),
                    "side_reg_scale": float(self.side_reg_scale),
                }
                self.history.append(row)

                if verbose:
                    print(
                        f"[Offline][{ep:03d}/{epochs}] "
                        f"train_mse={train_eval_mse:.6f} "
                        f"val_mse={val_mse:.6f} "
                        f"test_rmse={test_rmse:.6f} "
                        f"reg={train_stats['avg_reg_penalty']:.6f} "
                        f"lr={self._current_lr():.6g} "
                        f"wd={self._current_weight_decay():.6g} "
                    )

                if vl_pack is not None:
                    if val_mse < best_val - float(min_delta):
                        best_val = val_mse
                        bad = 0
                        self._save_best()
                    else:
                        bad += 1

                    if use_early_stopping and bad >= patience:
                        if verbose:
                            print(f"[Offline] early stop at epoch {ep} (best_val_mse={best_val:.6f})")
                        break

            if vl_pack is not None and self._best_state is not None:
                self._load_best()
                if verbose:
                    print(f"[Offline] restored best weights (best_val_mse={best_val:.6f})")

            return self.history.copy()
        finally:
            self.set_regularization_phase("disabled")


    def finetune_new_users(
        self,
        train_inc_df: pd.DataFrame,
        val_inc_df: Optional[pd.DataFrame],
        new_user_ids: np.ndarray | list[int],
        *,
        epochs: int,
        lr: float,
        reg_rate: float,
        batch_size: int,
        early_stop: bool = True,
        patience: int = 2,
        min_delta: float = 0.0,
        anchor_to_zero: float = 0.0,
        verbose: bool = True,
        strict_freeze_old: bool = True,
    ) -> list[dict]:
        new_user_ids = np.asarray(new_user_ids, dtype=np.int64)
        if new_user_ids.size == 0:
            return []

        uid_set = set(new_user_ids.tolist())
        tr_df = train_inc_df[train_inc_df.user_id.isin(uid_set)].copy()
        vl_df = None if (val_inc_df is None or val_inc_df.empty) else val_inc_df[val_inc_df.user_id.isin(uid_set)].copy()

        if tr_df.empty:
            return []

        self._set_loss_from_train_df(tr_df)

        old_hps = dict(self.model_hps)
        self.model_hps.update({
            "lr": float(lr),
            "batch_size": int(batch_size),
        })

        policy = self.side_info_cfg.usage.incremental_branch_policy
        predictive_policy = str(policy.predictive).lower()
        regularization_policy = str(policy.regularization).lower()

        predictive_trainable = (predictive_policy == "train")
        regularization_trainable = (regularization_policy == "train")

        def _set_module_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = bool(flag)

        self.backbone.user_embedding.weight.requires_grad = True
        if self.backbone.user_bias is not None:
            self.backbone.user_bias.weight.requires_grad = True

        self.backbone.item_embedding.weight.requires_grad = False
        self.backbone.item_bias.weight.requires_grad = False
        if self.backbone.global_bias is not None:
            self.backbone.global_bias.requires_grad = False

        _set_module_requires_grad(self.group_bias, predictive_trainable)
        _set_module_requires_grad(self.predictive_head, predictive_trainable)
        _set_module_requires_grad(self.reg_head, regularization_trainable)

        n_total = int(self.backbone.user_embedding.num_embeddings)
        new_mask = torch.zeros(n_total, dtype=torch.bool, device=self.device)
        new_mask[new_user_ids] = True
        old_mask = ~new_mask

        with torch.no_grad():
            U0 = self.backbone.user_embedding.weight.detach().clone()
            bU0 = None if self.backbone.user_bias is None else self.backbone.user_bias.weight.detach().clone()
            V0 = self.backbone.item_embedding.weight.detach().clone()
            bI0 = self.backbone.item_bias.weight.detach().clone()
            G0 = None if self.backbone.global_bias is None else self.backbone.global_bias.detach().clone()

            GB0 = None if self.group_bias is None else {
                k: v.detach().clone() for k, v in self.group_bias.state_dict().items()
            }
            PH0 = None if self.predictive_head is None else {
                k: v.detach().clone() for k, v in self.predictive_head.state_dict().items()
            }
            RH0 = None if self.reg_head is None else {
                k: v.detach().clone() for k, v in self.reg_head.state_dict().items()
            }

        tr_pack = self._df_to_tensorpack(tr_df)
        vl_pack = None if (vl_df is None or vl_df.empty) else self._df_to_tensorpack(vl_df)

        opt_params = [self.backbone.user_embedding.weight]
        if self.backbone.user_bias is not None:
            opt_params.append(self.backbone.user_bias.weight)

        if predictive_trainable:
            if self.group_bias is not None:
                opt_params += list(self.group_bias.parameters())
            if self.predictive_head is not None:
                opt_params += list(self.predictive_head.parameters())

        if regularization_trainable and self.reg_head is not None:
            opt_params += list(self.reg_head.parameters())

        self.optimizer = self._make_optimizer(
            lr=float(lr),
            reg=float(reg_rate),
            optimizer_type=self.optimizer_type,
            params=opt_params,
        )
        inc_scheduler_cfg = self.model_hps.get("incremental_scheduler", None)
        self.scheduler = self._make_scheduler(inc_scheduler_cfg)

        best_val = float("inf")
        bad = 0
        self.history.clear()
        self._reset_best()
        self.set_regularization_phase("incremental")

        try:
            for ep in range(1, int(epochs) + 1):
                train_stats = self._train_epoch_pack(tr_pack)

                if strict_freeze_old:
                    with torch.no_grad():
                        self.backbone.user_embedding.weight.data[old_mask] = U0[old_mask]
                        if self.backbone.user_bias is not None and bU0 is not None:
                            self.backbone.user_bias.weight.data[old_mask] = bU0[old_mask]

                        self.backbone.item_embedding.weight.data.copy_(V0)
                        self.backbone.item_bias.weight.data.copy_(bI0)
                        if self.backbone.global_bias is not None and G0 is not None:
                            self.backbone.global_bias.data.copy_(G0)

                        if not predictive_trainable:
                            if self.group_bias is not None and GB0 is not None:
                                self.group_bias.load_state_dict(GB0)
                            if self.predictive_head is not None and PH0 is not None:
                                self.predictive_head.load_state_dict(PH0)

                        if not regularization_trainable:
                            if self.reg_head is not None and RH0 is not None:
                                self.reg_head.load_state_dict(RH0)

                if anchor_to_zero > 0.0:
                    with torch.no_grad():
                        self.backbone.user_embedding.weight.data[new_mask] *= (1.0 / (1.0 + float(anchor_to_zero)))

                train_eval_mse = float(self._evaluate_pack(tr_pack))
                val_mse = train_eval_mse if vl_pack is None else float(self._evaluate_pack(vl_pack))

                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_mse)
                    else:
                        self.scheduler.step()

                row = {
                    "epoch": ep,
                    "train_mse": train_eval_mse,
                    "val_mse": val_mse,
                    "train_step_mse": float(train_stats["train_step_mse"]),
                    "avg_reg_penalty": float(train_stats["avg_reg_penalty"]),
                    "lr": self._current_lr(),
                    "weight_decay": self._current_weight_decay(),
                    "side_reg_scale": float(self.side_reg_scale),
                }

                for k, v in train_stats.items():
                    if k not in row:
                        row[k] = float(v)

                self.history.append(row)

                if verbose:
                    msg = (
                        f"[Finetune][{ep:03d}/{epochs}] "
                        f"train_mse={train_eval_mse:.6f} "
                        f"val_mse={val_mse:.6f} "
                        f"reg={train_stats['avg_reg_penalty']:.6f} "
                        f"pred_branch={predictive_policy} "
                        f"reg_branch={regularization_policy} "
                        f"lr={self._current_lr():.6g} "
                        f"wd={self._current_weight_decay():.6g} "
                    )
                    print(msg)

                if vl_pack is not None:
                    if val_mse < best_val - float(min_delta):
                        best_val = val_mse
                        bad = 0
                        self._save_best()
                    else:
                        bad += 1

                    if early_stop and bad >= int(patience):
                        if verbose:
                            print(f"[Finetune] early stop at epoch {ep} (best_val_mse={best_val:.6f})")
                        break

            if vl_pack is not None and self._best_state is not None:
                self._load_best()
                if verbose:
                    print(f"[Finetune] restored best weights (best_val_mse={best_val:.6f})")

            if strict_freeze_old:
                with torch.no_grad():
                    self.backbone.user_embedding.weight.data[old_mask] = U0[old_mask]
                    if self.backbone.user_bias is not None and bU0 is not None:
                        self.backbone.user_bias.weight.data[old_mask] = bU0[old_mask]

                    self.backbone.item_embedding.weight.data.copy_(V0)
                    self.backbone.item_bias.weight.data.copy_(bI0)
                    if self.backbone.global_bias is not None and G0 is not None:
                        self.backbone.global_bias.data.copy_(G0)

                    if not predictive_trainable:
                        if self.group_bias is not None and GB0 is not None:
                            self.group_bias.load_state_dict(GB0)
                        if self.predictive_head is not None and PH0 is not None:
                            self.predictive_head.load_state_dict(PH0)

                    if not regularization_trainable:
                        if self.reg_head is not None and RH0 is not None:
                            self.reg_head.load_state_dict(RH0)

            return self.history.copy()

        finally:
            self.backbone.user_embedding.weight.requires_grad = True
            if self.backbone.user_bias is not None:
                self.backbone.user_bias.weight.requires_grad = True

            self.backbone.item_embedding.weight.requires_grad = True
            self.backbone.item_bias.weight.requires_grad = True
            if self.backbone.global_bias is not None:
                self.backbone.global_bias.requires_grad = True

            _set_module_requires_grad(self.group_bias, True)
            _set_module_requires_grad(self.predictive_head, True)
            _set_module_requires_grad(self.reg_head, True)

            self.model_hps.update(old_hps)
            self._rebuild_optimizer()
            self.set_regularization_phase("disabled")

    # ---------- public prediction ----------

    def evaluate_mse(self, df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        return float(self._evaluate_pack(self._df_to_tensorpack(df)))

    @torch.no_grad()
    def predict_from_df(
        self,
        df: pd.DataFrame,
        *,
        batch_size: Optional[int] = None,
        clip_bounds: Optional[tuple[float, float]] = None,
        output_space: Literal["raw", "model"] = "raw",
    ) -> np.ndarray:
        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        bs = int(batch_size or self.model_hps.get("batch_size", 4096))
        user_ids = df["user_id"].to_numpy(dtype=np.int64, copy=True)
        item_ids = df["item_id"].to_numpy(dtype=np.int64, copy=True)
        out = np.empty((len(df),), dtype=np.float32)

        self.backbone.eval()
        if self.group_bias is not None:
            self.group_bias.eval()
        if self.predictive_head is not None:
            self.predictive_head.eval()
        if self.reg_head is not None:
            self.reg_head.eval()

        for s in range(0, len(df), bs):
            e = min(s + bs, len(df))
            u = torch.as_tensor(user_ids[s:e], dtype=torch.long, device=self.device)
            i = torch.as_tensor(item_ids[s:e], dtype=torch.long, device=self.device)

            pred_model = self.forward_model_space(u, i)
            pred = pred_model.float() if output_space == "model" else self._model_output_to_raw(pred_model, u)

            if clip_bounds is not None:
                pred = torch.clamp(pred, clip_bounds[0], clip_bounds[1])

            out[s:e] = pred.detach().cpu().numpy().astype(np.float32, copy=False)

        return out

    @torch.no_grad()
    def predict_user_item_grid(
        self,
        user_ids,
        item_ids,
        *,
        user_batch_size: int = 512,
        item_batch_size: int = 4096,
        clip_bounds: Optional[tuple[float, float]] = None,
        output_space: Literal["raw", "model"] = "raw",
    ) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)

        n_u = int(user_ids.shape[0])
        n_i = int(item_ids.shape[0])
        if n_u == 0 or n_i == 0:
            return np.empty((n_u, n_i), dtype=np.float32)

        out = np.empty((n_u, n_i), dtype=np.float32)

        self.backbone.eval()
        if self.group_bias is not None:
            self.group_bias.eval()
        if self.predictive_head is not None:
            self.predictive_head.eval()
        if self.reg_head is not None:
            self.reg_head.eval()

        for us in range(0, n_u, int(user_batch_size)):
            ue = min(us + int(user_batch_size), n_u)
            u = torch.as_tensor(user_ids[us:ue], dtype=torch.long, device=self.device)

            base_user_emb = self.backbone.user_embedding(u)
            user_emb_after_side, add = self._predictive_side_contrib(u, base_user_emb)

            for is_ in range(0, n_i, int(item_batch_size)):
                ie = min(is_ + int(item_batch_size), n_i)
                i = torch.as_tensor(item_ids[is_:ie], dtype=torch.long, device=self.device)

                item_emb = self.backbone.item_embedding(i)
                item_bias = self.backbone.item_bias(i).squeeze(-1)
                pred = user_emb_after_side @ item_emb.T + item_bias.unsqueeze(0)

                if self.backbone.user_bias is not None:
                    pred = pred + self.backbone.user_bias(u).squeeze(-1).unsqueeze(1)

                if self.backbone.global_bias is not None:
                    pred = pred + self.backbone.global_bias

                pred = pred + add.unsqueeze(1)

                if output_space == "raw" and self.residual_target:
                    pred = pred + self._get_user_means(u).unsqueeze(1)

                if clip_bounds is not None:
                    pred = torch.clamp(pred, clip_bounds[0], clip_bounds[1])

                out[us:ue, is_:ie] = pred.detach().cpu().numpy().astype(np.float32, copy=False)

        return out



# ===== Notebook cell 27 =====
# Side-Info-Aware Experiment

class JLRaceMFSideInfoExperiment(BaseExperiment):
    """
    New experiment family:
    - offline training with sketch-derived side info
    - incremental phase = finetune new users only
    - online phase = retrieve closest users from live sketch and use learned MF parameters

    Supports:
    - reference_mode = offline_frozen / appended_live
    - representation = group_id / scalar / d_vector / dr_matrix / engineered_features
    - predictive usage = group_bias / additive_scalar / embedding_residual
    - regularization usage = none / user_l2 / anchor_l2
    """

    # ----------------------------
    # init / validation
    # ----------------------------

    def _get_side_info_cfg_dict(self) -> dict:
        raw = getattr(self.params, "side_info", None)
        if raw is not None:
            return raw
        return dict(self.params.model_init.get("side_info", {}) or {})

    def _validate_required_params(self) -> None:
        required_model_init = ["rating_col", "num_epochs", "patience", "loss_type"]
        required_sketch = ["jl_components", "race_k", "race_d", "race_w", "race_R", "n_blocks", "n_perms", "batch_size"]
        required_model_hps = ["embedding_size", "reg_rate", "lr", "batch_size"]

        for k in required_model_init:
            if self.params.model_init.get(k, None) is None:
                raise ValueError(f"Missing required model_init parameter: {k}")

        for k in required_sketch:
            if self.params.sketch.get(k, None) is None:
                raise ValueError(f"Missing required sketch parameter: {k}")

        for k in required_model_hps:
            if self.params.model_hps.get(k, None) is None:
                raise ValueError(f"Missing required model_hps parameter: {k}")

        inc_hps = getattr(self.params, "incremental_hps", None) or {}
        required_inc = [
            "incremental_lr",
            "incremental_reg_rate",
            "incremental_batch_size",
            "incremental_epochs",
            "incremental_patience",
            "incremental_anchor",
        ]
        for k in required_inc:
            if inc_hps.get(k, None) is None:
                raise ValueError(f"Missing required incremental_hps parameter: {k}")

        loss_type = str(self.params.model_init.get("loss_type", "")).lower()
        if loss_type != "residual_mse":
            raise ValueError(
                "This experiment currently assumes loss_type='residual_mse'. "
                f"Got: {loss_type}"
            )


    def _global_anchor_from_existing_users(
        self,
        candidate_user_ids: np.ndarray,
        *,
        n_rows: int,
    ) -> np.ndarray:
        candidate_user_ids = np.asarray(candidate_user_ids, dtype=np.int64)
        if candidate_user_ids.size == 0:
            return np.zeros((n_rows, int(self.mf.embedding_size)), dtype=np.float32)

        emb = (
            self.mf.backbone.user_embedding.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        mu = emb[candidate_user_ids].mean(axis=0, keepdims=True)
        return np.repeat(mu, repeats=int(n_rows), axis=0).astype(np.float32, copy=False)


    def _build_group_centroid_anchors(
        self,
        *,
        new_group_ids: np.ndarray,
        candidate_user_ids: np.ndarray,
    ) -> np.ndarray:
        candidate_user_ids = np.asarray(candidate_user_ids, dtype=np.int64)
        new_group_ids = np.asarray(new_group_ids, dtype=np.int64)

        if candidate_user_ids.size == 0:
            return np.zeros((len(new_group_ids), int(self.mf.embedding_size)), dtype=np.float32)

        if self.mf.group_ids_per_user is None:
            raise RuntimeError("group_centroid anchor requires existing group ids on current users.")

        all_emb = self.mf.backbone.user_embedding.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        all_gids = self.mf.group_ids_per_user.detach().cpu().numpy().astype(np.int64, copy=False)

        cand_emb = all_emb[candidate_user_ids]
        cand_gid = all_gids[candidate_user_ids]

        global_mu = cand_emb.mean(axis=0, keepdims=True)
        centroids: dict[int, np.ndarray] = {}

        for gid in np.unique(cand_gid):
            mask = (cand_gid == gid)
            if np.any(mask):
                centroids[int(gid)] = cand_emb[mask].mean(axis=0).astype(np.float32, copy=False)

        out = np.empty((len(new_group_ids), cand_emb.shape[1]), dtype=np.float32)
        for idx, gid in enumerate(new_group_ids):
            out[idx] = centroids.get(int(gid), global_mu[0])

        return out.astype(np.float32, copy=False)


    def _build_neighbor_mean_anchors(
        self,
        *,
        query_vectors: torch.Tensor,
        candidate_user_ids: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        candidate_user_ids = np.asarray(candidate_user_ids, dtype=np.int64)
        n_rows = int(query_vectors.shape[0])

        if candidate_user_ids.size == 0:
            return np.zeros((n_rows, int(self.mf.embedding_size)), dtype=np.float32)

        top_vals, top_idx = self.reference_race.find_closest_users(
            query_vectors=query_vectors,
            top_k=int(top_k),
            return_kernels=False,
            candidate_user_ids=candidate_user_ids,
        )

        weights = normalize_rows_nonneg(top_vals).astype(np.float32, copy=False)
        all_emb = self.mf.backbone.user_embedding.weight.detach().cpu().numpy().astype(np.float32, copy=False)

        if top_idx.shape[1] == 0:
            return self._global_anchor_from_existing_users(candidate_user_ids, n_rows=n_rows)

        anchors = np.einsum("qk,qkd->qd", weights, all_emb[top_idx]).astype(np.float32, copy=False)
        return anchors


    def _compute_incremental_anchor_vectors(
        self,
        *,
        new_user_ids: np.ndarray,
        query_vectors: torch.Tensor,
        new_group_ids: Optional[np.ndarray],
        candidate_user_ids: np.ndarray,
    ) -> Optional[np.ndarray]:
        reg_cfg = self.side_info_cfg.usage.regularization

        if reg_cfg.mode != "anchor_l2":
            return None

        if not reg_cfg.apply_in_incremental:
            return None

        anchor = str(reg_cfg.anchor).lower()
        n_rows = int(len(new_user_ids))

        if anchor == "zero":
            return np.zeros((n_rows, int(self.mf.embedding_size)), dtype=np.float32)

        if anchor == "offline_global_mean":
            return self._global_anchor_from_existing_users(candidate_user_ids, n_rows=n_rows)

        if anchor == "group_centroid":
            if new_group_ids is None:
                raise RuntimeError("group_centroid anchor requires new_group_ids.")
            return self._build_group_centroid_anchors(
                new_group_ids=new_group_ids,
                candidate_user_ids=candidate_user_ids,
            )

        if anchor == "neighbor_mean":
            return self._build_neighbor_mean_anchors(
                query_vectors=query_vectors,
                candidate_user_ids=candidate_user_ids,
                top_k=int(self._sideinfo_neighbor_top_k),
            )

        raise ValueError(f"Unsupported incremental anchor='{anchor}'")

    def _log_old_user_side_info_drift(
        self,
        before: SideInfoBatch,
        after: SideInfoBatch,
        *,
        prefix: str,
    ) -> None:
        if not hasattr(self, "artifacts") or not hasattr(self.artifacts, "logs"):
            return

        logs = self.artifacts.logs

        if before.features is not None and after.features is not None:
            x0 = np.asarray(before.features, dtype=np.float32).reshape(before.features.shape[0], -1)
            x1 = np.asarray(after.features, dtype=np.float32).reshape(after.features.shape[0], -1)

            diff = x1 - x0
            abs_diff = np.abs(diff)

            logs[f"{prefix}.features.mean_abs_diff"] = float(abs_diff.mean()) if abs_diff.size else 0.0
            logs[f"{prefix}.features.max_abs_diff"] = float(abs_diff.max()) if abs_diff.size else 0.0

            denom = np.maximum(np.abs(x0), 1e-8)
            rel = abs_diff / denom
            logs[f"{prefix}.features.mean_rel_diff"] = float(rel.mean()) if rel.size else 0.0

            x0n = np.linalg.norm(x0, axis=1)
            x1n = np.linalg.norm(x1, axis=1)
            cos = np.sum(x0 * x1, axis=1) / np.maximum(x0n * x1n, 1e-8)
            logs[f"{prefix}.features.cosine_mean"] = float(cos.mean()) if cos.size else 0.0
            logs[f"{prefix}.features.cosine_std"] = float(cos.std()) if cos.size else 0.0

        if before.group_ids is not None and after.group_ids is not None:
            g0 = np.asarray(before.group_ids, dtype=np.int64)
            g1 = np.asarray(after.group_ids, dtype=np.int64)
            changed = (g0 != g1)
            logs[f"{prefix}.group_ids.changed_frac"] = float(changed.mean()) if changed.size else 0.0


    def _log_side_info_batch_stats(self, name: str, batch: SideInfoBatch) -> None:
        if not hasattr(self, "artifacts") or not hasattr(self.artifacts, "logs"):
            return

        logs = self.artifacts.logs

        raw = batch.raw
        logs[f"{name}.raw.scalar_freq.mean"] = float(np.mean(raw.scalar_freq)) if raw.scalar_freq.size else 0.0
        logs[f"{name}.raw.scalar_freq.std"] = float(np.std(raw.scalar_freq)) if raw.scalar_freq.size else 0.0
        logs[f"{name}.raw.scalar_freq.min"] = float(np.min(raw.scalar_freq)) if raw.scalar_freq.size else 0.0
        logs[f"{name}.raw.scalar_freq.max"] = float(np.max(raw.scalar_freq)) if raw.scalar_freq.size else 0.0

        logs[f"{name}.raw.row_mom.mean"] = float(np.mean(raw.row_mom)) if raw.row_mom.size else 0.0
        logs[f"{name}.raw.row_mom.std"] = float(np.std(raw.row_mom)) if raw.row_mom.size else 0.0

        logs[f"{name}.raw.raw_dr.mean"] = float(np.mean(raw.raw_dr)) if raw.raw_dr.size else 0.0
        logs[f"{name}.raw.raw_dr.std"] = float(np.std(raw.raw_dr)) if raw.raw_dr.size else 0.0

        if batch.features is not None:
            feats = np.asarray(batch.features, dtype=np.float32)
            flat = feats.reshape(feats.shape[0], -1)
            norms = np.linalg.norm(flat, axis=1)

            logs[f"{name}.features.shape"] = tuple(feats.shape)
            logs[f"{name}.features.mean"] = float(flat.mean()) if flat.size else 0.0
            logs[f"{name}.features.std"] = float(flat.std()) if flat.size else 0.0
            logs[f"{name}.features.norm_mean"] = float(norms.mean()) if norms.size else 0.0
            logs[f"{name}.features.norm.std"] = float(norms.std()) if norms.size else 0.0

        if batch.group_ids is not None:
            gids = np.asarray(batch.group_ids, dtype=np.int64)
            uniq, cnt = np.unique(gids, return_counts=True)
            logs[f"{name}.group_ids.n_unique"] = int(len(uniq))
            logs[f"{name}.group_ids.top_counts"] = {
                int(u): int(c) for u, c in zip(uniq[:10], cnt[:10])
            }

    def _init_models(self) -> None:
        self._validate_required_params()

        self.side_info_cfg = SideInfoConfig.from_dict(self._get_side_info_cfg_dict())

        self.rating_col = str(self.params.model_init["rating_col"])
        self.loss_type = str(self.params.model_init["loss_type"]).lower()
        self.verbose = bool(self.params.model_init.get("verbose", True))
        self.residual_mean_shrinkage = float(self.params.model_init.get("residual_mean_shrinkage", 5.0))

        self._sideinfo_neighbor_top_k = int(
            self.params.sketch.get(
                "neighbor_top_k",
                self.params.model_hps.get("neighbor_top_k", 20),
            )
        )
        self.score_user_batch = int(self.params.sketch.get("score_user_batch", 1024))
        self.score_item_batch = int(self.params.sketch.get("score_item_batch", 4096))

        self.inc_hps = dict(getattr(self.params, "incremental_hps", None) or {})
        self._use_centered_sketch = True  # locked for residual_mse

        off = self.ds["offline"]
        off_train = off["train"]
        off_val = off.get("val", pd.DataFrame())
        off_test = off.get("test", pd.DataFrame())

        n_off, n_items = _infer_shape([off_train, off_val, off_test])
        self._offline_shape = (n_off, n_items)
        self._offline_user_ids = np.arange(n_off, dtype=np.int64)
        self._offline_train_df = off_train.copy()

        self._offline_default_mean = float(off_train[self.rating_col].mean()) if off_train is not None and not off_train.empty else 0.0

        offline_user_means = self._user_means_vector(
            df=off_train,
            user_ids=self._offline_user_ids,
            default_mean=self._offline_default_mean,
        )

        train_csr = csr_from_df(
            df=off_train,
            n_users=n_off,
            n_items=n_items,
            rating_col=self.rating_col,
            center_by_user_mean=True,
            user_means=offline_user_means,
        )

        self.jl = CustomJL(
            n_components=int(self.params.sketch["jl_components"]),
            train_matrix=train_csr,
            device=self.device,
        )
        X_off_jl = self.jl.train_projected

        race_kwargs = dict(
            k=int(self.params.sketch["race_k"]),
            d=int(self.params.sketch["race_d"]),
            w=int(self.params.sketch["race_w"]),
            R=int(self.params.sketch["race_R"]),
            n_blocks=int(self.params.sketch["n_blocks"]),
            n_perms=int(self.params.sketch["n_perms"]),
            batch_size=int(self.params.sketch["batch_size"]),
            query_batch_size=int(self.params.sketch.get("query_batch_size", 256)),
            seed=self.params.seed,
            build_initial=True,
            use_gpu=(self.device.type == "cuda"),
        )

        # reference sketch for side info
        self.reference_race = VectorizedRaceLSH(
            data=X_off_jl,
            **race_kwargs,
        )

        # live sketch for online retrieval
        self.live_race = VectorizedRaceLSH(
            data=X_off_jl,
            **race_kwargs,
        )

        self.reference_readout = ReferenceSketchReadout(self.jl, self.reference_race, self.device)

        raw_off = self.reference_readout.raw_outputs_from_projected(
            projected_vectors=X_off_jl,
            user_ids=self._offline_user_ids,
        )

        self.side_info_processor = SideInfoProcessor(self.side_info_cfg)
        self.side_info_processor.fit_offline_reference(raw_off)
        off_side = self.side_info_processor.transform(raw_off)

        self._offline_reference_side_batch = off_side
        self._offline_reference_raw = raw_off
        self._log_side_info_batch_stats("offline.side_info", off_side)

        self.mf = SideInfoAwareMF(
            model_init_params={
                "optimizer_type": self.params.model_init.get("optimizer_type", "adam"),
                "num_users": n_off,
                "num_items": n_items,
                "rating_col": self.rating_col,
                "num_epochs": int(self.params.model_init["num_epochs"]),
                "patience": int(self.params.model_init["patience"]),
                "loss_type": self.loss_type,
                "wmse_alpha": self.params.model_init.get("wmse_alpha", 0.5),
                "wmse_cap": self.params.model_init.get("wmse_cap", None),
            },
            model_hps={
                "embedding_size": int(self.params.model_hps["embedding_size"]),
                "reg_rate": float(self.params.model_hps["reg_rate"]),
                "lr": float(self.params.model_hps["lr"]),
                "batch_size": int(self.params.model_hps["batch_size"]),
                "side_reg_scale": float(self.params.model_hps.get("side_reg_scale", 1.0)),
                "offline_scheduler": dict(self.params.model_hps.get("offline_scheduler", {})),
                "incremental_scheduler": dict(self.inc_hps.get("incremental_scheduler", {})),
            },
            side_info_cfg=self.side_info_cfg,
            num_groups=self.side_info_processor.num_groups,
            feature_shape=self.side_info_processor.feature_shape,
        )

        self.mf.set_all_side_info(
            features=off_side.features,
            group_ids=off_side.group_ids,
            user_means=offline_user_means,
        )

        self._live_candidate_user_ids = self._offline_user_ids.copy()

        if hasattr(self, "artifacts") and hasattr(self.artifacts, "logs"):
            self.artifacts.logs["side_info.reference_mode"] = self.side_info_cfg.reference_mode
            self.artifacts.logs["side_info.representation_type"] = self.side_info_cfg.representation.type
            self.artifacts.logs["side_info.predictive_mode"] = self.side_info_cfg.usage.predictive.mode
            self.artifacts.logs["side_info.regularization_mode"] = self.side_info_cfg.usage.regularization.mode

    # ----------------------------
    # user-mean helpers
    # ----------------------------

    def _shrunk_user_mean_series(
        self,
        df: pd.DataFrame,
        *,
        default_mean: float,
    ) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=np.float32)

        stats = df.groupby("user_id", sort=False)[self.rating_col].agg(["mean", "count"])
        lam = float(self.residual_mean_shrinkage)

        num = stats["count"].astype(np.float32) * stats["mean"].astype(np.float32) + np.float32(lam * default_mean)
        den = stats["count"].astype(np.float32) + np.float32(lam)
        shrunk = (num / den).astype(np.float32)
        return shrunk

    def _user_means_vector(
        self,
        df: pd.DataFrame,
        user_ids: np.ndarray,
        *,
        default_mean: float,
    ) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if user_ids.size == 0:
            return np.empty((0,), dtype=np.float32)

        if df is None or df.empty:
            return np.full(user_ids.shape[0], float(default_mean), dtype=np.float32)

        means = self._shrunk_user_mean_series(df, default_mean=float(default_mean))
        out = pd.Series(user_ids).map(means).to_numpy(dtype=np.float32, copy=False)
        out = np.where(np.isnan(out), np.float32(default_mean), out)
        return out.astype(np.float32, copy=False)

    def _known_global_mean(self) -> float:
        if self.mf is not None and self.mf.user_mean_per_user is not None:
            return float(self.mf.user_mean_per_user.mean().item())
        return float(self._offline_default_mean)

    # ----------------------------
    # sketch input builders
    # ----------------------------

    def _build_centered_dense_for_users(
        self,
        df: pd.DataFrame,
        user_ids: np.ndarray,
        *,
        n_items: int,
        default_mean: float,
    ) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if user_ids.size == 0:
            return np.zeros((0, n_items), dtype=np.float32)

        if df is None or df.empty:
            return np.zeros((user_ids.shape[0], n_items), dtype=np.float32)

        work = df[df["user_id"].isin(user_ids)][["user_id", "item_id", self.rating_col]].copy()
        if work.empty:
            return np.zeros((user_ids.shape[0], n_items), dtype=np.float32)

        remap = {int(uid): int(i) for i, uid in enumerate(user_ids.tolist())}
        work["user_id"] = work["user_id"].map(remap).astype(np.int64)

        user_means_local = self._user_means_vector(
            df=df[df["user_id"].isin(user_ids)],
            user_ids=user_ids,
            default_mean=float(default_mean),
        )

        return dense_from_df(
            df=work,
            n_users=int(user_ids.shape[0]),
            n_items=int(n_items),
            rating_col=self.rating_col,
            center_by_user_mean=True,
            user_means=user_means_local,
        )

    def _build_raw_outputs_for_users(
        self,
        train_df: pd.DataFrame,
        user_ids: np.ndarray,
        *,
        n_items: int,
        default_mean: float,
    ) -> tuple[np.ndarray, torch.Tensor, RawSketchOutputs]:
        X_dense = self._build_centered_dense_for_users(
            df=train_df,
            user_ids=user_ids,
            n_items=n_items,
            default_mean=default_mean,
        )
        X_jl = self.jl.project(X_dense, return_torch=True, device=self.device)
        raw = self.reference_readout.raw_outputs_from_projected(
            projected_vectors=X_jl,
            user_ids=user_ids,
        )
        return X_dense, X_jl, raw

    # ----------------------------
    # offline fit
    # ----------------------------

    def _fit_offline(self) -> None:
        off = self.ds["offline"]
        self.mf.fit_offline(
            train_df=off["train"],
            val_df=off.get("val", pd.DataFrame()),
            test_df=off.get("test", pd.DataFrame()),
            use_early_stopping=True,
            patience=int(self.params.model_init["patience"]),
            min_delta=float(self.params.model_init.get("min_delta", 0.0)),
            verbose=bool(self.verbose),
        )

    # ----------------------------
    # incremental prep / fit
    # ----------------------------

    def _assert_known_items_only(self, df: pd.DataFrame, *, phase_name: str) -> None:
        if df is None or df.empty:
            return
        _, n_items = self._offline_shape
        max_item = int(df["item_id"].max())
        if max_item >= int(n_items):
            raise ValueError(
                f"{phase_name} contains unseen item_ids >= {n_items}. "
                "This experiment assumes a fixed item catalog."
            )


    def _prepare_incremental(self) -> Optional[dict]:
        inc = self.ds["incremental"]
        train_df = inc.get("train", pd.DataFrame())
        val_df = inc.get("val", pd.DataFrame())
        test_df = inc.get("test", pd.DataFrame())

        if all(df is None or df.empty for df in (train_df, val_df, test_df)):
            return None

        self._assert_known_items_only(train_df, phase_name="incremental/train")
        self._assert_known_items_only(val_df, phase_name="incremental/val")
        self._assert_known_items_only(test_df, phase_name="incremental/test")

        _, n_items = self._offline_shape
        n_current = int(self.mf.backbone.user_embedding.num_embeddings)

        inc_all = pd.concat(
            [df for df in (train_df, val_df, test_df) if df is not None and not df.empty],
            ignore_index=True,
        )
        new_user_ids = np.array(sorted(inc_all["user_id"].unique()), dtype=np.int64)
        new_user_ids = new_user_ids[new_user_ids >= n_current]

        if new_user_ids.size == 0:
            return None

        expected = np.arange(n_current, n_current + len(new_user_ids), dtype=np.int64)
        if not np.array_equal(new_user_ids, expected):
            raise ValueError(
                "Incremental user_ids must be contiguous and appended after the current max user id."
            )

        default_mean = self._known_global_mean()

        X_new_dense = self._build_centered_dense_for_users(
            df=train_df,
            user_ids=new_user_ids,
            n_items=n_items,
            default_mean=default_mean,
        )
        X_new_jl = self.jl.project(X_new_dense, return_torch=True, device=self.device)

        have_train_user_ids = (
            np.array(sorted(train_df["user_id"].unique()), dtype=np.int64)
            if train_df is not None and not train_df.empty
            else np.empty((0,), dtype=np.int64)
        )
        have_train_user_ids = have_train_user_ids[np.isin(have_train_user_ids, new_user_ids)]
        mask_have = np.isin(new_user_ids, have_train_user_ids)

        self.live_race.extend_indices_for_new_users(count=len(new_user_ids), start_j=n_current)
        if mask_have.any():
            self.live_race.one_pass_sketch_A(
                user_ids=new_user_ids[mask_have],
                data=X_new_jl[mask_have],
                rebuild=False,
                overwrite=False,
            )

        self._live_candidate_user_ids = np.concatenate(
            [self._live_candidate_user_ids, new_user_ids[mask_have]],
            axis=0,
        ).astype(np.int64, copy=False)

        ref_mode = self.side_info_cfg.reference_mode
        rep_type = self.side_info_cfg.representation.type
        stats_mode = self.side_info_cfg.representation.transform.stats_mode

        new_features = None
        new_group_ids = None
        full_features = None
        full_group_ids = None
        old_candidate_user_ids = np.arange(n_current, dtype=np.int64)

        if ref_mode == "offline_frozen":
            raw_new = self.reference_readout.raw_outputs_from_projected(
                projected_vectors=X_new_jl,
                user_ids=new_user_ids,
            )
            batch_new = self.side_info_processor.transform(raw_new)
            self._log_side_info_batch_stats("incremental.new_users.side_info", batch_new)

            if rep_type == "group_id":
                new_group_ids = batch_new.group_ids
            else:
                new_features = batch_new.features

        elif ref_mode == "appended_live":
            self.reference_race.extend_indices_for_new_users(count=len(new_user_ids), start_j=n_current)
            if mask_have.any():
                self.reference_race.one_pass_sketch_A(
                    user_ids=new_user_ids[mask_have],
                    data=X_new_jl[mask_have],
                    rebuild=False,
                    overwrite=False,
                )

            old_user_ids = np.arange(n_current, dtype=np.int64)

            X_old_dense = self._build_centered_dense_for_users(
                df=self._offline_train_df,
                user_ids=old_user_ids,
                n_items=n_items,
                default_mean=default_mean,
            )
            X_old_jl = self.jl.project(X_old_dense, return_torch=True, device=self.device)
            raw_old_after = self.reference_readout.raw_outputs_from_projected(
                projected_vectors=X_old_jl,
                user_ids=old_user_ids,
            )

            if rep_type == "group_id":
                old_after = self.side_info_processor.transform(raw_old_after)
                self._log_old_user_side_info_drift(
                    before=self._offline_reference_side_batch,
                    after=old_after,
                    prefix="incremental.old_user_drift",
                )

                raw_new = self.reference_readout.raw_outputs_from_projected(
                    projected_vectors=X_new_jl,
                    user_ids=new_user_ids,
                )
                batch_new = self.side_info_processor.transform(raw_new)
                self._log_side_info_batch_stats("incremental.new_users.side_info", batch_new)
                new_group_ids = batch_new.group_ids

            else:
                use_current_ref_stats = (stats_mode == "current_reference")

                if use_current_ref_stats:
                    all_user_ids = np.arange(n_current + len(new_user_ids), dtype=np.int64)
                    combined_train = pd.concat([self._offline_train_df, train_df], ignore_index=True)

                    X_all_dense = self._build_centered_dense_for_users(
                        df=combined_train,
                        user_ids=all_user_ids,
                        n_items=n_items,
                        default_mean=default_mean,
                    )
                    X_all_jl = self.jl.project(X_all_dense, return_torch=True, device=self.device)
                    raw_all = self.reference_readout.raw_outputs_from_projected(
                        projected_vectors=X_all_jl,
                        user_ids=all_user_ids,
                    )

                    self.side_info_processor.fit_offline_reference(raw_all)
                    batch_all = self.side_info_processor.transform(raw_all)

                    full_features = batch_all.features
                    new_features = full_features[n_current:]

                    old_after = SideInfoBatch(
                        features=full_features[:n_current],
                        group_ids=None,
                        raw=RawSketchOutputs(
                            scalar_freq=raw_all.scalar_freq[:n_current],
                            row_mom=raw_all.row_mom[:n_current],
                            raw_dr=raw_all.raw_dr[:n_current],
                        ),
                    )
                    self._log_old_user_side_info_drift(
                        before=self._offline_reference_side_batch,
                        after=old_after,
                        prefix="incremental.old_user_drift",
                    )

                    new_batch = SideInfoBatch(
                        features=new_features,
                        group_ids=None,
                        raw=RawSketchOutputs(
                            scalar_freq=raw_all.scalar_freq[n_current:],
                            row_mom=raw_all.row_mom[n_current:],
                            raw_dr=raw_all.raw_dr[n_current:],
                        ),
                    )
                    self._log_side_info_batch_stats("incremental.new_users.side_info", new_batch)

                else:
                    old_after = self.side_info_processor.transform(raw_old_after, refit_if_current_reference=False)
                    self._log_old_user_side_info_drift(
                        before=self._offline_reference_side_batch,
                        after=old_after,
                        prefix="incremental.old_user_drift",
                    )

                    raw_new = self.reference_readout.raw_outputs_from_projected(
                        projected_vectors=X_new_jl,
                        user_ids=new_user_ids,
                    )
                    batch_new = self.side_info_processor.transform(
                        raw_new,
                        refit_if_current_reference=False,
                    )
                    self._log_side_info_batch_stats("incremental.new_users.side_info", batch_new)
                    new_features = batch_new.features

        else:
            raise ValueError(f"Unsupported side_info.reference_mode='{ref_mode}'")

        new_anchor_vectors = self._compute_incremental_anchor_vectors(
            new_user_ids=new_user_ids,
            query_vectors=X_new_jl,
            new_group_ids=new_group_ids,
            candidate_user_ids=old_candidate_user_ids,
        )

        new_user_means = self._user_means_vector(
            df=train_df,
            user_ids=new_user_ids,
            default_mean=default_mean,
        )

        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "new_user_ids": new_user_ids,
            "have_train_user_ids": have_train_user_ids,
            "new_features": new_features,
            "new_group_ids": new_group_ids,
            "full_features": full_features,
            "full_group_ids": full_group_ids,
            "new_user_means": new_user_means,
            "new_anchor_vectors": new_anchor_vectors,
        }

    def _fit_incremental(self) -> None:
        prep = self._prepare_incremental()
        if prep is None:
            return

        train_df = prep["train_df"]
        val_df = prep["val_df"]
        new_user_ids = prep["new_user_ids"]
        have_train_user_ids = prep["have_train_user_ids"]

        self.mf.add_users(
            n_new=len(new_user_ids),
            new_features=prep["new_features"],
            new_group_ids=prep["new_group_ids"],
            new_user_means=prep["new_user_means"],
            new_anchor_vectors=prep["new_anchor_vectors"],
        )

        if prep["full_features"] is not None:
            full_user_means = self.mf.user_mean_per_user.detach().cpu().numpy().astype(np.float32, copy=True)
            full_anchor_vectors = (
                None
                if self.mf.anchor_vectors_per_user is None
                else self.mf.anchor_vectors_per_user.detach().cpu().numpy().astype(np.float32, copy=True)
            )
            self.mf.set_all_side_info(
                features=prep["full_features"],
                group_ids=self.mf.group_ids_per_user.detach().cpu().numpy() if self.mf.group_ids_per_user is not None else None,
                user_means=full_user_means,
                anchor_vectors=full_anchor_vectors,
            )

        if have_train_user_ids.size > 0:
            self.mf.finetune_new_users(
                train_inc_df=train_df,
                val_inc_df=val_df,
                new_user_ids=have_train_user_ids,
                epochs=int(self.inc_hps["incremental_epochs"]),
                lr=float(self.inc_hps["incremental_lr"]),
                reg_rate=float(self.inc_hps["incremental_reg_rate"]),
                batch_size=int(self.inc_hps["incremental_batch_size"]),
                early_stop=True,
                patience=int(self.inc_hps["incremental_patience"]),
                min_delta=float(self.inc_hps.get("incremental_min_delta", 0.0)),
                anchor_to_zero=float(self.inc_hps["incremental_anchor"]),
                verbose=bool(self.verbose),
                strict_freeze_old=True,
            )

        if hasattr(self, "artifacts") and hasattr(self.artifacts, "logs"):
            self.artifacts.logs["incremental.n_users_total"] = int(len(new_user_ids))
            self.artifacts.logs["incremental.n_users_finetuned"] = int(len(have_train_user_ids))

    # ----------------------------
    # prediction helpers
    # ----------------------------

    def _predict_known_df(
        self,
        df: pd.DataFrame,
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        return self.mf.predict_from_df(
            df=df,
            batch_size=int(batch_size or self.params.model_hps["batch_size"]),
            clip_bounds=None,
            output_space="raw",
        )

    def _online_user_means(self, train_df: pd.DataFrame) -> pd.Series:
        return self._shrunk_user_mean_series(
            train_df,
            default_mean=float(self._known_global_mean()),
        )

    def _predict_online_df_batch(self, df: pd.DataFrame) -> np.ndarray:
        on = self.ds["online"]
        train_df = on.get("train", pd.DataFrame())

        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        self._assert_known_items_only(train_df, phase_name="online/train")
        self._assert_known_items_only(df, phase_name="online/test")

        _, n_items = self._offline_shape
        n_known = int(self.mf.backbone.user_embedding.num_embeddings)
        candidates = self._live_candidate_user_ids.astype(np.int64, copy=False)

        query_users = np.array(sorted(df["user_id"].unique()), dtype=np.int64)
        global_mean = self._known_global_mean()
        train_means = self._online_user_means(train_df)

        X_query_dense = self._build_centered_dense_for_users(
            df=train_df,
            user_ids=query_users,
            n_items=n_items,
            default_mean=global_mean,
        )
        X_query_jl = self.jl.project(X_query_dense, return_torch=True, device=self.device)

        top_vals, top_idx = self.live_race.find_closest_users(
            query_vectors=X_query_jl,
            top_k=int(self._sideinfo_neighbor_top_k),
            return_kernels=False,
            candidate_user_ids=candidates,
        )

        W = normalize_rows_nonneg(top_vals).astype(np.float32, copy=False)

        unique_items = np.array(sorted(df["item_id"].unique()), dtype=np.int64)
        item_to_col = {it: j for j, it in enumerate(unique_items)}

        output_space = "model" if self.mf.residual_target else "raw"
        score_cache = self.mf.predict_user_item_grid(
            user_ids=np.arange(n_known, dtype=np.int64),
            item_ids=unique_items,
            user_batch_size=int(self.score_user_batch),
            item_batch_size=int(self.score_item_batch),
            clip_bounds=None,
            output_space=output_space,
        )

        cold_item_scores = score_cache[candidates].mean(axis=0).astype(np.float32, copy=False)

        work = df[["user_id", "item_id"]].copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)

        out = np.empty((len(work),), dtype=np.float32)
        query_pos = {int(u): int(i) for i, u in enumerate(query_users.tolist())}

        for u, g in work.groupby("user_id", sort=False):
            row_pos = g["_row_pos"].to_numpy(dtype=np.int64, copy=False)
            item_ids = g["item_id"].to_numpy(dtype=np.int64, copy=True)
            item_cols = np.fromiter((item_to_col[it] for it in item_ids), dtype=np.int64, count=len(item_ids))

            if int(u) in query_pos:
                j = query_pos[int(u)]
                neigh = top_idx[j]
                w = W[j]
                preds_model = (score_cache[neigh][:, item_cols] * w[:, None]).sum(axis=0).astype(np.float32, copy=False)
            else:
                preds_model = cold_item_scores[item_cols]

            if self.mf.residual_target:
                user_mean = float(train_means.get(u, global_mean)) if not train_means.empty else global_mean
                out[row_pos] = preds_model + user_mean
            else:
                out[row_pos] = preds_model

        return out

    def _predict_online_df_per_user(self, df: pd.DataFrame) -> np.ndarray:
        on = self.ds["online"]
        train_df = on.get("train", pd.DataFrame())

        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        self._assert_known_items_only(train_df, phase_name="online/train")
        self._assert_known_items_only(df, phase_name="online/test")

        _, n_items = self._offline_shape
        n_known = int(self.mf.backbone.user_embedding.num_embeddings)
        candidates = self._live_candidate_user_ids.astype(np.int64, copy=False)

        global_mean = self._known_global_mean()
        train_means = self._online_user_means(train_df)

        unique_items = np.array(sorted(df["item_id"].unique()), dtype=np.int64)
        item_to_col = {it: j for j, it in enumerate(unique_items)}

        output_space = "model" if self.mf.residual_target else "raw"
        score_cache = self.mf.predict_user_item_grid(
            user_ids=np.arange(n_known, dtype=np.int64),
            item_ids=unique_items,
            user_batch_size=int(self.score_user_batch),
            item_batch_size=int(self.score_item_batch),
            clip_bounds=None,
            output_space=output_space,
        )

        cold_item_scores = score_cache[candidates].mean(axis=0).astype(np.float32, copy=False)

        work = df[["user_id", "item_id"]].copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)
        out = np.empty((len(work),), dtype=np.float32)

        train_by_user = {
            int(u): g.loc[:, ["user_id", "item_id", self.rating_col]].copy()
            for u, g in train_df.groupby("user_id", sort=False)
        }

        for u, g_test in work.groupby("user_id", sort=False):
            row_pos = g_test["_row_pos"].to_numpy(dtype=np.int64, copy=False)
            item_ids = g_test["item_id"].to_numpy(dtype=np.int64, copy=True)
            item_cols = np.fromiter((item_to_col[it] for it in item_ids), dtype=np.int64, count=len(item_ids))

            g_train = train_by_user.get(int(u))
            user_mean = float(train_means.get(u, global_mean)) if not train_means.empty else global_mean

            if g_train is None or g_train.empty:
                preds_model = cold_item_scores[item_cols]
            else:
                q_users = np.array([int(u)], dtype=np.int64)
                X_query_dense = self._build_centered_dense_for_users(
                    df=g_train,
                    user_ids=q_users,
                    n_items=n_items,
                    default_mean=global_mean,
                )
                X_query_jl = self.jl.project(X_query_dense, return_torch=True, device=self.device)

                top_vals, top_idx = self.live_race.find_closest_users(
                    query_vectors=X_query_jl,
                    top_k=int(self._sideinfo_neighbor_top_k),
                    return_kernels=False,
                    candidate_user_ids=candidates,
                )
                w = normalize_nonneg(top_vals[0]).astype(np.float32, copy=False)
                neigh = top_idx[0]
                preds_model = (score_cache[neigh][:, item_cols] * w[:, None]).sum(axis=0).astype(np.float32, copy=False)

            if self.mf.residual_target:
                out[row_pos] = preds_model + user_mean
            else:
                out[row_pos] = preds_model

        return out

    # ----------------------------
    # BaseExperiment hooks
    # ----------------------------

    def _predict_df(self, df: pd.DataFrame, *, phase: str) -> np.ndarray:
        inc_bs = int(self.inc_hps["incremental_batch_size"])

        if phase == "offline":
            return self._predict_known_df(df, batch_size=int(self.params.model_hps["batch_size"]))

        if phase == "incremental":
            return self._predict_known_df(df, batch_size=inc_bs)

        if phase == "offline_post_incremental":
            return self._predict_known_df(df, batch_size=inc_bs)

        if phase == "online":
            mode = str(getattr(self.params, "online_inference_pred_type", "batch")).lower()
            if mode in {"per_user", "per-user"}:
                return self._predict_online_df_per_user(df)
            return self._predict_online_df_batch(df)

        raise ValueError(f"Unsupported phase for _predict_df: {phase}")

    def collect_phase_prediction_dfs(
        self,
        *,
        include_split: bool = False,
    ) -> dict[str, pd.DataFrame]:
        def _empty_df() -> pd.DataFrame:
            cols = ["user_id", "item_id", "rating", "prediction"]
            if include_split:
                cols.append("split")
            return pd.DataFrame(columns=cols)

        def _build_from_ds(phase_name: str, ds_name: str) -> pd.DataFrame:
            phase_ds = self.ds.get(ds_name, {})
            parts = []
            for split in ("train", "val", "test"):
                split_df = phase_ds.get(split, pd.DataFrame())
                if split_df is None or split_df.empty:
                    continue

                preds = self._predict_df(split_df, phase=phase_name)
                out = split_df.loc[:, ["user_id", "item_id", self.rating_col]].copy()
                out = out.rename(columns={self.rating_col: "rating"})
                out["prediction"] = preds.astype(np.float32, copy=False)
                if include_split:
                    out["split"] = split
                parts.append(out)

            if not parts:
                return _empty_df()
            return pd.concat(parts, ignore_index=True)

        return {
            "offline": _build_from_ds("offline", "offline"),
            "incremental": _build_from_ds("incremental", "incremental"),
            "offline_post_incremental": _build_from_ds("offline_post_incremental", "offline"),
            "online": _build_from_ds("online", "online"),
        }



# ===== Notebook cell 29 =====
class SpaceTrackedJLRaceMFSideInfoExperiment(
    SpaceTrackedExperimentBase,
    JLRaceMFSideInfoExperiment,
):
    """
    Persisted logical state for JLRaceMFSideInfoExperiment.

    Counted:
      - JL SparseRandomProjection operator components_ (CSR payload only)
      - Reference sketch counters A + routing indices H
      - Live sketch counters A + routing indices H
      - Side-info processor persisted state:
          * group assigner state (edges + majority_group) for group_id
          * feature-transform stats (mean/std/min/max) for continuous reps
      - MF backbone params:
          * user_embedding
          * item_embedding
          * item_bias
          * user_bias      (only if present)
          * global_bias    (only if present)
      - Optional side heads:
          * group_bias
          * predictive_head
          * reg_head
      - Persisted per-user auxiliary state:
          * group_ids_per_user          (when representation=group_id)
          * side_features_per_user      (continuous representations)
          * user_mean_per_user          (residual mode)
          * anchor_vectors_per_user     (when anchor-based regularization stores them)

    Not counted:
      - dense completed matrices / RMSE-only artifacts
      - JL train caches / projected train matrices
      - sketch helper tensors (all_projections, _all_proj_t, user_buckets, offline_freqs)
      - optimizer state / minibatch temporaries / online query temporaries

    Convention:
      - This is logical persisted state, not peak runtime RAM.
      - Integer sketch / routing caches are counted with compact logical storage
        (int8 as requested), not the current in-memory numpy dtype.
    """

    def _build_space_snapshot(self, phase: Phase) -> SpaceSnapshot:
        snap = SpaceSnapshot()

        if self.mf is None or self.mf.backbone is None:
            raise ValueError("MF backbone is not initialized; cannot build space snapshot.")

        backbone = self.mf.backbone
        side_cfg = self.side_info_cfg
        rep_cfg = side_cfg.representation
        reg_cfg = side_cfg.usage.regularization

        U = int(backbone.user_embedding.num_embeddings)
        N = int(backbone.item_embedding.num_embeddings)
        k_mf = int(backbone.user_embedding.embedding_dim)

        def add_dense(
            *,
            key: str,
            name: str,
            shape: tuple[int, ...],
            dtype: str,
            formula: str,
            note: str | None = None,
        ) -> None:
            snap.components.append(
                SpaceComponent(
                    key=key,
                    name=name,
                    bytes=_bytes_dense(shape, dtype),
                    phase=phase,
                    shape=shape,
                    dtype=dtype,
                    formula=formula,
                    note=note,
                )
            )

        def add_bytes(
            *,
            key: str,
            name: str,
            bytes_: int,
            shape: tuple[int, ...] | None,
            dtype: str | None,
            formula: str,
            note: str | None = None,
        ) -> None:
            snap.components.append(
                SpaceComponent(
                    key=key,
                    name=name,
                    bytes=int(bytes_),
                    phase=phase,
                    shape=shape,
                    dtype=dtype,
                    formula=formula,
                    note=note,
                )
            )

        def add_tensor(
            *,
            key: str,
            name: str,
            tensor: torch.Tensor,
            formula: str,
            note: str | None = None,
        ) -> None:
            dtype = _torch_dtype_to_key(tensor.dtype)
            shape = tuple(int(s) for s in tensor.shape) if tensor.ndim > 0 else (1,)
            snap.components.append(
                SpaceComponent(
                    key=key,
                    name=name,
                    bytes=int(tensor.numel()) * int(tensor.element_size()),
                    phase=phase,
                    shape=shape,
                    dtype=dtype,
                    formula=formula,
                    note=note,
                )
            )

        def add_numpy_array(
            *,
            key: str,
            name: str,
            arr,
            formula: str,
            note: str | None = None,
        ) -> None:
            if arr is None:
                return

            arr_np = np.asarray(arr)
            shape = tuple(int(s) for s in arr_np.shape) if arr_np.ndim > 0 else (1,)
            dtype = str(arr_np.dtype)
            snap.components.append(
                SpaceComponent(
                    key=key,
                    name=name,
                    bytes=int(arr_np.nbytes),
                    phase=phase,
                    shape=shape,
                    dtype=dtype,
                    formula=formula,
                    note=note,
                )
            )

        def add_module_parameters(
            *,
            key_prefix: str,
            name_prefix: str,
            module: torch.nn.Module | None,
            note: str | None = None,
        ) -> None:
            if module is None:
                return

            for param_name, param in module.named_parameters(recurse=True):
                clean_name = param_name.replace(".", "/")
                add_tensor(
                    key=f"{key_prefix}.{param_name}",
                    name=f"{name_prefix}::{clean_name}",
                    tensor=param,
                    formula=f"sum(params in {name_prefix})",
                    note=note,
                )

        def add_sketch_state(
            *,
            key_prefix: str,
            human_name: str,
            race,
        ) -> None:
            d_s = int(race.d)
            w = int(race.w)
            R = int(race.R)
            m = 2 ** int(race.k)
            U_race = int(race.js)

            add_dense(
                key=f"{key_prefix}.A",
                name=f"{human_name} sketch counters A",
                shape=(d_s, w, R, m),
                dtype="int8",
                formula="d_s*w*R*2^k_bits",
                note=(
                    "Logical compact counter cache. "
                    "Counted as int8 by convention; helper/permutation materialization not persisted."
                ),
            )

            add_dense(
                key=f"{key_prefix}.H",
                name=f"{human_name} routing indices H",
                shape=(U_race, d_s),
                dtype="int8",
                formula="U_race*d_s",
                note=(
                    "Logical compact routing cache. "
                    "Counted as int8 by convention; excludes user_buckets/offline_freqs."
                ),
            )

        # -------------------------
        # JL operator (CSR payload only)
        # -------------------------
        P = self.jl.transformer.components_

        if sparse.issparse(P):
            bytes_P = int(P.data.nbytes + P.indices.nbytes + P.indptr.nbytes)
            add_bytes(
                key="jl.P_csr",
                name="JL projection operator (SparseRandomProjection.components_, CSR payload)",
                bytes_=bytes_P,
                shape=(int(P.shape[0]), int(P.shape[1])),
                dtype=str(P.data.dtype),
                formula="nnz*(data + indices) + (rows+1)*indptr",
                note=(
                    "CSR payload only. No dense P, no train_projected cache, no projected train matrix."
                ),
            )
        else:
            P_np = np.asarray(P)
            add_bytes(
                key="jl.P_dense",
                name="JL projection operator (dense payload)",
                bytes_=int(P_np.nbytes),
                shape=tuple(int(s) for s in P_np.shape),
                dtype=str(P_np.dtype),
                formula="rows*cols",
                note="Dense JL operator payload only.",
            )

        # -------------------------
        # Reference + live sketches
        # -------------------------
        add_sketch_state(
            key_prefix="reference_race",
            human_name="Reference RACE",
            race=self.reference_race,
        )
        add_sketch_state(
            key_prefix="live_race",
            human_name="Live RACE",
            race=self.live_race,
        )

        # -------------------------
        # Side-info processor persisted state
        # -------------------------
        if self.side_info_processor.group_assigner is not None:
            ga = self.side_info_processor.group_assigner

            if getattr(ga, "edges_", None) is not None:
                add_numpy_array(
                    key="side.group_assigner.edges",
                    name="Side-info group assigner edges",
                    arr=ga.edges_,
                    formula="n_bins+1",
                    note="Persisted only for group_id representation.",
                )

            add_dense(
                key="side.group_assigner.majority_group",
                name="Side-info group assigner majority_group",
                shape=(1,),
                dtype="int8",
                formula="1",
                note="Compact categorical cache for fallback group id.",
            )

        if self.side_info_processor.feature_transform is not None:
            ft = self.side_info_processor.feature_transform

            if ft.mean_ is not None:
                add_numpy_array(
                    key="side.feature_transform.mean",
                    name="Side-info feature-transform mean",
                    arr=ft.mean_,
                    formula="feature_dim",
                    note="Persisted transform statistics for continuous side-info.",
                )

            if ft.std_ is not None:
                add_numpy_array(
                    key="side.feature_transform.std",
                    name="Side-info feature-transform std",
                    arr=ft.std_,
                    formula="feature_dim",
                    note="Persisted transform statistics for continuous side-info.",
                )

            if ft.min_ is not None:
                add_numpy_array(
                    key="side.feature_transform.min",
                    name="Side-info feature-transform min",
                    arr=ft.min_,
                    formula="feature_dim",
                    note="Persisted transform statistics for continuous side-info.",
                )

            if ft.max_ is not None:
                add_numpy_array(
                    key="side.feature_transform.max",
                    name="Side-info feature-transform max",
                    arr=ft.max_,
                    formula="feature_dim",
                    note="Persisted transform statistics for continuous side-info.",
                )

        # -------------------------
        # MF backbone persisted params
        # -------------------------
        add_tensor(
            key="mf.user_embedding",
            name="MF user factors E_u",
            tensor=backbone.user_embedding.weight,
            formula="U*k_mf",
        )
        add_tensor(
            key="mf.item_embedding",
            name="MF item factors E_i",
            tensor=backbone.item_embedding.weight,
            formula="N*k_mf",
        )
        add_tensor(
            key="mf.item_bias",
            name="MF item bias b_i",
            tensor=backbone.item_bias.weight,
            formula="N",
            note="Always present in SideInfoMFBackbone.",
        )

        if backbone.user_bias is not None:
            add_tensor(
                key="mf.user_bias",
                name="MF user bias b_u",
                tensor=backbone.user_bias.weight,
                formula="U",
                note="Present only when raw-rating bias terms are enabled.",
            )

        if backbone.global_bias is not None:
            add_tensor(
                key="mf.global_bias",
                name="MF global bias b0",
                tensor=backbone.global_bias,
                formula="1",
                note="Present only when raw-rating bias terms are enabled.",
            )

        # -------------------------
        # Side heads / branch params
        # -------------------------
        if self.mf.group_bias is not None:
            add_tensor(
                key="side.group_bias",
                name="Side group-bias embedding",
                tensor=self.mf.group_bias.weight,
                formula="num_groups",
                note="Present only when predictive.mode='group_bias'.",
            )

        add_module_parameters(
            key_prefix="side.predictive_head",
            name_prefix="Predictive side head",
            module=self.mf.predictive_head,
            note="Present for additive_scalar / embedding_residual predictive usage.",
        )

        add_module_parameters(
            key_prefix="side.reg_head",
            name_prefix="Regularization head",
            module=self.mf.reg_head,
            note="Present only when regularization.mapping='mlp'.",
        )

        # -------------------------
        # Per-user side-info state
        # -------------------------
        if self.mf.group_ids_per_user is not None:
            add_dense(
                key="side.group_ids_per_user",
                name="Per-user side-info group ids",
                shape=(U,),
                dtype="int8",
                formula="U",
                note=(
                    "Logical compact categorical cache for representation=group_id."
                ),
            )

        if self.mf.side_features_per_user is not None:
            add_tensor(
                key="side.side_features_per_user",
                name="Per-user continuous side-info features",
                tensor=self.mf.side_features_per_user,
                formula="U*feature_shape",
                note=(
                    f"Persisted continuous side-info for representation={rep_cfg.type}."
                ),
            )

        if self.mf.user_mean_per_user is not None:
            add_tensor(
                key="mf.user_mean_per_user",
                name="Per-user mean rating cache",
                tensor=self.mf.user_mean_per_user,
                formula="U",
                note="Used in residual_mse mode to restore raw-rating predictions.",
            )

        if getattr(self.mf, "anchor_vectors_per_user", None) is not None:
            add_tensor(
                key="side.anchor_vectors_per_user",
                name="Per-user anchor vectors",
                tensor=self.mf.anchor_vectors_per_user,
                formula="U*k_mf",
                note=(
                    f"Persisted only when anchor-based regularization stores explicit anchors "
                    f"(anchor={reg_cfg.anchor})."
                ),
            )

        return snap
