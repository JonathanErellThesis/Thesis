"""
LEAF baseline and lifecycle experiment wrapper.

Generated from the original LEAF notebook with import paths adjusted for the
anonymous reproduction package. Class/function bodies are kept unchanged as much
as possible so reported runs can be reproduced from the original logic.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from recsys_edge.core import *  # noqa: F401,F403 - preserves notebook-style globals
from recsys_edge.core import _torch_dtype_to_key

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# =========================================================
# Packs / helpers
# =========================================================

@dataclass
class TripletPack:
    u: torch.Tensor  # int64 [N]
    i: torch.Tensor  # int64 [N]
    r: torch.Tensor  # float32 [N]


_LARGE_PRIME = 2_147_483_647


def _as_device_tensor_1d(
    np_arr: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    t = torch.as_tensor(np_arr, dtype=dtype, device="cpu")
    if device.type == "cuda":
        try:
            t = t.pin_memory()
        except Exception:
            pass
        t = t.to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t


def df_to_tripletpack(df: pd.DataFrame, device: torch.device) -> Optional[TripletPack]:
    if df is None or len(df) == 0:
        return None

    u_np = df["user_id"].to_numpy(dtype=np.int64, copy=False)
    i_np = df["item_id"].to_numpy(dtype=np.int64, copy=False)
    r_np = df["rating"].to_numpy(dtype=np.float32, copy=False)

    return TripletPack(
        u=_as_device_tensor_1d(u_np, dtype=torch.int64, device=device),
        i=_as_device_tensor_1d(i_np, dtype=torch.int64, device=device),
        r=_as_device_tensor_1d(r_np, dtype=torch.float32, device=device),
    )


def iter_minibatches(
    pack: TripletPack,
    batch_size: int,
    shuffle: bool,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    n = int(pack.r.numel())
    if n == 0:
        return
    dev = pack.r.device
    idx = torch.randperm(n, device=dev) if shuffle else torch.arange(n, device=dev)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield pack.u[b], pack.i[b], pack.r[b]


# =========================================================
# Sketch
# =========================================================

class SMEDSketch:
    """
    Practical SMED-like sketch:
      - Maintains up to K counters for ids
      - update_batch(ids): processes stream updates
      - frequent keys = ids with active counters
    """

    def __init__(self, K: int, *, seed: int = 0):
        self.K = int(K)
        self._rng = random.Random(int(seed))
        self._counters: Dict[int, int] = {}

    def state_dict(self) -> Dict[str, Any]:
        return {"K": self.K, "counters": dict(self._counters)}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.K = int(sd["K"])
        self._counters = dict(sd["counters"])

    def _decrement_all(self, c: int) -> None:
        if c <= 0:
            return
        dead = []
        for k in self._counters:
            self._counters[k] -= c
            if self._counters[k] <= 0:
                dead.append(k)
        for k in dead:
            del self._counters[k]

    def update_batch(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return

        uniq, cnt = torch.unique(ids, return_counts=True)
        uniq = uniq.detach().cpu().tolist()
        cnt = cnt.detach().cpu().tolist()

        total = int(ids.numel())
        l = max(1, int(math.log(max(2, total))))

        for x, add in zip(uniq, cnt):
            x = int(x)
            add = int(add)

            if x in self._counters:
                self._counters[x] += add
                continue

            if len(self._counters) < self.K:
                self._counters[x] = add
                continue

            while len(self._counters) >= self.K and x not in self._counters:
                vals = list(self._counters.values())
                if not vals:
                    break
                sample = [vals[self._rng.randrange(len(vals))] for _ in range(min(l, len(vals)))]
                sample.sort()
                c = int(sample[len(sample) // 2])
                if c <= 0:
                    c = 1
                self._decrement_all(c)

            if len(self._counters) < self.K:
                self._counters[x] = add

    def frequent_keys_sorted(self, device: torch.device) -> torch.Tensor:
        if not self._counters:
            return torch.empty(0, dtype=torch.int64, device=device)
        keys = torch.tensor(list(self._counters.keys()), dtype=torch.int64, device=device)
        return torch.sort(keys).values


# =========================================================
# Model
# =========================================================

class LEAFModel(nn.Module):
    """
    Unified LEAF model:
      - Two compressed tables per side (freq / infreq)
      - k independent affine hashes -> gather k rows -> mean pool
      - Routing is driven by sorted frequent-key sets
      - Optional per-id user/item bias (ridge-style)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_size: int,
        *,
        k: int,
        M_u_f: int,
        M_u_inf: int,
        M_i_f: int,
        M_i_inf: int,
        use_bias: bool = True,
    ):
        super().__init__()

        for name, v in dict(
            num_users=num_users,
            num_items=num_items,
            embedding_size=embedding_size,
            k=k,
            M_u_f=M_u_f,
            M_u_inf=M_u_inf,
            M_i_f=M_i_f,
            M_i_inf=M_i_inf,
        ).items():
            if v is None:
                raise ValueError(f"LEAFModel: '{name}' must be provided.")
        if num_users <= 0 or num_items <= 0 or embedding_size <= 0 or k <= 0:
            raise ValueError("LEAFModel: num_users/num_items/embedding_size/k must be > 0.")
        if min(M_u_f, M_u_inf, M_i_f, M_i_inf) <= 0:
            raise ValueError("LEAFModel: all M_* must be > 0.")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.d = int(embedding_size)
        self.k = int(k)
        self.M_u_f = int(M_u_f)
        self.M_u_inf = int(M_u_inf)
        self.M_i_f = int(M_i_f)
        self.M_i_inf = int(M_i_inf)
        self.use_bias = bool(use_bias)

        self.user_E_f = nn.Embedding(self.M_u_f, self.d)
        self.user_E_inf = nn.Embedding(self.M_u_inf, self.d)
        self.item_E_f = nn.Embedding(self.M_i_f, self.d)
        self.item_E_inf = nn.Embedding(self.M_i_inf, self.d)

        for emb in (self.user_E_f, self.user_E_inf, self.item_E_f, self.item_E_inf):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        if self.use_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)
            self.item_bias = nn.Embedding(self.num_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
        else:
            self.user_bias = None
            self.item_bias = None

        self.global_bias = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("freq_user_keys", torch.empty(0, dtype=torch.int64), persistent=False)
        self.register_buffer("freq_item_keys", torch.empty(0, dtype=torch.int64), persistent=False)

        self.register_buffer("a_u_f", torch.empty(self.k, dtype=torch.int64))
        self.register_buffer("b_u_f", torch.empty(self.k, dtype=torch.int64))
        self.register_buffer("a_u_inf", torch.empty(self.k, dtype=torch.int64))
        self.register_buffer("b_u_inf", torch.empty(self.k, dtype=torch.int64))

        self.register_buffer("a_i_f", torch.empty(self.k, dtype=torch.int64))
        self.register_buffer("b_i_f", torch.empty(self.k, dtype=torch.int64))
        self.register_buffer("a_i_inf", torch.empty(self.k, dtype=torch.int64))
        self.register_buffer("b_i_inf", torch.empty(self.k, dtype=torch.int64))

        self._hash_inited = False

    @torch.no_grad()
    def init_hash_params(self, seed: Optional[int] = None) -> None:
        dev = self.user_E_f.weight.device
        if seed is not None:
            g = torch.Generator(device=dev)
            g.manual_seed(int(seed))
            rand = lambda shape: torch.randint(0, _LARGE_PRIME, shape, dtype=torch.int64, generator=g, device=dev)
        else:
            rand = lambda shape: torch.randint(0, _LARGE_PRIME, shape, dtype=torch.int64, device=dev)

        def _make():
            a = rand((self.k,)) | 1
            b = rand((self.k,))
            return a, b

        self.a_u_f, self.b_u_f = _make()
        self.a_u_inf, self.b_u_inf = _make()
        self.a_i_f, self.b_i_f = _make()
        self.a_i_inf, self.b_i_inf = _make()
        self._hash_inited = True

    @torch.no_grad()
    def set_frequent_keys(self, user_keys_sorted: torch.Tensor, item_keys_sorted: torch.Tensor) -> None:
        dev = self.user_E_f.weight.device
        self.freq_user_keys = user_keys_sorted.to(dev, dtype=torch.int64)
        self.freq_item_keys = item_keys_sorted.to(dev, dtype=torch.int64)

    @staticmethod
    def _isin_sorted(x: torch.LongTensor, keys_sorted: torch.LongTensor) -> torch.BoolTensor:
        if keys_sorted.numel() == 0:
            return torch.zeros_like(x, dtype=torch.bool)
        pos = torch.searchsorted(keys_sorted, x)
        inrange = pos < keys_sorted.numel()
        pos = pos.clamp_max(keys_sorted.numel() - 1)
        return inrange & (keys_sorted[pos] == x)

    @staticmethod
    def _k_affine_hash(
        ids_1d: torch.LongTensor,
        a: torch.Tensor,
        b: torch.Tensor,
        mod: int,
    ) -> torch.LongTensor:
        x = ids_1d.view(-1, 1)
        vals = (x * a + b) % _LARGE_PRIME
        return vals % mod

    def _pooled_user(self, user_ids: torch.LongTensor) -> torch.Tensor:
        if not self._hash_inited:
            raise RuntimeError("LEAFModel: init_hash_params(...) must be called first.")

        B = int(user_ids.numel())
        dev = user_ids.device
        dtype = self.user_E_f.weight.dtype

        mask = self._isin_sorted(user_ids, self.freq_user_keys)
        out = torch.empty((B, self.d), device=dev, dtype=dtype)

        if mask.any():
            ids_f = user_ids[mask]
            idx = self._k_affine_hash(ids_f, self.a_u_f, self.b_u_f, self.M_u_f)
            out[mask] = self.user_E_f(idx).mean(dim=1)

        if (~mask).any():
            ids_i = user_ids[~mask]
            idx = self._k_affine_hash(ids_i, self.a_u_inf, self.b_u_inf, self.M_u_inf)
            out[~mask] = self.user_E_inf(idx).mean(dim=1)

        return out

    def _pooled_item(self, item_ids: torch.LongTensor) -> torch.Tensor:
        if not self._hash_inited:
            raise RuntimeError("LEAFModel: init_hash_params(...) must be called first.")

        B = int(item_ids.numel())
        dev = item_ids.device
        dtype = self.item_E_f.weight.dtype

        mask = self._isin_sorted(item_ids, self.freq_item_keys)
        out = torch.empty((B, self.d), device=dev, dtype=dtype)

        if mask.any():
            ids_f = item_ids[mask]
            idx = self._k_affine_hash(ids_f, self.a_i_f, self.b_i_f, self.M_i_f)
            out[mask] = self.item_E_f(idx).mean(dim=1)

        if (~mask).any():
            ids_i = item_ids[~mask]
            idx = self._k_affine_hash(ids_i, self.a_i_inf, self.b_i_inf, self.M_i_inf)
            out[~mask] = self.item_E_inf(idx).mean(dim=1)

        return out

    def forward(self, user: torch.LongTensor, item: torch.LongTensor) -> torch.Tensor:
        ue = self._pooled_user(user)
        ie = self._pooled_item(item)
        dot = (ue * ie).sum(dim=-1)

        if self.use_bias:
            ub = self.user_bias(user).squeeze(-1)
            ib = self.item_bias(item).squeeze(-1)
            return dot + ub + ib + self.global_bias

        return dot + self.global_bias

    @torch.no_grad()
    def embed_all_items(self, chunk_size: int = 65536) -> torch.Tensor:
        self.eval()
        dev = self.user_E_f.weight.device
        outs = []
        for lo in range(0, self.num_items, chunk_size):
            hi = min(self.num_items, lo + chunk_size)
            ids = torch.arange(lo, hi, device=dev, dtype=torch.long)
            outs.append(self._pooled_item(ids))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def item_bias_vector(self) -> torch.Tensor:
        dev = self.user_E_f.weight.device
        if not self.use_bias:
            return torch.zeros((self.num_items,), device=dev, dtype=torch.float32)
        return self.item_bias.weight.squeeze(-1)

    @torch.no_grad()
    def embed_new_user_infrequent(self, user_id: int) -> torch.Tensor:
        if not self._hash_inited:
            raise RuntimeError("LEAFModel: init_hash_params(...) must be called first.")
        dev = self.user_E_f.weight.device
        uid = torch.tensor([int(user_id)], device=dev, dtype=torch.long)
        idx = self._k_affine_hash(uid, self.a_u_inf, self.b_u_inf, self.M_u_inf)
        return self.user_E_inf(idx).mean(dim=1)[0]

# =========================================================
# Experiment
# =========================================================

LEAF_REQUIRED_HPS = [
    "leaf_approach",
    "lr",
    "lr_incremental",
    "weight_decay",
    "weight_decay_incremental",
    "batch_size",
    "epochs_offline",
    "epochs_incremental",
    "incremental_batch_size",
    "patience",
    "incremental_patience",
    "incremental_freeze_mode",
]


class LeafMFExperiment(BaseExperiment):
    """
    LEAF aligned to the new BaseExperiment.

    Supports:
      - mse
      - residual_mse

    Key behavior:
      - _predict_df returns RAW, UNCLIPPED predictions
      - eval clipping / per-rating rmse / ranking are delegated to BaseExperiment
      - ridge incremental fit uses incremental/train only (no val leakage)
      - paper online sgd_adapt is temporary: model/sketch state is restored
      - incremental sketch updates are disabled by default for stability
      - residual_mse uses persisted per-user mean lookups:
          * offline lookup for offline / offline_post_incremental
          * incremental lookup for incremental
          * temporary online lookup for online (not persisted)
    """
    def _init_models(self) -> None:
        meta = self.ds["meta"]
        self.n_items = int(meta["n_items"])
        self.n_off = int(meta["n_offline_users"])
        self.n_inc = int(meta["n_incremental_users"])
        self.n_on = int(meta["n_online_users"])

        hps = dict(self.params.model_hps or {})

        self.leaf_approach = str(hps.get("leaf_approach", "paper")).lower().strip()
        if self.leaf_approach not in {"ridge", "paper"}:
            raise ValueError("model_hps['leaf_approach'] must be 'ridge' or 'paper'.")

        missing = [hp for hp in LEAF_REQUIRED_HPS if hps.get(hp) is None]
        if missing:
            raise ValueError(f"Missing LEAF hyperparams in model_hps: {missing}")

        # Bias is independent of leaf_approach.
        default_use_bias = (self.leaf_approach == "ridge")
        self.use_bias = bool(hps.get("use_bias", default_use_bias))

        # Incremental behavior is independent of leaf_approach.
        self.incremental_mode = str(
            hps.get(
                "incremental_mode",
                "ridge" if self.leaf_approach == "ridge" else "sgd",
            )
        ).lower().strip()
        if self.incremental_mode not in {"ridge", "sgd"}:
            raise ValueError("model_hps['incremental_mode'] must be 'ridge' or 'sgd'.")

        # Unified online mode: prior from infrequent-user table + ridge fold-in
        self.online_mode = str(hps.get("online_mode", "ridge_prior")).lower().strip()
        if self.online_mode not in {"ridge_prior"}:
            raise ValueError("model_hps['online_mode'] must be 'ridge_prior'.")

        init = dict(self.params.model_init or {})

        self.loss_type = str(init.pop("loss_type", "mse")).lower().strip()
        if self.loss_type not in {"mse", "residual_mse"}:
            raise ValueError(
                f"Unsupported loss_type={self.loss_type!r}. "
                "Expected one of: {'mse', 'residual_mse'}"
            )
        self.residual_target = (self.loss_type == "residual_mse")

        for req in ("embedding_size", "k", "M_u_f", "M_u_inf", "M_i_f", "M_i_inf"):
            if req not in init:
                raise ValueError(f"LEAF model_init must contain '{req}'.")

        init["num_users"] = self.n_off + self.n_inc
        init["num_items"] = self.n_items

        self.use_amp = (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16

        self.model = LEAFModel(
            **init,
            use_bias=self.use_bias,
        ).to(self.device)
        self.model.init_hash_params(seed=int(self.params.seed or 0))

        self.optimizer = self._make_optimizer(
            lr=float(hps["lr"]),
            weight_decay=float(hps["weight_decay"]),
            params=self.model.parameters(),
        )

        self.batch_size = int(hps["batch_size"])
        self.incremental_batch_size = int(hps["incremental_batch_size"])
        self.epochs_offline = int(hps["epochs_offline"])
        self.epochs_incremental = int(hps["epochs_incremental"])

        self.patience = int(hps["patience"])
        self.incremental_patience = int(hps["incremental_patience"])
        self.min_delta = float(hps.get("min_delta", 1e-4))

        self.ridge_lam_p = float(hps.get("ridge_lambda_factors", 1e-2))
        self.ridge_lam_bu = float(hps.get("ridge_lambda_user_bias", 1e-2))

        self.lr_incremental = float(hps["lr_incremental"])
        self.weight_decay = float(hps["weight_decay"])
        self.weight_decay_incremental = float(hps["weight_decay_incremental"])

        self.incremental_freeze_mode = str(hps["incremental_freeze_mode"]).lower().strip()
        self.incremental_update_sketch = bool(hps.get("incremental_update_sketch", False))

        self._init_sketch_state(
            num_users=self.n_off + self.n_inc,
            n_items=self.n_items,
        )

        self._best_state: Optional[dict] = None
        self._inc_p: Optional[torch.Tensor] = None
        self._inc_bu: Optional[torch.Tensor] = None

        self._offline_user_mean_lookup: Optional[Dict[str, Any]] = None
        self._incremental_user_mean_lookup: Optional[Dict[str, Any]] = None

        # Optional cache for batch online inference
        self._online_completed: Optional[torch.Tensor] = None
        self._online_u_min: int = 0


    def _make_optimizer(self, *, lr: float, weight_decay: float, params):
        try:
            return torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                fused=(self.device.type == "cuda"),
            )
        except TypeError:
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # -------------------------
    # residual helpers
    # -------------------------

    def _make_dense_user_mean_lookup(
        self,
        df: Optional[pd.DataFrame],
        *,
        n_users: int,
        user_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Dense mean lookup for contiguous user ranges.

        Examples:
          - offline users:     ids in [0, n_off), offset=0
          - incremental users: ids in [n_off, n_off+n_inc), offset=n_off
        """
        n_users = int(n_users)
        user_offset = int(user_offset)

        if n_users < 0:
            raise ValueError("n_users must be >= 0.")

        if df is None or len(df) == 0:
            means = torch.zeros((n_users,), dtype=torch.float32, device=self.device)
            return {
                "kind": "dense",
                "means": means,
                "user_offset": user_offset,
                "global_mean": 0.0,
            }

        u_np = df["user_id"].to_numpy(dtype=np.int64, copy=False) - user_offset
        r_np = df["rating"].to_numpy(dtype=np.float32, copy=False)

        if np.any(u_np < 0) or np.any(u_np >= n_users):
            bad = u_np[(u_np < 0) | (u_np >= n_users)][:10].tolist()
            raise ValueError(
                f"Dense mean lookup received user ids outside [0, {n_users}) "
                f"after subtracting offset={user_offset}. Examples: {bad}"
            )

        global_mean = float(np.mean(r_np)) if r_np.size > 0 else 0.0
        means_np = np.full((n_users,), global_mean, dtype=np.float32)

        tmp = pd.DataFrame({"u": u_np, "r": r_np})
        stats = tmp.groupby("u", sort=False)["r"].mean()

        idx = stats.index.to_numpy(dtype=np.int64, copy=True)
        vals = stats.to_numpy(dtype=np.float32, copy=True)
        means_np[idx] = vals

        means = torch.as_tensor(means_np, dtype=torch.float32, device=self.device)
        return {
            "kind": "dense",
            "means": means,
            "user_offset": user_offset,
            "global_mean": global_mean,
        }

    def _make_sparse_user_mean_lookup(
        self,
        df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Sparse lookup for arbitrary user ids, mainly online users.
        """
        if df is None or len(df) == 0:
            return {
                "kind": "sparse",
                "user_ids": torch.empty(0, dtype=torch.int64, device=self.device),
                "means": torch.empty(0, dtype=torch.float32, device=self.device),
                "global_mean": 0.0,
            }

        stats = df.groupby("user_id", sort=True)["rating"].mean()
        user_ids = torch.as_tensor(
            stats.index.to_numpy(dtype=np.int64, copy=True),
            dtype=torch.int64,
            device=self.device,
        )
        means = torch.as_tensor(
            stats.to_numpy(dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device,
        )

        return {
            "kind": "sparse",
            "user_ids": user_ids,
            "means": means,
            "global_mean": float(df["rating"].mean()),
        }

    def _lookup_user_means(
        self,
        user_ids: torch.Tensor,
        user_mean_lookup: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if not self.residual_target or user_mean_lookup is None:
            return torch.zeros_like(user_ids, dtype=torch.float32, device=user_ids.device)

        kind = str(user_mean_lookup["kind"]).lower()

        if kind == "dense":
            means = user_mean_lookup["means"].to(user_ids.device)
            user_offset = int(user_mean_lookup["user_offset"])

            local_ids = user_ids.to(torch.int64) - user_offset
            if local_ids.numel() == 0:
                return torch.empty(0, dtype=torch.float32, device=user_ids.device)

            bad_mask = (local_ids < 0) | (local_ids >= means.numel())
            if bad_mask.any():
                bad = local_ids[bad_mask][:10].detach().cpu().tolist()
                raise ValueError(
                    f"Dense mean lookup received out-of-range local ids. Examples: {bad}, "
                    f"valid range=[0, {means.numel()})"
                )

            return means.index_select(0, local_ids)

        if kind == "sparse":
            ids_sorted = user_mean_lookup["user_ids"].to(user_ids.device)
            means_sorted = user_mean_lookup["means"].to(user_ids.device)
            fallback = float(user_mean_lookup.get("global_mean", 0.0))

            if ids_sorted.numel() == 0:
                return torch.full(
                    (user_ids.numel(),),
                    fill_value=fallback,
                    dtype=torch.float32,
                    device=user_ids.device,
                )

            pos = torch.searchsorted(ids_sorted, user_ids.to(torch.int64))
            inrange = pos < ids_sorted.numel()
            pos_clamped = pos.clamp_max(ids_sorted.numel() - 1)

            out = torch.full(
                (user_ids.numel(),),
                fill_value=fallback,
                dtype=torch.float32,
                device=user_ids.device,
            )

            hit = inrange & (ids_sorted[pos_clamped] == user_ids.to(torch.int64))
            if hit.any():
                out[hit] = means_sorted.index_select(0, pos_clamped[hit])

            return out

        raise ValueError(f"Unsupported user_mean_lookup kind: {kind!r}")

    def _target_from_raw(
        self,
        y_raw: torch.Tensor,
        user_means: torch.Tensor,
    ) -> torch.Tensor:
        y_raw = y_raw.float()
        if self.residual_target:
            return y_raw - user_means.float()
        return y_raw

    def _pred_model_to_raw(
        self,
        pred_model: torch.Tensor,
        user_means: torch.Tensor,
    ) -> torch.Tensor:
        pred_model = pred_model.float()
        if self.residual_target:
            return pred_model + user_means.float()
        return pred_model

    # -------------------------
    # sketch
    # -------------------------

    def _init_sketch_state(self, *, num_users: int, n_items: int) -> None:
        cfg = self.params.sketch or {}

        def _k_from_cfg(prefix: str, total: int, default_pct: float = 0.01) -> int:
            k_key = f"K_{prefix}"
            if k_key in cfg and cfg[k_key] is not None:
                return max(1, int(cfg[k_key]))
            pct_key = f"{prefix}_top_pct"
            pct = float(cfg.get(pct_key, default_pct))
            pct = min(max(pct, 1e-6), 1.0)
            return max(1, int(round(total * pct)))

        self.K_users = _k_from_cfg("users", num_users, default_pct=float(cfg.get("user_top_pct", 0.01)))
        self.K_items = _k_from_cfg("items", n_items, default_pct=float(cfg.get("item_top_pct", 0.01)))
        self.refresh_every = max(1, int(cfg.get("refresh_every", 1)))

        self.user_sketch = SMEDSketch(self.K_users, seed=int(self.params.seed or 0))
        self.item_sketch = SMEDSketch(self.K_items, seed=int(self.params.seed or 0))
        self._sketch_step = 0

        self.model.set_frequent_keys(
            user_keys_sorted=torch.empty(0, dtype=torch.int64, device=self.device),
            item_keys_sorted=torch.empty(0, dtype=torch.int64, device=self.device),
        )

    def _sketch_enabled_for_phase(self, phase: str) -> bool:
        phase = phase.lower()

        if phase == "offline":
            return True

        if phase == "incremental":
            return bool(self.incremental_update_sketch) and (self.incremental_mode == "sgd")

        if phase == "online_adapt":
            return False

        return False

    def _sketch_update_and_route(self, u: torch.Tensor, i: torch.Tensor) -> None:
        self.user_sketch.update_batch(u)
        self.item_sketch.update_batch(i)
        self._sketch_step += 1
        if (self._sketch_step % self.refresh_every) == 0:
            self.model.set_frequent_keys(
                self.user_sketch.frequent_keys_sorted(self.device),
                self.item_sketch.frequent_keys_sorted(self.device),
            )

    # -------------------------
    # best-state
    # -------------------------

    def _reset_best(self) -> None:
        self._best_state = None

    def _save_best(self) -> None:
        self._best_state = {
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "user_sketch": self.user_sketch.state_dict(),
            "item_sketch": self.item_sketch.state_dict(),
            "sketch_step": int(self._sketch_step),
        }

    def _load_best(self) -> None:
        if self._best_state is None:
            return

        sd = {k: v.to(self.device) for k, v in self._best_state["model"].items()}
        self.model.load_state_dict(sd, strict=True)

        self.user_sketch.load_state_dict(self._best_state["user_sketch"])
        self.item_sketch.load_state_dict(self._best_state["item_sketch"])
        self._sketch_step = int(self._best_state["sketch_step"])

        self.model.set_frequent_keys(
            self.user_sketch.frequent_keys_sorted(self.device),
            self.item_sketch.frequent_keys_sorted(self.device),
        )

    # -------------------------
    # train / eval loops
    # -------------------------

    def _train_epoch_sgd(
        self,
        *,
        phase: str,
        tr_pack: TripletPack,
        optimizer,
        batch_size: int,
        user_mean_lookup: Optional[Dict[str, Any]] = None,
    ) -> float:
        self.model.train()
        sse_raw = 0.0
        n_ex = 0

        sketch_on = self._sketch_enabled_for_phase(phase)
        device_type = self.device.type

        for u, i, r in iter_minibatches(tr_pack, batch_size, shuffle=True):
            if sketch_on:
                self._sketch_update_and_route(u, i)

            user_means = self._lookup_user_means(u, user_mean_lookup)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_model = self.model(u, i)
                y_target = self._target_from_raw(r, user_means)
                loss = torch.mean((pred_model.float() - y_target.float()).pow(2))

            loss.backward()
            optimizer.step()

            pred_raw = self._pred_model_to_raw(pred_model.detach(), user_means)
            diff_raw = pred_raw.float() - r.float()

            bs = int(r.numel())
            sse_raw += float(diff_raw.square().sum().item())
            n_ex += bs

        if sketch_on:
            self.model.set_frequent_keys(
                self.user_sketch.frequent_keys_sorted(self.device),
                self.item_sketch.frequent_keys_sorted(self.device),
            )

        return float("nan") if n_ex == 0 else sse_raw / n_ex

    @torch.no_grad()
    def _eval_pack(
        self,
        vl_pack: Optional[TripletPack],
        batch_size: int,
        user_mean_lookup: Optional[Dict[str, Any]] = None,
    ) -> float:
        if vl_pack is None or vl_pack.r.numel() == 0:
            return float("nan")

        self.model.eval()
        device_type = self.device.type

        sse_raw = 0.0
        n_ex = 0
        for u, i, r in iter_minibatches(vl_pack, batch_size, shuffle=False):
            user_means = self._lookup_user_means(u, user_mean_lookup)

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_model = self.model(u, i)

            pred_raw = self._pred_model_to_raw(pred_model, user_means)
            diff_raw = pred_raw.float() - r.float()

            bs = int(r.numel())
            sse_raw += float(diff_raw.square().sum().item())
            n_ex += bs

        return float("nan") if n_ex == 0 else sse_raw / n_ex

    def _run_epochs_sgd(
        self,
        *,
        phase: str,
        tr_pack: TripletPack,
        vl_pack: Optional[TripletPack],
        optimizer,
        epochs: int,
        batch_size: int,
        early_stop: bool,
        min_delta: float,
        patience: int,
        verbose: bool,
        tag: str,
        user_mean_lookup: Optional[Dict[str, Any]] = None,
    ) -> list[dict]:
        best_val = float("inf")
        bad = 0
        hist: list[dict] = []
        self._reset_best()

        for ep in range(1, epochs + 1):
            tr_mse = self._train_epoch_sgd(
                phase=phase,
                tr_pack=tr_pack,
                optimizer=optimizer,
                batch_size=batch_size,
                user_mean_lookup=user_mean_lookup,
            )
            vl_mse = tr_mse if (vl_pack is None) else self._eval_pack(
                vl_pack,
                batch_size,
                user_mean_lookup=user_mean_lookup,
            )

            hist.append(
                {
                    "epoch": ep,
                    "train_mse": float(tr_mse),
                    "val_mse": float(vl_mse),
                }
            )

            if verbose:
                print(f"[{tag}][{ep:03d}/{epochs}] train_mse={tr_mse:.6f}  val_mse={vl_mse:.6f}")

            try:
                self._log_epoch(
                    tag.lower(),
                    ep,
                    train_loss=float(tr_mse),
                    val_loss=float(vl_mse),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            except Exception:
                pass

            if vl_pack is not None and math.isfinite(vl_mse):
                if vl_mse < best_val - min_delta:
                    best_val = vl_mse
                    bad = 0
                    self._save_best()
                else:
                    bad += 1

                if early_stop and bad >= patience:
                    if verbose:
                        print(f"[{tag}] early stop at epoch {ep} (best_val_mse={best_val:.6f})")
                    break

        if vl_pack is not None and self._best_state is not None:
            self._load_best()
            if verbose:
                print(f"[{tag}] restored best weights (best_val_mse={best_val:.6f})")

        return hist

    # -------------------------
    # offline
    # -------------------------

    def _fit_offline(self) -> None:
        off = self.ds["offline"]

        self._offline_user_mean_lookup = self._make_dense_user_mean_lookup(
            off["train"],
            n_users=self.n_off,
            user_offset=0,
        )

        tr_pack = df_to_tripletpack(off["train"], self.device)
        vl_pack = df_to_tripletpack(off["val"], self.device) if len(off.get("val", [])) else None

        if tr_pack is None:
            self.artifacts.logs["offline.history"] = []
            return

        hist = self._run_epochs_sgd(
            phase="offline",
            tr_pack=tr_pack,
            vl_pack=vl_pack,
            optimizer=self.optimizer,
            epochs=self.epochs_offline,
            batch_size=self.batch_size,
            early_stop=True,
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=True,
            tag="Offline",
            user_mean_lookup=self._offline_user_mean_lookup,
        )
        self.artifacts.logs["offline.history"] = hist

    # -------------------------
    # ridge helpers
    # -------------------------

    @staticmethod
    def _sorted_slices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[int, int]]]:
        if df is None or len(df) == 0:
            return (
                np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.float32),
                {},
            )

        u_np = df["user_id"].to_numpy(dtype=np.int64, copy=False)
        i_np = df["item_id"].to_numpy(dtype=np.int64, copy=False)
        r_np = df["rating"].to_numpy(dtype=np.float32, copy=False)

        order = np.argsort(u_np, kind="mergesort")
        u_np = u_np[order]
        i_np = i_np[order]
        r_np = r_np[order]

        uniq_u, start_idx, counts = np.unique(u_np, return_index=True, return_counts=True)
        slices = {int(uu): (int(s), int(c)) for uu, s, c in zip(uniq_u, start_idx, counts)}
        return u_np, i_np, r_np, slices

    @staticmethod
    def _ridge_solve_theta(Vi: torch.Tensor, y: torch.Tensor, reg_vec: torch.Tensor) -> torch.Tensor:
        """
        Solve:
        min_{p, b} ||Vi p + b - y||^2
                    + sum_j reg_vec[j] * p_j^2
                    + reg_vec[d] * b^2

        via augmented least squares.
        """
        if Vi.ndim != 2:
            raise ValueError(f"Vi must be rank-2. Got shape={tuple(Vi.shape)}")
        if y.ndim != 1:
            raise ValueError(f"y must be rank-1. Got shape={tuple(y.shape)}")
        if reg_vec.ndim != 1:
            raise ValueError(f"reg_vec must be rank-1. Got shape={tuple(reg_vec.shape)}")

        m, d = Vi.shape
        if y.numel() != m:
            raise ValueError(f"y length mismatch: expected {m}, got {y.numel()}")
        if reg_vec.numel() != d + 1:
            raise ValueError(f"reg_vec length mismatch: expected {d + 1}, got {reg_vec.numel()}")

        dev = Vi.device
        dtype = torch.float64

        Vi64 = Vi.to(device=dev, dtype=dtype)
        y64 = y.to(device=dev, dtype=dtype)
        reg64 = reg_vec.to(device=dev, dtype=dtype).clone()

        reg64[:d] = reg64[:d].clamp_min(1e-6)
        reg64[d] = reg64[d].clamp_min(1e-6)

        sqrt_reg_p = torch.sqrt(reg64[:d])
        sqrt_reg_b = torch.sqrt(reg64[d]).view(1)

        X_data = torch.cat(
            [Vi64, torch.ones((m, 1), device=dev, dtype=dtype)],
            dim=1,
        )  # (m, d+1)
        y_data = y64

        X_reg_p = torch.cat(
            [torch.diag(sqrt_reg_p), torch.zeros((d, 1), device=dev, dtype=dtype)],
            dim=1,
        )  # (d, d+1)
        y_reg_p = torch.zeros((d,), device=dev, dtype=dtype)

        X_reg_b = torch.cat(
            [torch.zeros((1, d), device=dev, dtype=dtype), sqrt_reg_b.view(1, 1)],
            dim=1,
        )  # (1, d+1)
        y_reg_b = torch.zeros((1,), device=dev, dtype=dtype)

        X_aug = torch.cat([X_data, X_reg_p, X_reg_b], dim=0)
        y_aug = torch.cat([y_data, y_reg_p, y_reg_b], dim=0)

        sol = torch.linalg.lstsq(X_aug, y_aug.unsqueeze(-1)).solution.squeeze(-1)

        if not torch.isfinite(sol).all():
            raise RuntimeError("Non-finite solution in _ridge_solve_theta.")

        return sol.to(dtype=torch.float32)

    @staticmethod
    def _ridge_solve_theta_with_prior(
        Vi: torch.Tensor,
        y: torch.Tensor,
        reg_vec: torch.Tensor,
        p0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve:
        min_{p, b} ||Vi p + b - y||^2
                    + sum_j reg_vec[j] * (p_j - p0_j)^2
                    + reg_vec[d] * b^2

        via augmented least squares, which is more numerically robust than
        Cholesky on the normal equations.

        Inputs:
        Vi:      (m, d)
        y:       (m,)
        reg_vec: (d+1,)  where reg_vec[:d] are factor penalties and reg_vec[d] is bias penalty
        p0:      (d,)    prior user vector

        Returns:
        theta: (d+1,) = [p, b]
        """
        if Vi.ndim != 2:
            raise ValueError(f"Vi must be rank-2. Got shape={tuple(Vi.shape)}")
        if y.ndim != 1:
            raise ValueError(f"y must be rank-1. Got shape={tuple(y.shape)}")
        if reg_vec.ndim != 1:
            raise ValueError(f"reg_vec must be rank-1. Got shape={tuple(reg_vec.shape)}")
        if p0.ndim != 1:
            raise ValueError(f"p0 must be rank-1. Got shape={tuple(p0.shape)}")

        m, d = Vi.shape
        if y.numel() != m:
            raise ValueError(f"y length mismatch: expected {m}, got {y.numel()}")
        if reg_vec.numel() != d + 1:
            raise ValueError(f"reg_vec length mismatch: expected {d + 1}, got {reg_vec.numel()}")
        if p0.numel() != d:
            raise ValueError(f"p0 length mismatch: expected {d}, got {p0.numel()}")

        dev = Vi.device
        dtype = torch.float64  # use higher precision for the tiny linear solve

        Vi64 = Vi.to(device=dev, dtype=dtype)
        y64 = y.to(device=dev, dtype=dtype)
        p064 = p0.to(device=dev, dtype=dtype)

        reg64 = reg_vec.to(device=dev, dtype=dtype).clone()

        # Numerical safety floor: tiny or zero ridge can still become unstable.
        reg64[:d] = reg64[:d].clamp_min(1e-6)
        reg64[d] = reg64[d].clamp_min(1e-6)

        sqrt_reg_p = torch.sqrt(reg64[:d])                 # (d,)
        sqrt_reg_b = torch.sqrt(reg64[d]).view(1)          # (1,)

        # Data block: [Vi, 1]
        X_data = torch.cat(
            [Vi64, torch.ones((m, 1), device=dev, dtype=dtype)],
            dim=1,
        )  # (m, d+1)
        y_data = y64

        # Prior block on p: sqrt(lambda_p) * (p - p0)
        X_prior_p = torch.cat(
            [torch.diag(sqrt_reg_p), torch.zeros((d, 1), device=dev, dtype=dtype)],
            dim=1,
        )  # (d, d+1)
        y_prior_p = sqrt_reg_p * p064

        # Bias regularization block: sqrt(lambda_b) * b
        X_prior_b = torch.cat(
            [torch.zeros((1, d), device=dev, dtype=dtype), sqrt_reg_b.view(1, 1)],
            dim=1,
        )  # (1, d+1)
        y_prior_b = torch.zeros((1,), device=dev, dtype=dtype)

        X_aug = torch.cat([X_data, X_prior_p, X_prior_b], dim=0)   # (m+d+1, d+1)
        y_aug = torch.cat([y_data, y_prior_p, y_prior_b], dim=0)   # (m+d+1,)

        # lstsq is robust for these tiny per-user systems.
        sol = torch.linalg.lstsq(X_aug, y_aug.unsqueeze(-1)).solution.squeeze(-1)

        if not torch.isfinite(sol).all():
            raise RuntimeError("Non-finite solution in _ridge_solve_theta_with_prior.")

        return sol.to(dtype=torch.float32)

    # -------------------------
    # paper SGD trainability
    # -------------------------
    def _configure_incremental_sgd_trainability(self) -> list[torch.nn.Parameter]:
        """
        Controls what incremental SGD is allowed to update.

        Supported modes:
        - "full"
        - "train_items_only"
        - "train_items_plus_global"
        - "train_users_only"
        - "train_users_plus_global"
        - "train_global_only"
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        mode = self.incremental_freeze_mode

        for p in self.model.parameters():
            p.requires_grad = False

        trainable: list[torch.nn.Parameter] = []
        seen: set[int] = set()

        def add_module(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            for p in module.parameters(recurse=True):
                pid = id(p)
                if pid in seen:
                    continue
                p.requires_grad = True
                trainable.append(p)
                seen.add(pid)

        def add_param(param: Optional[torch.nn.Parameter]) -> None:
            if param is None:
                return
            pid = id(param)
            if pid in seen:
                return
            param.requires_grad = True
            trainable.append(param)
            seen.add(pid)

        if mode == "full":
            add_module(self.model.user_E_f)
            add_module(self.model.user_E_inf)
            add_module(self.model.item_E_f)
            add_module(self.model.item_E_inf)
            if self.model.use_bias:
                add_module(self.model.user_bias)
                add_module(self.model.item_bias)
            add_param(self.model.global_bias)

        elif mode == "train_items_only":
            add_module(self.model.item_E_f)
            add_module(self.model.item_E_inf)
            if self.model.use_bias:
                add_module(self.model.item_bias)

        elif mode == "train_items_plus_global":
            add_module(self.model.item_E_f)
            add_module(self.model.item_E_inf)
            if self.model.use_bias:
                add_module(self.model.item_bias)
            add_param(self.model.global_bias)

        elif mode == "train_users_only":
            add_module(self.model.user_E_f)
            add_module(self.model.user_E_inf)
            if self.model.use_bias:
                add_module(self.model.user_bias)

        elif mode == "train_users_plus_global":
            add_module(self.model.user_E_f)
            add_module(self.model.user_E_inf)
            if self.model.use_bias:
                add_module(self.model.user_bias)
            add_param(self.model.global_bias)

        elif mode == "train_global_only":
            add_param(self.model.global_bias)

        else:
            raise ValueError(
                f"Unsupported incremental_freeze_mode={mode!r}. "
                "Expected one of: "
                "{'full', 'train_items_only', 'train_items_plus_global', "
                "'train_users_only', 'train_users_plus_global', 'train_global_only'}"
            )

        if not trainable:
            raise ValueError(
                f"No trainable parameters selected for incremental_freeze_mode={mode!r}."
            )

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[incremental] freeze_mode={mode} | "
            f"trainable_params={n_trainable:,}/{n_total:,}"
        )

        return trainable

    def _configure_paper_sgd_trainability(self) -> list[torch.nn.Parameter]:
        """
        Controls what paper-mode SGD is allowed to update.

        Supported modes:
          - "full"
          - "train_items_only"
          - "train_items_plus_global"
          - "train_users_only"
          - "train_users_plus_global"
          - "train_global_only"
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        mode = self.incremental_freeze_mode

        for p in self.model.parameters():
            p.requires_grad = False

        trainable: list[torch.nn.Parameter] = []
        seen: set[int] = set()

        def add_module(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            for p in module.parameters(recurse=True):
                pid = id(p)
                if pid in seen:
                    continue
                p.requires_grad = True
                trainable.append(p)
                seen.add(pid)

        def add_param(param: Optional[torch.nn.Parameter]) -> None:
            if param is None:
                return
            pid = id(param)
            if pid in seen:
                return
            param.requires_grad = True
            trainable.append(param)
            seen.add(pid)

        if mode == "full":
            add_module(self.model.user_E_f)
            add_module(self.model.user_E_inf)
            add_module(self.model.item_E_f)
            add_module(self.model.item_E_inf)
            add_param(self.model.global_bias)

        elif mode == "train_items_only":
            add_module(self.model.item_E_f)
            add_module(self.model.item_E_inf)

        elif mode == "train_items_plus_global":
            add_module(self.model.item_E_f)
            add_module(self.model.item_E_inf)
            add_param(self.model.global_bias)

        elif mode == "train_users_only":
            add_module(self.model.user_E_f)
            add_module(self.model.user_E_inf)

        elif mode == "train_users_plus_global":
            add_module(self.model.user_E_f)
            add_module(self.model.user_E_inf)
            add_param(self.model.global_bias)

        elif mode == "train_global_only":
            add_param(self.model.global_bias)

        else:
            raise ValueError(
                f"Unsupported incremental_freeze_mode={mode!r}. "
                "Expected one of: "
                "{'full', 'train_items_only', 'train_items_plus_global', "
                "'train_users_only', 'train_users_plus_global', 'train_global_only'}"
            )

        if not trainable:
            raise ValueError(
                f"No trainable parameters selected for incremental_freeze_mode={mode!r}."
            )

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[incremental] freeze_mode={mode} | "
            f"trainable_params={n_trainable:,}/{n_total:,}"
        )

        return trainable

    # -------------------------
    # incremental
    # -------------------------
    def _fit_incremental(self) -> None:
        inc = self.ds["incremental"]

        self._incremental_user_mean_lookup = self._make_dense_user_mean_lookup(
            inc["train"],
            n_users=self.n_inc,
            user_offset=self.n_off,
        )

        # ---------------------------------
        # Incremental mode: ridge fold-in
        # ---------------------------------
        if self.incremental_mode == "ridge":
            inc_fit = inc["train"]

            self.model.eval()
            with torch.no_grad():
                V = self.model.embed_all_items().to(self.device).to(torch.float32)
                Vb = self.model.item_bias_vector().to(self.device).to(torch.float32)
                gb = float(self.model.global_bias.detach().item())

            d = int(V.shape[1])
            self._inc_p = torch.zeros((self.n_inc, d), dtype=torch.float32, device=self.device)
            self._inc_bu = torch.zeros((self.n_inc,), dtype=torch.float32, device=self.device)

            reg_vec = torch.empty((d + 1,), device=self.device, dtype=torch.float32)
            reg_vec[:d].fill_(self.ridge_lam_p)
            reg_vec[d] = self.ridge_lam_bu

            _, i_np, r_np, slices = self._sorted_slices(inc_fit)

            for uu, _g in inc_fit.groupby("user_id", sort=False):
                uu = int(uu)
                if uu < self.n_off or uu >= self.n_off + self.n_inc:
                    continue

                sl = slices.get(uu)
                if sl is None:
                    continue
                s, c = sl

                items = torch.as_tensor(i_np[s:s + c], device=self.device, dtype=torch.long)
                r = torch.as_tensor(r_np[s:s + c], device=self.device, dtype=torch.float32)

                mean_u = self._lookup_user_means(
                    torch.tensor([uu], device=self.device, dtype=torch.long),
                    self._incremental_user_mean_lookup,
                )[0]
                user_means = torch.full_like(r, fill_value=float(mean_u.item()))

                Vi = V[items]
                y = self._target_from_raw(r, user_means) - Vb[items] - gb

                theta = self._ridge_solve_theta(Vi, y, reg_vec)
                self._inc_p[uu - self.n_off].copy_(theta[:-1])
                self._inc_bu[uu - self.n_off] = theta[-1]

            self.artifacts.logs["incremental.history"] = []
            return

        # ---------------------------------
        # Incremental mode: SGD continuation
        # ---------------------------------
        if self.incremental_mode != "sgd":
            raise ValueError(
                f"Unsupported incremental_mode={self.incremental_mode!r}. "
                "Expected 'ridge' or 'sgd'."
            )

        tr_pack = df_to_tripletpack(inc["train"], self.device)
        vl_pack = df_to_tripletpack(inc["val"], self.device) if len(inc.get("val", [])) else None

        if tr_pack is None:
            self.artifacts.logs["incremental.history"] = []
            return

        old_req = {id(p): p.requires_grad for p in self.model.parameters()}
        old_user_sketch = self.user_sketch.state_dict()
        old_item_sketch = self.item_sketch.state_dict()
        old_sketch_step = int(self._sketch_step)

        trainable_params = self._configure_incremental_sgd_trainability()

        opt_inc = self._make_optimizer(
            lr=self.lr_incremental,
            weight_decay=self.weight_decay_incremental,
            params=trainable_params,
        )

        try:
            hist = self._run_epochs_sgd(
                phase="incremental",
                tr_pack=tr_pack,
                vl_pack=vl_pack,
                optimizer=opt_inc,
                epochs=self.epochs_incremental,
                batch_size=self.incremental_batch_size,
                early_stop=True,
                min_delta=self.min_delta,
                patience=self.incremental_patience,
                verbose=True,
                tag="Incremental",
                user_mean_lookup=self._incremental_user_mean_lookup,
            )
            self.artifacts.logs["incremental.history"] = hist
        finally:
            for p in self.model.parameters():
                p.requires_grad = old_req[id(p)]

            if self.incremental_update_sketch:
                self.user_sketch.load_state_dict(old_user_sketch)
                self.item_sketch.load_state_dict(old_item_sketch)
                self._sketch_step = old_sketch_step
                self.model.set_frequent_keys(
                    self.user_sketch.frequent_keys_sorted(self.device),
                    self.item_sketch.frequent_keys_sorted(self.device),
                )

    # -------------------------
    # prediction helpers
    # -------------------------

    @torch.no_grad()
    def _predict_model_df(
        self,
        df: pd.DataFrame,
        *,
        batch_size: Optional[int] = None,
        user_mean_lookup: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        u_np = df["user_id"].to_numpy(dtype=np.int64, copy=True)
        i_np = df["item_id"].to_numpy(dtype=np.int64, copy=True)

        bs = int(batch_size or self.batch_size)
        out = np.empty((len(df),), dtype=np.float32)

        self.model.eval()
        device_type = self.device.type

        for s in range(0, len(df), bs):
            e = min(s + bs, len(df))
            u = _as_device_tensor_1d(u_np[s:e], dtype=torch.int64, device=self.device)
            i = _as_device_tensor_1d(i_np[s:e], dtype=torch.int64, device=self.device)
            user_means = self._lookup_user_means(u, user_mean_lookup)

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_model = self.model(u, i)

            pred_raw = self._pred_model_to_raw(pred_model, user_means)
            out[s:e] = pred_raw.detach().float().cpu().numpy()

        return out

    @torch.no_grad()
    def _predict_incremental_ridge_df(
        self,
        df: pd.DataFrame,
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        if self._inc_p is None or self._inc_bu is None:
            raise ValueError("Incremental ridge factors are missing. Run _fit_incremental() first.")

        bs = int(batch_size or self.incremental_batch_size)

        self.model.eval()
        with torch.no_grad():
            V = self.model.embed_all_items().to(self.device).to(torch.float32)
            Vb = self.model.item_bias_vector().to(self.device).to(torch.float32)
            gb = float(self.model.global_bias.detach().item())

        u_global_np = df["user_id"].to_numpy(dtype=np.int64, copy=True)
        local_u_np = u_global_np - self.n_off
        i_np = df["item_id"].to_numpy(dtype=np.int64, copy=True)

        if local_u_np.min(initial=0) < 0 or local_u_np.max(initial=-1) >= self.n_inc:
            raise ValueError("Incremental ridge prediction received user ids outside incremental range.")

        out = np.empty((len(df),), dtype=np.float32)

        for s in range(0, len(df), bs):
            e = min(s + bs, len(df))
            u_global = _as_device_tensor_1d(u_global_np[s:e], dtype=torch.int64, device=self.device)
            u_local = _as_device_tensor_1d(local_u_np[s:e], dtype=torch.int64, device=self.device)
            items = _as_device_tensor_1d(i_np[s:e], dtype=torch.int64, device=self.device)

            user_means = self._lookup_user_means(u_global, self._incremental_user_mean_lookup)

            p = self._inc_p.index_select(0, u_local)
            bu = self._inc_bu.index_select(0, u_local)
            Vi = V.index_select(0, items)
            Vib = Vb.index_select(0, items)

            pred_model = (p * Vi).sum(dim=-1) + bu + Vib + gb
            pred_raw = self._pred_model_to_raw(pred_model, user_means)
            out[s:e] = pred_raw.detach().float().cpu().numpy()

        return out

    @torch.no_grad()
    def _predict_online_df_ridge_prior(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        on = self.ds["online"]
        on_train = on["train"]

        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        online_mean_lookup = self._make_sparse_user_mean_lookup(on_train)

        self.model.eval()
        with torch.no_grad():
            V = self.model.embed_all_items().to(self.device).to(torch.float32)
            gb = float(self.model.global_bias.detach().item())

            if self.model.use_bias:
                Vb = self.model.item_bias_vector().to(self.device).to(torch.float32)
            else:
                Vb = torch.zeros((self.n_items,), device=self.device, dtype=torch.float32)

        d = int(V.shape[1])
        reg_vec = torch.empty((d + 1,), device=self.device, dtype=torch.float32)
        reg_vec[:d].fill_(self.ridge_lam_p)
        reg_vec[d] = self.ridge_lam_bu

        _, i_np, r_np, slices = self._sorted_slices(on_train)

        work = df[["user_id", "item_id"]].copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)
        out = np.empty((len(work),), dtype=np.float32)

        for uu, g in work.groupby("user_id", sort=False):
            row_pos = g["_row_pos"].to_numpy(np.int64, copy=False)
            item_ids = g["item_id"].to_numpy(np.int64, copy=True)

            mean_u = float(
                self._lookup_user_means(
                    torch.tensor([int(uu)], device=self.device, dtype=torch.long),
                    online_mean_lookup,
                )[0].item()
            )

            # Prior from learned infrequent-user table
            p0 = self.model.embed_new_user_infrequent(int(uu)).to(self.device).to(torch.float32)

            sl = slices.get(int(uu))
            if sl is None:
                u_vec = p0
                b_u = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            else:
                s, c = sl
                items_tr = torch.as_tensor(i_np[s:s + c], device=self.device, dtype=torch.long)
                r_tr = torch.as_tensor(r_np[s:s + c], device=self.device, dtype=torch.float32)

                Vi = V[items_tr]
                user_means = torch.full_like(r_tr, fill_value=mean_u)

                # Respect whether the trained model actually uses item bias.
                y = self._target_from_raw(r_tr, user_means) - Vb[items_tr] - gb

                theta = self._ridge_solve_theta_with_prior(
                    Vi=Vi,
                    y=y,
                    reg_vec=reg_vec,
                    p0=p0,
                )
                u_vec = theta[:-1]
                b_u = theta[-1]

            items = torch.as_tensor(item_ids, device=self.device, dtype=torch.long)
            pred_model = (V[items] @ u_vec) + Vb[items] + b_u + gb

            pred_raw = self._pred_model_to_raw(
                pred_model,
                torch.full_like(pred_model, fill_value=mean_u),
            )
            out[row_pos] = pred_raw.detach().float().cpu().numpy()

        return out


    def _snapshot_leaf_state(self) -> dict:
        return {
            "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "user_sketch": self.user_sketch.state_dict(),
            "item_sketch": self.item_sketch.state_dict(),
            "sketch_step": int(self._sketch_step),
        }

    def _restore_leaf_state(self, snap: dict) -> None:
        sd = {k: v.to(self.device) for k, v in snap["model"].items()}
        self.model.load_state_dict(sd, strict=True)
        self.user_sketch.load_state_dict(snap["user_sketch"])
        self.item_sketch.load_state_dict(snap["item_sketch"])
        self._sketch_step = int(snap["sketch_step"])
        self.model.set_frequent_keys(
            self.user_sketch.frequent_keys_sorted(self.device),
            self.item_sketch.frequent_keys_sorted(self.device),
        )

    def _resolve_online_prediction_mode(self) -> str:
        mode = str(getattr(self.params, "online_inference_pred_type", "batch")).lower().strip()
        if mode not in {"batch", "per_user"}:
            raise ValueError("online_inference_pred_type must be 'batch' or 'per_user'.")
        return mode

    @torch.no_grad()
    def _prepare_online_foldin_state(self) -> dict:
        on = self.ds["online"]
        on_train = on["train"]

        online_mean_lookup = self._make_sparse_user_mean_lookup(on_train)

        self.model.eval()
        with torch.no_grad():
            V = self.model.embed_all_items().to(self.device).to(torch.float32)
            gb = float(self.model.global_bias.detach().item())

            if self.model.use_bias:
                Vb = self.model.item_bias_vector().to(self.device).to(torch.float32)
            else:
                Vb = torch.zeros((self.n_items,), device=self.device, dtype=torch.float32)

        d = int(V.shape[1])
        reg_vec = torch.empty((d + 1,), device=self.device, dtype=torch.float32)
        reg_vec[:d].fill_(self.ridge_lam_p)
        reg_vec[d] = self.ridge_lam_bu

        _, i_np, r_np, slices = self._sorted_slices(on_train)

        return {
            "online_mean_lookup": online_mean_lookup,
            "V": V,
            "Vb": Vb,
            "gb": gb,
            "reg_vec": reg_vec,
            "i_np": i_np,
            "r_np": r_np,
            "slices": slices,
        }

    @torch.no_grad()
    def _predict_online_df_ridge_prior_per_user(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        st = self._prepare_online_foldin_state()
        online_mean_lookup = st["online_mean_lookup"]
        V = st["V"]
        Vb = st["Vb"]
        gb = st["gb"]
        reg_vec = st["reg_vec"]
        i_np = st["i_np"]
        r_np = st["r_np"]
        slices = st["slices"]

        work = df[["user_id", "item_id"]].copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)
        out = np.empty((len(work),), dtype=np.float32)

        for uu, g in work.groupby("user_id", sort=False):
            row_pos = g["_row_pos"].to_numpy(np.int64, copy=False)
            item_ids = g["item_id"].to_numpy(np.int64, copy=True)

            mean_u = float(
                self._lookup_user_means(
                    torch.tensor([int(uu)], device=self.device, dtype=torch.long),
                    online_mean_lookup,
                )[0].item()
            )

            p0 = self.model.embed_new_user_infrequent(int(uu)).to(self.device).to(torch.float32)

            sl = slices.get(int(uu))
            if sl is None:
                u_vec = p0
                b_u = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            else:
                s, c = sl
                items_tr = torch.as_tensor(i_np[s:s + c], device=self.device, dtype=torch.long)
                r_tr = torch.as_tensor(r_np[s:s + c], device=self.device, dtype=torch.float32)

                Vi = V[items_tr]
                user_means = torch.full_like(r_tr, fill_value=mean_u)
                y = self._target_from_raw(r_tr, user_means) - Vb[items_tr] - gb

                theta = self._ridge_solve_theta_with_prior(
                    Vi=Vi,
                    y=y,
                    reg_vec=reg_vec,
                    p0=p0,
                )
                u_vec = theta[:-1]
                b_u = theta[-1]

            items = torch.as_tensor(item_ids, device=self.device, dtype=torch.long)
            pred_model = (V[items] @ u_vec) + Vb[items] + b_u + gb
            pred_raw = self._pred_model_to_raw(
                pred_model,
                torch.full_like(pred_model, fill_value=mean_u),
            )
            out[row_pos] = pred_raw.detach().float().cpu().numpy()

        return out

    @torch.no_grad()
    def _predict_online_df_ridge_prior_batch(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        on = self.ds["online"]
        on_train = on["train"]

        if df is None or df.empty:
            self._online_completed = None
            self._online_u_min = 0
            return np.empty((0,), dtype=np.float32)

        st = self._prepare_online_foldin_state()
        online_mean_lookup = st["online_mean_lookup"]
        V = st["V"]
        Vb = st["Vb"]
        gb = st["gb"]
        reg_vec = st["reg_vec"]
        i_np = st["i_np"]
        r_np = st["r_np"]
        slices = st["slices"]

        users = np.unique(pd.concat([on_train["user_id"], df["user_id"]]).to_numpy(dtype=np.int64))
        users.sort()

        u_min = int(users.min())
        u_max = int(users.max())
        B = u_max - u_min + 1

        C_online = torch.empty((B, self.n_items), dtype=torch.float32, device=self.device)

        for uu in users:
            row = int(uu - u_min)

            mean_u = float(
                self._lookup_user_means(
                    torch.tensor([int(uu)], device=self.device, dtype=torch.long),
                    online_mean_lookup,
                )[0].item()
            )

            p0 = self.model.embed_new_user_infrequent(int(uu)).to(self.device).to(torch.float32)

            sl = slices.get(int(uu))
            if sl is None:
                u_vec = p0
                b_u = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            else:
                s, c = sl
                items_tr = torch.as_tensor(i_np[s:s + c], device=self.device, dtype=torch.long)
                r_tr = torch.as_tensor(r_np[s:s + c], device=self.device, dtype=torch.float32)

                Vi = V[items_tr]
                user_means = torch.full_like(r_tr, fill_value=mean_u)
                y = self._target_from_raw(r_tr, user_means) - Vb[items_tr] - gb

                theta = self._ridge_solve_theta_with_prior(
                    Vi=Vi,
                    y=y,
                    reg_vec=reg_vec,
                    p0=p0,
                )
                u_vec = theta[:-1]
                b_u = theta[-1]

            pred_model = (V @ u_vec) + Vb + b_u + gb
            pred_raw = self._pred_model_to_raw(
                pred_model,
                torch.full_like(pred_model, fill_value=mean_u),
            )
            C_online[row].copy_(pred_raw.to(torch.float32))

        self._online_completed = C_online
        self._online_u_min = u_min

        rows_rel = df["user_id"].to_numpy(np.int64) - u_min
        cols = df["item_id"].to_numpy(np.int64)

        preds = C_online[
            torch.as_tensor(rows_rel, device=self.device, dtype=torch.long),
            torch.as_tensor(cols, device=self.device, dtype=torch.long),
        ].detach().cpu().numpy()

        return preds

    def _predict_df(
        self,
        df: pd.DataFrame,
        *,
        phase: str,
    ) -> np.ndarray:
        off_bs = int((self.params.model_hps or {}).get("batch_size", 4096))
        inc_bs = int((self.params.model_hps or {}).get("incremental_batch_size", off_bs))

        if phase == "offline":
            return self._predict_model_df(
                df,
                batch_size=off_bs,
                user_mean_lookup=self._offline_user_mean_lookup,
            )

        if phase == "offline_post_incremental":
            return self._predict_model_df(
                df,
                batch_size=inc_bs,
                user_mean_lookup=self._offline_user_mean_lookup,
            )

        if phase == "incremental":
            if self.incremental_mode == "ridge":
                return self._predict_incremental_ridge_df(df, batch_size=inc_bs)
            if self.incremental_mode == "sgd":
                return self._predict_model_df(
                    df,
                    batch_size=inc_bs,
                    user_mean_lookup=self._incremental_user_mean_lookup,
                )
            raise ValueError(
                f"Unsupported incremental_mode={self.incremental_mode!r}. "
                "Expected 'ridge' or 'sgd'."
            )

        if phase == "online":
            if self.online_mode != "ridge_prior":
                raise ValueError(
                    f"Unsupported online_mode={self.online_mode!r}. "
                    "Expected 'ridge_prior'."
                )

            pred_mode = self._resolve_online_prediction_mode()
            if pred_mode == "per_user":
                return self._predict_online_df_ridge_prior_per_user(df)
            if pred_mode == "batch":
                return self._predict_online_df_ridge_prior_batch(df)

            raise ValueError(
                f"Unsupported online_inference_pred_type={pred_mode!r}. "
                "Expected 'batch' or 'per_user'."
            )

        raise ValueError(f"Unsupported phase for _predict_df: {phase}")

def _as_tensor_payload(x: Any) -> Optional[torch.Tensor]:
    """
    Return the underlying tensor to count, if x is:
      - nn.Embedding -> weight
      - nn.Parameter -> itself
      - torch.Tensor -> itself
    """
    if x is None:
        return None
    if isinstance(x, torch.nn.Embedding):
        return x.weight
    if isinstance(x, torch.nn.Parameter):
        return x
    if torch.is_tensor(x):
        return x
    return None


class SpaceTrackedLeafMFExperiment(SpaceTrackedExperimentBase, LeafMFExperiment):
    """
    Space tracker for LEAF with support for:
      - leaf_approach = "ridge" | "paper"
      - loss_type     = "mse" | "residual_mse"

    Counting rules:
      - Ignore optimizer state.
      - Do not count temporary online prediction caches / online-adapt artifacts.
      - Count only persisted residual mean vectors, not temporary online mean lookups.

    Persistent components tracked:
      - Compressed LEAF tables (always)
      - Hash buffers (always)
      - Routing-key tensors (always)
      - Sketch-state payload approximation (always)
      - Bias terms only if present in the model
      - Incremental ridge fold-in storage (_inc_p, _inc_bu) only in ridge mode
      - Offline / incremental user-mean vectors only when residual_mse is enabled
    """

    def _build_space_snapshot(self, phase: Phase) -> SpaceSnapshot:
        snap = SpaceSnapshot()

        model = getattr(self, "model", None)
        if model is None:
            return snap

        leaf_approach = str(getattr(self, "leaf_approach", "ridge")).lower()
        use_bias = bool(getattr(model, "use_bias", False))
        residual_target = bool(getattr(self, "residual_target", False))

        def add_tensor(
            *,
            key: str,
            name: str,
            t: torch.Tensor,
            formula: str,
            note: Optional[str] = None,
        ) -> None:
            dtype_key = _torch_dtype_to_key(t.dtype)
            shape = tuple(int(s) for s in t.shape)
            snap.components.append(
                SpaceComponent(
                    key=key,
                    name=name,
                    bytes=_bytes_dense(shape, dtype_key),
                    phase=phase,
                    shape=shape,
                    dtype=dtype_key,
                    formula=formula,
                    note=note,
                )
            )

        def add_bytes(
            *,
            key: str,
            name: str,
            bytes_: int,
            formula: str,
            note: Optional[str] = None,
            shape: Optional[Tuple[int, ...]] = None,
            dtype: Optional[str] = None,
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

        def add_lookup_means(
            *,
            key: str,
            name: str,
            lookup: Optional[Dict[str, Any]],
            note: str,
        ) -> None:
            if lookup is None:
                return
            t = _as_tensor_payload(lookup.get("means"))
            if t is None:
                return
            add_tensor(
                key=key,
                name=name,
                t=t,
                formula="n_users_phase",
                note=note,
            )

        # 1) Core LEAF compressed tables
        table_attrs = [
            ("leaf.user_E_f",   "LEAF: user table (frequent) U_f",     "user_E_f",   "M_u^f·d"),
            ("leaf.user_E_inf", "LEAF: user table (infrequent) U_inf", "user_E_inf", "M_u^inf·d"),
            ("leaf.item_E_f",   "LEAF: item table (frequent) V_f",     "item_E_f",   "M_i^f·d"),
            ("leaf.item_E_inf", "LEAF: item table (infrequent) V_inf", "item_E_inf", "M_i^inf·d"),
        ]
        for key, name, attr, formula in table_attrs:
            t = _as_tensor_payload(getattr(model, attr, None))
            if t is not None:
                add_tensor(key=key, name=name, t=t, formula=formula)

        # 2) Optional biases
        if use_bias:
            bias_attrs = [
                ("leaf.user_bias", "LEAF: user bias b_u", "user_bias", "(M+M')"),
                ("leaf.item_bias", "LEAF: item bias b_i", "item_bias", "N"),
            ]
            for key, name, attr, formula in bias_attrs:
                t = _as_tensor_payload(getattr(model, attr, None))
                if t is not None:
                    add_tensor(
                        key=key,
                        name=name,
                        t=t,
                        formula=formula,
                        note=f"Present because leaf_approach='{leaf_approach}' is bias-aware.",
                    )

        # 3) Global bias
        gb = _as_tensor_payload(getattr(model, "global_bias", None))
        if gb is not None:
            add_tensor(
                key="leaf.global_bias",
                name="LEAF: global bias b0",
                t=gb,
                formula="1",
                note="Scalar global bias.",
            )

        # 4) Routing keys
        rk_user = _as_tensor_payload(getattr(model, "freq_user_keys", None))
        rk_item = _as_tensor_payload(getattr(model, "freq_item_keys", None))

        if rk_user is not None:
            add_tensor(
                key="leaf.routing.freq_user_keys",
                name="LEAF: routing keys (frequent users)",
                t=rk_user,
                formula="|F_u| ≤ K_u",
                note="Sorted frequent-user ids used for routing.",
            )
        if rk_item is not None:
            add_tensor(
                key="leaf.routing.freq_item_keys",
                name="LEAF: routing keys (frequent items)",
                t=rk_item,
                formula="|F_i| ≤ K_i",
                note="Sorted frequent-item ids used for routing.",
            )

        # 5) Hash buffers
        hash_param_names = [
            "a_u_f", "b_u_f", "a_u_inf", "b_u_inf",
            "a_i_f", "b_i_f", "a_i_inf", "b_i_inf",
        ]
        for hp in hash_param_names:
            t = _as_tensor_payload(getattr(model, hp, None))
            if t is None:
                continue
            add_tensor(
                key=f"leaf.hash.{hp}",
                name=f"LEAF: hash buffer {hp}",
                t=t,
                formula="k",
                note="Affine-hash parameter buffer.",
            )

        # 6) Sketch-state payload approximation
        Ku = int(getattr(self, "K_users", 0) or 0)
        Ki = int(getattr(self, "K_items", 0) or 0)

        if Ku > 0:
            add_bytes(
                key="leaf.sketch.user_counters",
                name="SMED sketch: user counters (payload-only)",
                bytes_=_bytes_dense((Ku, 2), "int64"),
                shape=(Ku, 2),
                dtype="int64",
                formula="K_u·(id:int64 + count:int64)",
                note="Approx payload only; ignores Python dict overhead.",
            )
        if Ki > 0:
            add_bytes(
                key="leaf.sketch.item_counters",
                name="SMED sketch: item counters (payload-only)",
                bytes_=_bytes_dense((Ki, 2), "int64"),
                shape=(Ki, 2),
                dtype="int64",
                formula="K_i·(id:int64 + count:int64)",
                note="Approx payload only; ignores Python dict overhead.",
            )

        # 7) Residual mean vectors
        if residual_target:
            off_lookup = getattr(self, "_offline_user_mean_lookup", None)
            inc_lookup = getattr(self, "_incremental_user_mean_lookup", None)

            if phase == Phase.OFFLINE:
                add_lookup_means(
                    key="leaf.residual.offline_user_means",
                    name="Residual support: offline user mean vector",
                    lookup=off_lookup,
                    note=(
                        "Persisted dense offline user-mean vector used to restore "
                        "raw predictions from residual-space outputs."
                    ),
                )

            if phase in (Phase.INCREMENTAL, Phase.ONLINE):
                add_lookup_means(
                    key="leaf.residual.offline_user_means",
                    name="Residual support: offline user mean vector",
                    lookup=off_lookup,
                    note=(
                        "Retained after incremental because offline and "
                        "offline_post_incremental predictions still use it."
                    ),
                )
                add_lookup_means(
                    key="leaf.residual.incremental_user_means",
                    name="Residual support: incremental user mean vector",
                    lookup=inc_lookup,
                    note=(
                        "Persisted dense incremental user-mean vector used for "
                        "incremental-phase residual reconstruction."
                    ),
                )

        # 8) Incremental ridge fold-in storage
        if phase in (Phase.INCREMENTAL, Phase.ONLINE) and leaf_approach == "ridge":
            inc_p = _as_tensor_payload(getattr(self, "_inc_p", None))
            inc_bu = _as_tensor_payload(getattr(self, "_inc_bu", None))

            if inc_p is not None:
                add_tensor(
                    key="leaf.ridge.inc_p",
                    name="Ridge fold-in: incremental user factors P_inc",
                    t=inc_p,
                    formula="M'·d",
                    note="Persisted only in ridge mode.",
                )
            if inc_bu is not None:
                add_tensor(
                    key="leaf.ridge.inc_bu",
                    name="Ridge fold-in: incremental user bias b_u_inc",
                    t=inc_bu,
                    formula="M'",
                    note="Persisted only in ridge mode.",
                )

        return snap