"""
YouTube-style explicit-feedback recommender and experiment wrapper.

Generated from the original YouTube notebook with import paths adjusted for the
anonymous reproduction package. The class/function bodies are intentionally kept
unchanged as much as possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Iterator, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from recsys_edge.core import *  # noqa: F401,F403 - preserves notebook-style globals
from recsys_edge.core import _torch_dtype_to_key



@dataclass
class HistoryEncoderBatch:
    """
    Mini-batch already on the target device.

    history_items:   (B, Lb) int64 item ids, padded with pad_idx
    history_ratings: (B, Lb) float32 ratings aligned with history_items
    target_items:    (B,)    int64 target item ids
    user_means:      (B,)    float32 effective per-example mean used for
                     residual centering / add-back.
    """
    history_items: torch.Tensor
    history_ratings: torch.Tensor
    target_items: torch.Tensor
    user_means: torch.Tensor


@dataclass
class UserHistoryStore:
    """
    Device-side padded history bank.

    items:        (U, Lg) int64 on device
    ratings:      (U, Lg) float32 on device
    user_means:   (U,)    float32 full-evidence user means (fallback)
    user_sums:    (U,)    float32 full-evidence user rating sums
    user_counts:  (U,)    int64 full-evidence user rating counts
    global_mean:  scalar  float32 global rating mean for this evidence split
    """
    items: torch.Tensor
    ratings: torch.Tensor
    user_means: torch.Tensor
    user_sums: torch.Tensor
    user_counts: torch.Tensor
    global_mean: torch.Tensor
    pad_idx: int

    @property
    def n_users(self) -> int:
        return int(self.items.shape[0])

    @property
    def max_history(self) -> int:
        return int(self.items.shape[1])

    @property
    def device(self) -> torch.device:
        return self.items.device

    def _batch_max_valid_len(self, hist_items: torch.Tensor) -> int:
        if hist_items.ndim != 2:
            raise ValueError("hist_items must be rank-2.")

        valid_counts = hist_items.ne(self.pad_idx).sum(dim=1)
        max_len = int(valid_counts.max().item()) if valid_counts.numel() > 0 else 0
        return max(1, max_len)

    def _history_means_after_exclusion(
        self,
        users: torch.Tensor,
        hist_items: torch.Tensor,
        hist_ratings: torch.Tensor,
    ) -> torch.Tensor:
        fallback_means = self.user_means.index_select(0, users).float()

        valid_mask = hist_items.ne(self.pad_idx)
        valid_counts = valid_mask.sum(dim=1)
        valid_mask_f = valid_mask.float()

        hist_sums = (hist_ratings.float() * valid_mask_f).sum(dim=1)
        hist_means = hist_sums / valid_counts.float().clamp_min(1.0)

        return torch.where(valid_counts > 0, hist_means, fallback_means)

    def _leave_one_out_means_from_full_stats(
        self,
        users: torch.Tensor,
        target_ratings: torch.Tensor,
    ) -> torch.Tensor:
        full_sums = self.user_sums.index_select(0, users).float()
        full_counts = self.user_counts.index_select(0, users).float()
        target_ratings = target_ratings.float().to(full_sums.device)

        loo_counts = full_counts - 1.0
        loo_sums = full_sums - target_ratings

        global_mean = float(self.global_mean.item())
        fallback = torch.full_like(full_sums, fill_value=global_mean)

        loo_means = loo_sums / loo_counts.clamp_min(1.0)
        return torch.where(loo_counts > 0, loo_means, fallback)

    def gather(
        self,
        users: torch.Tensor,
        target_items: torch.Tensor,
        *,
        exclude_target: bool = True,
        target_ratings: Optional[torch.Tensor] = None,
    ) -> HistoryEncoderBatch:
        if users.device != self.device:
            users = users.to(self.device, dtype=torch.long, non_blocking=True)
        else:
            users = users.long()

        if target_items.device != self.device:
            target_items = target_items.to(self.device, dtype=torch.long, non_blocking=True)
        else:
            target_items = target_items.long()

        if target_ratings is not None:
            if target_ratings.device != self.device:
                target_ratings = target_ratings.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            else:
                target_ratings = target_ratings.float()

        hist_items = self.items.index_select(0, users)      # (B, Lg)
        hist_ratings = self.ratings.index_select(0, users)  # (B, Lg)

        batch_max_len = self._batch_max_valid_len(hist_items)
        hist_items = hist_items[:, :batch_max_len]
        hist_ratings = hist_ratings[:, :batch_max_len]

        if exclude_target:
            hit = hist_items.eq(target_items.unsqueeze(1))
            if hit.any():
                first_hit = hit & hit.int().cumsum(dim=1).eq(1)
                if first_hit.any():
                    hist_items = hist_items.clone()
                    hist_ratings = hist_ratings.clone()
                    hist_items[first_hit] = int(self.pad_idx)
                    hist_ratings[first_hit] = 0.0

        if target_ratings is not None:
            user_means = self._leave_one_out_means_from_full_stats(
                users=users,
                target_ratings=target_ratings,
            )
        else:
            user_means = self._history_means_after_exclusion(
                users=users,
                hist_items=hist_items,
                hist_ratings=hist_ratings,
            )

        return HistoryEncoderBatch(
            history_items=hist_items,
            history_ratings=hist_ratings,
            target_items=target_items,
            user_means=user_means,
        )

    def slice_users(
        self,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_items = self.items[start:end]
        hist_ratings = self.ratings[start:end]
        user_means = self.user_means[start:end]

        batch_max_len = self._batch_max_valid_len(hist_items)
        hist_items = hist_items[:, :batch_max_len]
        hist_ratings = hist_ratings[:, :batch_max_len]

        return hist_items, hist_ratings, user_means


@dataclass
class ExamplePack:
    """
    Device-side triplet pack.
    """
    u: torch.Tensor   # (N,) int64 on device
    i: torch.Tensor   # (N,) int64 on device
    r: torch.Tensor   # (N,) float32 on device

    @property
    def n(self) -> int:
        return int(self.u.numel())

    @property
    def device(self) -> torch.device:
        return self.u.device


# =========================================================
# Model
# =========================================================

class HistoryEncoderRec(nn.Module):
    """
    Explicit-feedback history-encoder recommender:

      1) item embedding lookup over user history
      2) rating-aware weighted pooling
      3) MLP encoder -> user embedding
      4) dot-product scorer against target item embedding

    Supports:
      - shared_table=True  : one embedding table for both history + target
      - shared_table=False : separate history and target embedding tables
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int = 64,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        activation: str = "relu",
        pad_idx: Optional[int] = None,
        shared_table: bool = True,
        use_item_bias: bool = True,
        use_global_bias: bool = True,
        rating_mode: str = "centered",   # {"none", "raw", "centered", "normalized", "binary"}
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_items = int(num_items)
        self.emb_dim = int(emb_dim)
        self.pad_idx = self.num_items if pad_idx is None else int(pad_idx)
        self.shared_table = bool(shared_table)
        self.use_item_bias = bool(use_item_bias)
        self.use_global_bias = bool(use_global_bias)
        self.rating_mode = str(rating_mode).lower()
        self.rating_min = float(rating_min)
        self.rating_max = float(rating_max)
        self.eps = float(eps)

        if self.pad_idx < 0:
            raise ValueError("pad_idx must be >= 0.")
        if self.pad_idx > self.num_items:
            raise ValueError("pad_idx must be <= num_items.")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one dimension.")

        n_embeddings = self.num_items + 1 if self.pad_idx == self.num_items else self.num_items

        self.hist_item_emb = nn.Embedding(
            num_embeddings=n_embeddings,
            embedding_dim=self.emb_dim,
            padding_idx=self.pad_idx,
        )

        if self.shared_table:
            self.target_item_emb = self.hist_item_emb
        else:
            self.target_item_emb = nn.Embedding(
                num_embeddings=n_embeddings,
                embedding_dim=self.emb_dim,
                padding_idx=self.pad_idx,
            )

        self.item_bias = (
            nn.Embedding(
                num_embeddings=n_embeddings,
                embedding_dim=1,
                padding_idx=self.pad_idx,
            )
            if self.use_item_bias else None
        )

        self.global_bias = nn.Parameter(torch.zeros(1)) if self.use_global_bias else None

        dims = [self.emb_dim] + list(hidden_dims)
        act_factory = self._build_activation_factory(activation)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                layers.append(act_factory())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.user_encoder = nn.Sequential(*layers)

        self.user_dim = int(dims[-1])
        self.target_proj = (
            nn.Linear(self.emb_dim, self.user_dim, bias=False)
            if self.user_dim != self.emb_dim
            else nn.Identity()
        )

        self._reset_parameters()

    @staticmethod
    def _build_activation_factory(name: str):
        name = str(name).lower()
        if name == "relu":
            return nn.ReLU
        if name == "gelu":
            return nn.GELU
        if name == "tanh":
            return nn.Tanh
        if name == "sigmoid":
            return nn.Sigmoid
        raise ValueError(f"Unsupported activation: {name}")

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.hist_item_emb.weight)
        if self.hist_item_emb.padding_idx is not None:
            with torch.no_grad():
                self.hist_item_emb.weight[self.hist_item_emb.padding_idx].zero_()

        if not self.shared_table:
            nn.init.xavier_uniform_(self.target_item_emb.weight)
            if self.target_item_emb.padding_idx is not None:
                with torch.no_grad():
                    self.target_item_emb.weight[self.target_item_emb.padding_idx].zero_()

        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias.weight)

        for module in self.user_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        if isinstance(self.target_proj, nn.Linear):
            nn.init.xavier_uniform_(self.target_proj.weight)

        if self.global_bias is not None:
            nn.init.zeros_(self.global_bias)

    def _rating_weights(self, ratings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = ratings

        if self.rating_mode == "none":
            w = torch.ones_like(x)
        elif self.rating_mode == "raw":
            w = x
        elif self.rating_mode == "centered":
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_u = (x * valid_mask).sum(dim=1, keepdim=True) / denom
            w = x - mean_u
        elif self.rating_mode == "normalized":
            scale = max(self.rating_max - self.rating_min, self.eps)
            w = 2.0 * (x - self.rating_min) / scale - 1.0
        elif self.rating_mode == "binary":
            midpoint = 0.5 * (self.rating_min + self.rating_max)
            w = torch.where(x > midpoint, torch.ones_like(x), -torch.ones_like(x))
        else:
            raise ValueError(f"Unsupported rating_mode: {self.rating_mode}")

        return w * valid_mask

    def pool_history(
        self,
        history_items: torch.Tensor,
        history_ratings: torch.Tensor,
    ) -> torch.Tensor:
        if history_items.ndim != 2 or history_ratings.ndim != 2:
            raise ValueError("history_items and history_ratings must be rank-2 tensors.")
        if history_items.shape != history_ratings.shape:
            raise ValueError("history_items and history_ratings must have the same shape.")

        valid_mask = history_items.ne(self.pad_idx).float()                  # (B, L)
        hist_emb = self.hist_item_emb(history_items)                         # (B, L, D)
        weights = self._rating_weights(history_ratings, valid_mask)          # (B, L)
        weighted_hist = hist_emb * weights.unsqueeze(-1)                     # (B, L, D)

        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)           # (B, 1)
        pooled = weighted_hist.sum(dim=1) / denom                            # (B, D)
        return pooled

    def encode_from_pooled(self, pooled_history: torch.Tensor) -> torch.Tensor:
        if pooled_history.ndim != 2:
            raise ValueError("pooled_history must be rank-2.")
        return self.user_encoder(pooled_history)

    def encode_user(
        self,
        history_items: torch.Tensor,
        history_ratings: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool_history(history_items, history_ratings)
        return self.encode_from_pooled(pooled)

    def score_items(
        self,
        user_vec: torch.Tensor,
        target_items: torch.Tensor,
    ) -> torch.Tensor:
        if target_items.ndim != 1:
            raise ValueError("target_items must be rank-1.")
        if user_vec.shape[0] != target_items.shape[0]:
            raise ValueError("Batch size mismatch between user_vec and target_items.")

        target_vec = self.target_item_emb(target_items)                      # (B, emb_dim)
        target_vec = self.target_proj(target_vec)                            # (B, user_dim)

        scores = (user_vec * target_vec).sum(dim=1)                          # (B,)

        if self.item_bias is not None:
            scores = scores + self.item_bias(target_items).squeeze(-1)

        if self.global_bias is not None:
            scores = scores + self.global_bias

        return scores

    def forward(
        self,
        history_items: torch.Tensor,
        history_ratings: torch.Tensor,
        target_items: torch.Tensor,
    ) -> torch.Tensor:
        user_vec = self.encode_user(history_items, history_ratings)
        return self.score_items(user_vec, target_items)

    @torch.no_grad()
    def score_all_items(
        self,
        history_items: torch.Tensor,
        history_ratings: torch.Tensor,
    ) -> torch.Tensor:
        user_vec = self.encode_user(history_items, history_ratings)          # (B, user_dim)

        all_item_ids = torch.arange(self.num_items, device=history_items.device, dtype=torch.long)
        all_item_vecs = self.target_item_emb(all_item_ids)                   # (I, emb_dim)
        all_item_vecs = self.target_proj(all_item_vecs)                      # (I, user_dim)

        scores = user_vec @ all_item_vecs.T                                  # (B, I)

        if self.item_bias is not None:
            scores = scores + self.item_bias(all_item_ids).squeeze(-1).unsqueeze(0)

        if self.global_bias is not None:
            scores = scores + self.global_bias

        return scores


# =========================================================
# Experiment
# =========================================================

YOUTUBE_REQUIRED_HPS = [
    "emb_dim",
    "hidden_dims",
    "lr",
    "weight_decay",
    "lr_incremental",
    "weight_decay_incremental",
    "batch_size",
    "incremental_batch_size",
    "epochs_offline",
    "epochs_incremental",
    "incremental_patience",
    "max_history",
]


class YoutubeExperiment(BaseExperiment):
    """
    GPU-oriented explicit-feedback history-encoder recommender aligned to the
    new BaseExperiment contract.

    Supports:
      - mse
      - wmse
      - residual_mse

    Important:
      - _predict_df(...) returns RAW, UNCLIPPED predictions
      - BaseExperiment handles RMSE clipping and ranking metrics
      - In residual_mse:
          target = rating - user_mean
          raw_prediction = model_prediction + user_mean
    """

    # ----------------------------
    # init
    # ----------------------------

    def _init_models(self) -> None:
        hps = self.params.model_hps or {}
        meta = self.ds["meta"]
        n_items = int(meta["n_items"])

        missing = [hp for hp in YOUTUBE_REQUIRED_HPS if hps.get(hp) is None]
        if missing:
            raise ValueError(f"Missing Youtube hyperparams in model_hps: {missing}")

        self.loss_type = str((self.params.model_init or {}).get("loss_type", "mse")).lower()
        if self.loss_type not in {"mse", "wmse", "residual_mse"}:
            raise ValueError(
                f"Unsupported loss_type='{self.loss_type}'. "
                "Expected one of: mse, wmse, residual_mse"
            )
        self.residual_target = self.loss_type == "residual_mse"

        model_init = self.params.model_init or {}
        if model_init.get("patience") is None:
            raise ValueError("Missing offline patience in model_init['patience'].")

        use_global_bias = bool(hps.get("use_global_bias", not self.residual_target))

        self.model = HistoryEncoderRec(
            num_items=n_items,
            emb_dim=int(hps["emb_dim"]),
            hidden_dims=tuple(hps["hidden_dims"]),
            dropout=float(hps.get("dropout", 0.1)),
            activation=str(hps.get("activation", "relu")),
            shared_table=bool(hps.get("shared_table", True)),
            use_item_bias=bool(hps.get("use_item_bias", True)),
            use_global_bias=use_global_bias,
            rating_mode=str(hps.get("rating_mode", "centered")),
            rating_min=float(hps.get("rating_min", RATING_BOUNDS[0])),
            rating_max=float(hps.get("rating_max", RATING_BOUNDS[1])),
        ).to(self.device)

        self.opt = optim.Adam(
            self.model.parameters(),
            lr=float(hps["lr"]),
            weight_decay=float(hps["weight_decay"]),
        )

        self.n_off = int(meta["n_offline_users"])
        self.n_inc = int(meta["n_incremental_users"])
        self.n_on = int(meta["n_online_users"])
        self.n_items = n_items

        self.max_history = int(hps["max_history"])
        if self.max_history <= 0:
            raise ValueError("model_hps.max_history must be > 0.")

        self.history_sampling_mode = str(hps.get("history_sampling_mode", "random")).lower()
        if self.history_sampling_mode not in {"random", "first"}:
            raise ValueError("history_sampling_mode must be one of {'random', 'first'}.")

        self.history_sampling_seed = int(hps.get("history_sampling_seed", self.params.seed))
        self.online_use_full_history = bool(hps.get("online_use_full_history", True))

        self.use_amp = (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16

        self._offline_history: Optional[UserHistoryStore] = None
        self._incremental_history: Optional[UserHistoryStore] = None

        self._loss_rating_values: Optional[torch.Tensor] = None
        self._loss_rating_weights: Optional[torch.Tensor] = None

    # ----------------------------
    # packs
    # ----------------------------

    def _df_to_pack(self, df: pd.DataFrame) -> Optional[ExamplePack]:
        if df is None or len(df) == 0:
            return None

        return ExamplePack(
            u=torch.as_tensor(
                df["user_id"].to_numpy(np.int64, copy=True),
                dtype=torch.long,
                device=self.device,
            ),
            i=torch.as_tensor(
                df["item_id"].to_numpy(np.int64, copy=True),
                dtype=torch.long,
                device=self.device,
            ),
            r=torch.as_tensor(
                df["rating"].to_numpy(np.float32, copy=True),
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _iter_example_batches(
        self,
        pack: ExamplePack,
        batch_size: int,
        shuffle: bool,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        n = pack.n
        if n == 0:
            return

        device = pack.device
        idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            yield pack.u[b], pack.i[b], pack.r[b]

    # ----------------------------
    # loss helpers
    # ----------------------------

    def _clear_loss_state(self) -> None:
        self._loss_rating_values = None
        self._loss_rating_weights = None

    def _set_loss_from_train_df(self, df: pd.DataFrame) -> None:
        if self.loss_type != "wmse":
            self._clear_loss_state()
            return

        if df is None or df.empty:
            raise ValueError("Cannot build weighted MSE weights from an empty training dataframe.")

        vc = df["rating"].value_counts().sort_index()
        rating_values = vc.index.to_numpy(dtype=np.float32, copy=True)
        rating_counts = vc.to_numpy(dtype=np.float32, copy=True)

        alpha = float((self.params.model_init or {}).get("wmse_alpha", 0.5))
        weights = 1.0 / np.power(rating_counts, alpha)

        cap = (self.params.model_init or {}).get("wmse_cap", None)
        if cap is not None:
            weights = np.minimum(weights, float(cap))

        weights = weights / weights.mean()

        self._loss_rating_values = torch.as_tensor(
            rating_values,
            dtype=torch.float32,
            device=self.device,
        )
        self._loss_rating_weights = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )

    def _configure_incremental_trainability(self) -> list[torch.nn.Parameter]:
        """
        Freeze/unfreeze parameter groups for incremental training.

        Supported modes via:
            self.params.model_init["incremental_freeze_mode"]

        Modes:
        - "full"
        - "item_bias_only"
        - "last_encoder_layer_plus_bias"
        - "encoder_only_plus_bias"
        - "embeddings_plus_bias"
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        mode = str((self.params.model_init or {}).get(
            "incremental_freeze_mode",
            "item_bias_only",
        )).lower()

        # Freeze everything first
        for p in self.model.parameters():
            p.requires_grad = False

        trainable: list[torch.nn.Parameter] = []
        seen: set[int] = set()

        def add_param(p: Optional[torch.nn.Parameter]) -> None:
            if p is None:
                return
            pid = id(p)
            if pid in seen:
                return
            p.requires_grad = True
            trainable.append(p)
            seen.add(pid)

        def add_module(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            if isinstance(module, nn.Identity):
                return
            for p in module.parameters(recurse=True):
                add_param(p)

        if mode == "full":
            for p in self.model.parameters():
                add_param(p)

        elif mode == "item_bias_only":
            add_module(self.model.item_bias)
            if self.model.global_bias is not None:
                add_param(self.model.global_bias)

        elif mode == "last_encoder_layer_plus_bias":
            linear_layers = [m for m in self.model.user_encoder if isinstance(m, nn.Linear)]
            if not linear_layers:
                raise ValueError("user_encoder has no Linear layers to unfreeze.")
            add_module(linear_layers[-1])
            add_module(self.model.item_bias)
            if isinstance(self.model.target_proj, nn.Linear):
                add_module(self.model.target_proj)
            if self.model.global_bias is not None:
                add_param(self.model.global_bias)

        elif mode == "encoder_only_plus_bias":
            add_module(self.model.user_encoder)
            if isinstance(self.model.target_proj, nn.Linear):
                add_module(self.model.target_proj)
            add_module(self.model.item_bias)
            if self.model.global_bias is not None:
                add_param(self.model.global_bias)

        elif mode == "embeddings_plus_bias":
            add_module(self.model.hist_item_emb)
            if not bool(self.model.shared_table):
                add_module(self.model.target_item_emb)
            add_module(self.model.item_bias)
            if self.model.global_bias is not None:
                add_param(self.model.global_bias)

        else:
            raise ValueError(
                f"Unsupported incremental_freeze_mode='{mode}'. "
                "Expected one of: "
                "{'full', 'item_bias_only', 'last_encoder_layer_plus_bias', "
                "'encoder_only_plus_bias', 'embeddings_plus_bias'}"
            )

        if not trainable:
            raise ValueError(
                f"No trainable parameters selected for incremental_freeze_mode='{mode}'."
            )

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[incremental] freeze_mode={mode} | "
            f"trainable_params={n_trainable:,}/{n_total:,}"
        )

        return trainable

    def _get_sample_weights(self, y_true_raw: torch.Tensor) -> torch.Tensor:
        if self.loss_type != "wmse":
            return torch.ones_like(y_true_raw, dtype=torch.float32, device=y_true_raw.device)

        if self._loss_rating_values is None or self._loss_rating_weights is None:
            raise RuntimeError("Weighted MSE is enabled, but loss weights were not initialized from train_df.")

        y_true_raw = y_true_raw.float()
        weights = torch.ones_like(y_true_raw, dtype=torch.float32, device=y_true_raw.device)

        vals = self._loss_rating_values.to(y_true_raw.device)
        wts = self._loss_rating_weights.to(y_true_raw.device)

        for rv, rw in zip(vals, wts):
            weights = torch.where(y_true_raw == rv, rw, weights)

        return weights

    def _target_from_raw(self, y_raw: torch.Tensor, user_means: torch.Tensor) -> torch.Tensor:
        y_raw = y_raw.float()
        if self.residual_target:
            return y_raw - user_means.float()
        return y_raw

    def _pred_model_to_raw(self, pred_model: torch.Tensor, user_means: torch.Tensor) -> torch.Tensor:
        pred_model = pred_model.float()
        if self.residual_target:
            return pred_model + user_means.float()
        return pred_model

    def _compute_objective(
        self,
        pred_model: torch.Tensor,
        y_raw: torch.Tensor,
        user_means: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_target = self._target_from_raw(y_raw, user_means)

        if self.loss_type == "wmse":
            per_sample_loss = (pred_model.float() - y_target.float()).square()
            weights = self._get_sample_weights(y_raw)
            loss_num = (per_sample_loss * weights).sum()
            loss_den = weights.sum().clamp_min(1e-12)
            loss = loss_num / loss_den
            return loss, loss_num, loss_den

        diff = pred_model.float() - y_target.float()
        loss_num = diff.square().sum()
        loss_den = torch.tensor(float(y_target.numel()), dtype=torch.float32, device=y_target.device)
        loss = loss_num / loss_den
        return loss, loss_num, loss_den

    # ----------------------------
    # history sampling
    # ----------------------------

    def _sample_history_indices(
        self,
        n: int,
        *,
        keep: int,
        user_id: int,
        phase_seed: int,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        if keep <= 0:
            return np.empty((0,), dtype=np.int64)
        if n <= keep:
            return np.arange(n, dtype=np.int64)

        mode = (mode or self.history_sampling_mode).lower()

        if mode == "first":
            return np.arange(keep, dtype=np.int64)

        if mode != "random":
            raise ValueError(f"Unsupported history sampling mode: {mode}")

        user_seed = int(phase_seed) + int(user_id) * 1_000_003
        rng = np.random.default_rng(user_seed)
        idx = rng.choice(n, size=keep, replace=False)
        idx.sort()
        return idx.astype(np.int64, copy=False)

    def _sample_user_history_arrays(
        self,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        *,
        user_id: int,
        phase_seed: int,
        max_history: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        item_ids = np.asarray(item_ids, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float32)

        if item_ids.ndim != 1 or ratings.ndim != 1:
            raise ValueError("item_ids and ratings must be rank-1 arrays.")
        if item_ids.shape[0] != ratings.shape[0]:
            raise ValueError("item_ids and ratings must have the same length.")

        keep = int(self.max_history if max_history is None else max_history)
        idx = self._sample_history_indices(
            n=int(item_ids.size),
            keep=keep,
            user_id=int(user_id),
            phase_seed=int(phase_seed),
            mode=mode,
        )
        return item_ids[idx], ratings[idx]

    def _build_single_user_history_tensors(
        self,
        *,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        user_id: int,
        phase_seed: int,
        use_full_history: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        item_ids = np.asarray(item_ids, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float32)

        if not use_full_history and item_ids.size > self.max_history:
            item_ids, ratings = self._sample_user_history_arrays(
                item_ids=item_ids,
                ratings=ratings,
                user_id=user_id,
                phase_seed=phase_seed,
                max_history=self.max_history,
            )

        if item_ids.size == 0:
            hist_items = torch.full(
                (1, 1),
                fill_value=int(self.model.pad_idx),
                dtype=torch.long,
                device=self.device,
            )
            hist_ratings = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            return hist_items, hist_ratings

        hist_items = torch.as_tensor(item_ids.reshape(1, -1), dtype=torch.long, device=self.device)
        hist_ratings = torch.as_tensor(ratings.reshape(1, -1), dtype=torch.float32, device=self.device)
        return hist_items, hist_ratings

    # ----------------------------
    # history store
    # ----------------------------

    def _build_history_store(
        self,
        evidence_df: pd.DataFrame,
        n_users: int,
        *,
        phase_seed: Optional[int] = None,
    ) -> UserHistoryStore:
        """
        Build a device-side fixed-width history bank.

        Users with > max_history interactions are reduced according to
        history_sampling_mode.

        In addition to the sampled history tensors, this stores full-evidence
        per-user sum/count/mean statistics. Those are used to compute an exact
        leave-one-out mean during training when the target example itself came
        from the same evidence_df.
        """
        pad_idx = int(self.model.pad_idx)
        phase_seed = int(self.history_sampling_seed if phase_seed is None else phase_seed)

        items = np.full((n_users, self.max_history), pad_idx, dtype=np.int64)
        ratings = np.zeros((n_users, self.max_history), dtype=np.float32)

        means = np.zeros((n_users,), dtype=np.float32)
        sums = np.zeros((n_users,), dtype=np.float32)
        counts = np.zeros((n_users,), dtype=np.int64)
        global_mean = 0.0

        if evidence_df is not None and len(evidence_df) > 0:
            global_mean = float(evidence_df["rating"].mean())
            means.fill(global_mean)

            stats = evidence_df.groupby("user_id", sort=False)["rating"].agg(["sum", "count", "mean"])
            stat_uids = stats.index.to_numpy(dtype=np.int64, copy=True)

            if stat_uids.size > 0:
                if stat_uids.min() < 0 or stat_uids.max() >= n_users:
                    raise ValueError(
                        f"user_id out of range in evidence_df stats: "
                        f"min={stat_uids.min()} max={stat_uids.max()} n_users={n_users}"
                    )

                sums[stat_uids] = stats["sum"].to_numpy(dtype=np.float32, copy=True)
                counts[stat_uids] = stats["count"].to_numpy(dtype=np.int64, copy=True)
                means[stat_uids] = stats["mean"].to_numpy(dtype=np.float32, copy=True)

            for u, g in evidence_df.groupby("user_id", sort=False):
                uu = int(u)
                if uu < 0 or uu >= n_users:
                    raise ValueError(f"user_id {uu} out of range [0, {n_users}).")

                item_ids = g["item_id"].to_numpy(np.int64, copy=True)
                vals = g["rating"].to_numpy(np.float32, copy=True)

                item_ids, vals = self._sample_user_history_arrays(
                    item_ids=item_ids,
                    ratings=vals,
                    user_id=uu,
                    phase_seed=phase_seed,
                    max_history=self.max_history,
                )

                L = int(item_ids.size)
                if L == 0:
                    continue

                items[uu, :L] = item_ids
                ratings[uu, :L] = vals

        return UserHistoryStore(
            items=torch.as_tensor(items, dtype=torch.long, device=self.device),
            ratings=torch.as_tensor(ratings, dtype=torch.float32, device=self.device),
            user_means=torch.as_tensor(means, dtype=torch.float32, device=self.device),
            user_sums=torch.as_tensor(sums, dtype=torch.float32, device=self.device),
            user_counts=torch.as_tensor(counts, dtype=torch.long, device=self.device),
            global_mean=torch.tensor(global_mean, dtype=torch.float32, device=self.device),
            pad_idx=pad_idx,
        )

    # ----------------------------
    # eval helpers
    # ----------------------------

    @torch.no_grad()
    def _eval_mse_examples(
        self,
        *,
        pack: Optional[ExamplePack],
        history: UserHistoryStore,
        batch_size: int,
    ) -> float:
        if pack is None:
            return float("nan")

        self.model.eval()
        sse = 0.0
        cnt = 0
        device_type = self.device.type

        for users, items, ratings in self._iter_example_batches(
            pack,
            batch_size=batch_size,
            shuffle=False,
        ):
            batch = history.gather(
                users=users,
                target_items=items,
                exclude_target=True,
            )

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_model = self.model(
                    history_items=batch.history_items,
                    history_ratings=batch.history_ratings,
                    target_items=batch.target_items,
                )

            pred_raw = self._pred_model_to_raw(pred_model, batch.user_means)
            diff2 = (pred_raw - ratings).pow(2)
            sse += float(diff2.sum().item())
            cnt += int(ratings.numel())

        return float("nan") if cnt == 0 else float(sse / cnt)

    # ----------------------------
    # training
    # ----------------------------
    def _fit_phase(
        self,
        *,
        phase: str,
        history: UserHistoryStore,
        train_pack: Optional[ExamplePack],
        val_pack: Optional[ExamplePack],
        test_pack: Optional[ExamplePack],
        epochs: int,
        patience: int,
        batch_size: int,
    ) -> None:
        device_type = self.device.type
        hist = []
        use_es = (val_pack is not None)
        best_val = math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        min_delta = 1e-6
        waited = 0

        def _sync() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        if train_pack is None:
            final_test_mse = self._eval_mse_examples(
                pack=test_pack,
                history=history,
                batch_size=batch_size,
            )
            self.artifacts.logs[f"{phase}.history"] = hist
            self._log_metrics(
                phase,
                final_val_mse=float("nan"),
                final_test_mse=final_test_mse,
            )
            return

        for ep in range(1, epochs + 1):
            self.model.train()

            sse_raw = 0.0
            n_obs = 0
            loss_num_sum = 0.0
            loss_den_sum = 0.0

            t_epoch0 = time.perf_counter()
            t_batch_total = 0.0
            t_gather = 0.0
            t_forward = 0.0
            t_backward = 0.0
            n_batches = 0

            for users, items, ratings in self._iter_example_batches(
                train_pack,
                batch_size=batch_size,
                shuffle=True,
            ):
                n_batches += 1
                t_batch0 = time.perf_counter()

                _sync()
                t0 = time.perf_counter()

                batch = history.gather(
                    users=users,
                    target_items=items,
                    exclude_target=True,
                    target_ratings=ratings,
                )

                _sync()
                t1 = time.perf_counter()
                t_gather += (t1 - t0)

                _sync()
                t0 = time.perf_counter()

                with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                    pred_model = self.model(
                        history_items=batch.history_items,
                        history_ratings=batch.history_ratings,
                        target_items=batch.target_items,
                    )
                    loss, loss_num, loss_den = self._compute_objective(
                        pred_model=pred_model,
                        y_raw=ratings,
                        user_means=batch.user_means,
                    )

                _sync()
                t1 = time.perf_counter()
                t_forward += (t1 - t0)

                _sync()
                t0 = time.perf_counter()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                _sync()
                t1 = time.perf_counter()
                t_backward += (t1 - t0)

                pred_raw = self._pred_model_to_raw(pred_model.detach(), batch.user_means)
                diff_raw = pred_raw.float() - ratings.float()
                sse_raw += float(diff_raw.square().sum().item())
                n_obs += int(ratings.numel())

                loss_num_sum += float(loss_num.detach().item())
                loss_den_sum += float(loss_den.detach().item())

                t_batch_total += (time.perf_counter() - t_batch0)

            train_mse = float("nan") if n_obs == 0 else float(sse_raw / n_obs)
            train_step_loss = float("nan") if loss_den_sum <= 0 else float(loss_num_sum / loss_den_sum)

            _sync()
            t_val0 = time.perf_counter()
            val_mse = self._eval_mse_examples(
                pack=val_pack,
                history=history,
                batch_size=batch_size,
            )
            _sync()
            t_val = time.perf_counter() - t_val0

            _sync()
            t_epoch = time.perf_counter() - t_epoch0

            hist_row = {
                "epoch": ep,
                "train_mse": train_mse,
                "train_step_loss": train_step_loss,
                "val_mse": val_mse,
                "epoch_sec": float(t_epoch),
                "train_loop_sec": float(t_batch_total),
                "gather_sec": float(t_gather),
                "forward_sec": float(t_forward),
                "backward_sec": float(t_backward),
                "val_sec": float(t_val),
                "n_batches": int(n_batches),
            }
            hist.append(hist_row)

            print(
                f"[{phase}] epoch {ep}/{epochs} | "
                f"train_mse={train_mse:.6f} | "
                f"val_mse={val_mse:.6f} | "
                f"step_loss={train_step_loss:.6f}"
            )

            try:
                cur_lr = float(self.opt.param_groups[0]["lr"])
            except Exception:
                cur_lr = None

            log_kwargs = {
                "train_mse": train_mse,
                "train_step_loss": train_step_loss,
                "val_mse": val_mse,
                "epoch_sec": float(t_epoch),
                "train_loop_sec": float(t_batch_total),
                "gather_sec": float(t_gather),
                "forward_sec": float(t_forward),
                "backward_sec": float(t_backward),
                "val_sec": float(t_val),
                "n_batches": int(n_batches),
            }
            if cur_lr is not None:
                log_kwargs["lr"] = cur_lr
            self._log_epoch(phase, ep, **log_kwargs)

            if use_es and math.isfinite(val_mse) and (val_mse < best_val - min_delta):
                best_val = val_mse
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                waited = 0
            elif use_es:
                waited += 1
                hist[-1]["_wait"] = waited
                if waited >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        final_test_mse = self._eval_mse_examples(
            pack=test_pack,
            history=history,
            batch_size=batch_size,
        )

        final_val_mse = (
            float(best_val)
            if use_es and math.isfinite(best_val)
            else (hist[-1]["val_mse"] if hist else float("nan"))
        )

        self.artifacts.logs[f"{phase}.history"] = hist
        if hist:
            self._log_metrics(
                phase,
                final_val_mse=final_val_mse,
                final_test_mse=final_test_mse,
            )

    # ----------------------------
    # offline
    # ----------------------------

    def _fit_offline(self) -> None:
        off = self.ds["offline"]

        self._offline_history = self._build_history_store(
            evidence_df=off["train"],
            n_users=self.n_off,
            phase_seed=self.history_sampling_seed + 11,
        )

        if off["train"] is not None and not off["train"].empty:
            self._set_loss_from_train_df(off["train"])
        else:
            self._clear_loss_state()

        hps = self.params.model_hps or {}
        self.opt = optim.Adam(
            self.model.parameters(),
            lr=float(hps["lr"]),
            weight_decay=float(hps["weight_decay"]),
        )

        train_pack = self._df_to_pack(off["train"])
        val_pack = self._df_to_pack(off["val"])
        test_pack = self._df_to_pack(off["test"])

        model_init = self.params.model_init or {}

        self._fit_phase(
            phase="offline",
            history=self._offline_history,
            train_pack=train_pack,
            val_pack=val_pack,
            test_pack=test_pack,
            epochs=int(hps["epochs_offline"]),
            patience=int(model_init.get("patience", 7)),
            batch_size=int(hps["batch_size"]),
        )

    # ----------------------------
    # incremental
    # ----------------------------

    def _fit_incremental(self) -> None:
        hps = self.params.model_hps or {}
        inc = self.ds["incremental"]

        inc_train = inc["train"].copy()
        inc_val = inc["val"].copy()
        inc_test = inc["test"].copy()

        inc_train["user_id"] = inc_train["user_id"] - self.n_off
        inc_val["user_id"] = inc_val["user_id"] - self.n_off
        inc_test["user_id"] = inc_test["user_id"] - self.n_off

        self._incremental_history = self._build_history_store(
            evidence_df=inc_train,
            n_users=self.n_inc,
            phase_seed=self.history_sampling_seed + 29,
        )

        if inc_train is not None and not inc_train.empty:
            self._set_loss_from_train_df(inc_train)
        else:
            self._clear_loss_state()

        trainable_params = self._configure_incremental_trainability()

        self.opt = optim.Adam(
            trainable_params,
            lr=float(hps["lr_incremental"]),
            weight_decay=float(hps["weight_decay_incremental"]),
        )

        train_pack = self._df_to_pack(inc_train)
        val_pack = self._df_to_pack(inc_val)
        test_pack = self._df_to_pack(inc_test)

        self._fit_phase(
            phase="incremental",
            history=self._incremental_history,
            train_pack=train_pack,
            val_pack=val_pack,
            test_pack=test_pack,
            epochs=int(hps["epochs_incremental"]),
            patience=int(hps.get("incremental_patience", 7)),
            batch_size=int(hps.get("incremental_batch_size", hps["batch_size"])),
        )

        # Restore all params to trainable state for future reuse/debugging convenience
        for p in self.model.parameters():
            p.requires_grad = True

    # ----------------------------
    # prediction helpers
    # ----------------------------

    @torch.no_grad()
    def _predict_from_history_df(
        self,
        *,
        df: pd.DataFrame,
        history: UserHistoryStore,
        batch_size: int,
        user_offset: int = 0,
    ) -> np.ndarray:
        """
        Return RAW, UNCLIPPED predictions.
        """
        if df is None or len(df) == 0:
            return np.empty((0,), dtype=np.float32)

        local_users = df["user_id"].to_numpy(np.int64, copy=True) - int(user_offset)
        item_ids = df["item_id"].to_numpy(np.int64, copy=True)

        pack = ExamplePack(
            u=torch.as_tensor(local_users, dtype=torch.long, device=self.device),
            i=torch.as_tensor(item_ids, dtype=torch.long, device=self.device),
            r=torch.zeros(len(df), dtype=torch.float32, device=self.device),
        )

        self.model.eval()
        device_type = self.device.type
        preds = []

        for users, items, _ in self._iter_example_batches(
            pack,
            batch_size=batch_size,
            shuffle=False,
        ):
            batch = history.gather(
                users=users,
                target_items=items,
                exclude_target=True,
            )

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_model = self.model(
                    history_items=batch.history_items,
                    history_ratings=batch.history_ratings,
                    target_items=batch.target_items,
                )

            pred_raw = self._pred_model_to_raw(pred_model, batch.user_means)
            preds.append(pred_raw.detach().float().cpu())

        if not preds:
            return np.empty((0,), dtype=np.float32)

        return torch.cat(preds, dim=0).numpy()

    # ----------------------------
    # online
    # ----------------------------

    @torch.no_grad()
    def _predict_online_df_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch online scoring using a temporary history bank built from online/train.
        Returns RAW, UNCLIPPED predictions.
        """
        if df is None or len(df) == 0:
            return np.empty((0,), dtype=np.float32)

        on_train = self.ds["online"]["train"]

        users = np.unique(
            pd.concat([on_train["user_id"], df["user_id"]], ignore_index=True).to_numpy(np.int64, copy=True)
        )
        users.sort()
        u_to_row = {int(u): i for i, u in enumerate(users)}

        on_train_local = on_train.copy()
        on_train_local["user_id"] = on_train_local["user_id"].map(u_to_row)

        df_local = df[["user_id", "item_id"]].copy()
        df_local["user_id"] = df_local["user_id"].map(u_to_row)

        history = self._build_history_store(
            evidence_df=on_train_local,
            n_users=int(users.size),
            phase_seed=self.history_sampling_seed + 53,
        )

        hps = self.params.model_hps or {}
        bs = int(
            hps.get(
                "eval_batch_size",
                hps.get("incremental_batch_size", hps.get("batch_size", 1024)),
            )
        )

        return self._predict_from_history_df(
            df=df_local,
            history=history,
            batch_size=bs,
            user_offset=0,
        )

    @torch.no_grad()
    def _predict_online_df_per_user(self, df: pd.DataFrame) -> np.ndarray:
        """
        Per-user online scoring.
        Returns RAW, UNCLIPPED predictions.

        This version mirrors the batch path:
        - target item is excluded from the effective history for that row
        - user_mean is recomputed from the post-exclusion history when possible
        - if no history remains, it falls back to the user's train mean (or the
            online global mean when the user has no train history at all)
        """
        if df is None or len(df) == 0:
            return np.empty((0,), dtype=np.float32)

        on_train = self.ds["online"]["train"]
        tr_groups = {u: g for u, g in on_train.groupby("user_id", sort=False)}

        global_on_mean = float(on_train["rating"].mean()) if on_train is not None and not on_train.empty else 0.0

        out = np.empty((len(df),), dtype=np.float32)
        work = df[["user_id", "item_id"]].copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)

        device_type = self.device.type
        self.model.eval()

        for u, g_test in work.groupby("user_id", sort=False):
            g_tr = tr_groups.get(u)

            if g_tr is None or g_tr.empty:
                item_ids = np.empty((0,), dtype=np.int64)
                rating_vals = np.empty((0,), dtype=np.float32)
                fallback_user_mean = float(global_on_mean)
            else:
                item_ids = g_tr["item_id"].to_numpy(np.int64, copy=True)
                rating_vals = g_tr["rating"].to_numpy(np.float32, copy=True)

                if not self.online_use_full_history and item_ids.size > self.max_history:
                    item_ids, rating_vals = self._sample_user_history_arrays(
                        item_ids=item_ids,
                        ratings=rating_vals,
                        user_id=int(u),
                        phase_seed=self.history_sampling_seed + 71,
                        max_history=self.max_history,
                    )

                fallback_user_mean = float(g_tr["rating"].mean())

            if item_ids.size == 0:
                hist_items = torch.full(
                    (1, 1),
                    fill_value=int(self.model.pad_idx),
                    dtype=torch.long,
                    device=self.device,
                )
                hist_ratings = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            else:
                hist_items = torch.as_tensor(item_ids.reshape(1, -1), dtype=torch.long, device=self.device)
                hist_ratings = torch.as_tensor(rating_vals.reshape(1, -1), dtype=torch.float32, device=self.device)

            target_items = torch.as_tensor(
                g_test["item_id"].to_numpy(np.int64, copy=True),
                dtype=torch.long,
                device=self.device,
            )

            hist_items = hist_items.expand(target_items.shape[0], -1).clone()
            hist_ratings = hist_ratings.expand(target_items.shape[0], -1).clone()

            hit = hist_items.eq(target_items.unsqueeze(1))
            if hit.any():
                first_hit = hit & hit.int().cumsum(dim=1).eq(1)
                if first_hit.any():
                    hist_items[first_hit] = int(self.model.pad_idx)
                    hist_ratings[first_hit] = 0.0

            valid_mask = hist_items.ne(self.model.pad_idx)
            valid_counts = valid_mask.sum(dim=1)
            valid_mask_f = valid_mask.float()

            hist_sums = (hist_ratings.float() * valid_mask_f).sum(dim=1)
            hist_means = hist_sums / valid_counts.float().clamp_min(1.0)

            fallback_means = torch.full(
                (target_items.shape[0],),
                fill_value=fallback_user_mean,
                dtype=torch.float32,
                device=self.device,
            )
            user_means = torch.where(valid_counts > 0, hist_means, fallback_means)

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_model = self.model(
                    history_items=hist_items,
                    history_ratings=hist_ratings,
                    target_items=target_items,
                )

            pred_raw = self._pred_model_to_raw(pred_model, user_means)
            out[g_test["_row_pos"].to_numpy(np.int64, copy=False)] = (
                pred_raw.detach().float().cpu().numpy()
            )

        return out

    # ----------------------------
    # BaseExperiment contract
    # ----------------------------

    def _predict_df(
        self,
        df: pd.DataFrame,
        *,
        phase: str,
    ) -> np.ndarray:
        """
        Return RAW, UNCLIPPED predictions in the same order as df.
        """
        hps = self.params.model_hps or {}

        off_bs = int(hps.get("eval_batch_size", hps.get("batch_size", 1024)))
        inc_bs = int(
            hps.get(
                "eval_batch_size",
                hps.get("incremental_batch_size", hps.get("batch_size", 1024)),
            )
        )

        if phase == "offline":
            if self._offline_history is None:
                raise ValueError("Offline history is missing.")
            return self._predict_from_history_df(
                df=df,
                history=self._offline_history,
                batch_size=off_bs,
                user_offset=0,
            )

        if phase == "incremental":
            if self._incremental_history is None:
                raise ValueError("Incremental history is missing.")
            return self._predict_from_history_df(
                df=df,
                history=self._incremental_history,
                batch_size=inc_bs,
                user_offset=self.n_off,
            )

        if phase == "offline_post_incremental":
            if self._offline_history is None:
                raise ValueError("Offline history is missing for offline_post_incremental.")
            return self._predict_from_history_df(
                df=df,
                history=self._offline_history,
                batch_size=off_bs,
                user_offset=0,
            )

        if phase == "online":
            mode = str(getattr(self.params, "online_inference_pred_type", "batch")).lower()
            if mode in {"per_user", "per-user"}:
                return self._predict_online_df_per_user(df)
            return self._predict_online_df_batch(df)

        raise ValueError(f"Unsupported phase for _predict_df: {phase}")

class SpaceTrackedYoutubeExperiment(SpaceTrackedExperimentBase, YoutubeExperiment):
    """
    YouTube-style history-encoder persisted state.

    Base persisted objects that count:
      - history item embedding table
      - target item embedding table (ONLY if shared_table=False)
      - item bias embedding (if enabled)
      - global bias scalar (if enabled)
      - user encoder MLP parameters
      - target projection (ONLY if emb_dim != user_dim)

    Additional persisted objects for residual_mse only:
      - offline user mean vector
      - incremental user mean vector

    Not counted:
      - raw ratings / original matrices
      - padded history tensors
      - optimizer state
      - temporary online history bank
      - temporary evaluation artifacts

    Delta behavior:
      mse / wmse:
        - OFFLINE:     + all model parameter bytes
        - INCREMENTAL: 0
        - ONLINE:      0

      residual_mse:
        - OFFLINE:     + all model parameter bytes + offline user mean vector
        - INCREMENTAL: + incremental user mean vector
                        (offline user mean vector is still kept)
        - ONLINE:      0
    """

    @staticmethod
    def _param_group_stats(params) -> tuple[int, int, str, Optional[str]]:
        """
        Returns:
          total_bytes, total_numel, dtype_key, dtype_note
        """
        params = list(params)
        if not params:
            return 0, 0, "float32", None

        total_bytes = 0
        total_numel = 0
        dtypes = set()

        for p in params:
            total_numel += int(p.numel())
            total_bytes += int(p.numel()) * int(p.element_size())
            dtypes.add(p.dtype)

        dtype_note = None
        dtype_key = _torch_dtype_to_key(next(iter(dtypes)))
        if len(dtypes) > 1:
            dtype_key = "mixed"
            dtype_note = f"Mixed dtypes: {[str(dt) for dt in sorted(dtypes, key=lambda x: str(x))]}"

        return total_bytes, total_numel, dtype_key, dtype_note

    def _append_model_components(self, snap: SpaceSnapshot, phase: Phase) -> None:
        if self.model is None:
            raise ValueError("Youtube model is not initialized.")

        m = self.model
        n_embeddings = int(m.hist_item_emb.num_embeddings)
        emb_dim = int(m.hist_item_emb.embedding_dim)

        # 1) History/input item embedding table
        hist_bytes, hist_numel, hist_dtype, hist_note = self._param_group_stats(
            [m.hist_item_emb.weight]
        )
        snap.components.append(
            SpaceComponent(
                key="youtube.hist_item_emb",
                name="YouTube history item embedding table",
                bytes=hist_bytes,
                phase=phase,
                shape=tuple(m.hist_item_emb.weight.shape),
                dtype=hist_dtype,
                formula="I_emb * d",
                note=(
                    f"Input/history embedding table with I_emb={n_embeddings}, d={emb_dim}. "
                    f"Used for weighted pooling over user history."
                    + (f" {hist_note}" if hist_note else "")
                ),
            )
        )

        # 2) Target/output item embedding table (only if not shared)
        if not bool(m.shared_table):
            tgt_bytes, tgt_numel, tgt_dtype, tgt_note = self._param_group_stats(
                [m.target_item_emb.weight]
            )
            snap.components.append(
                SpaceComponent(
                    key="youtube.target_item_emb",
                    name="YouTube target item embedding table",
                    bytes=tgt_bytes,
                    phase=phase,
                    shape=tuple(m.target_item_emb.weight.shape),
                    dtype=tgt_dtype,
                    formula="I_emb * d",
                    note=(
                        f"Separate target/output embedding table with I_emb={n_embeddings}, d={emb_dim}. "
                        "Counted because shared_table=False."
                        + (f" {tgt_note}" if tgt_note else "")
                    ),
                )
            )

        # 3) Item bias embedding
        if m.item_bias is not None:
            ib_bytes, ib_numel, ib_dtype, ib_note = self._param_group_stats(
                [m.item_bias.weight]
            )
            snap.components.append(
                SpaceComponent(
                    key="youtube.item_bias",
                    name="YouTube item bias table",
                    bytes=ib_bytes,
                    phase=phase,
                    shape=tuple(m.item_bias.weight.shape),
                    dtype=ib_dtype,
                    formula="I_emb * 1",
                    note=(
                        f"Per-item bias table with I_emb={n_embeddings}. "
                        "Counted because use_item_bias=True."
                        + (f" {ib_note}" if ib_note else "")
                    ),
                )
            )

        # 4) Global bias scalar
        if m.global_bias is not None:
            gb_bytes, gb_numel, gb_dtype, gb_note = self._param_group_stats([m.global_bias])
            snap.components.append(
                SpaceComponent(
                    key="youtube.global_bias",
                    name="YouTube global bias",
                    bytes=gb_bytes,
                    phase=phase,
                    shape=tuple(m.global_bias.shape),
                    dtype=gb_dtype,
                    formula="1",
                    note=(
                        "Single global scalar bias. "
                        "Counted because use_global_bias=True."
                        + (f" {gb_note}" if gb_note else "")
                    ),
                )
            )

        # 5) User encoder MLP
        enc_params = list(m.user_encoder.parameters())
        enc_bytes, enc_numel, enc_dtype, enc_note = self._param_group_stats(enc_params)

        linear_layers = [mod for mod in m.user_encoder if isinstance(mod, nn.Linear)]
        if linear_layers:
            enc_formula = " + ".join(
                [f"({lin.in_features}*{lin.out_features} + {lin.out_features})" for lin in linear_layers]
            )
        else:
            enc_formula = "0"

        snap.components.append(
            SpaceComponent(
                key="youtube.user_encoder",
                name="YouTube user encoder MLP",
                bytes=enc_bytes,
                phase=phase,
                shape=(enc_numel,),
                dtype=enc_dtype,
                formula=enc_formula,
                note=(
                    "MLP mapping pooled history vector to final user embedding."
                    + (f" {enc_note}" if enc_note else "")
                ),
            )
        )

        # 6) Target projection
        if isinstance(m.target_proj, nn.Linear):
            tp_bytes, tp_numel, tp_dtype, tp_note = self._param_group_stats(
                list(m.target_proj.parameters())
            )
            snap.components.append(
                SpaceComponent(
                    key="youtube.target_proj",
                    name="YouTube target projection",
                    bytes=tp_bytes,
                    phase=phase,
                    shape=(tp_numel,),
                    dtype=tp_dtype,
                    formula=f"{m.target_proj.in_features}*{m.target_proj.out_features}",
                    note=(
                        "Projection from target item embedding space to user embedding space. "
                        "Present because emb_dim != final user dim."
                        + (f" {tp_note}" if tp_note else "")
                    ),
                )
            )

    def _build_user_mean_component(
        self,
        *,
        phase: Phase,
        key: str,
        name: str,
        means_tensor: torch.Tensor,
        note_suffix: str,
    ) -> SpaceComponent:
        if not isinstance(means_tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor.")

        return SpaceComponent(
            key=key,
            name=name,
            bytes=int(means_tensor.numel()) * int(means_tensor.element_size()),
            phase=phase,
            shape=tuple(int(x) for x in means_tensor.shape),
            dtype=_torch_dtype_to_key(means_tensor.dtype),
            formula="U_phase * sizeof(dtype)",
            note=(
                "Persisted dense user-mean vector required only for residual_mse "
                "to reconstruct raw predictions from residual-space outputs."
                f" {note_suffix}"
            ),
        )

    def _append_residual_components(self, snap: SpaceSnapshot, phase: Phase) -> None:
        """
        Under the user's counting convention, we add only the persisted mean vectors
        needed by residual_mse, not the entire history bank.

        Important:
          - OFFLINE snapshot keeps offline means
          - INCREMENTAL snapshot keeps BOTH offline and incremental means,
            because both history stores are still retained by the experiment
          - ONLINE snapshot is the same persisted state as after incremental;
            no temporary online means are counted
        """
        if not getattr(self, "residual_target", False):
            return

        if phase == Phase.OFFLINE:
            if self._offline_history is None:
                raise ValueError("Offline history is missing for residual space accounting.")

            snap.components.append(
                self._build_user_mean_component(
                    phase=phase,
                    key="youtube.offline_user_means",
                    name="YouTube offline user mean vector",
                    means_tensor=self._offline_history.user_means,
                    note_suffix="One mean per offline user.",
                )
            )
            return

        if phase in {Phase.INCREMENTAL, Phase.ONLINE}:
            if self._offline_history is None:
                raise ValueError("Offline history is missing for residual space accounting.")
            if self._incremental_history is None:
                raise ValueError("Incremental history is missing for residual space accounting.")

            snap.components.append(
                self._build_user_mean_component(
                    phase=phase,
                    key="youtube.offline_user_means",
                    name="YouTube offline user mean vector",
                    means_tensor=self._offline_history.user_means,
                    note_suffix="Retained because offline/offline_post_incremental still use the offline history bank.",
                )
            )
            snap.components.append(
                self._build_user_mean_component(
                    phase=phase,
                    key="youtube.incremental_user_means",
                    name="YouTube incremental user mean vector",
                    means_tensor=self._incremental_history.user_means,
                    note_suffix="Retained because incremental predictions use the incremental history bank.",
                )
            )
            return

        raise ValueError(f"Unsupported phase: {phase}")

    def _build_space_snapshot(self, phase: Phase) -> SpaceSnapshot:
        snap = SpaceSnapshot()

        # Base model parameters
        self._append_model_components(snap, phase)

        # Residual-only persisted user mean vectors
        self._append_residual_components(snap, phase)

        return snap
