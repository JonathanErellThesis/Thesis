"""
User AutoRec baseline and lifecycle experiment wrapper.

Generated from the original AutoRec notebook with import paths adjusted for the
anonymous reproduction package. Class/function bodies are kept unchanged as much
as possible so reported runs can be reproduced from the original logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from recsys_edge.core import *  # noqa: F401,F403 - preserves notebook-style globals
from recsys_edge.core import _torch_dtype_to_key



# =========================================================
# Packs
# =========================================================

@dataclass
class UserIndexPack:
    u: torch.Tensor  # int64 on device


# =========================================================
# Model
# =========================================================

class UserAutoRec(nn.Module):
    """
    Multi-layer AutoRec:
      input d=num_items -> hidden_dims -> d
      activation on hidden layers, linear output.

    Regularization is handled via optimizer weight_decay.
    """

    def __init__(
        self,
        num_items: int,
        hidden_dims: list[int],
        activation: str = "sigmoid",
    ):
        super().__init__()
        dims = [num_items] + list(hidden_dims) + [num_items]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)])
        self.act = nn.ReLU() if activation.lower() == "relu" else nn.Sigmoid()

        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


# =========================================================
# Experiment
# =========================================================

AUTOREC_REQUIRED_HPS = [
    "hidden_dims",
    "lr",
    "weight_decay",
    "lr_incremental",
    "weight_decay_incremental",
    "batch_size",
    "epochs_offline",
    "epochs_incremental",
    "patience",
    "incremental_batch_size",
]


class AutoRecExperiment(BaseExperiment):
    """
    User-based AutoRec aligned to the new BaseExperiment.

    Supports:
      - mse
      - residual_mse

    Behavior:
      - _predict_df(...) returns RAW, UNCLIPPED predictions
      - BaseExperiment handles RMSE clipping and ranking metrics
      - online supports batch and true per-user inference

    residual_mse logic:
      - compute one evidence-based mean per user row
      - center only observed entries in the input/target rows
      - model predicts in residual space
      - add user mean back at prediction/eval time
    """

    # ----------------------------
    # init
    # ----------------------------

    def _init_models(self) -> None:
        hps = self.params.model_hps or {}
        meta = self.ds["meta"]
        n_items = int(meta["n_items"])

        missing = [hp for hp in AUTOREC_REQUIRED_HPS if hps.get(hp) is None]
        if missing:
            raise ValueError(f"Missing AutoRec hyperparams in model_hps: {missing}")

        self.loss_type = str((self.params.model_init or {}).get("loss_type", "mse")).lower()
        if self.loss_type not in {"mse", "residual_mse"}:
            raise ValueError(
                f"Unsupported loss_type='{self.loss_type}'. "
                "Expected one of: {'mse', 'residual_mse'}"
            )
        self.residual_target = self.loss_type == "residual_mse"

        self.model = UserAutoRec(
            num_items=n_items,
            hidden_dims=hps["hidden_dims"],
            activation=hps.get("activation", "sigmoid"),
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(hps["lr"]),
            weight_decay=float(hps["weight_decay"]),
        )

        self.n_off = int(meta["n_offline_users"])
        self.n_inc = int(meta["n_incremental_users"])
        self.n_on = int(meta["n_online_users"])
        self.n_items = n_items

        self.use_amp = (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16

        self._best_state: Optional[Dict[str, torch.Tensor]] = None
        self._phase_mats: Dict[str, Dict[str, torch.Tensor]] = {}

    # ----------------------------
    # checkpoint helpers
    # ----------------------------

    def _reset_best(self) -> None:
        self._best_state = None

    def _save_best(self) -> None:
        self._best_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }

    def _load_best(self) -> None:
        if self._best_state is None:
            return
        self.model.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})

    # ----------------------------
    # user pack helpers
    # ----------------------------

    def _np_users_to_pack(self, u: np.ndarray) -> Optional[UserIndexPack]:
        u = np.asarray(u, dtype=np.int64)
        if u.size == 0:
            return None
        t = torch.as_tensor(u, device="cpu", dtype=torch.long).to(self.device, non_blocking=True)
        return UserIndexPack(u=t)

    def _iter_user_batches(
        self,
        pack: UserIndexPack,
        batch_size: int,
        shuffle: bool,
    ) -> Iterator[torch.Tensor]:
        n = int(pack.u.numel())
        if n == 0:
            return
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for s in range(0, n, batch_size):
            yield pack.u[idx[s:s + batch_size]]

    # ----------------------------
    # residual helpers
    # ----------------------------

    def _compute_user_means(self, M: torch.Tensor) -> torch.Tensor:
        """
        M is dense user-item matrix with 0 meaning missing.
        Mean is computed only over observed entries.
        Users with no observations fall back to the global observed mean.
        """
        M = M.float()
        obs = M.ne(0).float()

        counts = obs.sum(dim=1)            # (U,)
        sums = (M * obs).sum(dim=1)        # (U,)

        total_cnt = counts.sum()
        if float(total_cnt.item()) > 0:
            global_mean = sums.sum() / total_cnt.clamp_min(1.0)
        else:
            global_mean = torch.tensor(0.0, dtype=torch.float32, device=M.device)

        means = sums / counts.clamp_min(1.0)
        means = torch.where(
            counts > 0,
            means,
            torch.full_like(means, float(global_mean.item())),
        )
        return means.float()

    def _center_observed_entries(
        self,
        M_raw: torch.Tensor,
        user_means: torch.Tensor,
    ) -> torch.Tensor:
        """
        Center observed entries only; keep missing entries at zero.
        """
        M_raw = M_raw.float()
        obs = M_raw.ne(0).float()
        return (M_raw - user_means.unsqueeze(1).float()) * obs

    def _matrix_to_model_input(
        self,
        M_raw: torch.Tensor,
        user_means: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_target:
            return self._center_observed_entries(M_raw, user_means)
        return M_raw.float()

    def _matrix_to_target(
        self,
        M_raw: torch.Tensor,
        user_means: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_target:
            return self._center_observed_entries(M_raw, user_means)
        return M_raw.float()

    def _pred_model_to_raw_matrix(
        self,
        pred_model: torch.Tensor,
        user_means: torch.Tensor,
    ) -> torch.Tensor:
        pred_model = pred_model.float()
        if self.residual_target:
            return pred_model + user_means.unsqueeze(1).float()
        return pred_model

    # ----------------------------
    # eval helpers
    # ----------------------------

    @torch.no_grad()
    def _eval_mse_users(
        self,
        *,
        M_infer: torch.Tensor,
        infer_user_means: torch.Tensor,
        M_label: torch.Tensor,
        users_pack: Optional[UserIndexPack],
        batch_size: int,
    ) -> float:
        if users_pack is None:
            return float("nan")

        self.model.eval()
        sse = 0.0
        cnt = 0.0
        device_type = self.device.type

        for users in self._iter_user_batches(users_pack, batch_size=batch_size, shuffle=False):
            r_in_raw = M_infer[users]
            r_lab_raw = M_label[users]
            user_means = infer_user_means[users]
            mask = torch.sign(r_lab_raw).abs()

            r_in_model = self._matrix_to_model_input(r_in_raw, user_means)

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                r_hat_model = self.model(r_in_model)

            r_hat_raw = self._pred_model_to_raw_matrix(r_hat_model, user_means)

            diff2 = ((r_hat_raw - r_lab_raw) * mask).pow(2)
            sse += float(diff2.sum().item())
            cnt += float(mask.sum().item())

        return float("nan") if cnt == 0.0 else float(sse / cnt)

    # ----------------------------
    # shared training loop
    # ----------------------------

    def _fit_phase(
        self,
        *,
        u_train: np.ndarray,
        u_val: np.ndarray,
        u_test: np.ndarray,
        epochs: int,
        patience: int,
        phase: str,   # "offline" | "incremental"
    ) -> None:
        mats = self._phase_mats[phase]
        M_infer: torch.Tensor = mats["infer"]
        infer_user_means: torch.Tensor = mats["infer_user_means"]
        M_tr: torch.Tensor = mats["train"]
        M_va: torch.Tensor = mats["val"]
        M_te: torch.Tensor = mats["test"]

        hps = self.params.model_hps or {}
        bs = (
            int(hps.get("incremental_batch_size", hps["batch_size"]))
            if phase == "incremental"
            else int(hps["batch_size"])
        )

        tr_pack = self._np_users_to_pack(u_train)
        va_pack = self._np_users_to_pack(u_val)
        te_pack = self._np_users_to_pack(u_test)

        hist: list[dict] = []
        use_es = (va_pack is not None)
        best_val = math.inf
        min_delta = 1e-6
        waited = 0
        device_type = self.device.type

        if tr_pack is None:
            self.artifacts.logs[f"{phase}.history"] = hist
            return

        self._reset_best()

        for ep in range(1, epochs + 1):
            self.model.train()

            sse_train_raw = 0.0
            cnt_train = 0.0
            loss_num_sum = 0.0
            loss_den_sum = 0.0

            for users in self._iter_user_batches(tr_pack, batch_size=bs, shuffle=True):
                r_in_raw = M_infer[users]
                r_tr_raw = M_tr[users]
                user_means = infer_user_means[users]
                mask = torch.sign(r_tr_raw).abs()

                r_in_model = self._matrix_to_model_input(r_in_raw, user_means)
                r_tr_target = self._matrix_to_target(r_tr_raw, user_means)

                with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                    r_hat_model = self.model(r_in_model)
                    diff_target = (r_hat_model - r_tr_target) * mask
                    loss_num = diff_target.pow(2).sum()
                    loss_den = mask.sum().clamp_min(1.0)
                    loss = loss_num / loss_den

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                r_hat_raw = self._pred_model_to_raw_matrix(r_hat_model.detach(), user_means)
                diff_raw = (r_hat_raw - r_tr_raw) * mask

                sse_train_raw += float(diff_raw.pow(2).sum().item())
                cnt_train += float(mask.sum().item())
                loss_num_sum += float(loss_num.detach().item())
                loss_den_sum += float(loss_den.detach().item())

            train_mse = float("nan") if cnt_train == 0.0 else float(sse_train_raw / cnt_train)
            train_step_loss = float("nan") if loss_den_sum <= 0.0 else float(loss_num_sum / loss_den_sum)

            val_mse = (
                self._eval_mse_users(
                    M_infer=M_infer,
                    infer_user_means=infer_user_means,
                    M_label=M_va,
                    users_pack=va_pack,
                    batch_size=bs,
                )
                if va_pack else float("nan")
            )
            print(
                f"[{phase}] epoch {ep}/{epochs} | "
                f"train_mse={train_mse:.6f} | "
                f"val_mse={val_mse:.6f}"
            )

            test_mse = (
                self._eval_mse_users(
                    M_infer=M_infer,
                    infer_user_means=infer_user_means,
                    M_label=M_te,
                    users_pack=te_pack,
                    batch_size=bs,
                )
                if te_pack else float("nan")
            )

            hist.append(
                {
                    "epoch": ep,
                    "train_mse": train_mse,
                    "train_step_loss": train_step_loss,
                    "val_mse": val_mse,
                    "test_mse": test_mse,
                }
            )

            self._log_epoch(
                phase,
                ep,
                train_mse=float(train_mse),
                train_step_loss=float(train_step_loss),
                val_mse=float(val_mse),
                test_mse=float(test_mse),
                lr=float(self.optimizer.param_groups[0]["lr"]),
            )

            if use_es and math.isfinite(val_mse):
                if val_mse < best_val - min_delta:
                    best_val = val_mse
                    self._save_best()
                    waited = 0
                else:
                    waited += 1
                    hist[-1]["_wait"] = waited
                    if waited >= patience:
                        break

        if self._best_state is not None:
            self._load_best()

        self.artifacts.logs[f"{phase}.history"] = hist

    # ----------------------------
    # offline
    # ----------------------------

    def _fit_offline(self) -> None:
        off = self.ds["offline"]

        M_tr = convert_to_sparse_arr(
            [off["train"]],
            n_users=self.n_off,
            n_items=self.n_items,
            rating_col="rating",
            as_dense=True,
        )[0].to(self.device)

        M_va = convert_to_sparse_arr(
            [off["val"]],
            n_users=self.n_off,
            n_items=self.n_items,
            rating_col="rating",
            as_dense=True,
        )[0].to(self.device)

        M_te = convert_to_sparse_arr(
            [off["test"]],
            n_users=self.n_off,
            n_items=self.n_items,
            rating_col="rating",
            as_dense=True,
        )[0].to(self.device)

        M_infer = M_tr
        infer_user_means = self._compute_user_means(M_infer)

        u_train = np.unique(off["train"]["user_id"].to_numpy()) if len(off["train"]) else np.array([], dtype=np.int64)
        u_val = np.unique(off["val"]["user_id"].to_numpy()) if len(off["val"]) else np.array([], dtype=np.int64)
        u_test = np.unique(off["test"]["user_id"].to_numpy()) if len(off["test"]) else np.array([], dtype=np.int64)

        self._phase_mats["offline"] = {
            "infer": M_infer,
            "infer_user_means": infer_user_means,
            "train": M_tr,
            "val": M_va,
            "test": M_te,
        }

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.params.model_hps["lr"]),
            weight_decay=float(self.params.model_hps["weight_decay"]),
        )

        self._fit_phase(
            u_train=u_train,
            u_val=u_val,
            u_test=u_test,
            epochs=int(self.params.model_hps["epochs_offline"]),
            patience=int(self.params.model_hps.get("patience", 7)),
            phase="offline",
        )

    # ----------------------------
    # incremental
    # ----------------------------

    def _fit_incremental(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.params.model_hps["lr_incremental"]),
            weight_decay=float(self.params.model_hps["weight_decay_incremental"]),
        )

        off = self.ds["offline"]
        inc = self.ds["incremental"]
        U_total = self.n_off + self.n_inc

        M_tr = convert_to_sparse_arr(
            [pd.concat([off["train"], inc["train"]], ignore_index=True)],
            n_users=U_total,
            n_items=self.n_items,
            rating_col="rating",
            as_dense=True,
        )[0].to(self.device)

        M_va = convert_to_sparse_arr(
            [pd.concat([off["val"], inc["val"]], ignore_index=True)],
            n_users=U_total,
            n_items=self.n_items,
            rating_col="rating",
            as_dense=True,
        )[0].to(self.device)

        M_te = convert_to_sparse_arr(
            [pd.concat([off["test"], inc["test"]], ignore_index=True)],
            n_users=U_total,
            n_items=self.n_items,
            rating_col="rating",
            as_dense=True,
        )[0].to(self.device)

        M_infer = M_tr
        infer_user_means = self._compute_user_means(M_infer)

        u_train = np.unique(inc["train"]["user_id"].to_numpy()) if len(inc["train"]) else np.array([], dtype=np.int64)
        u_val = np.unique(inc["val"]["user_id"].to_numpy()) if len(inc["val"]) else np.array([], dtype=np.int64)
        u_test = np.unique(inc["test"]["user_id"].to_numpy()) if len(inc["test"]) else np.array([], dtype=np.int64)

        self._phase_mats["incremental"] = {
            "infer": M_infer,
            "infer_user_means": infer_user_means,
            "train": M_tr,
            "val": M_va,
            "test": M_te,
        }

        self._fit_phase(
            u_train=u_train,
            u_val=u_val,
            u_test=u_test,
            epochs=int(self.params.model_hps["epochs_incremental"]),
            patience=int(self.params.model_hps.get("incremental_patience", self.params.model_hps["patience"])),
            phase="incremental",
        )

    # ----------------------------
    # prediction helpers
    # ----------------------------

    @torch.no_grad()
    def _predict_from_infer_df(
        self,
        *,
        infer_matrix: torch.Tensor,
        infer_user_means: torch.Tensor,
        df: pd.DataFrame,
        batch_size: int,
    ) -> np.ndarray:
        """
        Return RAW, UNCLIPPED predictions.
        """
        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        work = df[["user_id", "item_id"]].copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)

        groups = {
            int(u): (
                g["_row_pos"].to_numpy(np.int64, copy=False),
                g["item_id"].to_numpy(np.int64, copy=True),
            )
            for u, g in work.groupby("user_id", sort=False)
        }

        users = np.array(sorted(groups.keys()), dtype=np.int64)
        pack = self._np_users_to_pack(users)
        if pack is None:
            return np.empty((0,), dtype=np.float32)

        out = np.empty((len(df),), dtype=np.float32)
        self.model.eval()
        device_type = self.device.type

        for batch_users in self._iter_user_batches(pack, batch_size=batch_size, shuffle=False):
            user_means = infer_user_means[batch_users]
            r_in_model = self._matrix_to_model_input(infer_matrix[batch_users], user_means)

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_block_model = self.model(r_in_model)

            pred_block_raw = self._pred_model_to_raw_matrix(pred_block_model, user_means)

            batch_users_np = batch_users.detach().cpu().numpy()
            for bi, u in enumerate(batch_users_np):
                row_pos, item_ids = groups[int(u)]
                items_t = torch.as_tensor(item_ids, device=self.device, dtype=torch.long)
                vals = pred_block_raw[bi].index_select(0, items_t).detach().float().cpu().numpy()
                out[row_pos] = vals

        return out

    @torch.no_grad()
    def _predict_online_df_batch(
        self,
        df: pd.DataFrame,
        *,
        batch_size: int,
    ) -> np.ndarray:
        on = self.ds["online"]
        on_train = on["train"]

        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        users = np.unique(pd.concat([on_train["user_id"], df["user_id"]], ignore_index=True).to_numpy())
        users.sort()
        u_to_row = {u: i for i, u in enumerate(users)}
        B = int(users.size)

        on_tr_loc = on_train.copy()
        on_tr_loc["user_id"] = on_tr_loc["user_id"].map(u_to_row)

        Ru = convert_to_sparse_arr(
            [on_tr_loc],
            n_users=B,
            n_items=self.n_items,
            rating_col="rating",
            as_dense=True,
            device=str(self.device),
        )[0].to(self.device)

        infer_user_means = self._compute_user_means(Ru)

        df_loc = df[["user_id", "item_id"]].copy()
        df_loc["user_id"] = df_loc["user_id"].map(u_to_row)

        return self._predict_from_infer_df(
            infer_matrix=Ru,
            infer_user_means=infer_user_means,
            df=df_loc,
            batch_size=batch_size,
        )

    @torch.no_grad()
    def _predict_online_df_per_user(self, df: pd.DataFrame) -> np.ndarray:
        on = self.ds["online"]
        on_train = on["train"]

        if df is None or df.empty:
            return np.empty((0,), dtype=np.float32)

        train_cols = ["user_id", "item_id", "rating"]
        missing = [c for c in train_cols if c not in on_train.columns]
        if missing:
            raise ValueError(f"online/train missing columns: {missing}")

        tr_groups = {u: g for u, g in on_train[train_cols].groupby("user_id", sort=False)}
        global_on_mean = float(on_train["rating"].mean()) if len(on_train) else 0.0

        work = df[["user_id", "item_id"]].copy()
        work["_row_pos"] = np.arange(len(work), dtype=np.int64)

        out = np.empty((len(work),), dtype=np.float32)
        self.model.eval()
        device_type = self.device.type

        for u, g in work.groupby("user_id", sort=False):
            row_pos = g["_row_pos"].to_numpy(np.int64, copy=False)
            item_ids = g["item_id"].to_numpy(np.int64, copy=True)

            g_tr = tr_groups.get(u)
            if g_tr is None or g_tr.empty:
                Ru_u = torch.zeros((1, self.n_items), dtype=torch.float32, device=self.device)
                user_mean = torch.tensor([global_on_mean], dtype=torch.float32, device=self.device)
            else:
                tr_item_ids = g_tr["item_id"].to_numpy(dtype=np.int64)
                tr_ratings = g_tr["rating"].to_numpy(dtype=np.float32)

                Ru_u = torch.zeros((1, self.n_items), dtype=torch.float32, device=self.device)
                Ru_u[0, torch.as_tensor(tr_item_ids, device=self.device, dtype=torch.long)] = (
                    torch.as_tensor(tr_ratings, device=self.device, dtype=torch.float32)
                )

                user_mean = self._compute_user_means(Ru_u)

            Ru_u_model = self._matrix_to_model_input(Ru_u, user_mean)

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                pred_row_model = self.model(Ru_u_model)[0:1]

            pred_row_raw = self._pred_model_to_raw_matrix(pred_row_model, user_mean)[0]

            vals = pred_row_raw.index_select(
                0,
                torch.as_tensor(item_ids, device=self.device, dtype=torch.long),
            ).detach().float().cpu().numpy()

            out[row_pos] = vals

        return out

    # ----------------------------
    # BaseExperiment hook
    # ----------------------------

    def _predict_df(
        self,
        df: pd.DataFrame,
        *,
        phase: str,
    ) -> np.ndarray:
        hps = self.params.model_hps or {}
        off_bs = int(hps.get("batch_size", 256))
        inc_bs = int(hps.get("incremental_batch_size", off_bs))
        on_bs = int(hps.get("online_batch_size", off_bs))

        if phase == "offline":
            mats = self._phase_mats.get("offline")
            if mats is None:
                raise ValueError("Offline phase mats are missing.")
            return self._predict_from_infer_df(
                infer_matrix=mats["infer"],
                infer_user_means=mats["infer_user_means"],
                df=df,
                batch_size=off_bs,
            )

        if phase == "incremental":
            mats = self._phase_mats.get("incremental")
            if mats is None:
                raise ValueError("Incremental phase mats are missing.")
            return self._predict_from_infer_df(
                infer_matrix=mats["infer"],
                infer_user_means=mats["infer_user_means"],
                df=df,
                batch_size=inc_bs,
            )

        if phase == "offline_post_incremental":
            mats = self._phase_mats.get("incremental")
            if mats is None:
                raise ValueError("Incremental phase mats are missing for offline_post_incremental.")
            return self._predict_from_infer_df(
                infer_matrix=mats["infer"],
                infer_user_means=mats["infer_user_means"],
                df=df,
                batch_size=inc_bs,
            )

        if phase == "online":
            mode = str(getattr(self.params, "online_inference_pred_type", "batch")).lower()
            if mode in {"per_user", "per-user"}:
                return self._predict_online_df_per_user(df)
            return self._predict_online_df_batch(df, batch_size=on_bs)

        raise ValueError(f"Unsupported phase for _predict_df: {phase}")

class SpaceTrackedAutoRecExperiment(SpaceTrackedExperimentBase, AutoRecExperiment):
    """
    AutoRec persistent state = global network parameters θ only.

    With the delta-based SpaceTrackedExperimentBase:
      - OFFLINE delta: +P bytes (θ appears first time)
      - INCREMENTAL delta: 0 (same θ)
      - ONLINE delta: 0 (same θ)
    """

    def _build_space_snapshot(self, phase: Phase) -> SpaceSnapshot:
        snap = SpaceSnapshot()

        params = list(self.model.parameters())
        if not params:
            raise ValueError("AutoRec model has no parameters?")

        # Robust bytes count (handles mixed dtypes correctly)
        total_param_bytes = 0
        P = 0
        dtypes = set()

        for p in params:
            P += int(p.numel())
            dtypes.add(p.dtype)
            total_param_bytes += int(p.numel()) * int(p.element_size())

        dtype_note = None
        dtype_key = _torch_dtype_to_key(next(iter(dtypes)))
        if len(dtypes) > 1:
            dtype_key = "mixed"
            dtype_note = f"Mixed dtypes: {[str(dt) for dt in sorted(dtypes, key=lambda x: str(x))]}"

        snap.components.append(
            SpaceComponent(
                key="autorec.theta",  # same key across phases => delta works automatically
                name="AutoRec network parameters θ",
                bytes=total_param_bytes,
                phase=phase,
                shape=(P,),
                dtype=dtype_key,
                formula="P = Σ_t (d_t*d_{t+1} + d_{t+1})",
                note=(
                    "Global encoder–decoder weights+biases; no per-user storage, no KNN pool."
                    + (f" {dtype_note}" if dtype_note else "")
                ),
            )
        )

        return snap

