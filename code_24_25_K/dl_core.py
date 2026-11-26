# ============================================================
# dl_core.py — shared deep-learning core for the multimodal configs
#              M5 (CNN fusion) and M7 (temporal multimodal Transformer).
# ------------------------------------------------------------
# This module factors the reusable machinery out of
# 25c_interval_encoding_transformer_allmodes.py so that M5 and M7 are
# thin, separate driver scripts that stay head-to-head comparable with
# each other AND with the classical-ML configs (M1..M4, M6):
#
#   * SAME samples / labels        -> data_prep.load_samples
#   * SAME stage-aware 3 classes   -> data_prep.classify_k_value
#   * SAME regression metrics      -> data_prep.regression_metrics
#   * SAME 5-fold GroupKFold       -> GroupKFold on Image_ID (plot)
#   * image / scalar / sequence normalization fit on the TRAIN fold ONLY
#   * DDP via torchrun (multi-GPU), single-GPU / CPU fallback
#
# Model variants (one class, two switches):
#   M5  use_image_temporal=False, use_sequences=False
#       -> per-frame CNN(ortho)+CNN(dem) embeddings, attention-pooled across
#          the available image dates (CNN FUSION), fused with C+P+T+Fagg+Wagg
#          scalars. No daily Fseq/Wseq. (Learned image reps replace V,S.)
#   M7  use_image_temporal=True,  use_sequences=True
#       -> temporal Transformer over the image-date sequence + TCN-encoded
#          daily fertilizer (Fseq) and weather (Wseq) branches + scalars.
#
# Heavy paper-grade explainability is intentionally NOT duplicated here;
# use 25c_run_explainability_from_saved_pt.py on the saved .pt checkpoints.
#
# Requires: torch, torchvision, rasterio, scikit-learn, pandas, numpy.
# ============================================================

from __future__ import annotations

import os
import re
import copy
import gc
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler
from torchvision import models
from sklearn.model_selection import GroupKFold

import rasterio

# Shared preprocessing / labels / metrics (single source of truth)
import data_prep as dp


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
GLOBAL_SEED = dp.GLOBAL_SEED
FOLDS = dp.FOLDS
GROUP_COL = dp.GROUP_COL          # "Image_ID"
TARGET_COL = dp.TARGET_COL        # "K (%)"
DATE_COL = dp.DATE_COL            # "conc_date"
N_CLASSES = dp.N_CLASSES

NODATA_VALUE = -9999.0
MIN_IMAGES_PER_SAMPLE = 1
TIME_FEAT_DIM = 4
TIME_NORM_DAYS = 150.0

# Per-season transplant dates (DAP 0)
TRANSPLANT_2324 = date(2023, 10, 16)
TRANSPLANT_2425 = date(2024, 10, 15)
WEATHER_SEASON_END_2324 = date(2024, 3, 31)

# Per-season (transplant, weather-season-end) windows. The image-DAP feature and
# the daily-weather loader pick whichever season actually matches the data, so
# the SAME code runs correctly for 23-24, 24-25 and 25-26 without hardcoding one
# season. (Previously these were pinned to 23-24, which silently produced wrong
# image-DAP values and an all-zero weather sequence on any other season.)
SEASON_WINDOWS = [
    (date(2023, 10, 16), date(2024, 3, 31)),   # 2023-24
    (date(2024, 10, 15), date(2025, 3, 31)),   # 2024-25
    (date(2025, 10, 8),  date(2026, 3, 31)),   # 2025-26
]


def _season_window_for(d: Optional[date]) -> Tuple[date, date]:
    """(transplant, season_end) for the season containing date d (nearest if none)."""
    if d is not None:
        for tp, end in SEASON_WINDOWS:
            if tp <= d <= end:
                return tp, end
        return min(SEASON_WINDOWS, key=lambda w: abs((w[0] - d).days))
    return SEASON_WINDOWS[0]


def _transplant_of(d: Optional[date]) -> date:
    """Season transplant date (DAP 0) for an arbitrary image/sampling date d."""
    return _season_window_for(d)[0]

# FAWN daily weather variables (15) for the weather TCN
# TCN sequence-branch architecture — single source of truth, shared by the M7
# deep model's fertilizer/weather branches AND the M6 Eseq sequence encoder so
# the two are an apples-to-apples comparison.
TCN_HIDDEN = 64       # hidden channels per dilated residual block
TCN_LEVELS = 4        # number of residual blocks (dilations 1,2,4,8)
TCN_KSIZE = 3         # causal convolution kernel size
TCN_FERT_OUT = 32     # projected output dim of the fertilizer branch
TCN_WEATHER_OUT = 32  # projected output dim of the weather branch

WEATHER_VAR_COLS = [
    "60cm T avg (C)", "60cm T min (C)", "60cm T max (C)",
    "Tsoil avg-10cm  (C)", "Tsoil min(avg)-10cm  (C)", "Tsoil max(avg)-10cm  (C)",
    "2m DewPt avg (C)", "RelHum avg 2m  (pct)",
    "2m Rain tot (mm)", "2m Rain max over 15min (mm)",
    "SolRad avg2m  (w/m^2)",
    "10m Wind avg (mps)", "10m Wind min (mps)", "10m Wind max (mps)",
    "ET (mm)",
]
N_WEATHER_VARS = len(WEATHER_VAR_COLS)

DATE_FOLDER_RE = re.compile(r"^(?P<yyyymmdd>\d{8})_")


# ─────────────────────────────────────────────────────────────
# RUN CONFIG  (each model script fills one of these)
# ─────────────────────────────────────────────────────────────
@dataclass
class DLRunConfig:
    model_tag: str                       # "M5_CNN_fusion" | "M7_Transformer"
    config_name: str                     # "M5" | "M7"  (data_prep.MODEL_CONFIGS)
    image_config: Literal["closest_date", "on_before_closest",
                          "single_date", "all_dates", "interval_dates"] = "closest_date"
    use_image_temporal: bool = False     # temporal encoder over frames (M7) vs pool (M5)
    temporal_encoder: str = "transformer"  # image-date sequence encoder: "transformer" (M7) | "lstm"
    use_sequences: bool = False          # TCN fert + weather branches (M7)
    task_modes: Tuple[str, ...] = ("reg", "cls")

    # paths (EDIT for your environment)
    root_images: str = ""
    ortho_filter: str = "ortho_by_plot"
    dem_filter: str = "dem_by_plot"
    excel_features_path: str = ""
    fert_path: str = ""                  # daily fertilizer xlsx (sheet 'Combined')
    weather_path: str = ""               # FAWN daily weather xlsx
    output_dir: str = ""

    # training
    folds: int = FOLDS
    epochs: int = 80
    batch_size: int = 4
    ortho_backbone: str = "resnet18"
    dem_backbone: str = "convnext_tiny"
    ortho_ch: int = 5
    dem_ch: int = 1
    lr_backbone: float = 5e-5
    lr_mid: float = 5e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 15
    grad_clip: float = 2.0
    freeze_backbone_epochs: int = 5
    hidden: int = 128
    mlp_head_hidden: Tuple[int, ...] = (256, 64)
    mlp_head_drop: float = 0.2
    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = True      # set False to free workers between folds

    # memory / precision controls (default off => unchanged behavior for M5)
    use_amp: bool = False                # mixed-precision (autocast + GradScaler)
    max_frames: Optional[int] = None     # cap frames/sample (keep most recent); None=all
    crop_cap: Optional[int] = None       # cap padded tile H and W (center-crop larger)

    # scalar feature groups (resolved from data_prep config if empty)
    scalar_groups: Tuple[str, ...] = ()

    # Disk control. run_cv writes one .pt per fold per task (~35 MB for M7-class
    # models => ~175 MB per config per season). Only M7_Transformer checkpoints are
    # consumed downstream (by M7_explainability.py). Set False for configs whose
    # checkpoints nothing reads, to avoid
    # filling a storage quota. Metrics, per-fold CSVs and predictions are unaffected.
    save_checkpoints: bool = True


# ─────────────────────────────────────────────────────────────
# LABEL / METRIC WRAPPERS  (delegate to data_prep for consistency)
# ─────────────────────────────────────────────────────────────
def infer_sampling_stage(v):
    return dp.infer_sampling_stage(v)


def k_to_class_label(k_value: float, conc_date_val) -> int:
    stage = dp.infer_sampling_stage(conc_date_val)
    if stage is None:
        return -1
    return dp.classify_k_value(k_value, stage)


def class_idx_to_name(i: int) -> str:
    return dp.class_idx_to_name(i)


regression_metrics = dp.regression_metrics
classification_metrics = dp.classification_metrics


def seed_everything(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    """Seed each DataLoader worker deterministically so per-sample RNG
    (shuffling/augmentation across num_workers>0) is reproducible run-to-run."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader_generator(seed: int) -> torch.Generator:
    """Fixed generator so the train DataLoader's shuffle order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ─────────────────────────────────────────────────────────────
# DDP UTILITIES
# ─────────────────────────────────────────────────────────────
def init_distributed_if_needed():
    if not torch.cuda.is_available():
        return False, 0, 1, 0
    env = os.environ
    world_size = int(env.get("WORLD_SIZE", "1"))
    rank = int(env.get("RANK", "0"))
    local_rank = int(env.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        torch.cuda.set_device(0)
        return False, 0, 1, 0
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    return True, rank, world_size, local_rank


def is_rank0(rank: int) -> bool:
    return rank == 0


def ddp_barrier(distributed: bool):
    if distributed:
        dist.barrier()


def ddp_cleanup(distributed: bool):
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def normalize_image_id(x):
    x = str(x).strip().upper()
    m = re.match(r'^([A-Z0-9]+)_(\d+)$', x)
    return f"{m.group(1)}_{int(m.group(2)):02d}" if m else x


# ─────────────────────────────────────────────────────────────
# IMAGE IO / STATS
# ─────────────────────────────────────────────────────────────
def _read_multiband_with_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)
        roi = src.dataset_mask() > 0
    x[:, ~roi] = np.nan
    return x


def pad_to_min_size(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    c, h, w = x.shape
    ph, pw = max(out_h - h, 0), max(out_w - w, 0)
    if ph == 0 and pw == 0:
        return x
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    return F.pad(x, (pl, pr, pt, pb), mode="constant", value=0.0)


def fit_to_size(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Center-crop dimensions larger than (out_h,out_w), then pad smaller ones.
    With crop_cap=None the dataset's crop_hw is the global max, so x is never
    larger than the target and this reduces to pad_to_min_size (unchanged M5)."""
    c, h, w = x.shape
    if h > out_h:
        top = (h - out_h) // 2
        x = x[:, top:top + out_h, :]
    if w > out_w:
        left = (w - out_w) // 2
        x = x[:, :, left:left + out_w]
    return pad_to_min_size(x, out_h, out_w)


def compute_global_crop_hw(df_imgs, path_col="ortho_path") -> Tuple[int, int]:
    max_h, max_w = 0, 0
    for p in df_imgs[path_col]:
        paths = p if isinstance(p, list) else [p]
        for path in paths:
            with rasterio.open(path) as src:
                max_h, max_w = max(max_h, src.height), max(max_w, src.width)
    return (max_h, max_w)


def compute_band_stats_listcol(df, col, n_bands):
    s = np.zeros(n_bands); sq = np.zeros(n_bands); cnt = np.zeros(n_bands)
    for paths in df[col].tolist():
        paths = paths if isinstance(paths, list) else [paths]
        for p in paths:
            with rasterio.open(p) as src:
                x = src.read().astype(np.float64)
                roi = src.dataset_mask().astype(bool)
            for c in range(n_bands):
                v = x[c][roi & np.isfinite(x[c])]
                if v.size:
                    s[c] += v.sum(); sq[c] += (v * v).sum(); cnt[c] += v.size
    mean = s / np.maximum(cnt, 1.0)
    std = np.sqrt(np.clip(sq / np.maximum(cnt, 1.0) - mean ** 2, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_scalar_stats_listcol(df, col):
    count = 0
    mean = 0.0
    m2 = 0.0
    for paths in df[col].tolist():
        paths = paths if isinstance(paths, list) else [paths]
        for p in paths:
            with rasterio.open(p) as src:
                for _, window in src.block_windows(1):
                    x = src.read(1, window=window).astype(np.float64)
                    roi = src.dataset_mask(window=window).astype(bool)
                    v = x[roi & np.isfinite(x)]
                    if not v.size:
                        continue
                    batch_count = v.size
                    batch_mean = float(v.mean())
                    batch_m2 = float(((v - batch_mean) ** 2).sum())
                    new_count = count + batch_count
                    delta = batch_mean - mean
                    mean += delta * batch_count / new_count
                    m2 += batch_m2 + delta * delta * count * batch_count / new_count
                    count = new_count
    if count == 0:
        return 0.0, 0.0
    return float(mean), float(np.sqrt(max(m2 / count, 0.0)))


def compute_target_stats(df, col="value"):
    v = df[col].astype(np.float64).values
    return float(v.mean()), float(np.std(v, ddof=0))


# ─────────────────────────────────────────────────────────────
# IMAGE TIME FEATURES + FILE SCANNER + SAMPLE BUILDERS
# ─────────────────────────────────────────────────────────────
def parse_date_from_folder(name: str):
    m = DATE_FOLDER_RE.match(name)
    return datetime.strptime(m.group("yyyymmdd"), "%Y%m%d").date() if m else None


def _safe_to_date(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


def _parse_date_from_path(path: str):
    for p in os.path.normpath(str(path)).split(os.sep)[::-1]:
        d = parse_date_from_folder(p)
        if d is not None:
            return d
    return None


def build_image_time_features(row, ortho_paths) -> torch.Tensor:
    """(T,4): [delta_days_norm, image_DAP_norm, sin(doy), cos(doy)]."""
    conc_d = _safe_to_date(row.get("conc_date", None))
    image_dates = row.get("image_dates", None)
    if isinstance(image_dates, str):
        try:
            image_dates = [pd.to_datetime(x).date() for x in image_dates.split("|") if x]
        except Exception:
            image_dates = None
    if not isinstance(image_dates, list) or len(image_dates) != len(ortho_paths):
        image_dates = [_parse_date_from_path(p) for p in ortho_paths]
    feats = []
    for d in image_dates:
        if d is None:
            delta, img_dap, doy = 0.0, 0.0, 1
        else:
            delta = float((conc_d - d).days) if conc_d is not None else 0.0
            delta = max(min(delta, TIME_NORM_DAYS), -TIME_NORM_DAYS) / TIME_NORM_DAYS
            img_dap = float((d - _transplant_of(d)).days) / TIME_NORM_DAYS
            doy = int(d.timetuple().tm_yday)
        phase = 2.0 * math.pi * float(doy) / 366.0
        feats.append([delta, img_dap, math.sin(phase), math.cos(phase)])
    return torch.tensor(feats, dtype=torch.float32)


def find_all_paired_tifs(root, ortho_filter, dem_filter) -> pd.DataFrame:
    ortho_rows, dem_rows = [], []
    for dirpath, _, fnames in os.walk(root):
        for fn in fnames:
            if not fn.lower().endswith(".tif"):
                continue
            full = os.path.join(dirpath, fn)
            folder_date = folder_name = None
            for p in os.path.normpath(full).split(os.sep)[::-1]:
                d = parse_date_from_folder(p)
                if d:
                    folder_date, folder_name = d, p
                    break
            if folder_date is None:
                continue
            if ortho_filter.lower() in dirpath.lower():
                ortho_rows.append((folder_name, folder_date, fn, full))
            elif dem_filter.lower() in dirpath.lower():
                dem_rows.append((folder_name, folder_date, fn, full))
    df_o = pd.DataFrame(ortho_rows, columns=["date_folder", "folder_date", "filename", "ortho_path"])
    df_d = pd.DataFrame(dem_rows, columns=["date_folder", "folder_date", "filename", "dem_path"])
    df_o["stem"] = df_o["filename"].apply(lambda x: os.path.splitext(x)[0])
    df_d["stem"] = df_d["filename"].apply(lambda x: os.path.splitext(x)[0])
    df = df_o.merge(df_d[["date_folder", "stem", "dem_path"]], on=["date_folder", "stem"], how="inner")
    df["exp_ID"] = df["stem"].apply(lambda s: s.split("_")[0])
    df["plot_num"] = df["stem"].apply(lambda s: int(s.split("_")[1]))
    return df


def _coerce_date_col(df, col):
    if np.issubdtype(df[col].dtype, np.number):
        # NaN-safe: rows with a missing date stay NaT so downstream
        # (pd.isna(D)) skips them instead of crashing on int() cast
        s = pd.to_numeric(df[col], errors="coerce")
        out = pd.Series(pd.NaT, index=df.index, dtype="object")
        mask = s.notna()
        out.loc[mask] = pd.to_datetime(
            s[mask].astype("int64").astype(str), format="%Y%m%d"
        ).dt.date
        df[col] = out
    else:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def build_single_date_samples(df_imgs, df_all):
    df_imgs, df_all = df_imgs.copy(), df_all.copy()
    df_imgs["Image_ID"] = df_imgs["Image_ID"].map(normalize_image_id)
    df_all["Image_ID"] = df_all["Image_ID"].map(normalize_image_id)
    df_imgs["folder_date"] = pd.to_datetime(df_imgs["folder_date"]).dt.date
    df_all = _coerce_date_col(df_all, "uav_date")
    imgs_by_plot = {k: g.sort_values("folder_date") for k, g in df_imgs.groupby("Image_ID")}
    out = []
    for _, m in df_all.iterrows():
        iid, uav_d = m["Image_ID"], m["uav_date"]
        if iid not in imgs_by_plot or pd.isna(uav_d):
            continue
        exact = imgs_by_plot[iid][imgs_by_plot[iid]["folder_date"] == uav_d]
        if len(exact) == 0:
            continue
        row = m.to_dict()
        row["ortho_paths"] = exact["ortho_path"].tolist()
        row["dem_paths"] = exact["dem_path"].tolist()
        row["image_dates"] = exact["folder_date"].tolist()
        row["n_imgs"] = len(exact)
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def build_all_dates_samples(df_imgs, df_all, min_images=1):
    df_imgs, df_all = df_imgs.copy(), df_all.copy()
    df_imgs["Image_ID"] = df_imgs["Image_ID"].map(normalize_image_id)
    df_all["Image_ID"] = df_all["Image_ID"].map(normalize_image_id)
    df_imgs["folder_date"] = pd.to_datetime(df_imgs["folder_date"]).dt.date
    df_all = _coerce_date_col(df_all, "conc_date")
    imgs_by_plot = {k: g.sort_values("folder_date") for k, g in df_imgs.groupby("Image_ID")}
    out = []
    for _, m in df_all.iterrows():
        iid, D = m["Image_ID"], m["conc_date"]
        if iid not in imgs_by_plot or pd.isna(D):
            continue
        cand = imgs_by_plot[iid][imgs_by_plot[iid]["folder_date"] <= D].sort_values("folder_date")
        if len(cand) < min_images:
            continue
        row = m.to_dict()
        row["ortho_paths"] = cand["ortho_path"].tolist()
        row["dem_paths"] = cand["dem_path"].tolist()
        row["image_dates"] = cand["folder_date"].tolist()
        row["n_imgs"] = len(cand)
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def build_interval_dates_samples(df_imgs, df_all, min_images=1,
                                 transplant_2324=TRANSPLANT_2324,
                                 transplant_2425=TRANSPLANT_2425,
                                 debug_csv_path=None):
    df_imgs, df_all = df_imgs.copy(), df_all.copy()
    df_imgs["Image_ID"] = df_imgs["Image_ID"].map(normalize_image_id)
    df_all["Image_ID"] = df_all["Image_ID"].map(normalize_image_id)
    df_imgs["folder_date"] = pd.to_datetime(df_imgs["folder_date"]).dt.date
    df_all = _coerce_date_col(df_all, "conc_date")
    df_all = df_all[df_all["conc_date"].notna()].copy()

    def _transplant_for(d):
        # season-general: picks 23-24 / 24-25 / 25-26 transplant from the date
        if d is None:
            return transplant_2324
        return _transplant_of(d)

    imgs_by_plot = {k: g.sort_values("folder_date") for k, g in df_imgs.groupby("Image_ID")}
    df_sorted = df_all.sort_values(["Image_ID", "conc_date"], kind="mergesort").reset_index(drop=True)
    prev_date: Dict[str, Optional[date]] = {}
    out, dbg = [], []
    for _, m in df_sorted.iterrows():
        iid, D = m["Image_ID"], m["conc_date"]
        if iid not in imgs_by_plot:
            prev_date[iid] = D
            continue
        D_prev = prev_date.get(iid, None)
        g = imgs_by_plot[iid]
        if D_prev is None:
            t0 = _transplant_for(D)
            mask = (g["folder_date"] >= t0) & (g["folder_date"] <= D)
        else:
            mask = (g["folder_date"] > D_prev) & (g["folder_date"] <= D)
        cand = g[mask].sort_values("folder_date")
        if len(cand) >= min_images:
            row = m.to_dict()
            row["ortho_paths"] = cand["ortho_path"].tolist()
            row["dem_paths"] = cand["dem_path"].tolist()
            row["image_dates"] = cand["folder_date"].tolist()
            row["n_imgs"] = int(len(cand))
            row["prev_conc_date"] = D_prev
            out.append(row)
            dbg.append({"Image_ID": iid, "conc_date": D, "prev_conc_date": D_prev,
                        "n_imgs": int(len(cand)),
                        "first_image_date": cand["folder_date"].iloc[0],
                        "last_image_date": cand["folder_date"].iloc[-1]})
        prev_date[iid] = D
    if debug_csv_path and dbg:
        os.makedirs(os.path.dirname(debug_csv_path), exist_ok=True)
        pd.DataFrame(dbg).to_csv(debug_csv_path, index=False)
    return pd.DataFrame(out).reset_index(drop=True)


def _closest_image_date(folder_dates: Sequence[date], conc_date: date) -> Optional[date]:
    """Pick the UAV image date closest to conc_date.

    Tie-break: prefer the flight BEFORE the conc date (the earlier date).
    Sort key = (|days diff| asc, folder_date asc) -> closest, then earliest.
    """
    cands = [d for d in folder_dates if d is not None]
    if not cands:
        return None
    return sorted(cands, key=lambda d: (abs((d - conc_date).days), d))[0]


def build_closest_date_samples(df_imgs, df_all):
    """M5: ONE image per (plot, conc_date) = the closest UAV date (tie -> earlier).

    Mirrors the closest-flight rule used for the tabular features, but applied
    to the image tiles actually present on disk."""
    df_imgs, df_all = df_imgs.copy(), df_all.copy()
    df_imgs["Image_ID"] = df_imgs["Image_ID"].map(normalize_image_id)
    df_all["Image_ID"] = df_all["Image_ID"].map(normalize_image_id)
    df_imgs["folder_date"] = pd.to_datetime(df_imgs["folder_date"]).dt.date
    df_all = _coerce_date_col(df_all, "conc_date")
    imgs_by_plot = {k: g.sort_values("folder_date") for k, g in df_imgs.groupby("Image_ID")}
    out = []
    for _, m in df_all.iterrows():
        iid, D = m["Image_ID"], m["conc_date"]
        if iid not in imgs_by_plot or pd.isna(D):
            continue
        g = imgs_by_plot[iid]
        cd = _closest_image_date(g["folder_date"].tolist(), D)
        if cd is None:
            continue
        sel = g[g["folder_date"] == cd]
        row = m.to_dict()
        row["ortho_paths"] = sel["ortho_path"].tolist()
        row["dem_paths"] = sel["dem_path"].tolist()
        row["image_dates"] = sel["folder_date"].tolist()
        row["closest_image_date"] = cd
        row["n_imgs"] = len(sel)
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def build_on_before_closest_samples(df_imgs, df_all, min_images=1):
    """M7: ALL images with folder_date <= the closest UAV date for that conc.

    The image sequence ends at the same closest flight M5 uses as its single
    image (e.g. if the closest flight is conc-7, M7 sees every image on/before
    conc-7), giving the temporal Transformer the full pre-sampling history."""
    df_imgs, df_all = df_imgs.copy(), df_all.copy()
    df_imgs["Image_ID"] = df_imgs["Image_ID"].map(normalize_image_id)
    df_all["Image_ID"] = df_all["Image_ID"].map(normalize_image_id)
    df_imgs["folder_date"] = pd.to_datetime(df_imgs["folder_date"]).dt.date
    df_all = _coerce_date_col(df_all, "conc_date")
    imgs_by_plot = {k: g.sort_values("folder_date") for k, g in df_imgs.groupby("Image_ID")}
    out = []
    for _, m in df_all.iterrows():
        iid, D = m["Image_ID"], m["conc_date"]
        if iid not in imgs_by_plot or pd.isna(D):
            continue
        g = imgs_by_plot[iid]
        cd = _closest_image_date(g["folder_date"].tolist(), D)
        if cd is None:
            continue
        cand = g[g["folder_date"] <= cd].sort_values("folder_date")
        if len(cand) < min_images:
            continue
        row = m.to_dict()
        row["ortho_paths"] = cand["ortho_path"].tolist()
        row["dem_paths"] = cand["dem_path"].tolist()
        row["image_dates"] = cand["folder_date"].tolist()
        row["closest_image_date"] = cd
        row["n_imgs"] = len(cand)
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# DAILY SEQUENCE LOADERS  (shared with M6 Eseq)
# ─────────────────────────────────────────────────────────────
def load_fert_lookup(path: str):
    """fert_by_plot_daily xlsx (sheet 'Combined'): Image_ID/plot_Num + day 1.. day N.
    Returns (dict Image_ID -> np.ndarray[max_dap], max_dap)."""
    df = pd.read_excel(path, sheet_name="Combined")
    df.columns = [str(c).strip() for c in df.columns]
    id_col = next((c for c in df.columns
                   if c.lower() in ("plot_num", "plot num", "plotnum", "image_id", "imageid")), None)
    if id_col is None:
        raise ValueError(f"No plot/Image_ID column in {path}. cols={list(df.columns)}")
    day_cols = sorted([c for c in df.columns if re.match(r"^day\s*\d+$", c, re.IGNORECASE)],
                      key=lambda x: int(re.search(r"\d+", x).group()))
    lookup = {}
    for _, row in df.iterrows():
        lookup[normalize_image_id(str(row[id_col]).strip())] = \
            row[day_cols].fillna(0.0).astype(np.float32).values
    return lookup, len(day_cols)


def load_weather_array(path: str, transplant: Optional[date] = None,
                       season_end: Optional[date] = None,
                       normalize: bool = True):
    """FAWN daily weather xlsx -> ((N_days,15) array, transplant date).

    The season window (transplant -> season_end, = day 0 of the returned array)
    is AUTO-DETECTED from the file's own dates unless given explicitly, so the
    same loader works for 23-24, 24-25 or 25-26. Day 0 of the array is the
    season transplant, matching the DAP indexing used by M6/M7.

    normalize : if True (default, backward-compatible) the returned array is
      z-scored using FULL-WINDOW means/stds. **This is leakage-prone inside a
      CV loop** because the full window spans post-sampling dates and is fit on
      all data, not the train fold. Pass normalize=False to get the RAW array;
      run_cv then fits the weather scaler on the train (fit) fold only, over
      days <= the latest training sampling date (see _fold_seq_norm).
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    # FAWN "Period" is "12 Oct 2023" in some exports and "12-Oct-23" in others
    per = df["Period"].astype(str).str.strip()
    dt = pd.to_datetime(per, format="%d %b %Y", errors="coerce")
    dt = dt.fillna(pd.to_datetime(per, format="%d-%b-%y", errors="coerce"))
    if dt.isna().any():
        raise ValueError(f"unparsed FAWN dates in {path}: {per[dt.isna()].head(3).tolist()}")
    df["date_parsed"] = dt.dt.date
    if transplant is None or season_end is None:
        # choose the season window that contains the most rows of this file
        best, best_n = SEASON_WINDOWS[1], -1
        for tp, end in SEASON_WINDOWS:
            n = int(((df["date_parsed"] >= tp) & (df["date_parsed"] <= end)).sum())
            if n > best_n:
                best, best_n = (tp, end), n
        transplant, season_end = best
    df = df[(df["date_parsed"] >= transplant) &
            (df["date_parsed"] <= season_end)].sort_values("date_parsed")
    n_days = (season_end - transplant).days + 1
    raw = np.zeros((n_days, N_WEATHER_VARS), dtype=np.float32)
    for _, row in df.iterrows():
        idx = (row["date_parsed"] - transplant).days
        if 0 <= idx < n_days:
            for ci, col in enumerate(WEATHER_VAR_COLS):
                v = row.get(col, np.nan)
                raw[idx, ci] = float(v) if pd.notna(v) else 0.0
    if not normalize:
        return raw.astype(np.float32), transplant
    means = raw.mean(0, keepdims=True)
    stds = np.where(raw.std(0, keepdims=True) < 1e-6, 1.0, raw.std(0, keepdims=True))
    return ((raw - means) / stds).astype(np.float32), transplant


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class OrthoDemTabularDataset(Dataset):
    """Returns ortho_seq(T,5,H,W), dem_seq(T,1,H,W), feat(F,), img_time(T,4),
    seq_len, y_reg(norm), y_cls, fert_seq(1,Tf), weather_seq(15,Tw)."""

    def __init__(self, df, ortho_mean, ortho_std, dem_mean, dem_std,
                 y_mean, y_std, crop_hw, feature_cols,
                 feat_mean=None, feat_std=None, use_sequences=False,
                 fert_lookup=None, fert_mean=0.0, fert_std=1.0,
                 weather_array=None, weather_transplant=None, max_frames=None):
        self.df = df.reset_index(drop=True)
        self.max_frames = max_frames
        self.ortho_mean = ortho_mean.astype(np.float32)
        self.ortho_std = np.clip(ortho_std.astype(np.float32), 1e-6, None)
        self.dem_mean, self.dem_std = float(dem_mean), float(max(dem_std, 1e-6))
        self.y_mean, self.y_std = float(y_mean), float(max(y_std, 1e-6))
        self.crop_hw = crop_hw
        self.feature_cols = feature_cols or []
        self.feat_mean = None if feat_mean is None else feat_mean.astype(np.float32)
        self.feat_std = None if feat_std is None else np.clip(feat_std.astype(np.float32), 1e-6, None)
        self.use_sequences = use_sequences
        self.fert_lookup = fert_lookup or {}
        self.fert_mean, self.fert_std = float(fert_mean), float(max(fert_std, 1e-6))
        self.weather_array = weather_array
        self.weather_transplant = weather_transplant

    def __len__(self):
        return len(self.df)

    def _load_norm(self, op, dp_):
        ot = torch.nan_to_num(torch.from_numpy(_read_multiband_with_mask(op)))
        dt = torch.nan_to_num(torch.from_numpy(_read_multiband_with_mask(dp_)))
        h, w = self.crop_hw
        ot, dt = fit_to_size(ot, h, w), fit_to_size(dt, h, w)
        ot = (ot - torch.from_numpy(self.ortho_mean)[:, None, None]) / torch.from_numpy(self.ortho_std)[:, None, None]
        dt = (dt - self.dem_mean) / self.dem_std
        return ot, dt

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        op = row["ortho_paths"] if isinstance(row["ortho_paths"], list) else [row["ortho_paths"]]
        dps = row["dem_paths"] if isinstance(row["dem_paths"], list) else [row["dem_paths"]]
        # cap sequence length: keep the most recent frames (closest to sampling)
        if self.max_frames is not None and len(op) > self.max_frames:
            op = op[-self.max_frames:]
            dps = dps[-self.max_frames:]
        ol, dl = [], []
        for o, d in zip(op, dps):
            a, b = self._load_norm(o, d)
            ol.append(a); dl.append(b)
        ortho_seq = torch.stack(ol, 0)
        dem_seq = torch.stack(dl, 0)
        img_time = build_image_time_features(row, op)
        seq_len = len(ol)

        y_val = float(row["value"])
        y_reg = torch.tensor((y_val - self.y_mean) / self.y_std, dtype=torch.float32)

        if "K_label_idx" in row.index and not pd.isna(row["K_label_idx"]):
            y_cls_int = int(row["K_label_idx"])
        else:
            y_cls_int = k_to_class_label(y_val, _safe_to_date(row.get("conc_date")))
        y_cls = torch.tensor(y_cls_int, dtype=torch.long)

        if self.feature_cols:
            fs = row[self.feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).values
            fs = np.nan_to_num(fs)
            ft = torch.from_numpy(fs)
            if self.feat_mean is not None:
                ft = (ft - torch.from_numpy(self.feat_mean)) / torch.from_numpy(self.feat_std)
        else:
            ft = torch.empty(0, dtype=torch.float32)

        dap = int(row["DAP"]) if "DAP" in row.index and not pd.isna(row.get("DAP")) else \
            (int(row["day"]) if "day" in row.index and not pd.isna(row.get("day")) else 0)

        if self.use_sequences:
            iid = normalize_image_id(row.get("Image_ID", ""))
            if iid in self.fert_lookup and dap > 0:
                arr = self.fert_lookup[iid][:dap].astype(np.float32)
                arr = (arr - self.fert_mean) / self.fert_std
            else:
                arr = np.zeros(max(dap, 1), dtype=np.float32)
            fert_t = torch.from_numpy(arr).unsqueeze(0)
            if self.weather_array is not None and dap > 0:
                n = min(dap + 1, self.weather_array.shape[0])
                wt = torch.from_numpy(self.weather_array[:n].T.astype(np.float32))
            else:
                wt = torch.zeros(N_WEATHER_VARS, 1, dtype=torch.float32)
        else:
            fert_t = torch.zeros(1, 1, dtype=torch.float32)
            wt = torch.zeros(N_WEATHER_VARS, 1, dtype=torch.float32)

        return ortho_seq, dem_seq, ft, img_time, torch.tensor(seq_len), y_reg, y_cls, fert_t, wt


def temporal_collate_fn(batch):
    o, d, f, t, sl, yr, yc, fe, we = zip(*batch)
    o = pad_sequence(o, batch_first=True)
    d = pad_sequence(d, batch_first=True)
    f = torch.stack(f, 0)
    t = pad_sequence(t, batch_first=True)
    sl = torch.stack(sl, 0)
    yr = torch.stack(yr, 0)
    yc = torch.stack(yc, 0)
    mtf = max(x.shape[-1] for x in fe)
    fe = torch.stack([F.pad(x, (0, mtf - x.shape[-1])) for x in fe], 0)
    mtw = max(x.shape[-1] for x in we)
    we = torch.stack([F.pad(x, (0, mtw - x.shape[-1])) for x in we], 0)
    return o, d, f, t, sl, yr, yc, fe, we


# ─────────────────────────────────────────────────────────────
# ENCODERS / TCN / POOLING
# ─────────────────────────────────────────────────────────────
def _adapt_first_conv(module, in_ch):
    if hasattr(module, "conv1") and isinstance(module.conv1, nn.Conv2d):
        old = module.conv1
        new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        module.conv1 = new
        return new, old
    if hasattr(module, "features"):
        first = module.features[0][0]
        if isinstance(first, nn.Conv2d):
            old = first
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            module.features[0][0] = new
            return new, old
    raise ValueError("Cannot find first conv to adapt.")


def _init_first_conv(new_conv, old_conv):
    with torch.no_grad():
        ic = new_conv.in_channels
        if ic == old_conv.in_channels:
            new_conv.weight.copy_(old_conv.weight)
        elif ic == 1:
            new_conv.weight.copy_(old_conv.weight.mean(1, keepdim=True))
        elif ic >= 3:
            new_conv.weight[:, :3].copy_(old_conv.weight)
            if ic > 3:
                avg = old_conv.weight.mean(1, keepdim=True)
                for c in range(3, ic):
                    new_conv.weight[:, c:c + 1].copy_(avg)
        else:
            new_conv.weight[:, :ic].copy_(old_conv.weight[:, :ic])


def build_encoder(backbone, in_ch):
    b = backbone.lower()
    if b == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        nc, oc = _adapt_first_conv(m, in_ch); _init_first_conv(nc, oc)
        fd = m.fc.in_features; m.fc = nn.Identity(); return m, fd
    if b == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        nc, oc = _adapt_first_conv(m, in_ch); _init_first_conv(nc, oc)
        fd = m.fc.in_features; m.fc = nn.Identity(); return m, fd
    if b == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        nc, oc = _adapt_first_conv(m, in_ch); _init_first_conv(nc, oc)
        fd = m.classifier[-1].in_features; m.classifier = nn.Identity(); return m, fd
    if b == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        nc, oc = _adapt_first_conv(m, in_ch); _init_first_conv(nc, oc)
        fd = m.classifier[-1].in_features; m.classifier = nn.Identity(); return m, fd
    raise ValueError(f"Unknown backbone: {backbone}")


def _build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers, prev = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation))
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.chomp = pad
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.relu(self.conv1(x)[..., :-self.chomp] if self.chomp else self.conv1(x))
        y = self.drop(y)
        y = self.relu(self.conv2(y)[..., :-self.chomp] if self.chomp else self.conv2(y))
        y = self.drop(y)
        res = self.down(x) if self.down is not None else x
        return self.relu(y + res)


class TCNEncoder(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, levels=4, kernel_size=3, out_dim=32, dropout=0.1):
        super().__init__()
        blocks, ch = [], in_ch
        for i in range(levels):
            blocks.append(_TCNBlock(ch, hidden_ch, kernel_size, 2 ** i, dropout))
            ch = hidden_ch
        self.net = nn.Sequential(*blocks)
        self.project = nn.Linear(hidden_ch, out_dim)

    def forward(self, x):
        h = self.net(x)[..., -1]
        return self.project(h)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, seq_len=None):
        logits = self.score(x).squeeze(-1)
        if seq_len is not None:
            T = x.size(1)
            mask = torch.arange(T, device=x.device)[None, :] >= seq_len[:, None]
            # dtype-aware fill: -1e9 overflows fp16 under AMP autocast
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (x * w).sum(1)


class SinusoidalTimeMLP(nn.Module):
    def __init__(self, in_dim=TIME_FEAT_DIM, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
                                 nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim))

    def forward(self, x_time, B, T, device):
        if x_time is None:
            x_time = torch.zeros(B, T, TIME_FEAT_DIM, device=device)
        return self.net(x_time)


# ─────────────────────────────────────────────────────────────
# DUAL-TASK MODEL  (CNN fusion [M5] or temporal Transformer [M7])
# ─────────────────────────────────────────────────────────────
class DualTaskModel(nn.Module):
    def __init__(self, ortho_backbone="resnet18", dem_backbone="convnext_tiny",
                 ortho_ch=5, dem_ch=1, n_features=0, hidden=128,
                 mlp_head_hidden=(256, 64), mlp_head_drop=0.2, n_classes=3,
                 use_image_temporal=True, use_sequences=True,
                 temporal_encoder="transformer",
                 tcn_fert_out=TCN_FERT_OUT, tcn_weather_out=TCN_WEATHER_OUT):
        super().__init__()
        self.use_image_temporal = use_image_temporal
        self.use_sequences = use_sequences
        self.temporal_encoder = temporal_encoder

        self.ortho_cnn, ofeat = build_encoder(ortho_backbone, ortho_ch)
        self.dem_cnn, dfeat = build_encoder(dem_backbone, dem_ch)
        self.frame_feat_dim = ofeat + dfeat

        self.time_mlp = SinusoidalTimeMLP(TIME_FEAT_DIM, 32)
        self.frame_proj = nn.Sequential(
            nn.Linear(self.frame_feat_dim + 32, hidden), nn.LayerNorm(hidden),
            nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.temporal_transformer = None
        self.temporal_lstm = None
        if use_image_temporal:
            if temporal_encoder == "lstm":
                # LSTM swapped in for the Transformer over the image-date
                # sequence. Bidirectional + 2 layers mirrors the 2-layer,
                # full-context (non-causal) Transformer encoder; hidden_size =
                # hidden//2 so the concatenated output stays `hidden`-dim. Every
                # downstream module (attention pooling, TCN fusion, tabular MLP,
                # trunks) is therefore byte-for-byte identical to M7_Transformer,
                # keeping the two models directly comparable.
                self.temporal_lstm = nn.LSTM(
                    input_size=hidden, hidden_size=hidden // 2, num_layers=2,
                    batch_first=True, dropout=0.1, bidirectional=True)
            else:
                enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=256,
                                                 dropout=0.1, batch_first=True, activation="gelu")
                self.temporal_transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.attn_pool = MaskedAttentionPooling(hidden, hidden=128)

        self.n_features = int(n_features)
        tab_out = 64 if self.n_features > 0 else 0
        self.tab = nn.Sequential(
            nn.Linear(self.n_features, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, tab_out), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ) if self.n_features > 0 else None

        if use_sequences:
            self.tcn_fert = TCNEncoder(1, TCN_HIDDEN, TCN_LEVELS, TCN_KSIZE, tcn_fert_out)
            self.tcn_weather = TCNEncoder(N_WEATHER_VARS, TCN_HIDDEN, TCN_LEVELS, TCN_KSIZE, tcn_weather_out)
            self._fert_out, self._weather_out = tcn_fert_out, tcn_weather_out
        else:
            self.tcn_fert = self.tcn_weather = None
            self._fert_out = self._weather_out = 0

        trunk_in = hidden + tab_out + self._fert_out + self._weather_out
        self.trunk_reg = _build_mlp(trunk_in, list(mlp_head_hidden), 1, mlp_head_drop)
        self.trunk_cls = _build_mlp(trunk_in, list(mlp_head_hidden), n_classes, mlp_head_drop)

    def set_backbone_trainable(self, trainable):
        for p in self.ortho_cnn.parameters():
            p.requires_grad = trainable
        for p in self.dem_cnn.parameters():
            p.requires_grad = trainable

    def forward(self, x_ortho, x_dem, x_feat=None, seq_len=None, x_time=None,
                x_fert=None, x_weather=None):
        B, T = x_ortho.shape[:2]
        zo = self.ortho_cnn(x_ortho.view(B * T, *x_ortho.shape[2:]))
        zd = self.dem_cnn(x_dem.view(B * T, *x_dem.shape[2:]))
        if zo.dim() > 2:
            zo = zo.flatten(1)
        if zd.dim() > 2:
            zd = zd.flatten(1)
        z = torch.cat([zo, zd], 1).view(B, T, self.frame_feat_dim)
        z = self.frame_proj(torch.cat([z, self.time_mlp(x_time, B, T, z.device)], -1))
        pad_mask = None
        if seq_len is not None:
            pad_mask = torch.arange(T, device=z.device)[None, :] >= seq_len[:, None]
        if self.temporal_transformer is not None:
            z = self.temporal_transformer(z, src_key_padding_mask=pad_mask)
        elif self.temporal_lstm is not None:
            if seq_len is not None:
                # pack so the LSTM never reads padded frames; pad back to T so the
                # attention-pooling mask (seq_len) still lines up exactly.
                lengths = seq_len.detach().to("cpu").clamp(min=1)
                packed = nn.utils.rnn.pack_padded_sequence(
                    z, lengths, batch_first=True, enforce_sorted=False)
                packed_out, _ = self.temporal_lstm(packed)
                z, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True, total_length=T)
            else:
                z, _ = self.temporal_lstm(z)
        z_temporal = self.attn_pool(z, seq_len)

        parts = [z_temporal]
        if self.tab is not None and x_feat is not None:
            parts.append(self.tab(x_feat))
        if self.use_sequences:
            parts.append(self.tcn_fert(x_fert) if x_fert is not None
                         else torch.zeros(B, self._fert_out, device=z.device))
            parts.append(self.tcn_weather(x_weather) if x_weather is not None
                         else torch.zeros(B, self._weather_out, device=z.device))
        z = torch.cat(parts, 1)
        return self.trunk_reg(z), self.trunk_cls(z)


# ─────────────────────────────────────────────────────────────
# TRAIN ONE FOLD
# ─────────────────────────────────────────────────────────────
@dataclass
class BestState:
    val_loss: float = float("inf")
    epoch: int = -1
    state: Optional[dict] = None
    metrics_reg: dict = field(default_factory=dict)
    metrics_cls: dict = field(default_factory=dict)


def train_one_fold(cfg: DLRunConfig, fold_id, df, tr_idx, va_idx, device, crop_hw,
                   feature_cols, task, cls_weights=None,
                   distributed=False, rank=0, world_size=1, local_rank=0,
                   fert_lookup=None, fert_mean=0.0, fert_std=1.0,
                   weather_array=None, weather_transplant=None):
    df_tr = df.iloc[tr_idx].copy().reset_index(drop=True)
    df_va = df.iloc[va_idx].copy().reset_index(drop=True)

    feat_mean = (df_tr[feature_cols].apply(pd.to_numeric, errors="coerce")
                 .astype(np.float64).mean(0).values.astype(np.float32)) if feature_cols else None
    feat_std = (df_tr[feature_cols].apply(pd.to_numeric, errors="coerce")
                .astype(np.float64).std(0, ddof=0).values.astype(np.float32)) if feature_cols else None

    ortho_mean, ortho_std = compute_band_stats_listcol(df_tr, "ortho_paths", cfg.ortho_ch)
    dem_mean, dem_std = compute_scalar_stats_listcol(df_tr, "dem_paths")
    y_mean, y_std = compute_target_stats(df_tr)

    def make_ds(d):
        return OrthoDemTabularDataset(
            d, ortho_mean, ortho_std, dem_mean, dem_std, y_mean, y_std, crop_hw,
            feature_cols, feat_mean, feat_std, use_sequences=cfg.use_sequences,
            fert_lookup=fert_lookup, fert_mean=fert_mean, fert_std=fert_std,
            weather_array=weather_array, weather_transplant=weather_transplant,
            max_frames=cfg.max_frames)

    train_ds, val_ds = make_ds(df_tr), make_ds(df_va)

    if distributed:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    # deterministic shuffle order + worker RNG -> reproducible reruns
    train_gen = _make_loader_generator(GLOBAL_SEED + int(fold_id) + rank)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory, collate_fn=temporal_collate_fn,
                              drop_last=True,
                              worker_init_fn=_seed_worker, generator=train_gen,
                              persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=temporal_collate_fn,
                            worker_init_fn=_seed_worker) \
        if is_rank0(rank) else None

    model = DualTaskModel(
        ortho_backbone=cfg.ortho_backbone, dem_backbone=cfg.dem_backbone,
        ortho_ch=cfg.ortho_ch, dem_ch=cfg.dem_ch, n_features=len(feature_cols),
        hidden=cfg.hidden, mlp_head_hidden=cfg.mlp_head_hidden,
        mlp_head_drop=cfg.mlp_head_drop, n_classes=N_CLASSES,
        use_image_temporal=cfg.use_image_temporal, use_sequences=cfg.use_sequences,
        temporal_encoder=cfg.temporal_encoder,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if cfg.freeze_backbone_epochs > 0:
        unwrap_model(model).set_backbone_trainable(False)

    def split_params(m):
        bb, mid, head = [], [], []
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("ortho_cnn.") or name.startswith("dem_cnn."):
                bb.append(p)
            elif name.startswith("tab.") or name.startswith("frame_proj.") or \
                    name.startswith("temporal_transformer.") or \
                    name.startswith("temporal_lstm."):
                mid.append(p)
            else:
                head.append(p)
        return bb, mid, head

    def make_opt(m):
        bb, mid, head = split_params(m)
        opt = torch.optim.AdamW(
            [{"params": bb, "lr": cfg.lr_backbone},
             {"params": mid, "lr": cfg.lr_mid},
             {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)
        return opt, sch

    optimizer, scheduler = make_opt(unwrap_model(model))
    loss_reg = nn.SmoothL1Loss()
    loss_cls = nn.CrossEntropyLoss(
        weight=cls_weights.to(device) if cls_weights is not None else None, ignore_index=-1)

    def task_loss(out_reg, out_cls, yr, yc):
        if task == "reg":
            return loss_reg(out_reg, yr)
        valid = yc >= 0
        if valid.sum() > 0:
            return loss_cls(out_cls[valid], yc[valid])
        return None

    amp_enabled = bool(cfg.use_amp) and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=amp_enabled)

    best = BestState()
    bad = 0
    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_backbone_epochs > 0 and epoch == cfg.freeze_backbone_epochs + 1:
            unwrap_model(model).set_backbone_trainable(True)
            optimizer, scheduler = make_opt(unwrap_model(model))
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        tr_losses = []
        for xo, xd, fb, xt, sl, yr, yc, fe, we in train_loader:
            xo, xd, fb, xt, sl = xo.to(device), xd.to(device), fb.to(device), xt.to(device), sl.to(device)
            yr = yr.to(device).view(-1, 1); yc = yc.to(device)
            fe, we = fe.to(device), we.to(device)
            with autocast("cuda", enabled=amp_enabled):
                out_reg, out_cls = model(xo, xd, fb, sl, x_time=xt, x_fert=fe, x_weather=we)
                loss = task_loss(out_reg, out_cls, yr, yc)
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            tr_losses.append(loss.item())
        scheduler.step(epoch)

        stop = 0
        if is_rank0(rank):
            model.eval()
            va_losses, yt_r, yp_r, yt_c, yp_c = [], [], [], [], []
            with torch.no_grad():
                for xo, xd, fb, xt, sl, yr, yc, fe, we in val_loader:
                    xo, xd, fb, xt, sl = xo.to(device), xd.to(device), fb.to(device), xt.to(device), sl.to(device)
                    yr = yr.to(device).view(-1, 1); yc = yc.to(device)
                    fe, we = fe.to(device), we.to(device)
                    with autocast("cuda", enabled=amp_enabled):
                        out_reg, out_cls = unwrap_model(model)(xo, xd, fb, sl, x_time=xt, x_fert=fe, x_weather=we)
                    out_reg = out_reg.float(); out_cls = out_cls.float()
                    loss = task_loss(out_reg, out_cls, yr, yc)
                    if loss is not None:
                        va_losses.append(loss.item())
                    if task == "reg":
                        yp_r.extend((out_reg.cpu().numpy().reshape(-1) * y_std + y_mean).tolist())
                        yt_r.extend((yr.cpu().numpy().reshape(-1) * y_std + y_mean).tolist())
                    else:
                        v = yc >= 0
                        if v.sum() > 0:
                            yp_c.extend(torch.argmax(out_cls[v], 1).cpu().numpy().tolist())
                            yt_c.extend(yc[v].cpu().numpy().tolist())
            mets_reg = regression_metrics(np.array(yt_r), np.array(yp_r)) if task == "reg" and yt_r else {}
            mets_cls = classification_metrics(np.array(yt_c), np.array(yp_c)) if task == "cls" and yt_c else {}
            va_loss = float(np.mean(va_losses)) if va_losses else float("inf")
            tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
            msg = (f"R2={mets_reg.get('R2', float('nan')):.3f} RMSE={mets_reg.get('RMSE', float('nan')):.3f}"
                   if task == "reg" else
                   f"Acc={mets_cls.get('Acc', 0):.3f} F1m={mets_cls.get('F1_macro', 0):.3f}")
            print(f"  [F{fold_id} {task}] ep{epoch:03d} tr={tr_loss:.4f} va={va_loss:.4f} | {msg}", flush=True)
            if va_loss < best.val_loss - 1e-6:
                best.val_loss, best.epoch = va_loss, epoch
                best.state = copy.deepcopy(unwrap_model(model).state_dict())
                best.metrics_reg, best.metrics_cls = mets_reg, mets_cls
                bad = 0
            else:
                bad += 1
                stop = 1 if bad >= cfg.patience else 0
        flag = torch.tensor([stop], dtype=torch.int32, device=device)
        if distributed:
            dist.broadcast(flag, src=0)
        if int(flag.item()) == 1:
            break
        ddp_barrier(distributed)

    if best.state:
        unwrap_model(model).load_state_dict(best.state)
    stats = {"ortho_mean": ortho_mean, "ortho_std": ortho_std, "dem_mean": dem_mean,
             "dem_std": dem_std, "y_mean": y_mean, "y_std": y_std,
             "feature_cols": feature_cols, "feat_mean": feat_mean, "feat_std": feat_std,
             "fert_lookup": fert_lookup, "fert_mean": fert_mean, "fert_std": fert_std,
             "weather_array": weather_array, "weather_transplant": weather_transplant,
             "crop_hw": crop_hw}
    return model, stats, best.metrics_reg or {}, best.metrics_cls or {}


# ─────────────────────────────────────────────────────────────
# LEAKAGE CONTROLS  (nested CV + train-fold sequence normalization)
# ─────────────────────────────────────────────────────────────
def _dap_of_row(row) -> int:
    """Days-after-planting for a sample row (DAP or 'day'; 0 if absent)."""
    for c in ("DAP", "day"):
        if c in row.index and pd.notna(row.get(c)):
            try:
                return int(row[c])
            except Exception:
                pass
    return 0


def _inner_earlystop_split(df, tr_idx, group_col="Image_ID"):
    """Carve a plot-grouped inner validation split from an OUTER TRAIN fold.

    nested-CV fix: early stopping and checkpoint selection must NOT read
    the outer (scoring) fold. We split the outer TRAIN indices into
      * fit_idx  — rows the model trains on
      * es_idx   — held-out rows used ONLY for early stopping / best-epoch pick
    grouped on Image_ID so a plot never straddles fit/es. Deterministic
    (GroupKFold, first split ~= 1/k of train plots). Falls back to (tr, tr) only
    in the degenerate case of a single plot.
    """
    tr_idx = np.asarray(tr_idx)
    sub_groups = df.iloc[tr_idx][group_col].astype(str).values
    n_groups = len(np.unique(sub_groups))
    if n_groups < 2:
        return tr_idx, tr_idx
    k = max(2, min(5, n_groups))
    fit_local, es_local = list(
        GroupKFold(n_splits=k).split(np.arange(len(tr_idx)), groups=sub_groups))[0]
    return tr_idx[fit_local], tr_idx[es_local]


def weather_stats_from_rows(df, fit_idx, weather_raw):
    """Per-variable (mean, std) for the weather branch, fit on FIT rows only.

    Uses only days on/before the LATEST sampling date among the fit rows, so no
    post-sampling day contributes to the scaler. Returned as (1,15) arrays so
    they can be applied to ANY raw weather array with the same variable order —
    including a *different season's* array, which is what the external
    (cross-season) validation needs to honor "train-season statistics only".
    """
    if weather_raw is None:
        return None, None
    max_fit_dap = 0
    for _, row in df.iloc[np.asarray(fit_idx)].iterrows():
        max_fit_dap = max(max_fit_dap, _dap_of_row(row))
    cutoff = max(min(max_fit_dap + 1, weather_raw.shape[0]), 1)
    w = weather_raw[:cutoff]
    w_mean = w.mean(0, keepdims=True)
    w_std = np.where(w.std(0, keepdims=True) < 1e-6, 1.0, w.std(0, keepdims=True))
    return w_mean, w_std


def apply_weather_norm(weather_raw, w_mean, w_std):
    """Z-score a RAW weather array with pre-computed stats (see above)."""
    if weather_raw is None or w_mean is None:
        return None
    return ((weather_raw - w_mean) / w_std).astype(np.float32)


def _fold_seq_norm(df, fit_idx, fert_lookup, weather_raw):
    """Fit fertilizer + weather normalization on the FIT (train) rows ONLY.

    sequence-branch fix. Both scalers are fit on training rows and only
    over days on/before each training sample's sampling date (never the full
    season, never all folds):

      * fertilizer: pool fert_lookup[iid][:dap] (positive doses) across fit rows
        -> (mean, std)   [mirrors the already-correct M6 Eseq normalization]
      * weather:    fit per-variable (mean, std) over weather_raw[:max_fit_dap+1]
        i.e. only days up to the LATEST training sampling date, then z-score the
        whole raw array with those stats. Slicing a pre-normalized array in the
        dataset is identical to normalizing after slicing (affine per channel),
        so downstream code is unchanged.

    Returns (fert_mean, fert_std, weather_norm_or_None).
    """
    fit_rows = df.iloc[np.asarray(fit_idx)]

    # ── fertilizer: train-fold pooled dose stats (<= sampling via [:dap]) ──
    fert_mean, fert_std = 0.0, 1.0
    if fert_lookup:
        tv = []
        for _, row in fit_rows.iterrows():
            dap = _dap_of_row(row)
            iid = normalize_image_id(row.get("Image_ID", ""))
            if iid in fert_lookup and dap > 0:
                v = np.asarray(fert_lookup[iid][:dap], dtype=np.float64)
                tv.append(v[v > 0])
        if tv:
            allv = np.concatenate(tv)
            if allv.size:
                fert_mean, fert_std = float(allv.mean()), max(float(allv.std()), 1e-6)

    # ── weather: train-fold stats over days <= latest training sampling date ──
    w_mean, w_std = weather_stats_from_rows(df, fit_idx, weather_raw)
    weather_norm = apply_weather_norm(weather_raw, w_mean, w_std)

    return fert_mean, fert_std, weather_norm


# ─────────────────────────────────────────────────────────────
# FULL CROSS-VALIDATION RUN  (used by M5 / M7 driver scripts)
# ─────────────────────────────────────────────────────────────
def run_cv(cfg: DLRunConfig):
    distributed, rank, world_size, local_rank = init_distributed_if_needed()
    seed_everything(GLOBAL_SEED + rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if is_rank0(rank):
        print(f"[{cfg.model_tag}] device={device} world_size={world_size} distributed={distributed}", flush=True)
        os.makedirs(cfg.output_dir, exist_ok=True)
    ddp_barrier(distributed)

    # scalar feature groups from data_prep config (C+P+T+Fagg+Wagg for M5/M7)
    mcfg = dp.MODEL_CONFIGS[cfg.config_name]
    groups = cfg.scalar_groups or mcfg.scalar_groups

    # 1) scan images
    df_imgs = find_all_paired_tifs(cfg.root_images, cfg.ortho_filter, cfg.dem_filter)
    df_imgs["Image_ID"] = (df_imgs["exp_ID"].astype(str) + "_" + df_imgs["plot_num"].astype(str))
    if is_rank0(rank):
        print(f"[{cfg.model_tag}] image pairs: {len(df_imgs)}", flush=True)

    # 2) labeled table (shared loader => same samples/labels/stages as ML)
    df_all = dp.load_samples(cfg.excel_features_path, require_target=True)
    df_all["Image_ID"] = df_all["Image_ID"].astype(str)
    feature_cols = dp.get_scalar_feature_columns(df_all, groups)
    if is_rank0(rank):
        print(f"[{cfg.model_tag}] scalar features ({'+'.join(groups)}): {len(feature_cols)}", flush=True)

    # class weights (inverse frequency)
    vc = df_all[df_all["K_label_idx"] >= 0]["K_label_idx"].value_counts().sort_index()
    counts = np.array([vc.get(i, 1) for i in range(N_CLASSES)], dtype=np.float32)
    cls_weights = torch.from_numpy(1.0 / (counts / counts.sum())).float()
    cls_weights = cls_weights / cls_weights.sum() * N_CLASSES

    # 3) sequence sources (M7 / use_sequences)
    #    fix: load RAW fert doses + RAW weather here. The fert/weather
    #    normalization is NO LONGER fit globally; it is fit per-fold on the
    #    train (fit) rows only, over days <= sampling, inside the fold loop
    #    below (see _fold_seq_norm). This removes the full-season / all-fold
    #    weather-standardization leak the plan flags for the F_seq/W_seq branches.
    fert_lookup = None
    weather_raw, weather_transplant = None, None
    if cfg.use_sequences:
        try:
            fert_lookup, _ = load_fert_lookup(cfg.fert_path)
        except Exception as e:
            if is_rank0(rank):
                print(f"  WARN fert seq unavailable ({e}); fert TCN uses zeros.", flush=True)
        try:
            weather_raw, weather_transplant = load_weather_array(cfg.weather_path, normalize=False)
        except Exception as e:
            if is_rank0(rank):
                print(f"  WARN weather seq unavailable ({e}); weather TCN uses zeros.", flush=True)

    # 4) build image samples for this config
    if cfg.image_config == "closest_date":
        df_s = build_closest_date_samples(df_imgs, df_all)              # M5
    elif cfg.image_config == "on_before_closest":
        df_s = build_on_before_closest_samples(df_imgs, df_all, MIN_IMAGES_PER_SAMPLE)  # M7
    elif cfg.image_config == "single_date":
        df_s = build_single_date_samples(df_imgs, df_all)
    elif cfg.image_config == "interval_dates":
        dbg = os.path.join(cfg.output_dir, "interval_dates_sample_summary.csv")
        df_s = build_interval_dates_samples(df_imgs, df_all, MIN_IMAGES_PER_SAMPLE,
                                            debug_csv_path=(dbg if is_rank0(rank) else None))
    else:
        df_s = build_all_dates_samples(df_imgs, df_all, MIN_IMAGES_PER_SAMPLE)

    df_s = df_s[df_s[TARGET_COL].notna()].copy()
    df_s["value"] = df_s[TARGET_COL].astype(float)
    if "sampling_stage" not in df_s.columns:
        df_s["sampling_stage"] = [infer_sampling_stage(_safe_to_date(r["conc_date"]))
                                  for _, r in df_s.iterrows()]
    feature_cols = [c for c in feature_cols if c in df_s.columns]
    if is_rank0(rank):
        print(f"[{cfg.model_tag}] labeled samples: {len(df_s)} "
              f"(imgs/sample min={df_s['n_imgs'].min()} max={df_s['n_imgs'].max()})", flush=True)

    crop_hw = compute_global_crop_hw(df_imgs, "ortho_path")
    if cfg.crop_cap:
        crop_hw = (min(crop_hw[0], cfg.crop_cap), min(crop_hw[1], cfg.crop_cap))
        if is_rank0(rank):
            print(f"[{cfg.model_tag}] crop_hw capped to {crop_hw} (crop_cap={cfg.crop_cap})", flush=True)
    folds = list(GroupKFold(n_splits=cfg.folds).split(np.arange(len(df_s)), groups=df_s["Image_ID"].values))

    all_summary = []
    for task in cfg.task_modes:
        fold_results = []
        oof_true, oof_pred, oof_stage = [], [], []
        for fold_id, (tr, va) in enumerate(folds, start=1):
            if is_rank0(rank):
                print(f"\n── {cfg.model_tag} | task={task} | fold {fold_id}/{cfg.folds} ──", flush=True)

            # ── NESTED CV ──────────────────────────────────────────
            # Carve a plot-grouped inner early-stop split from the OUTER TRAIN
            # fold. Early stopping + checkpoint selection see ONLY `es_idx`;
            # the outer fold `va` is never touched during training and is used
            # for SCORING ONLY. This removes the dev-set-optimism that inflated
            # the DL metrics relative to the classical models.
            fit_idx, es_idx = _inner_earlystop_split(df_s, tr)
            # sequence fix: fit fert/weather scalers on FIT rows only.
            fold_fert_mean, fold_fert_std, fold_weather = (
                _fold_seq_norm(df_s, fit_idx, fert_lookup, weather_raw)
                if cfg.use_sequences else (0.0, 1.0, None))
            if is_rank0(rank):
                print(f"   nested-CV: train {len(fit_idx)} / early-stop {len(es_idx)}"
                      f" / score {len(va)} (outer) rows", flush=True)

            # NOTE: `es_idx` is passed as train_one_fold's validation arg, so its
            # early stopping / best-checkpoint logic reads the inner split, not
            # the outer fold. train_one_fold itself is UNCHANGED.
            model, stats, mreg_in, mcls_in = train_one_fold(
                cfg, fold_id, df_s, fit_idx, es_idx, device, crop_hw, feature_cols, task,
                cls_weights=cls_weights, distributed=distributed, rank=rank,
                world_size=world_size, local_rank=local_rank,
                fert_lookup=fert_lookup, fert_mean=fold_fert_mean, fert_std=fold_fert_std,
                weather_array=fold_weather, weather_transplant=weather_transplant)

            if is_rank0(rank):
                if getattr(cfg, "save_checkpoints", True):
                    torch.save({"state_dict": unwrap_model(model).state_dict()},
                               os.path.join(cfg.output_dir, f"model_{cfg.model_tag}_{task}_fold{fold_id}.pt"))
                else:
                    print(f"   [disk] save_checkpoints=False -> skipped "
                          f"model_{cfg.model_tag}_{task}_fold{fold_id}.pt", flush=True)
                # Score the UNTOUCHED outer fold — these are the REPORTED metrics.
                tt, tp = _infer_val(model, stats, df_s.iloc[va], cfg, device, task)
                meta = df_s.iloc[va].copy().reset_index(drop=True)
                stages = meta["sampling_stage"].tolist()
                if task == "reg":
                    meta["y_true_reg"], meta["y_pred_reg"] = tt, tp
                    oof_true += list(tt); oof_pred += list(tp); oof_stage += stages
                    fm = regression_metrics(np.array(tt), np.array(tp)) if tt else {}
                else:
                    meta = meta.iloc[:len(tt)].copy()
                    meta["y_true_cls"], meta["y_pred_cls"] = tt, tp
                    oof_true += list(tt); oof_pred += list(tp); oof_stage += stages[:len(tt)]
                    fm = classification_metrics(np.array(tt), np.array(tp)) if tt else {}
                # per-fold metrics now come from the outer fold, not the
                # early-stop split (was: fm = mreg/mcls from train_one_fold).
                fold_results.append({"fold": fold_id, **fm})
                meta.to_csv(os.path.join(
                    cfg.output_dir, f"val_predictions_{cfg.model_tag}_{task}_fold{fold_id}.csv"), index=False)
            ddp_barrier(distributed)

            # free per-fold model/loaders before the next fold (host RAM + VRAM)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if is_rank0(rank):
            df_task = pd.DataFrame(fold_results)
            mcols = [c for c in df_task.columns if c != "fold"]
            mean_row = df_task[mcols].mean().to_frame().T; mean_row.insert(0, "fold", "mean")
            std_row = df_task[mcols].std(ddof=0).to_frame().T; std_row.insert(0, "fold", "std")
            pd.concat([df_task, mean_row, std_row], ignore_index=True).to_csv(
                os.path.join(cfg.output_dir, f"cv_results_{cfg.model_tag}_{task}.csv"), index=False)

            # pooled OOF (+ stage-aware classification from regression)
            row = {"model": cfg.model_tag, "task": task}
            if task == "reg" and oof_true:
                pooled = regression_metrics(np.array(oof_true), np.array(oof_pred))
                _, _, cls_from_reg = dp.stage_aware_classification(oof_true, oof_pred, oof_stage)
                row.update({f"{k}_pooled": v for k, v in pooled.items()})
                row.update({f"{k}_fromreg": v for k, v in (cls_from_reg or {}).items()})
                pd.DataFrame({"sampling_stage": oof_stage, "y_true_reg": oof_true,
                              "y_pred_reg": oof_pred}).to_csv(
                    os.path.join(cfg.output_dir, f"val_predictions_{cfg.model_tag}_{task}_ALL.csv"), index=False)
            elif task == "cls" and oof_true:
                pooled = classification_metrics(np.array(oof_true), np.array(oof_pred))
                row.update({f"{k}_pooled": v for k, v in pooled.items()})
            for k in mcols:
                row[f"{k}_cvmean"] = float(df_task[k].mean())
                row[f"{k}_cvstd"] = float(df_task[k].std(ddof=0))
            all_summary.append(row)
            print(f"[{cfg.model_tag} {task}] CV: " +
                  "  ".join(f"{k}={df_task[k].mean():.3f}±{df_task[k].std(ddof=0):.3f}" for k in mcols), flush=True)

    if is_rank0(rank) and all_summary:
        pd.DataFrame(all_summary).to_csv(
            os.path.join(cfg.output_dir, f"summary_{cfg.model_tag}.csv"), index=False)
        print(f"\n[{cfg.model_tag}] all outputs -> {cfg.output_dir}", flush=True)
    ddp_cleanup(distributed)


def _infer_val(model, stats, df_va, cfg, device, task):
    eval_model = unwrap_model(model)
    eval_model.eval()
    ds = OrthoDemTabularDataset(
        df_va.copy().reset_index(drop=True),
        stats["ortho_mean"], stats["ortho_std"], stats["dem_mean"], stats["dem_std"],
        stats["y_mean"], stats["y_std"], stats["crop_hw"], stats["feature_cols"],
        stats["feat_mean"], stats["feat_std"], use_sequences=cfg.use_sequences,
        fert_lookup=stats["fert_lookup"], fert_mean=stats["fert_mean"], fert_std=stats["fert_std"],
        weather_array=stats["weather_array"], weather_transplant=stats["weather_transplant"],
        max_frames=cfg.max_frames)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                        collate_fn=temporal_collate_fn)
    amp_enabled = bool(cfg.use_amp) and (device.type == "cuda")
    tt, tp = [], []
    with torch.no_grad():
        for xo, xd, fb, xt, sl, yr, yc, fe, we in loader:
            xo, xd, fb, xt, sl = xo.to(device), xd.to(device), fb.to(device), xt.to(device), sl.to(device)
            fe, we = fe.to(device), we.to(device)
            with autocast("cuda", enabled=amp_enabled):
                out_reg, out_cls = eval_model(xo, xd, fb, sl, x_time=xt, x_fert=fe, x_weather=we)
            out_reg = out_reg.float(); out_cls = out_cls.float()
            if task == "reg":
                tp += (out_reg.cpu().numpy().reshape(-1) * stats["y_std"] + stats["y_mean"]).tolist()
                tt += (yr.numpy().reshape(-1) * stats["y_std"] + stats["y_mean"]).tolist()
            else:
                v = yc >= 0
                if v.sum() > 0:
                    tp += torch.argmax(out_cls[v], 1).cpu().numpy().tolist()
                    tt += yc[v].numpy().tolist()
    return tt, tp
