# ============================================================
# data_prep.py — Unified preprocessing for strawberry leaf-K prediction
# ------------------------------------------------------------
# ONE preprocessing layer shared by the classical-ML baselines
# (M1 PLSR, M2 RF, M3/M4 +Fagg/Wagg, M6 RF+Eseq) AND by the
# deep-learning model (M5 CNN fusion, M7 multimodal Transformer).
#
# Why a single module?
#   The manuscript ("Dataset construction and leakage-safe splitting",
#   "Weather/fertilizer/cultivar feature construction", Tables 2.4 & 2.7)
#   defines ONE canonical sample representation and ONE leakage-safe
#   evaluation protocol. Re-implementing it per script invites drift.
#   Everything below mirrors that single description so that ML and DL
#   models are compared on identical samples, labels, folds, and metrics.
#
# Manuscript input groups (Table 2.4) implemented here:
#   I    multispectral image tiles          (DL only — manifest of tiles)
#   D    DEM tiles                          (DL only — manifest of tiles)
#   V    spectral descriptors (VIs/stats)   *_Average / *_StdDev
#   S    structural descriptors             canopy coverage / height / volume
#   C    cultivar identity                  cv_* one-hot
#   P    plot position                      centroid x/y
#   T    timing scalars                     DAP (day) + |conc - uav date diff|
#   Fagg fertilizer scalar summaries        cumulative K (K_total)
#   Wagg weather scalar summaries           GDD / cumulative rain / heat-stress
#                                           / consecutive-dry / acc-solrad ...
#   Fseq daily fertilizer sequence          transplant -> sampling date
#   Wseq daily weather sequence             transplant -> sampling date
#   Eseq TCN-encoded fixed-length embedding of Fseq/Wseq (built downstream)
#
# Model configurations (Table 2.7):
#   M1 PLSR  : V+S+C+P+T
#   M2 RF    : V+S+C+P+T
#   M3 PLSR  : V+S+C+P+T+Fagg+Wagg
#   M4 RF    : V+S+C+P+T+Fagg+Wagg
#   M5 DL    : I+D+C+P+T+Fagg+Wagg          (learned image reps replace V,S)
#   M6 RF    : V+S+C+P+T+Fagg+Wagg+Eseq
#   M7 DL    : I_seq+D_seq+C+P+T+Fagg+Wagg+Fseq+Wseq  (temporal fusion)
#
# Leakage-safe protocol (shared):
#   * pre-sampling policy — only info on/before the leaf-sampling date
#   * 5-fold GroupKFold on Image_ID (a plot never splits across folds)
#   * scalers / sequence stats fit on the TRAIN fold only
#
# Standalone core depends only on numpy / pandas / scikit-learn.
# Sequence builders (Fseq/Wseq) and the DL sample index read extra daily
# files when their paths are provided, and degrade gracefully otherwise.
# ============================================================

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:  # only needed for stage-aware classification metrics
    from sklearn.metrics import balanced_accuracy_score, f1_score
    _HAS_SK_METRICS = True
except Exception:  # pragma: no cover
    _HAS_SK_METRICS = False


# ─────────────────────────────────────────────────────────────
# 0.  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
FOLDS       = 5
GROUP_COL   = "Image_ID"
TARGET_COL  = "K (%)"
DATE_COL    = "conc_date"          # leaf-sampling date, used for stage labels

# Outlier FLIGHT rows dropped (one specific UAV flight), matched on
# (Image_ID, htp_date). Applied BEFORE the closest-flight selection, so the
# conc-date target survives and its next-closest flight is used instead.
# NOTE: the entries below were specific to the 2023-24 NITROGEN dataset and have
# been cleared for the 2024-25 POTASSIUM dataset. Repopulate with any K/24-25
# flight outliers you identify (see 2_analysis outlier-detection step).
ROWS_TO_DROP: List[Tuple[str, int]] = []

# Outlier TARGET rows dropped (a whole tissue-K measurement = one conc date on
# one plot, all its flights), matched on (Image_ID, conc_date). Use this to
# remove an outlier sample entirely. Example: ("SB3_24", 20250229)
# NOTE: cleared for the 2024-25 potassium dataset — repopulate as needed.
CONC_ROWS_TO_DROP: List[Tuple[str, int]] = []

# Per-season transplant date (DAP 0). Used to bound the daily sequences.
TRANSPLANT_DATES: Dict[str, date] = {
    "2023_2024": date(2023, 10, 16),
    "2024_2025": date(2024, 10, 15),
    "2025_2026": date(2025, 10, 8),
}


# ─────────────────────────────────────────────────────────────
# 1.  STAGE-AWARE POTASSIUM LABELS
#     (identical thresholds across M1..M7 so the secondary
#      classification interpretation is comparable everywhere)
# ─────────────────────────────────────────────────────────────
CLASS_NAMES  = ["deficient", "adequate", "high"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
N_CLASSES    = 3   # number of classes (3), not nitrogen-specific

# Stage-specific deficiency / sufficiency thresholds for POTASSIUM, K (%)
# (strawberry, MRM leaf — critical/adequate-range table, "macronutrient_value").
# Convention: deficient if K < def_low; high if K > high_low; adequate in between.
#   def_low  = lower bound of the adequate range (= the deficiency/critical value)
#   high_low = upper bound of the adequate range (above it = high/excessive)
K_STAGE_THRESHOLDS = {
    "transplants":     {"def_low": 1.5, "high_low": 3.0},
    "initial_flower":  {"def_low": 1.5, "high_low": 3.0},
    "initial_harvest": {"def_low": 1.5, "high_low": 2.5},
    "midseason":       {"def_low": 1.1, "high_low": 2.5},
    "end_of_season":   {"def_low": 1.1, "high_low": 2.0},
}


def to_date(v) -> Optional[date]:
    """Accept int YYYYMMDD, str, datetime, Timestamp, or NaN -> datetime.date."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(int(v)) if isinstance(v, (int, np.integer, float)) else str(v)
    s = s.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def infer_sampling_stage(conc_date_val) -> Optional[str]:
    """Map a leaf-sampling date to a phenological stage (3 seasons supported)."""
    d = to_date(conc_date_val)
    if d is None:
        return None
    # 2023-24 season
    if   date(2023, 10, 16) <= d <= date(2023, 10, 30): return "transplants"
    elif date(2023, 10, 31) <= d <= date(2023, 11, 29): return "initial_flower"
    elif date(2023, 11, 30) <= d <= date(2023, 12, 31): return "initial_harvest"
    elif date(2024, 1, 1)   <= d <= date(2024, 2, 14):  return "midseason"
    elif date(2024, 2, 15)  <= d <= date(2024, 3, 31):  return "end_of_season"
    # 2024-25 season
    elif date(2024, 10, 15) <= d <= date(2024, 10, 19): return "transplants"
    elif date(2024, 10, 20) <= d <= date(2024, 11, 18): return "initial_flower"
    elif date(2024, 11, 19) <= d <= date(2024, 12, 31): return "initial_harvest"
    elif date(2025, 1, 1)   <= d <= date(2025, 2, 14):  return "midseason"
    elif date(2025, 2, 15)  <= d <= date(2025, 3, 31):  return "end_of_season"
    # 2025-26 season
    elif date(2025, 10, 8)  <= d <= date(2025, 10, 19): return "transplants"
    elif date(2025, 10, 20) <= d <= date(2025, 11, 19): return "initial_flower"
    elif date(2025, 11, 20) <= d <= date(2025, 12, 31): return "initial_harvest"
    elif date(2026, 1, 1)   <= d <= date(2026, 2, 14):  return "midseason"
    elif date(2026, 2, 15)  <= d <= date(2026, 3, 31):  return "end_of_season"
    return None


def classify_k_value(k_value: float, stage: Optional[str]) -> int:
    """Stage-aware 3-class potassium label. Returns -1 when stage/value is unusable."""
    if stage is None or stage not in K_STAGE_THRESHOLDS:
        return -1
    if k_value is None or (isinstance(k_value, float) and math.isnan(k_value)):
        return -1
    t = K_STAGE_THRESHOLDS[stage]
    if k_value < t["def_low"]:
        return CLASS_TO_IDX["deficient"]
    if k_value > t["high_low"]:
        return CLASS_TO_IDX["high"]
    return CLASS_TO_IDX["adequate"]


def class_idx_to_name(i: int) -> str:
    i = int(i)
    return CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else "unknown"


# ─────────────────────────────────────────────────────────────
# 2.  METRICS  (one definition used by every model)
# ─────────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    r2   = float(1.0 - ss_res / ss_tot)
    smape = float(100 * np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "sMAPE": smape}


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {"Acc": float(np.mean(y_true == y_pred))}
    if _HAS_SK_METRICS:
        out["BalAcc"]      = float(balanced_accuracy_score(y_true, y_pred))
        out["F1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        out["F1_macro"]    = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return out


# ─────────────────────────────────────────────────────────────
# 3.  COLUMN RESOLUTION  (robust to ML- and DL-file naming)
#     The ML file uses sanitized names ('60cm_T_avg_C', 'acc_rain_tot_mm',
#     'Centroid_X', 'day'); the DL labeled file uses raw names
#     ('60cm T avg (C)', 'total_rain_mm', 'centroid_x', 'DAP').
#     We resolve each logical feature to whichever name is present.
# ─────────────────────────────────────────────────────────────
def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def _resolve_one(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# Logical timing / management columns, each with the alternative names seen
# across the two source files.
_TIMING_DAP   = ["day", "DAP"]
_TIMING_DIFF  = ["abs_conc_minus_uav_date_diff", "conc_minus_uav_date_diff",
                 "conc_uav_diff"]
_FAGG_COLS    = ["K_total"]
_WAGG_COLS    = [  # cumulative / summary weather scalars (either naming)
    "total_temp_avg", "total_rain_mm", "total_solRad_w_m2",
    "gdd_tbase5", "gdd_tbase7", "gdd_tbase10",
    "heat_stress_days", "max_consecutive_dry_days",
]
_STRUCT_COLS  = [
    "Canopy_Coverage_pixel", "Canopy_Coverage_cm_2",
    "Relative_Average_Height_Top_5_cm", "Relative_Volume_cm_3",
    # raw-name variants
    "Canopy Coverage pixel", "Canopy Coverage (cm^2)",
    "Relative Average Height (Top 5%) (cm)", "Relative Volume (cm^3)",
]
_POS_COLS     = ["Centroid_X", "Centroid_Y", "centroid_x", "centroid_y"]


def build_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return the manuscript input groups present as columns in ``df``.

    Groups requiring image tiles (I, D) are NOT column-based and are handled
    by the DL sample index (see build_dl_sample_index); they are returned as
    empty lists here so callers can rely on a stable key set.
    """
    g: Dict[str, List[str]] = {}

    # V — spectral descriptors / vegetation indices
    g["V"] = [c for c in df.columns if c.endswith("_Average") or c.endswith("_StdDev")]
    # S — structural descriptors
    g["S"] = _first_present(df, _STRUCT_COLS)
    # C — cultivar identity (one-hot)
    g["C"] = [c for c in df.columns if c.startswith("cv_")]
    # P — plot position
    g["P"] = _first_present(df, _POS_COLS)
    # T — timing scalars (prefer absolute date-diff; keep exactly one diff col)
    dap  = _resolve_one(df, _TIMING_DAP)
    diff = _resolve_one(df, _TIMING_DIFF)
    g["T"] = [c for c in (dap, diff) if c is not None]
    # Fagg / Wagg — scalar fertilizer & weather summaries
    g["Fagg"] = _first_present(df, _FAGG_COLS)
    g["Wagg"] = _first_present(df, _WAGG_COLS)
    # I / D — image-tile groups (DL only; not columns)
    g["I"] = []
    g["D"] = []
    return g


# ─────────────────────────────────────────────────────────────
# 4.  MODEL CONFIG REGISTRY  (Table 2.7)
#     scalar_groups : column-based groups assembled into the feature matrix
#     image         : whether the model consumes image tiles  (I, D)
#     sequence      : whether the model consumes daily sequences (Fseq, Wseq)
#     kind          : 'ml' or 'dl'  (drives which builder you call)
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    kind: str                       # 'ml' | 'dl'
    scalar_groups: Tuple[str, ...]  # column-based groups for the feature matrix
    image: bool = False             # uses I + D tiles
    sequence: bool = False          # uses Fseq + Wseq daily sequences
    note: str = ""


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "M1": ModelConfig("M1", "PLSR", "ml", ("V", "S", "C", "P", "T")),
    "M2": ModelConfig("M2", "RF",   "ml", ("V", "S", "C", "P", "T")),
    "M3": ModelConfig("M3", "PLSR", "ml", ("V", "S", "C", "P", "T", "Fagg", "Wagg")),
    "M4": ModelConfig("M4", "RF",   "ml", ("V", "S", "C", "P", "T", "Fagg", "Wagg")),
    "M5": ModelConfig("M5", "CNN_fusion", "dl",
                      ("C", "P", "T", "Fagg", "Wagg"), image=True,
                      note="learned image reps (I,D) replace handcrafted V,S"),
    "M6": ModelConfig("M6", "RF", "ml",
                      ("V", "S", "C", "P", "T", "Fagg", "Wagg"), sequence=True,
                      note="adds Eseq = TCN-encoded Fseq/Wseq embeddings"),
    "M7": ModelConfig("M7", "Transformer", "dl",
                      ("C", "P", "T", "Fagg", "Wagg"), image=True, sequence=True,
                      note="temporal multimodal fusion: I_seq,D_seq,Fseq,Wseq + scalars"),
}


def get_scalar_feature_columns(df: pd.DataFrame, groups: Sequence[str]) -> List[str]:
    """Assemble the (de-duplicated, order-preserving) scalar feature column
    list for the requested groups."""
    gdef = build_feature_groups(df)
    cols: List[str] = []
    for g in groups:
        cols += gdef.get(g, [])
    return list(dict.fromkeys(cols))


# ─────────────────────────────────────────────────────────────
# 5.  DATA LOADING  (single source of truth for the sample table)
# ─────────────────────────────────────────────────────────────
SIGNED_DIFF_CANDIDATES = ("conc_minus_uav_date_diff", "conc_uav_diff")


def select_closest_flight(
    df: pd.DataFrame,
    group_col: str = GROUP_COL,
    date_col: str = DATE_COL,
    diff_candidates: Sequence[str] = SIGNED_DIFF_CANDIDATES,
) -> pd.DataFrame:
    """Collapse the table to ONE row per unique (Image_ID, conc_date) target.

    The merged table carries one row per (plot, conc_date, UAV-flight) triple,
    so a single tissue-K measurement (the label) is repeated across every flight
    near it. The manuscript predicts each *unique concentration date*, using the
    CLOSEST flight to that date. We therefore keep, per (Image_ID, conc_date):

        the flight with the smallest |conc - uav date diff|,
        breaking ties in favour of the flight BEFORE the conc date.

    Sign convention in this dataset: ``conc_minus_uav_date_diff`` is NEGATIVE
    when the flight precedes the conc date (e.g. a flight 7 days before -> -7,
    7 days after -> +7). So among |diff| ties we prefer the more negative value
    (the earlier / pre-sampling flight). Selection key: (|diff| asc, diff asc).
    """
    signed = next((c for c in diff_candidates if c in df.columns), None)
    if signed is None or date_col not in df.columns or group_col not in df.columns:
        return df  # cannot determine closeness -> leave untouched
    d = df.copy()
    d["_signed"] = pd.to_numeric(d[signed], errors="coerce")
    d["_abs"] = d["_signed"].abs()
    # stable sort so that, within |diff| ties, the more negative (earlier) wins
    d = d.sort_values([group_col, date_col, "_abs", "_signed"], kind="mergesort")
    d = d.drop_duplicates(subset=[group_col, date_col], keep="first")
    return d.drop(columns=["_signed", "_abs"]).reset_index(drop=True)


def load_samples(
    excel_path: str,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
    group_col: str = GROUP_COL,
    rows_to_drop: Sequence[Tuple[str, int]] = tuple(ROWS_TO_DROP),
    conc_rows_to_drop: Sequence[Tuple[str, int]] = tuple(CONC_ROWS_TO_DROP),
    require_target: bool = True,
    one_per_conc: bool = True,
) -> pd.DataFrame:
    """Load the merged plot-sampling table and attach stage + class labels.

    Steps (mirrors the manuscript dataset-construction section):
      1. read xlsx, strip column whitespace
      2. add abs timing-diff scalar if a signed one exists
      3. drop configured outlier rows (Image_ID, htp_date)
      4. require a valid regression target (optional)
      5. collapse to one row per unique (Image_ID, conc_date) = closest flight
         (one_per_conc=True; the manuscript predicts each unique conc date)
      6. attach sampling_stage + stage-aware K_label_idx
    """
    df = pd.read_excel(excel_path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df = df.copy()  # defragment internal memory blocks after Excel read

    # coerce the regression target to numeric so missing-value placeholders
    # (e.g. the string ".") become real NaN instead of surviving notna() and
    # later crashing float(); mirrors manuscript "valid target" definition
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # absolute timing-offset scalar
    for signed in ("conc_minus_uav_date_diff", "conc_uav_diff"):
        if signed in df.columns and "abs_conc_minus_uav_date_diff" not in df.columns:
            df["abs_conc_minus_uav_date_diff"] = pd.to_numeric(
                df[signed], errors="coerce"
            ).abs()
            break

    # drop outlier rows matched on (Image_ID, htp_date)
    if rows_to_drop and "htp_date" in df.columns and group_col in df.columns:
        drop_set = {(str(a), int(b)) for a, b in rows_to_drop}

        def _is_drop(r):
            try:
                return (str(r[group_col]), int(r["htp_date"])) in drop_set
            except Exception:
                return False

        df = df[~df.apply(_is_drop, axis=1)].copy()

    # drop whole outlier targets matched on (Image_ID, conc_date)
    if conc_rows_to_drop and date_col in df.columns and group_col in df.columns:
        cdrop = {(str(a), int(b)) for a, b in conc_rows_to_drop}

        def _is_conc_drop(r):
            try:
                return (str(r[group_col]), int(r[date_col])) in cdrop
            except Exception:
                return False

        df = df[~df.apply(_is_conc_drop, axis=1)].copy()

    if require_target:
        df = df[df[target_col].notna()].copy()

    # collapse to one row per unique (Image_ID, conc_date): the closest flight
    if one_per_conc:
        df = select_closest_flight(df, group_col=group_col, date_col=date_col)

    df = df.reset_index(drop=True)

    # stage + stage-aware class label
    if date_col in df.columns:
        df["sampling_stage"] = df[date_col].apply(infer_sampling_stage)
    else:
        df["sampling_stage"] = None
    df["K_label_idx"] = [
        classify_k_value(float(n) if pd.notna(n) else float("nan"), s)
        for n, s in zip(df[target_col], df["sampling_stage"])
    ]
    return df


# ─────────────────────────────────────────────────────────────
# 6.  LEAKAGE-SAFE SPLITTING  (plot-grouped, shared by ML & DL)
# ─────────────────────────────────────────────────────────────
def group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = FOLDS,
    group_col: str = GROUP_COL,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """5-fold GroupKFold on plot identity. A plot never appears in both
    train and validation within a fold (manuscript leakage-safe protocol)."""
    groups = df[group_col].astype(str).values
    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold needs an X with the right number of rows
    return list(gkf.split(np.zeros((len(df), 1)), groups=groups))


def assert_no_group_leakage(
    df: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    group_col: str = GROUP_COL,
) -> None:
    """Raise if any plot appears in both train and val of any fold."""
    groups = df[group_col].astype(str).values
    for k, (tr, va) in enumerate(splits, 1):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f"Group leakage detected in fold {k}")


# ─────────────────────────────────────────────────────────────
# 7.  ML PATH — fixed-length feature matrix for PLSR / RF / etc.
# ─────────────────────────────────────────────────────────────
@dataclass
class MLDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_cols: List[str]
    stages: List[Optional[str]]
    splits: List[Tuple[np.ndarray, np.ndarray]]
    df: pd.DataFrame
    config: ModelConfig

    def summary(self) -> str:
        return (f"[{self.config.name} {self.config.model}] "
                f"samples={len(self.df)} plots={pd.Series(self.groups).nunique()} "
                f"features={len(self.feature_cols)} folds={len(self.splits)}")


def build_ml_dataset(
    df: pd.DataFrame,
    config: ModelConfig,
    target_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
    n_splits: int = FOLDS,
) -> MLDataset:
    """Assemble X / y / groups / folds for a classical-ML configuration.

    Note: Fagg/Wagg are included automatically via config.scalar_groups.
    For M6 (Eseq) the caller appends the TCN sequence embeddings to X *inside
    each fold* (train-fold-fit encoder), so they are NOT added here — keeping
    this builder leakage-safe and model-agnostic.
    """
    if config.kind != "ml":
        raise ValueError(f"{config.name} is a '{config.kind}' model; "
                         f"use build_dl_sample_index() instead.")
    feat_cols = get_scalar_feature_columns(df, config.scalar_groups)
    if not feat_cols:
        raise ValueError(f"No feature columns resolved for groups "
                         f"{config.scalar_groups}.")
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    y = df[target_col].astype(np.float64).values
    groups = df[group_col].astype(str).values
    stages = df["sampling_stage"].tolist()
    splits = group_kfold_splits(df, n_splits=n_splits, group_col=group_col)
    assert_no_group_leakage(df, splits, group_col=group_col)
    return MLDataset(X, y, groups, feat_cols, stages, splits, df, config)


def fit_fold_scaler(X_train: np.ndarray) -> StandardScaler:
    """Standardizer fit on the TRAIN fold only (manuscript: PLSR uses
    standardized predictors; scaling fit on training fold)."""
    return StandardScaler().fit(X_train)


# ─────────────────────────────────────────────────────────────
# 8.  DL PATH — per-sample multimodal index (images + sequences)
# ─────────────────────────────────────────────────────────────
@dataclass
class DLSample:
    image_id: str
    conc_date: date
    transplant_date: Optional[date]
    dap: Optional[int]
    stage: Optional[str]
    y_reg: float
    y_cls: int
    scalar_cols: List[str]
    scalars: np.ndarray
    image_dates: List[date]          # I/D tiles with image_date <= conc_date
    fert_seq: Optional[np.ndarray]   # Fseq  (n_days,) or (n_days, n_fert_vars)
    weather_seq: Optional[np.ndarray]  # Wseq (n_days, n_weather_vars)


def _season_of(d: date) -> Optional[str]:
    for name, t0 in TRANSPLANT_DATES.items():
        # season runs roughly Oct(t0) -> Sep next year
        if t0 <= d <= date(t0.year + 1, 9, 30):
            return name
    return None


def _transplant_for(d: date) -> Optional[date]:
    s = _season_of(d)
    return TRANSPLANT_DATES.get(s) if s else None


def _load_daily_table(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _slice_daily_sequence(
    daily: pd.DataFrame,
    image_id: str,
    transplant: date,
    sampling: date,
    id_col: str,
    date_col: Optional[str],
    value_cols: Optional[Sequence[str]],
) -> Optional[np.ndarray]:
    """Return the daily values for one plot over transplant..sampling (inclusive),
    truncated at the sampling date (pre-sampling policy)."""
    if daily is None:
        return None
    sub = daily
    if id_col in sub.columns:
        sub = sub[sub[id_col].astype(str) == str(image_id)]
    if sub.empty:
        return None
    if date_col and date_col in sub.columns:
        dts = sub[date_col].apply(to_date)
        keep = [(dd is not None) and (transplant <= dd <= sampling) for dd in dts]
        sub = sub[pd.Series(keep, index=sub.index).values]
        if value_cols:
            cols = [c for c in value_cols if c in sub.columns]
        else:
            cols = [c for c in sub.columns if c not in (id_col, date_col)]
        if sub.empty or not cols:
            return None
        return sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    # wide layout: one row per plot, day columns day_1..day_N
    n_days = (sampling - transplant).days + 1
    daycols = [c for c in sub.columns if str(c).lower().startswith(("day", "d"))]
    daycols = daycols[:max(0, n_days)]
    if not daycols:
        return None
    return sub[daycols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64).ravel()


def build_dl_sample_index(
    df: pd.DataFrame,
    config: ModelConfig,
    image_dates_by_plot: Optional[Dict[str, List[date]]] = None,
    fert_daily_path: Optional[str] = None,
    weather_daily_path: Optional[str] = None,
    fert_id_col: str = "Image_ID",
    fert_date_col: Optional[str] = None,
    fert_value_cols: Optional[Sequence[str]] = None,
    weather_id_col: str = "Image_ID",
    weather_date_col: Optional[str] = None,
    weather_value_cols: Optional[Sequence[str]] = None,
    target_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
    date_col: str = DATE_COL,
) -> List[DLSample]:
    """Build the per-sample multimodal index for a DL configuration (M5/M7).

    For each plot-sampling event this records:
      * the scalar covariates (C, P, T, Fagg, Wagg) used by the fusion trunk
      * the ordered list of UAV image dates with image_date <= conc_date
        (the temporal image sequence; pre-sampling policy)
      * the truncated daily fertilizer / weather sequences (Fseq/Wseq), if the
        daily source files are supplied

    `image_dates_by_plot` maps Image_ID -> sorted list of acquisition dates.
    When omitted, image_dates is left empty (fill it from your tile manifest).
    Sequence files are optional; missing inputs yield None for that modality.
    """
    scalar_cols = get_scalar_feature_columns(df, config.scalar_groups)
    fert_daily    = _load_daily_table(fert_daily_path) if config.sequence else None
    weather_daily = _load_daily_table(weather_daily_path) if config.sequence else None

    samples: List[DLSample] = []
    for _, row in df.iterrows():
        iid = str(row[group_col])
        cd = to_date(row[date_col])
        transplant = _transplant_for(cd) if cd else None
        dap = (cd - transplant).days if (cd and transplant) else None

        # temporal image sequence: tiles on/before the sampling date
        img_dates: List[date] = []
        if image_dates_by_plot and iid in image_dates_by_plot and cd is not None:
            img_dates = sorted(d for d in image_dates_by_plot[iid] if d <= cd)

        fert_seq = weather_seq = None
        if config.sequence and cd is not None and transplant is not None:
            fert_seq = _slice_daily_sequence(
                fert_daily, iid, transplant, cd,
                fert_id_col, fert_date_col, fert_value_cols)
            weather_seq = _slice_daily_sequence(
                weather_daily, iid, transplant, cd,
                weather_id_col, weather_date_col, weather_value_cols)

        scalars = (row[scalar_cols].apply(pd.to_numeric, errors="coerce")
                   .astype(np.float64).values if scalar_cols else np.array([]))

        samples.append(DLSample(
            image_id=iid, conc_date=cd, transplant_date=transplant, dap=dap,
            stage=row.get("sampling_stage"),
            y_reg=float(row[target_col]) if pd.notna(row[target_col]) else float("nan"),
            y_cls=int(row["K_label_idx"]) if "K_label_idx" in row else -1,
            scalar_cols=scalar_cols, scalars=scalars,
            image_dates=img_dates, fert_seq=fert_seq, weather_seq=weather_seq,
        ))
    return samples


# ─────────────────────────────────────────────────────────────
# 9.  STAGE-AWARE SCORING OF REGRESSION OUTPUTS
#     (secondary agronomic interpretation, shared by ML & DL)
# ─────────────────────────────────────────────────────────────
def stage_aware_classification(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    stages: Sequence[Optional[str]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Convert continuous K% predictions+truth into stage-aware classes and
    score them. Returns (true_cls, pred_cls, metrics)."""
    true_cls = np.array([classify_k_value(float(t), s) for t, s in zip(y_true, stages)])
    pred_cls = np.array([classify_k_value(float(p), s) for p, s in zip(y_pred, stages)])
    valid = (true_cls >= 0) & (pred_cls >= 0)
    metrics = classification_metrics(true_cls[valid], pred_cls[valid]) if valid.sum() else {}
    return true_cls, pred_cls, metrics


# ─────────────────────────────────────────────────────────────
# 10.  CONVENIENCE: one-call dataset for any config name
# ─────────────────────────────────────────────────────────────
def prepare(
    excel_path: str,
    config_name: str,
    **dl_kwargs,
):
    """High-level entry point.

    Returns an MLDataset for ML configs (M1-M4, M6) or a list[DLSample] for DL
    configs (M5, M7). The same loaded/labeled/split-ready sample table backs
    both, so every model sees identical samples and labels.
    """
    if config_name not in MODEL_CONFIGS:
        raise KeyError(f"Unknown config '{config_name}'. "
                       f"Choose from {sorted(MODEL_CONFIGS)}.")
    cfg = MODEL_CONFIGS[config_name]
    df = load_samples(excel_path)
    if cfg.kind == "ml":
        return build_ml_dataset(df, cfg)
    return build_dl_sample_index(df, cfg, **dl_kwargs)


if __name__ == "__main__":
    # Smoke test against the combined ML file if present.
    here = os.path.dirname(os.path.abspath(__file__))
    xlsx = os.path.join(os.path.dirname(here), "data", "24_25_K",
                        "trait_K_weather_yield_combined_conc.xlsx")
    if os.path.exists(xlsx):
        df = load_samples(xlsx)
        print(f"Loaded {len(df)} samples / {df[GROUP_COL].nunique()} plots")
        g = build_feature_groups(df)
        for k in ["V", "S", "C", "P", "T", "Fagg", "Wagg"]:
            print(f"  {k:5s}: {len(g[k]):3d} cols  {g[k][:4]}")
        for name in ["M1", "M2", "M3", "M4", "M6"]:
            ds = build_ml_dataset(df, MODEL_CONFIGS[name])
            print("  ", ds.summary())
    else:
        print("Combined xlsx not found; module import OK.")
