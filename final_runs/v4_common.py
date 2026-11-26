"""
v4_common.py: shared setup and patches for the final analysis runs.

Imports the base pipeline READ-ONLY and changes behaviour only through
in-process monkeypatches. No base pipeline file is edited, and every output goes to
outputs/<nutrient>/<run_name>/.

Variant flags (one token each in the run name):
  seqfix   TCN sequence branch fixed (M6 and M7):
             - padded steps are marked NaN and zeroed inside every TCN layer, and
               sequences are LEFT-padded so the read-out step is the sample's own
               last day. Embeddings no longer depend on the other samples in the
               batch (base pipeline: right zero padding + read-out of the last step).
             - TCN levels 4 -> 6, so the receptive field grows from 61 to 253 days
               and covers the whole season (base pipeline only saw the last 61 days).
  (all v4 runs except no-flag / --nested-only reproduction runs) the 8 weather
           aggregates are recomputed from the transplant date (fixes the 4-day-early
           2023-24 window) with the Methods definitions: mean daily temperature and
           the longest dry spell so far (base pipeline: running temperature sum, current spell).
  sampEnd  every non-image input ends at the LEAF-SAMPLING date instead of the matched
           UAV flight date: fertilizer/weather sequences, cumulative fertilizer (Fagg),
           the 8 weather aggregates (Wagg, recomputed from the daily FAWN file counted
           from transplant; this also fixes a 4-day shift in the 2023-24 Wagg) and the
           DAP timing feature. Only the image (and the |sampling - flight| offset that
           describes it) keeps the flight date.
  noXY     plot centroid X/Y removed (group P).
  noUAV    UAV inputs removed: V+S for RF/PLSR; image branch for M7.
  single   M7 uses only the matched acquisition (max_frames = 1).
  noW      weather sequence removed (weather TCN gets zeros).
  nested   M6: TCN encoder refit inside every inner RF-tuning split.
  noF      no fertilizer or weather inputs at all (Fagg, Wagg and both sequences off):
           M7 with images + cultivar + position + timing only, i.e. the case where
           management records are missing (commercial-field argument).
  lstm     M7 with a bidirectional LSTM over the image dates (Table S3).
  noTemp   M7 without the temporal image encoder (Table S3).
  treat    treatment-held-out CV instead of plot-held-out CV (exploratory sensitivity
           check; not used for any result in the article).
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import legacy_groupkfold
_SK_GROUPKFOLD = legacy_groupkfold.install()   # fold rule of the article (see legacy_groupkfold.py)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))           # <repo>/final_runs
BASE = os.path.dirname(CODE_DIR)                                # repository root
DATA_ROOT = os.path.join(BASE, "data")                          # Zenodo sample-dataset layout
OUT_ROOT = os.environ.get("NUTRINET_OUT", os.path.join(BASE, "outputs"))
# season folder and modelling table of each season (names as in the Zenodo deposit)
SEASON_DIRS = {"23_24": "23_24_N", "24_25": "24_25_K", "25_26": "25_26_K"}
TABLES = {"23_24": "trait_N_weather_yield_combined_conc.xlsx",
          "24_25": "trait_K_weather_yield_combined_conc.xlsx",
          "25_26": "trait_K_weather_yield_combined_conc.xlsx"}
EXT_CODE_DIR = os.path.join(BASE, "code_25_26_external")

NUTRIENTS: Dict[str, dict] = {
    "N": dict(code_dir="code_23_24_N", season="23_24",
              transplant=date(2023, 10, 16), fagg_col="N_total",
              # the manuscript's N M7 is the seqwin run: frames on/after 2023-12-05
              min_image_date=date(2023, 12, 5)),
    "K": dict(code_dir="code_24_25_K", season="24_25",
              transplant=date(2024, 10, 15), fagg_col="K_total",
              min_image_date=None),
}
TOKENS = ["seqfix", "sampEnd", "noXY", "noUAV", "single", "noW", "nested", "treat",
          "noF", "lstm", "noTemp"]
# transplant date (DAP 0) of every season; the season of a table is read from its
# sampling dates, so 2025-26 external-test rows get their own transplant, fert and
# weather files.
SEASONS: Dict[str, date] = {"23_24": date(2023, 10, 16), "24_25": date(2024, 10, 15),
                            "25_26": date(2025, 10, 8)}
SEASON_ENDS: Dict[str, date] = {"23_24": date(2024, 6, 30), "24_25": date(2025, 6, 30),
                                "25_26": date(2026, 6, 30)}
TCN_LEVELS_V4 = 6


@dataclass
class Variant:
    seqfix: bool = False
    sampEnd: bool = False
    noXY: bool = False
    noUAV: bool = False
    single: bool = False
    noW: bool = False
    nested: bool = False
    treat: bool = False
    noF: bool = False
    lstm: bool = False
    noTemp: bool = False

    @classmethod
    def from_args(cls, args) -> "Variant":
        return cls(**{t: bool(getattr(args, t, False)) for t in TOKENS})

    @property
    def fix_wagg(self) -> bool:
        """v4 Wagg (correct 2023-24 window + Methods definitions) for every run except
        the base pipeline-reproduction runs (no flags, or --nested only), which keep base pipeline inputs."""
        return any(getattr(self, t) for t in TOKENS if t != "nested")

    def tag(self) -> str:
        on = [t for t in TOKENS if getattr(self, t)]
        return "_".join(on) if on else "base"


# ─────────────────────────────────────────────────────────────
# treatment-held-out folds (--treat)
# ─────────────────────────────────────────────────────────────
def treatment_table(nut: str) -> pd.DataFrame:
    """Image_ID -> block, cultivar, treatment number, treatment id (from the fert file)."""
    f = pd.read_excel(fert_path(nut), sheet_name="Combined")
    f.columns = [str(c).strip() for c in f.columns]
    t = pd.DataFrame({"Image_ID": f["plot_Num"].astype(str).str.strip()})
    t["blk"] = t["Image_ID"].str.split("_").str[0]
    t["cv"] = f["Cv"].astype(str).str.strip()
    t["trt"] = f["trt_Num"].astype(str).str.strip()
    t["bc"] = t["blk"] + "|" + t["cv"]
    t["tid"] = t["bc"] + "|" + t["trt"]
    return t.set_index("Image_ID")


def make_treatment_kfold(nut: str, seed: int = 42):
    """Returns a GroupKFold drop-in whose split(X, y, groups) takes PLOT ids (as all
    base pipeline callers pass) and splits by TREATMENT. With a full design (5 treatments
    in every block x cultivar present and n_splits == 5) fold k holds out one random
    treatment per block x cultivar; otherwise (inner splits on a training subset) it
    is a plain GroupKFold on treatment id."""
    _SkGKF = _SK_GROUPKFOLD
    tab = treatment_table(nut)
    rng = np.random.RandomState(seed)
    fold_of_tid = {}
    for bc, g in tab.groupby("bc"):
        tids = sorted(g["tid"].unique())
        for k, tid in enumerate(rng.permutation(tids)):
            fold_of_tid[tid] = k

    class TreatmentGroupKFold:
        def __init__(self, n_splits=5, **_):
            self.n_splits = n_splits

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

        def split(self, X, y=None, groups=None):
            ids = pd.Series(np.asarray(groups)).astype(str).str.strip()
            miss = sorted(set(ids) - set(tab.index))
            if miss:
                raise KeyError(f"[v4 treat] plot ids without a treatment: {miss[:5]}")
            tid = tab.loc[ids, "tid"].to_numpy()
            per_bc = tab.loc[ids].groupby("bc")["tid"].nunique()
            idx = np.arange(len(tid))
            if self.n_splits == 5 and (per_bc == 5).all():
                fold = np.array([fold_of_tid[t] for t in tid])
                for k in range(5):
                    yield idx[fold != k], idx[fold == k]
            else:
                yield from _SkGKF(n_splits=self.n_splits).split(idx, groups=tid)

    return TreatmentGroupKFold


def patch_treatfold(dp, dc, nut: str):
    """Route every GroupKFold in data_prep / dl_core / ml_runner (which imports it
    from sklearn inside its factories) through the treatment splitter."""
    import sklearn.model_selection as skms
    cls = make_treatment_kfold(nut)
    dp.GroupKFold = cls
    skms.GroupKFold = cls
    if dc is not None:
        dc.GroupKFold = cls
    print("[v4] treat: TREATMENT-held-out CV (one rate per block x cultivar per fold)", flush=True)
    return cls


def add_variant_args(ap):
    ap.add_argument("--nutrient", required=True, choices=list(NUTRIENTS))
    for t in TOKENS:
        ap.add_argument(f"--{t}", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the data and print the configuration only")


def run_dir(nut: str, model: str, v: Variant, seed: Optional[int] = None,
            external: bool = False) -> str:
    """outputs/<nut>/<model>__<tag>          within-season CV
       outputs/<nut>_multiseed/<..>__seed<s> extra seeds (kept apart from the
                                                paired-test discovery)
       outputs/<nut>_ext2526/<model>__<tag>  train 24-25 -> test 25-26"""
    if external:
        d = os.path.join(OUT_ROOT, f"{nut}_ext2526", f"{model}__{v.tag()}")
    elif seed is not None:
        d = os.path.join(OUT_ROOT, f"{nut}_multiseed", f"{model}__{v.tag()}__seed{seed}")
    else:
        d = os.path.join(OUT_ROOT, nut, f"{model}__{v.tag()}")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────
# import the base pipeline modules for one nutrient (one per process)
# ─────────────────────────────────────────────────────────────
def setup(nut: str, need_dl: bool):
    code = os.path.join(BASE, NUTRIENTS[nut]["code_dir"])
    if not os.path.isdir(code):
        raise SystemExit(f"base-pipeline code dir not found: {code}")
    sys.path.insert(0, code)
    import data_prep as dp            # noqa: E402
    import ml_runner as mlr           # noqa: E402
    dc = None
    if need_dl:
        import dl_core as dc          # noqa: E402
    print(f"[v4] {nut}: imported base-pipeline modules from {code}", flush=True)
    return dp, mlr, dc


# ─────────────────────────────────────────────────────────────
# fertilizer lookup (same file/format as dl_core.load_fert_lookup, no torch)
# ─────────────────────────────────────────────────────────────
def _season(key: str) -> str:
    """'N' / 'K' -> that nutrient's modelling season; a season key passes through."""
    return NUTRIENTS[key]["season"] if key in NUTRIENTS else key


def season_of_df(df: pd.DataFrame, dp) -> str:
    """The season a sample table belongs to, from its sampling dates (one season per table)."""
    ds = {d for d in (_to_date(dp, v) for v in df["conc_date"]) if d is not None}
    found = {s for s in SEASONS for d in ds if SEASONS[s] <= d <= SEASON_ENDS[s]}
    if len(found) != 1:
        raise RuntimeError(f"[v4] table spans seasons {sorted(found)}; expected exactly one")
    return found.pop()


def table_path(key: str) -> str:
    s = _season(key)
    return os.path.join(DATA_ROOT, SEASON_DIRS[s], TABLES[s])


def fert_path(key: str) -> str:
    return os.path.join(DATA_ROOT, SEASON_DIRS[_season(key)], "fert_by_plot_daily_lb2kg.xlsx")


def weather_path(key: str) -> str:
    s = _season(key)
    return os.path.join(DATA_ROOT, SEASON_DIRS[s], f"FAWN_report_metric_{s}.xlsx")


def load_fert_lookup(path: str) -> Dict[str, np.ndarray]:
    df = pd.read_excel(path, sheet_name="Combined")
    df.columns = [str(c).strip() for c in df.columns]
    id_col = next(c for c in df.columns
                  if c.lower() in ("plot_num", "plot num", "plotnum", "image_id", "imageid"))
    day_cols = sorted([c for c in df.columns if re.match(r"^day\s*\d+$", c, re.I)],
                      key=lambda x: int(re.search(r"\d+", x).group()))
    out = {}
    for _, r in df.iterrows():
        out[str(r[id_col]).strip()] = pd.to_numeric(r[day_cols], errors="coerce") \
            .fillna(0.0).to_numpy(np.float64)
    return out


# ─────────────────────────────────────────────────────────────
# sequence end date (flight vs leaf sampling) -> 'DAP' column
#   base pipeline reads the sequence length from row['DAP'] if present, else row['day']
#   (= UAV flight DAP). The table has no 'DAP' column, so base pipeline = flight end.
#   The T feature group resolves to 'day' first, so adding 'DAP' does NOT change T.
# ─────────────────────────────────────────────────────────────
def _to_date(dp, v) -> Optional[date]:
    try:
        return dp.to_date(v)
    except Exception:
        return None


def _replace_wagg(df: pd.DataFrame, season: str, end_dap) -> pd.DataFrame:
    """Overwrite the 8 Wagg columns with v4 values (from transplant, Methods definitions)."""
    wt = wagg_table(season, "methods")
    idx = np.clip(pd.to_numeric(end_dap, errors="coerce").astype(int).to_numpy(), 0, len(wt) - 1)
    for c in WAGG_COLS:
        if c in df.columns:
            df[c] = wt[c].to_numpy()[idx]
    return df


def add_seq_end_columns(df: pd.DataFrame, dp, nut: str, sampling_end: bool,
                        fix_wagg: bool = True) -> pd.DataFrame:
    cfg = NUTRIENTS[nut]
    season = season_of_df(df, dp)
    tp = SEASONS[season]
    df = df.copy()
    dap_s = [(_to_date(dp, v) - tp).days if _to_date(dp, v) else np.nan for v in df["conc_date"]]
    df["DAP_sampling"] = dap_s                 # bookkeeping only; not in any feature group
    if not sampling_end:
        df["DAP"] = pd.to_numeric(df["day"], errors="coerce")
        if fix_wagg:
            df = _replace_wagg(df, season, df["DAP"])
            print(f"[v4] {season}: Wagg recomputed at the FLIGHT date (from transplant, Methods definitions)",
                  flush=True)
        return df
    df["DAP"] = df["DAP_sampling"]
    # Fagg to the same end date: rebuild from the daily records, keeping the
    # table's own units (ratio estimated from the base pipeline flight-date values).
    fcol = cfg["fagg_col"]
    if fcol in df.columns:
        look = load_fert_lookup(fert_path(season))
        ids = df["Image_ID"].astype(str)
        sum_flight = np.array([look[i][:int(d)].sum() if i in look and pd.notna(d) else np.nan
                               for i, d in zip(ids, pd.to_numeric(df["day"], errors="coerce"))])
        old = pd.to_numeric(df[fcol], errors="coerce").to_numpy(float)
        ok = (old > 0) & np.isfinite(sum_flight)
        ratio = float(np.median(sum_flight[ok] / old[ok]))
        sum_samp = np.array([look[i][:int(d)].sum() if i in look and pd.notna(d) else np.nan
                             for i, d in zip(ids, df["DAP"])])
        df[fcol] = sum_samp / ratio
        print(f"[v4] {season} sampEnd: {fcol} rebuilt to the sampling date "
              f"(daily-sum / table ratio = {ratio:.4f}; 1.1209 = lb/ac -> kg/ha)", flush=True)
    # Wagg at the sampling date, counted from transplant (Methods definitions)
    df = _replace_wagg(df, season, df["DAP"])
    # DAP timing feature (group T) = sampling DAP; data_prep resolves T from the
    # first present name in _TIMING_DAP, so put DAP_T first (see patch_load_samples)
    df["DAP_T"] = df["DAP"]
    print("[v4] sampEnd: Wagg and the DAP feature moved to the sampling date", flush=True)
    return df


WAGG_COLS = ["total_temp_avg", "total_rain_mm", "total_solRad_w_m2", "gdd_tbase5",
             "gdd_tbase7", "gdd_tbase10", "heat_stress_days", "max_consecutive_dry_days"]


def wagg_table(key: str, definitions: str = "methods") -> pd.DataFrame:
    """The 8 weather aggregates for every DAP (row index = DAP, 0 = transplant day),
    accumulated from the transplant date.

    definitions="base": exactly base pipeline make_fawn_metric_*.py (build_modified);
        used only to verify this function (reproduces 24-25 to 0.000).
    definitions="methods": as written in manuscript Section 2.4. Two columns differ:
        total_temp_avg           mean daily air temperature since transplant
                                 (base pipeline: running SUM of daily means)
        max_consecutive_dry_days longest dry spell since transplant
                                 (base pipeline: length of the CURRENT dry spell)
    Column names are kept so data_prep's group W resolves them unchanged."""
    w = pd.read_excel(weather_path(key))
    w.columns = [str(c).strip() for c in w.columns]
    per = w["Period"].astype(str).str.strip()
    dt = pd.to_datetime(per, format="%d %b %Y", errors="coerce")          # '15 Oct 2024'
    for fmt in ("%d-%b-%y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):             # '12-Oct-23', ...
        dt = dt.fillna(pd.to_datetime(per, format=fmt, errors="coerce"))
    if dt.isna().any():
        raise RuntimeError(f"unparsed weather dates: {per[dt.isna()].head().tolist()}")
    w["date"] = dt.dt.date
    tp = SEASONS[_season(key)]
    w = w[w["date"] >= tp].sort_values("date").reset_index(drop=True)
    if w["date"].iloc[0] != tp:
        raise RuntimeError(f"daily weather file does not start at transplant {tp}")
    col = lambda name: pd.to_numeric(w[[c for c in w.columns if re.sub(r"\s+", " ", c) ==
                                         re.sub(r"\s+", " ", name)][0]], errors="coerce")
    tmax, tmin, tavg = col("60cm T max (C)"), col("60cm T min (C)"), col("60cm T avg (C)")
    rain, rmax = col("2m Rain tot (mm)"), col("2m Rain max over 15min (mm)")
    out = pd.DataFrame(index=w.index)
    out["heat_stress_days"] = (tmax > 30).cumsum()
    out["total_temp_avg"] = tavg.cumsum() if definitions == "base" else tavg.expanding().mean()
    out["total_rain_mm"] = rain.cumsum()
    out["total_solRad_w_m2"] = col("SolRad avg2m  (w/m^2)").cumsum()
    spell = (rmax == 0).astype(int).groupby(rmax.ne(0).cumsum()).cumsum()   # current dry spell
    out["max_consecutive_dry_days"] = spell if definitions == "base" else spell.cummax()
    t2 = (tmax + tmin) / 2
    for b in (5, 7, 10):
        out[f"gdd_tbase{b}"] = np.maximum(t2 - b, 0).cumsum()
    return out[WAGG_COLS]


def patch_load_samples(dp, nut: str, v: Variant):
    orig = dp.load_samples
    if v.sampEnd:
        dp._TIMING_DAP = ["DAP_T"] + [c for c in dp._TIMING_DAP if c != "DAP_T"]

    def load_samples_v4(*a, **k):
        df = orig(*a, **k)
        return add_seq_end_columns(df, dp, nut, v.sampEnd, fix_wagg=v.fix_wagg)

    dp.load_samples = load_samples_v4


def register_ml_config(dp, base_name: str, v: Variant) -> str:
    base = dp.MODEL_CONFIGS[base_name]
    groups = [g for g in base.scalar_groups
              if not (v.noXY and g == "P") and not (v.noUAV and g in ("V", "S"))
              and not (v.noF and g in ("Fagg", "Wagg"))]
    name = f"{base_name}__{v.tag()}"
    dp.MODEL_CONFIGS[name] = dataclasses.replace(base, name=name, scalar_groups=tuple(groups))
    return name


def scalar_groups_for(dp, config_name: str, v: Variant):
    return tuple(g for g in dp.MODEL_CONFIGS[config_name].scalar_groups
                 if not (v.noXY and g == "P") and not (v.noF and g in ("Fagg", "Wagg")))


# ─────────────────────────────────────────────────────────────
# TCN fix (shared by M6 v4 and M7 v4)
# ─────────────────────────────────────────────────────────────
def masked_tcn_forward(self, x):
    """Replacement for dl_core.TCNEncoder.forward (tested in test_seq_fix.py).
    Padded steps arrive as NaN and are zeroed at EVERY layer, which is exactly the
    causal conv's own zero padding for an unpadded sequence. With LEFT padding the
    read-out step [-1] is the sample's own last day, so outputs are batch-invariant."""
    import torch
    valid = ~torch.isnan(x[:, :1, :])
    m = valid.to(torch.float32)
    h = torch.nan_to_num(x) * m
    for blk in self.net:
        y = blk.conv1(h)
        y = blk.relu(y[..., :-blk.chomp] if blk.chomp else y) * m
        y = blk.drop(y)
        y = blk.conv2(y)
        y = blk.relu(y[..., :-blk.chomp] if blk.chomp else y) * m
        y = blk.drop(y)
        res = blk.down(h) if blk.down is not None else h
        h = blk.relu(y + res) * m
    return self.project(h[..., -1])


def left_pad_nan(seqs):
    """Stack 1-D (L,) or 2-D (C, L) arrays/tensors, LEFT-padded with NaN."""
    import torch
    import torch.nn.functional as F
    ts = [s if torch.is_tensor(s) else torch.from_numpy(np.asarray(s, np.float32)) for s in seqs]
    m = max(t.shape[-1] for t in ts)
    return torch.stack([F.pad(t, (m - t.shape[-1], 0), value=float("nan")) for t in ts], 0)


def patch_seqfix(dc):
    """Apply the TCN fix to dl_core in this process."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    dc.TCN_LEVELS = TCN_LEVELS_V4                 # read at model construction time
    dc.TCNEncoder.forward = masked_tcn_forward

    def temporal_collate_fn_v4(batch):
        o, d, f, t, sl, yr, yc, fe, we = zip(*batch)
        o = pad_sequence(o, batch_first=True)
        d = pad_sequence(d, batch_first=True)
        f = torch.stack(f, 0)
        t = pad_sequence(t, batch_first=True)
        sl = torch.stack(sl, 0)
        yr = torch.stack(yr, 0)
        yc = torch.stack(yc, 0)
        mtf = max(x.shape[-1] for x in fe)
        fe = torch.stack([F.pad(x, (mtf - x.shape[-1], 0), value=float("nan")) for x in fe], 0)
        mtw = max(x.shape[-1] for x in we)
        we = torch.stack([F.pad(x, (mtw - x.shape[-1], 0), value=float("nan")) for x in we], 0)
        return o, d, f, t, sl, yr, yc, fe, we

    dc.temporal_collate_fn = temporal_collate_fn_v4
    print(f"[v4] seqfix: masked TCN, left NaN padding, levels={dc.TCN_LEVELS} "
          f"(receptive field {1 + 4 * (2 ** dc.TCN_LEVELS - 1)} d)", flush=True)


# ─────────────────────────────────────────────────────────────
# DL-only patches
# ─────────────────────────────────────────────────────────────
def patch_dl_samples(dc, dp, nut: str, single: bool):
    """Wrap the M7 image-sequence builder: (1) the N seqwin window used in the
    manuscript (frames >= 2023-12-05, anchor always kept); (2) make sure the 'DAP'
    column added by load_samples survives into the DL sample table."""
    min_d = NUTRIENTS[nut]["min_image_date"]
    orig = dc.build_on_before_closest_samples

    def builder(df_imgs, df_all, min_images=1):
        df = orig(df_imgs, df_all, min_images)
        if df.empty:
            return df
        if min_d is not None and not single:
            rows = []
            for _, r in df.iterrows():
                r = r.to_dict()
                keep = [i for i, dd in enumerate(r["image_dates"]) if dd >= min_d] or \
                       [len(r["image_dates"]) - 1]
                for c in ("image_dates", "ortho_paths", "dem_paths"):
                    r[c] = [r[c][i] for i in keep]
                r["n_imgs"] = len(keep)
                rows.append(r)
            df = pd.DataFrame(rows).reset_index(drop=True)
        return _ensure_dap(df, df_all, dp)

    dc.build_on_before_closest_samples = builder
    orig_c = dc.build_closest_date_samples
    dc.build_closest_date_samples = lambda df_imgs, df_all: _ensure_dap(
        orig_c(df_imgs, df_all), df_all, dp)


def _ensure_dap(df, df_all, dp):
    if df.empty or ("DAP" in df.columns and df["DAP"].notna().all()):
        return df
    key = lambda d: d["Image_ID"].astype(str) + "|" + d["conc_date"].map(lambda v: str(_to_date(dp, v)))
    m = dict(zip(key(df_all), df_all["DAP"]))
    df = df.copy()
    df["DAP"] = key(df).map(m)
    miss = int(df["DAP"].isna().sum())
    if miss:
        raise RuntimeError(f"[v4] DAP missing for {miss} DL samples after merge")
    return df


def patch_no_uav(dc, ortho_ch=5):
    """M7 without any UAV input: the image branch is removed from the network, and
    image I/O is replaced by 1x1 dummies (the same samples/folds are still built
    from the image index, so the comparison with M7 stays paired)."""
    import torch
    import torch.nn as nn

    class NoImageModel(nn.Module):
        def __init__(self, n_features=0, hidden=128, mlp_head_hidden=(256, 64),
                     mlp_head_drop=0.2, n_classes=3, use_sequences=True,
                     tcn_fert_out=None, tcn_weather_out=None, **_ignored):
            super().__init__()
            fo = tcn_fert_out or dc.TCN_FERT_OUT
            wo = tcn_weather_out or dc.TCN_WEATHER_OUT
            self.use_sequences = use_sequences
            self.n_features = int(n_features)
            tab_out = 64 if self.n_features > 0 else 0
            self.tab = nn.Sequential(        # identical to DualTaskModel.tab
                nn.Linear(self.n_features, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(128, tab_out), nn.ReLU(inplace=True), nn.Dropout(0.1),
            ) if self.n_features > 0 else None
            if use_sequences:
                self.tcn_fert = dc.TCNEncoder(1, dc.TCN_HIDDEN, dc.TCN_LEVELS, dc.TCN_KSIZE, fo)
                self.tcn_weather = dc.TCNEncoder(dc.N_WEATHER_VARS, dc.TCN_HIDDEN, dc.TCN_LEVELS,
                                                 dc.TCN_KSIZE, wo)
                self._fo, self._wo = fo, wo
            else:
                self.tcn_fert = self.tcn_weather = None
                self._fo = self._wo = 0
            trunk_in = tab_out + self._fo + self._wo
            self.trunk_reg = dc._build_mlp(trunk_in, list(mlp_head_hidden), 1, mlp_head_drop)
            self.trunk_cls = dc._build_mlp(trunk_in, list(mlp_head_hidden), n_classes, mlp_head_drop)

        def set_backbone_trainable(self, trainable):
            pass

        def forward(self, x_ortho, x_dem, x_feat=None, seq_len=None, x_time=None,
                    x_fert=None, x_weather=None):
            B, dev = x_ortho.shape[0], x_ortho.device
            parts = []
            if self.tab is not None and x_feat is not None:
                parts.append(self.tab(x_feat))
            if self.use_sequences:
                parts.append(self.tcn_fert(x_fert) if x_fert is not None
                             else torch.zeros(B, self._fo, device=dev))
                parts.append(self.tcn_weather(x_weather) if x_weather is not None
                             else torch.zeros(B, self._wo, device=dev))
            z = torch.cat(parts, 1)
            return self.trunk_reg(z), self.trunk_cls(z)

    dc.DualTaskModel = NoImageModel
    dc.OrthoDemTabularDataset._load_norm = \
        lambda self, o, d: (torch.zeros(ortho_ch, 1, 1), torch.zeros(1, 1, 1))
    dc.compute_band_stats_listcol = \
        lambda df, col, n_bands: (np.zeros(n_bands, np.float32), np.ones(n_bands, np.float32))
    dc.compute_scalar_stats_listcol = lambda df, col: (0.0, 1.0)
    dc.compute_global_crop_hw = lambda df_imgs, path_col="ortho_path": (1, 1)
    print("[v4] noUAV: image branch removed; image I/O stubbed", flush=True)


# ─────────────────────────────────────────────────────────────
# shared helpers for the revision-2 additions
# ─────────────────────────────────────────────────────────────
N_BACKGROUND = 100


def central_background(X_tr: np.ndarray, n: int = N_BACKGROUND) -> np.ndarray:
    """Indices of the n training rows closest to the training median (standardized,
    Euclidean): the 'typical plot' SHAP background.
    A SHAP value then reads as how a feature moves the prediction away from a typical
    training plot."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    mu, sd = np.nanmean(X_tr, 0), np.nanstd(X_tr, 0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = np.nan_to_num((X_tr - mu) / sd)
    d = np.sqrt(((Z - np.median(Z, 0)) ** 2).sum(1))
    return np.argsort(d, kind="stable")[:min(n, len(X_tr))]


def transfer_metrics(y_true, y_pred) -> Dict[str, float]:
    """Table 4 quantities for zero-label transfer: R2, RMSE, MAE, Pearson r, bias
    (mean predicted - measured) and the OLS slope of measured on predicted."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    ss = ((y - y.mean()) ** 2).sum()
    out = {"n": int(len(y)),
           "R2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan"),
           "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
           "MAE": float(np.mean(np.abs(y - p))),
           "pearson_r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else float("nan"),
           "bias": float(np.mean(p - y))}
    vp = np.var(p)
    out["slope_measured_on_pred"] = float(np.cov(p, y, ddof=0)[0, 1] / vp) if vp > 0 else float("nan")
    return out


def use_fork_workers():
    """Python >= 3.14 defaults DataLoader workers to 'forkserver', which loses the
    in-process patches (the M7 runs fail in fold 1 without them). 'fork' keeps them."""
    import multiprocessing as mp
    try:
        import torch.multiprocessing as tmp
        mods = (mp, tmp)
    except Exception:
        mods = (mp,)
    for m in mods:
        try:
            m.set_start_method("fork", force=True)
        except (RuntimeError, ValueError):
            pass
