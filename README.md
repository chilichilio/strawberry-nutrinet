# strawberry-nutrinet

Code for **"Multimodal UAV Phenomics Integrating Canopy, Fertilizer, and Weather Data
in Time-Series for Strawberry Nutrient Level Prediction"** (Smart Agricultural Technology).

Seven model configurations (M1–M7) predict strawberry leaf nutrient concentration
(% dry mass) from UAV multispectral imagery and digital elevation models (DEMs),
cultivar, plot position, timing, and daily fertilizer and weather records:

- leaf **nitrogen**, 2023–2024 season (1,799 observations, 300 plots);
- leaf **potassium**, 2024–2025 season (1,259 observations, 195 plots);
- leaf **potassium**, independent 2025–2026 season (316 observations, 180 plots),
  used only for external validation.

| Model | Estimator | Inputs |
|---|---|---|
| M1 | PLSR | spectral (V) + structural (S) descriptors, cultivar (C), plot position (P), timing (T) |
| M2 | Random forest | as M1 |
| M3 | PLSR | M1 + cumulative nutrient applied (Fagg) + weather aggregates (Wagg) |
| M4 | Random forest | as M3 |
| M5 | CNN fusion | image and DEM tiles of the matched flight + C, P, T, Fagg, Wagg |
| M6 | Random forest | M4 + TCN embeddings of the daily fertilizer and weather sequences |
| M7 | Temporal multimodal Transformer | image/DEM sequence + daily fertilizer and weather sequences + C, P, T, Fagg, Wagg |

## Repository layout

```text
strawberry-nutrinet/
├── final_runs/            entry points for every result in the article (start here)
│   ├── run_all.sh         the complete list of runs, in order
│   ├── run_ml_v4.py       M1–M4, M6 (CPU)
│   ├── run_dl_v4.py       M5, M7 (GPU)
│   ├── explain_v4.py      M7 static-branch SHAP and Integrated Gradients
│   ├── paired_tests.py    corrected paired t-tests (Supplementary Tables S3, S12, S13)
│   ├── collate_external_v4.py, learning_curve_v4.py   Table 4, Supplementary Figure S11
│   ├── aggregate_seeds_v4.py                         seed-to-seed variability of M7
│   ├── v4_common.py       shared configuration (data paths, model variants, sequence fix)
│   ├── legacy_groupkfold.py                          fold-assignment rule (see Evaluation protocol)
│   └── fold_membership_ML_vs_DL.csv                  outer-fold assignment of every plot
├── code_23_24_N/          base pipeline, nitrogen season (data loading, CV engine, deep models)
├── code_24_25_K/          base pipeline, potassium season
├── code_25_26_external/   train 2024–2025 → test 2025–2026 harness and recalibration
├── slurm/                 job templates for a SLURM cluster
└── data/                  put the Zenodo data here (see data/README.md)
```

`final_runs/` imports the season pipelines in `code_*` and changes their behaviour
only through the variant flags below, so one code path produces every table.

## Data

The data are deposited on Zenodo: <https://doi.org/10.5281/zenodo.24410726>.
Download the deposit and place its three season folders in `data/`
(layout in [data/README.md](data/README.md)).

The deposit contains, for each season, the plot-level modelling table
(handcrafted spectral and structural descriptors, cultivar, plot position,
timing, cumulative fertilizer, weather aggregates and the measured leaf
concentration used in the article: N for 2023–2024, K for 2024–2025 and
2025–2026), the daily per-plot fertilizer records, the daily FAWN weather
records, and orthomosaic and DEM tiles for a subset of plots on representative
flight dates. Other measured variables are not part of this release.

- **Classical models (M1–M4, M6)** run on the deposited tables alone and reproduce
  the reported cross-validation results (small numerical differences can arise
  across library versions).
- **Deep models (M5, M7)** need the tiles of every plot and flight; the deposit
  contains a sample of tiles for testing the image pipeline. The complete
  image set is available from the corresponding author on reasonable request.

## Installation

```bash
conda create -n nutrinet python=3.10 -y
conda activate nutrinet
pip install -r requirements.txt     # install the torch build matching your CUDA first
```

## Reproducing the article

All commands run from `final_runs/`; results are written to `outputs/` at the
repository root (override with `NUTRINET_OUT`).

```bash
cd final_runs
python test_seq_fix.py                                        # unit test, prints PASS
python run_ml_v4.py --nutrient K --model M2 --sampEnd --dry-run   # check the data
bash run_all.sh --print                                       # list every run
bash run_all.sh                                               # run everything
```

Single runs, for example:

```bash
# N classical models are evaluated on the deep-learning outer folds
V4_FOLD_MANIFEST=$PWD/fold_membership_ML_vs_DL.csv \
    python run_ml_v4.py --nutrient N --model M6 --seqfix --sampEnd --nested --shap
python run_ml_v4.py --nutrient K --model M4 --sampEnd
torchrun --standalone --nproc_per_node=2 run_dl_v4.py --nutrient K --model M7 --seqfix --sampEnd
python run_ml_v4.py --nutrient K --model M4 --sampEnd --external     # 2025–2026 season
```

| Result in the article | Runs (see `run_all.sh`) | Output used |
|---|---|---|
| Table 3, Supplementary Table S12 | M1–M4 `--sampEnd`; M6 `--seqfix --sampEnd --nested`; M5 `--sampEnd`; M7 `--seqfix --sampEnd` | `*_fold_metrics.csv`, `cv_results_*_reg.csv`, `paired_tests.csv` |
| Figure 5, Supplementary Tables S5, Figures S4, S6 | M7 (and M2, M6) above | out-of-fold predictions |
| Supplementary Table S3 | M7 `--single`, `--lstm`, `--noTemp` | `paired_tests.csv` |
| Supplementary Table S13 | `--noUAV`, `--noXY`, `--noW`, `--noF` variants | `paired_tests.csv` |
| Figure 6, Supplementary Table S10 | M6 with `--shap` | `M6_shap_*.csv` |
| Figure 7, Supplementary Figure S7, Tables S8–S9 | `explain_v4.py` on the saved M7 checkpoints | `explain_v4/` |
| Table 4, Supplementary Table S14 | `--external` runs + `recalibrate_external.py` + `collate_external_v4.py` | `K_ext2526/table4_v4.csv` |
| Supplementary Figure S11 | `learning_curve_v4.py` | `K_ext2526/recalibration_learning_curve.csv` |

### Variant flags

| Flag | Meaning |
|---|---|
| `--sampEnd` | every non-image input (fertilizer and weather sequences, cumulative fertilizer, weather aggregates, days after transplanting) ends at the leaf-sampling date; the image keeps its flight date |
| `--seqfix` | masked TCN sequence branch with left padding and a 253-day receptive field (M6, M7) |
| `--nested` | M6: the sequence encoder is refit inside every inner tuning split |
| `--noUAV` | no UAV inputs (V + S for M2/M4/M6; image branch for M7) |
| `--noXY` | no plot coordinates |
| `--noW` | no daily weather sequence |
| `--noF` | no fertilizer or weather inputs |
| `--single` | M7 uses only the flight matched to the sampling date |
| `--lstm`, `--noTemp` | M7 with a bidirectional LSTM, or without a temporal encoder, over the image dates |
| `--external` | K: train on all 2024–2025, predict 2025–2026 |
| `--seed` | M7 training seed (default 42) |

## Evaluation protocol

- Five-fold cross-validation grouped by plot: all observations of a plot are in
  the same fold. M1–M7 are compared on identical outer folds
  (`final_runs/fold_membership_ML_vs_DL.csv`).
- scikit-learn 1.6 changed how `GroupKFold` breaks ties between plots with the same
  number of observations, which changes the folds. `final_runs/legacy_groupkfold.py`
  keeps the original rule for all outer and inner splits, so the results do not
  depend on the installed scikit-learn version.
- Scalers, hyperparameter searches, the M6 sequence encoder and the sequence
  normalization are fit on training-fold data only; deep-model early stopping uses
  a plot-grouped inner split of the training fold.
- Only information available on or before the leaf-sampling date is used.
- Regression metrics: R², RMSE, MAE, sMAPE. Stage-specific nutrient-status classes
  (deficient, adequate, high) are derived from the continuous predictions.
- Paired model comparisons use the Nadeau–Bengio corrected resampled t-test with
  Benjamini–Hochberg adjustment within each comparison family.
- For N, one erroneous laboratory reading (plot SB2_08, 14 February 2024) and one
  UAV flight record (plot SB3_24, 7 February 2024) are excluded in
  `code_23_24_N/data_prep.py`.

## License and citation

Released under the MIT License. If you use this code, please cite the article
(full reference to be added on publication) and the Zenodo data deposit
(<https://doi.org/10.5281/zenodo.24410726>).
