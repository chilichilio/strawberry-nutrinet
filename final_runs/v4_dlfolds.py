"""
v4_dlfolds.py: run the classical models (run_ml_v4.py: M1-M4, M6 and their variants) on the
SAME outer folds as the deep-learning models (M5, M7), read from a manifest
(Image_ID -> deep-learning fold 1..5; fold_membership_ML_vs_DL.csv).

Only the OUTER split (all plots present, n_splits = 5) is replaced; every inner
GroupKFold (RF grid search, PLSR components, nested TCN refits) is left unchanged.
run_ml_v4.py applies it when V4_FOLD_MANIFEST is set and then writes to outputs_dlfold/:
    V4_FOLD_MANIFEST=$PWD/fold_membership_ML_vs_DL.csv \
        python run_ml_v4.py --nutrient N --model M6 --seqfix --sampEnd --nested
All N classical results in the article use these folds. For K the classical and
deep-learning folds are identical, so no manifest is needed.
"""
import numpy as np
import pandas as pd
import sklearn.model_selection as skms

_SK = skms.GroupKFold


def make(nut, manifest):
    m = pd.read_csv(manifest)
    m = m[m.nutrient == nut]
    fold_of = dict(zip(m.Image_ID.astype(str).str.strip(), m.DL_fold_M5_M7.astype(int)))

    class ManifestGroupKFold:
        def __init__(self, n_splits=5, **kw):
            self.n_splits, self.kw = n_splits, kw

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

        def split(self, X, y=None, groups=None):
            ids = pd.Series(np.asarray(groups)).astype(str).str.strip().to_numpy()
            if self.n_splits == 5 and set(ids) == set(fold_of):        # the outer split
                f = np.array([fold_of[i] for i in ids]); idx = np.arange(len(ids))
                for k in range(1, 6):
                    yield idx[f != k], idx[f == k]
            else:                                                       # any inner split
                yield from _SK(n_splits=self.n_splits, **self.kw).split(X, y, groups)

    return ManifestGroupKFold


def patch(dp, nut, manifest):
    cls = make(nut, manifest)
    dp.GroupKFold = cls
    skms.GroupKFold = cls
    print(f"[v4] outer folds read from {manifest} (DL fold assignment)", flush=True)
    return cls
