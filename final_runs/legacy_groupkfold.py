"""
legacy_groupkfold.py: scikit-learn GroupKFold with the fold-assignment rule used for
every result in the article.

GroupKFold assigns groups (plots) to folds by distributing the largest groups first.
Groups of equal size are ties, and how ties are ordered changed in scikit-learn 1.6
(np.argsort(..., kind="stable") instead of the default quicksort). With many plots of
equal size, the newer rule produces a different plot-to-fold assignment, so the same
code gives different folds, and slightly different results, on newer versions.

This class restores the original rule (quicksort ordering of group sizes) for the outer
folds and for every inner split (hyperparameter tuning, PLSR components, early stopping).
v4_common installs it before the season pipelines are imported. The outer folds it
produces are listed in fold_membership_ML_vs_DL.csv.
"""
import numpy as np
from sklearn.model_selection import GroupKFold as _SkGroupKFold


class LegacyGroupKFold(_SkGroupKFold):
    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = np.asarray(groups)
        unique_groups, group_idx = np.unique(groups, return_inverse=True)
        if self.n_splits > len(unique_groups):
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} greater "
                             f"than the number of groups: {len(unique_groups)}.")
        n_per_group = np.bincount(group_idx)
        order = np.argsort(n_per_group)[::-1]          # quicksort, as scikit-learn < 1.6
        n_per_fold = np.zeros(self.n_splits)
        group_to_fold = np.zeros(len(unique_groups))
        for gi in order:
            lightest = np.argmin(n_per_fold)
            n_per_fold[lightest] += n_per_group[gi]
            group_to_fold[gi] = lightest
        fold_of_sample = group_to_fold[group_idx]
        for f in range(self.n_splits):
            yield np.where(fold_of_sample == f)[0]


def install():
    """Make `from sklearn.model_selection import GroupKFold` return LegacyGroupKFold."""
    import sklearn.model_selection as skms
    import sklearn.model_selection._split as sks
    skms.GroupKFold = LegacyGroupKFold
    sks.GroupKFold = LegacyGroupKFold
    return LegacyGroupKFold
