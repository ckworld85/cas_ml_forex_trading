"""
Feature Selection Module

This module provides advanced feature selection methods:
1. Mutual Information (MI) Pre-Filter
2. Permutation Feature Importance (PFI)
3. Combined feature selection pipeline

All methods support time-series cross-validation to prevent lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
import warnings


class XGBoostWrapper:
    """
    Wrapper to make XGBoost Booster sklearn-compatible for permutation_importance.
    
    XGBoost's native Booster uses predict(DMatrix), while sklearn expects predict(X).
    This wrapper handles the conversion.
    """
    
    def __init__(self, booster):
        """
        Initialize wrapper.
        
        Args:
            booster: XGBoost Booster object or sklearn-compatible estimator
        """
        self.booster = booster
        self._is_xgb_booster = hasattr(booster, 'save_model') and not hasattr(booster, 'fit')
    
    def fit(self, X, y):
        """
        Dummy fit method for sklearn compatibility.
        The model is already trained, so this does nothing.
        """
        return self
    
    def predict(self, X):
        """
        Predict using the wrapped model.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            
        Returns:
            Predictions (binary for classification)
        """
        if self._is_xgb_booster:
            import xgboost as xgb
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            dmatrix = xgb.DMatrix(X_array)
            proba = self.booster.predict(dmatrix)
            return (proba > 0.5).astype(int)
        else:
            return self.booster.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities using the wrapped model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        if self._is_xgb_booster:
            import xgboost as xgb
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            dmatrix = xgb.DMatrix(X_array)
            proba = self.booster.predict(dmatrix)
            # Return 2D array for binary classification
            return np.column_stack([1 - proba, proba])
        else:
            return self.booster.predict_proba(X)
    
    def score(self, X, y):
        """
        Calculate accuracy score for sklearn compatibility.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        y_true = y.values if isinstance(y, pd.Series) else y
        return (y_pred == y_true).mean()


class MutualInformationFilter:
    """
    Filter features based on Mutual Information with the target.

    MI measures the mutual dependence between a feature and the target.

    **Ties matter.** The kNN MI estimator needs distinct neighbour distances. A
    feature carried from a coarser timeframe repeats its value on every bar of the
    finer grid (a daily feature is identical on all 6 bars of a 4h day, all 96 bars
    of an M15 day), the neighbour radius collapses to ~0 and the estimate explodes.
    sklearn breaks ties with 1e-10 relative noise, which is far too small to help.
    Measured on run feature_eval (37 features, 4h cadence): every one of the 22
    daily features landed in the band 0.227-0.270 regardless of what it measured,
    and deduplicating dropped ``daily_rsi`` from 0.2562 to 0.0000 and ``daily_adx``
    from 0.2534 to 0.0000. Left uncorrected, any non-zero ``min_mi`` selects on bar
    granularity rather than on information.

    Two corrections, both on by default:

    * ``dedup_runs`` collapses runs of identical values per feature before scoring.
    * ``n_permutations`` scores each feature against its own shuffled-label noise
      floor, so features left with very different effective sample sizes stay
      comparable. A feature is informative when it beats that floor — a
      self-calibrating test, unlike a hand-set ``min_mi`` that has to mean the same
      thing at n=187 and n=17500.

    **The noise floor cannot replace the dedup.** It is tempting to think the
    permutation null absorbs the tie inflation on its own, since it keeps x (and
    therefore the ties) fixed. It does not: the inflation comes from x's tie blocks
    *coinciding* with y's persistence blocks — both are driven by the same daily
    cycle — and with tied values the estimator effectively measures "do identical
    values share a label". Shuffling y destroys that block structure, so the null
    loses the inflation while the observed score keeps it, and the floor drops
    instead of rising. Measured on long_slow: without dedup, 20 of 37 features clear
    their floor (daily_rsi 0.2562 vs floor 0.0185); with dedup, 6 do (daily_rsi
    0.0000 vs floor 0.0488). Running the floor on undeduplicated scores is worse
    than running neither.

    Set ``legacy=True`` to restore the pre-fix behaviour for reproducing old runs.
    """

    def __init__(
        self,
        min_mi: float = 0.001,
        task: str = 'classification',
        n_neighbors: int = 3,
        random_state: int = 42,
        dedup_runs: bool = True,
        n_permutations: int = 20,
        min_eff_samples: int = 50,
        legacy: bool = False,
    ):
        """
        Initialize MI filter.

        Args:
            min_mi: Minimum MI threshold. Applied *in addition to* the noise-floor
                test when permutations are enabled; the default 0.0 makes it vacuous.
            task: 'classification' or 'regression'
            n_neighbors: Number of neighbors for MI estimation
            random_state: Random seed for reproducibility
            dedup_runs: Collapse runs of identical values per feature before scoring.
            n_permutations: Shuffled-label repeats for the per-feature noise floor
                (95th percentile). 0 disables the test and falls back to ``min_mi``.
            min_eff_samples: Below this many post-dedup rows the estimate is not
                evaluable; such features are KEPT (absence of evidence is not
                evidence of absence) and reported with NaN scores.
            legacy: Restore pre-fix behaviour — no dedup, no noise floor, plain
                ``mi >= min_mi`` on the raw matrix.
        """
        self.min_mi = min_mi
        self.task = task
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.dedup_runs = dedup_runs and not legacy
        self.n_permutations = 0 if legacy else n_permutations
        self.min_eff_samples = min_eff_samples
        self.legacy = legacy

        self.mi_scores_ = None
        self.mi_scores_trend_ = None
        self.mi_scores_range_ = None
        self.mi_scores_trend_raw_ = None
        self.mi_scores_range_raw_ = None
        self.mi_null_trend_ = None
        self.mi_null_range_ = None
        self.n_eff_trend_ = None
        self.n_eff_range_ = None
        self.mi_informative_ = None
        self.selected_features_ = None
        self.removed_features_ = None

    # ------------------------------------------------------------------ scoring

    def _mi_func(self):
        return mutual_info_classif if self.task == 'classification' else mutual_info_regression

    def _score_bucket(self, X_array, y_array, feature_names):
        """Score every feature on one bucket of rows.

        Returns a DataFrame indexed by feature with columns mi, mi_raw, null_p95,
        n_eff, informative, evaluable. ``evaluable`` is False when the feature has
        too few post-dedup rows or no variation — those features carry no verdict
        and must not be removed on this bucket's evidence.
        """
        mi_func = self._mi_func()
        rows = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_all = mi_func(X_array, y_array, n_neighbors=self.n_neighbors,
                              random_state=self.random_state)

            for j, feat in enumerate(feature_names):
                x, y = X_array[:, j], y_array
                if self.dedup_runs:
                    keep = np.r_[True, x[1:] != x[:-1]]
                    x, y = x[keep], y[keep]
                n_eff = len(x)

                if (n_eff < self.min_eff_samples or np.unique(x).size < 2
                        or np.unique(y).size < 2):
                    rows[feat] = (np.nan, raw_all[j], np.nan, n_eff, False, False)
                    continue

                xr = x.reshape(-1, 1)
                mi = float(mi_func(xr, y, n_neighbors=self.n_neighbors,
                                   random_state=self.random_state)[0])

                if self.n_permutations > 0:
                    rng = np.random.RandomState(self.random_state)
                    null = [
                        float(mi_func(xr, rng.permutation(y), n_neighbors=self.n_neighbors,
                                      random_state=self.random_state + i)[0])
                        for i in range(self.n_permutations)
                    ]
                    floor = float(np.percentile(null, 95))
                    informative = mi > floor and mi >= self.min_mi
                else:
                    floor = np.nan
                    informative = mi >= self.min_mi

                rows[feat] = (mi, raw_all[j], floor, n_eff, informative, True)

        return pd.DataFrame.from_dict(
            rows, orient='index',
            columns=['mi', 'mi_raw', 'null_p95', 'n_eff', 'informative', 'evaluable'])

    @staticmethod
    def _bucket_keeps(res):
        """Features this bucket votes to keep: informative, or not evaluable here."""
        return set(res.index[res['informative'] | ~res['evaluable']])

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        regime_mask: Optional[pd.Series] = None,
    ) -> 'MutualInformationFilter':
        """
        Calculate MI scores for all features.

        When ``regime_mask`` is provided (regime-conditional label mode), MI is
        computed independently on the signal-regime bars and the remaining bars.
        A feature is kept if it clears ``min_mi`` in **either** bucket, preventing
        cancellation artefacts where opposite-direction signals average out to
        near-zero MI on the full dataset.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Optional list of feature names
            regime_mask: Optional Series aligned to X's index selecting the
                *signal regime* — the bars in which this target's positive labels
                can occur.  Truthy (``> 0.5``) = signal bar, everything else =
                non-signal bar.  For direction-agnostic regimes that is trend vs
                range; for direction-aware regimes it is uptrend vs rest (long
                targets) or downtrend vs rest (short targets), which is why the
                caller resolves it per model — see ``signal_regime_mask`` in
                advanced_train.py.  When provided, per-bucket MI is calculated and
                the union of selected features is returned.  Stored as
                ``mi_scores_trend_`` (signal) / ``mi_scores_range_`` (non-signal)
                for inspection; ``mi_scores_`` holds the element-wise max.

        Returns:
            self
        """
        if feature_names is None:
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f'f{i}' for i in range(X.shape[1])]

        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # NaN mask (shared across all branches)
        valid = ~(np.isnan(X_array).any(axis=1) | np.isnan(y_array))

        if regime_mask is not None:
            # Align mask to X's positional index.
            # Use > 0.5 instead of astype(bool): pandas converts NaN floats to
            # True with bool(), so NaN regime bars would silently land in the
            # trend bucket.  With > 0.5, NaN comparisons evaluate to False,
            # treating unknown-regime bars as range (the safe default).
            regime_aligned = (
                regime_mask.reindex(
                    X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X_array))
                ) > 0.5
            ).values

            trend_mask = valid & regime_aligned
            range_mask = valid & ~regime_aligned

            MIN_SAMPLES = max(self.n_neighbors + 1, 50)
            if trend_mask.sum() < MIN_SAMPLES or range_mask.sum() < MIN_SAMPLES:
                print(
                    f"[MI] Regime-aware mode: insufficient samples in one regime "
                    f"(trend={trend_mask.sum()}, range={range_mask.sum()}, min={MIN_SAMPLES}). "
                    "Falling back to global MI."
                )
                regime_mask = None  # fall through to global path

        if regime_mask is not None:
            print(
                f"[MI] Regime-aware: signal={trend_mask.sum()} bars, "
                f"other={range_mask.sum()} bars — selecting union of per-bucket features."
            )
            res_t = self._score_bucket(X_array[trend_mask], y_array[trend_mask], feature_names)
            res_r = self._score_bucket(X_array[range_mask], y_array[range_mask], feature_names)

            # A bucket whose labels are single-class carries no verdict at all — under
            # trend_only/regime_conditional every range bar is a forced 0, so the
            # non-signal bucket is structurally uninformative and must not get a vote.
            buckets = []
            for res, y_b, name in ((res_t, y_array[trend_mask], 'signal'),
                                   (res_r, y_array[range_mask], 'other')):
                if np.unique(y_b).size < 2:
                    print(f"[MI] {name} bucket is single-class — it casts no vote.")
                else:
                    buckets.append(res)

            self.mi_scores_trend_ = res_t['mi']
            self.mi_scores_range_ = res_r['mi']
            self.mi_scores_trend_raw_ = res_t['mi_raw']
            self.mi_scores_range_raw_ = res_r['mi_raw']
            self.mi_null_trend_ = res_t['null_p95']
            self.mi_null_range_ = res_r['null_p95']
            self.n_eff_trend_ = res_t['n_eff']
            self.n_eff_range_ = res_r['n_eff']

            combined = pd.concat([res_t['mi'], res_r['mi']], axis=1).max(axis=1)
            self.mi_scores_ = combined.sort_values(ascending=False, na_position='last')
            self.mi_informative_ = (
                pd.concat([b['informative'] for b in buckets], axis=1).any(axis=1)
                if buckets else pd.Series(True, index=feature_names)
            )

            if buckets:
                keep = set().union(*(self._bucket_keeps(b) for b in buckets))
            else:
                print("[MI] No bucket could cast a vote — keeping all features.")
                keep = set(feature_names)
        else:
            self.mi_scores_trend_ = None
            self.mi_scores_range_ = None

            res = self._score_bucket(X_array[valid], y_array[valid], feature_names)
            self.mi_scores_trend_raw_ = None
            self.mi_scores_range_raw_ = None
            self.mi_null_trend_ = res['null_p95']
            self.n_eff_trend_ = res['n_eff']
            self.mi_scores_ = res['mi'].sort_values(ascending=False, na_position='last')
            self.mi_informative_ = res['informative']
            keep = self._bucket_keeps(res)

        # Emit in the INPUT column order, never in score order. XGBoost's
        # colsample_bytree picks columns by index, so a reshuffled feature list
        # trains different trees even when the selected set is identical: permuting
        # the 37 long_slow columns moved predicted probabilities by up to 0.087 at
        # colsample_bytree=0.9, and not at all at 1.0. Ordering by score would make
        # every change to the MI computation silently reshuffle the exported model
        # and confound any ablation against it. mi_scores_ stays score-sorted for
        # reporting; only this list feeds the scaler, the ONNX input and training.
        self.selected_features_ = [f for f in feature_names if f in keep]
        self.removed_features_ = [f for f in feature_names if f not in keep]

        # MI is univariate: it cannot see a feature that only pays off in combination
        # with another. Measured on run feature_eval, long_slow: daily_adx scored 0.0000
        # against a 0.0434 floor yet carries 10.5% of the model's within-trend SHAP
        # spread (14.1% in short_slow, where it is the single largest contributor).
        # A sweeping cut is therefore a prompt to check the multivariate signals, not a
        # result to adopt unseen.
        n = len(self.mi_scores_)
        if n and len(self.removed_features_) > n / 2:
            print(
                f"[MI] WARNING: dropping {len(self.removed_features_)}/{n} features. MI is "
                f"univariate and blind to interaction-only features — verify against gain / "
                f"SHAP before accepting, or run with --mi-permutations 0 to report the noise "
                f"floor without gating on it."
            )
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on MI threshold.
        
        Args:
            X: Feature matrix
            
        Returns:
            Filtered feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before transform()")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_results(self) -> Dict:
        """
        Get detailed MI results.

        Returns:
            Dictionary with MI scores and selection results.
            In regime-aware mode also includes ``mi_scores_trend`` and
            ``mi_scores_range``.
        """
        result = {
            'mi_scores': self.mi_scores_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_,
            'n_selected': len(self.selected_features_),
            'n_removed': len(self.removed_features_),
            'min_mi_threshold': self.min_mi,
            'mi_informative': self.mi_informative_,
            'mi_null_trend': self.mi_null_trend_,
            'n_eff_trend': self.n_eff_trend_,
            'dedup_runs': self.dedup_runs,
            'n_permutations': self.n_permutations,
        }
        if self.mi_scores_trend_ is not None:
            result['mi_scores_trend'] = self.mi_scores_trend_
            result['mi_scores_range'] = self.mi_scores_range_
            result['mi_scores_trend_raw'] = self.mi_scores_trend_raw_
            result['mi_scores_range_raw'] = self.mi_scores_range_raw_
            result['mi_null_range'] = self.mi_null_range_
            result['n_eff_range'] = self.n_eff_range_
        return result

    @staticmethod
    def _fmt(v, width=8, prec=6):
        """Format a possibly-NaN score; NaN means 'not evaluable', not 'zero'."""
        return f"{'n/a':>{width}}" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:{width}.{prec}f}"

    def print_summary(self) -> None:
        """Print a summary of MI filtering results."""
        regime_aware = self.mi_scores_trend_ is not None
        print("\n" + "=" * 60)
        print("MUTUAL INFORMATION FILTER RESULTS"
              + (" (regime-aware)" if regime_aware else ""))
        print("=" * 60)
        mode = ("legacy (raw, no dedup, no noise floor)" if self.legacy else
                f"dedup={self.dedup_runs}, noise floor from {self.n_permutations} permutations "
                f"(p95), min_eff_samples={self.min_eff_samples}")
        print(f"Scoring: {mode}")
        print(f"MI Threshold: {self.min_mi}")
        print(f"Total features: {len(self.mi_scores_)}")
        print(f"Selected features: {len(self.selected_features_)}")
        print(f"Removed features: {len(self.removed_features_)}")

        def line(i, feat):
            m = self.mi_scores_.get(feat, np.nan)
            tag = "[sig]" if bool(self.mi_informative_.get(feat, False)) else "     "
            if regime_aware:
                t, r = self.mi_scores_trend_[feat], self.mi_scores_range_[feat]
                floor, n_eff = self.mi_null_trend_[feat], self.n_eff_trend_[feat]
                return (f"  {i:2d}. {feat:40s} {tag} signal {self._fmt(t)} "
                        f"(floor {self._fmt(floor)}, n_eff {int(n_eff):5d}) / "
                        f"other {self._fmt(r)}")
            floor, n_eff = self.mi_null_trend_[feat], self.n_eff_trend_[feat]
            return (f"  {i:2d}. {feat:40s} {tag} MI {self._fmt(m)} "
                    f"(floor {self._fmt(floor)}, n_eff {int(n_eff):5d})")

        print("\nSelected Features:")
        for i, feat in enumerate(self.selected_features_, 1):
            print(line(i, feat))

        if self.removed_features_:
            print(f"\nRemoved Features ({len(self.removed_features_)}):")
            for i, feat in enumerate(self.removed_features_[:99], 1):
                print(line(i, feat))
            if len(self.removed_features_) > 99:
                print(f"  ... and {len(self.removed_features_) - 99} more")

        print("=" * 60 + "\n")


class PermutationImportanceFilter:
    """
    Filter features based on Permutation Feature Importance (PFI).
    
    PFI measures the decrease in model performance when a feature is shuffled.
    Features with low or negative importance are considered uninformative.
    """
    
    def __init__(
        self,
        min_importance: float = 0.0,
        n_repeats: int = 10,
        scoring: Optional[str] = None,
        random_state: int = 42,
        n_cv_splits: int = 5
    ):
        """
        Initialize PFI filter.
        
        Args:
            min_importance: Minimum importance threshold. Features below are removed.
            n_repeats: Number of times to permute each feature
            scoring: Scoring metric (None uses model default)
            random_state: Random seed for reproducibility
            n_cv_splits: Number of cross-validation splits for time-series CV
        """
        self.min_importance = min_importance
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.random_state = random_state
        self.n_cv_splits = n_cv_splits
        self.importance_scores_ = None
        self.importance_std_ = None
        self.selected_features_ = None
        self.removed_features_ = None
    
    def fit(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        use_cv: bool = True
    ) -> 'PermutationImportanceFilter':
        """
        Calculate PFI on out-of-sample data using time-series CV.
        
        Args:
            model: Trained model with predict/predict_proba method (or XGBoost Booster)
            X: Feature matrix
            y: Target variable
            use_cv: If True, use time-series CV; if False, use full dataset
            
        Returns:
            self
        """
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f'f{i}' for i in range(X.shape[1])]
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Handle NaN values
        mask = ~(np.isnan(X_array).any(axis=1) | np.isnan(y_array))
        X_clean = X_array[mask]
        y_clean = y_array[mask]
        
        # Wrap model if it's an XGBoost Booster (not sklearn-compatible)
        wrapped_model = XGBoostWrapper(model)
        
        if use_cv and len(X_clean) > self.n_cv_splits * 100:
            # Use time-series CV for out-of-sample evaluation
            importances_list = []
            tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
            
            for train_idx, test_idx in tscv.split(X_clean):
                X_test = X_clean[test_idx]
                y_test = y_clean[test_idx]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = permutation_importance(
                        wrapped_model,
                        X_test,
                        y_test,
                        n_repeats=self.n_repeats,
                        random_state=self.random_state,
                        scoring=self.scoring
                    )
                
                importances_list.append(result.importances_mean)
            
            # Average across CV folds
            importance_mean = np.mean(importances_list, axis=0)
            importance_std = np.std(importances_list, axis=0)
        else:
            # Use full dataset (less reliable but works for small datasets)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = permutation_importance(
                    wrapped_model,
                    X_clean,
                    y_clean,
                    n_repeats=self.n_repeats,
                    random_state=self.random_state,
                    scoring=self.scoring
                )
            
            importance_mean = result.importances_mean
            importance_std = result.importances_std
        
        # Store results. As in MutualInformationFilter, the selected list keeps the
        # INPUT column order — score order would reshuffle the trained model via
        # colsample_bytree without changing which features are used.
        self.importance_scores_ = pd.Series(importance_mean, index=feature_names).sort_values(ascending=False)
        self.importance_std_ = pd.Series(importance_std, index=feature_names)
        keep = set(self.importance_scores_.index[self.importance_scores_ >= self.min_importance])
        self.selected_features_ = [f for f in feature_names if f in keep]
        self.removed_features_ = [f for f in feature_names if f not in keep]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on PFI threshold.
        
        Args:
            X: Feature matrix
            
        Returns:
            Filtered feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before transform()")
        
        return X[self.selected_features_]
    
    def get_results(self) -> Dict:
        """
        Get detailed PFI results.
        
        Returns:
            Dictionary with importance scores and selection results
        """
        return {
            'importance_scores': self.importance_scores_,
            'importance_std': self.importance_std_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_,
            'n_selected': len(self.selected_features_),
            'n_removed': len(self.removed_features_),
            'min_importance_threshold': self.min_importance
        }
    
    def print_summary(self) -> None:
        """Print a summary of PFI filtering results."""
        print("\n" + "=" * 60)
        print("PERMUTATION FEATURE IMPORTANCE RESULTS")
        print("=" * 60)
        print(f"Importance Threshold: {self.min_importance}")
        print(f"Total features: {len(self.importance_scores_)}")
        print(f"Selected features: {len(self.selected_features_)}")
        print(f"Removed features: {len(self.removed_features_)}")
        
        print("\nTop 15 Features by Importance:")
        for i, (feat, score) in enumerate(self.importance_scores_.head(15).items(), 1):
            std = self.importance_std_[feat]
            status = "[OK]" if score >= self.min_importance else "[X]"
            print(f"  {i:2d}. {feat:40s} Imp: {score:+.6f} (±{std:.6f}) {status}")
        
        # Show features with negative importance
        negative_features = self.importance_scores_[self.importance_scores_ < 0]
        if len(negative_features) > 0:
            print(f"\nFeatures with Negative Importance ({len(negative_features)}):")
            for feat, score in negative_features.head(10).items():
                print(f"  - {feat:40s} Imp: {score:+.6f}")
            if len(negative_features) > 10:
                print(f"  ... and {len(negative_features) - 10} more")
        
        print("=" * 60 + "\n")


class FeatureSelectionPipeline:
    """
    Combined feature selection pipeline using MI and PFI.
    
    The pipeline first filters features using MI, then refines using PFI.
    """
    
    def __init__(
        self,
        mi_threshold: float = 0.001,
        pfi_threshold: float = 0.0,
        task: str = 'classification',
        n_cv_splits: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the feature selection pipeline.
        
        Args:
            mi_threshold: Minimum MI threshold
            pfi_threshold: Minimum PFI threshold
            task: 'classification' or 'regression'
            n_cv_splits: Number of CV splits for PFI
            random_state: Random seed
        """
        self.mi_filter = MutualInformationFilter(
            min_mi=mi_threshold,
            task=task,
            random_state=random_state
        )
        self.pfi_filter = PermutationImportanceFilter(
            min_importance=pfi_threshold,
            n_cv_splits=n_cv_splits,
            random_state=random_state
        )
        self.task = task
        self.selected_features_ = None
        self.removed_by_mi_ = None
        self.removed_by_pfi_ = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model=None,
        run_pfi: bool = True
    ) -> 'FeatureSelectionPipeline':
        """
        Run the full feature selection pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model for PFI (required if run_pfi=True)
            run_pfi: Whether to run PFI after MI filtering
            
        Returns:
            self
        """
        # Step 1: MI filtering
        print("\n[Step 1/2] Running Mutual Information Filter...")
        self.mi_filter.fit(X, y)
        self.mi_filter.print_summary()
        
        X_after_mi = self.mi_filter.transform(X)
        self.removed_by_mi_ = self.mi_filter.removed_features_
        
        if run_pfi and model is not None:
            # Step 2: PFI filtering (on MI-selected features)
            print("[Step 2/2] Running Permutation Feature Importance...")
            self.pfi_filter.fit(model, X_after_mi, y)
            self.pfi_filter.print_summary()
            
            self.selected_features_ = self.pfi_filter.selected_features_
            self.removed_by_pfi_ = self.pfi_filter.removed_features_
        else:
            print("[Step 2/2] Skipping PFI (no model provided)")
            self.selected_features_ = self.mi_filter.selected_features_
            self.removed_by_pfi_ = []
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on the pipeline results.
        
        Args:
            X: Feature matrix
            
        Returns:
            Filtered feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before transform()")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, model)
        return self.transform(X)
    
    def get_summary(self) -> Dict:
        """Get summary of feature selection results."""
        return {
            'original_features': len(self.mi_filter.mi_scores_),
            'after_mi_filter': len(self.mi_filter.selected_features_),
            'final_selected': len(self.selected_features_),
            'removed_by_mi': len(self.removed_by_mi_),
            'removed_by_pfi': len(self.removed_by_pfi_) if self.removed_by_pfi_ else 0,
            'selected_features': self.selected_features_,
            'mi_scores': self.mi_filter.mi_scores_,
            'pfi_scores': self.pfi_filter.importance_scores_ if self.pfi_filter.importance_scores_ is not None else None
        }
    
    def print_final_summary(self) -> None:
        """Print final summary of feature selection."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("FEATURE SELECTION PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Original features:     {summary['original_features']}")
        print(f"After MI filter:       {summary['after_mi_filter']} (removed {summary['removed_by_mi']})")
        print(f"Final selected:        {summary['final_selected']} (removed {summary['removed_by_pfi']} by PFI)")
        print(f"Total removed:         {summary['removed_by_mi'] + summary['removed_by_pfi']}")
        print("=" * 60 + "\n")


def time_series_cross_val_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = 'accuracy'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform time-series cross-validation.
    
    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target variable
        n_splits: Number of CV splits
        scoring: Scoring metric
        
    Returns:
        Tuple of (train_scores, test_scores)
    """
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.base import clone
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    train_scores = []
    test_scores = []
    
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y
    
    for train_idx, test_idx in tscv.split(X_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        train_score = model_clone.score(X_train, y_train)
        test_score = model_clone.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    return np.array(train_scores), np.array(test_scores)


if __name__ == "__main__":
    # Example usage / test
    import sys
    import os
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create features with varying importance
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some features being important
    y = (
        0.5 * X['feature_0'] +
        0.3 * X['feature_1'] +
        0.1 * X['feature_2'] +
        np.random.randn(n_samples) * 0.5
    )
    y = (y > y.median()).astype(int)
    y = pd.Series(y, name='target')
    
    # Test MI filter
    mi_filter = MutualInformationFilter(min_mi=0.01, task='classification')
    mi_filter.fit(X, y)
    mi_filter.print_summary()
    
    print("\nMutual Information test completed successfully!")
