"""
SHAP Analysis Module

This module provides comprehensive SHAP (SHapley Additive exPlanations) analysis:
1. Global feature importance using SHAP values
2. Local SHAP values for individual predictions
3. SHAP summary plots
4. SHAP dependence plots for top features
5. Regime-specific SHAP analysis (Trend/Range, High/Low Volatility)

SHAP provides a unified measure of feature importance based on game theory.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
import warnings
import os


def get_shap_explainer(model, X_background: Optional[np.ndarray] = None, model_type: str = 'tree'):
    """
    Create appropriate SHAP explainer for the model type.

    Args:
        model: Trained model
        X_background: Background data for non-tree explainers
        model_type: 'tree', 'linear', 'kernel', or 'auto'

    Returns:
        SHAP explainer object
    """
    import shap
    import logging

    if model_type == 'auto':
        # Try to determine model type - check for tree-based models
        model_class = type(model).__name__.lower()
        # Check for common tree-based model names including XGBoost Booster
        if any(name in model_class for name in ['xgb', 'lgb', 'forest', 'tree', 'booster', 'gradient']):
            model_type = 'tree'
        elif 'linear' in model_class or 'logistic' in model_class:
            model_type = 'linear'
        # Also check for XGBoost by module
        elif hasattr(model, '__module__') and 'xgboost' in model.__module__:
            model_type = 'tree'
        elif hasattr(model, '__module__') and 'lightgbm' in model.__module__:
            model_type = 'tree'
        else:
            model_type = 'kernel'
    
    if model_type == 'tree':
        # Suppress XGBoost deprecated binary format warning from SHAP internals
        # This is a C-level warning, so we need to set XGBoost verbosity
        import xgboost as xgb
        prev_verbosity = xgb.config_context
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            xgb.set_config(verbosity=0)
            try:
                explainer = shap.TreeExplainer(model)
            finally:
                xgb.set_config(verbosity=1)
        return explainer
    elif model_type == 'linear':
        return shap.LinearExplainer(model, X_background)
    else:
        return shap.KernelExplainer(model.predict, X_background)


class SHAPAnalyzer:
    """
    Comprehensive SHAP analysis for trained models.
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        model_type: str = 'tree',
        max_samples: int = 1000
    ):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained model (XGBoost, LightGBM, sklearn, etc.)
            feature_names: List of feature names
            model_type: 'tree', 'linear', 'kernel', or 'auto'
            max_samples: Maximum samples for SHAP calculation
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.max_samples = max_samples
        self.explainer = None
        self.shap_values_ = None
        self.X_sample_ = None
        self.expected_value_ = None
    
    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_frac: float = 1.0
    ) -> np.ndarray:
        """
        Compute SHAP values for the given data.
        
        Args:
            X: Feature matrix
            sample_frac: Fraction of samples to use (for large datasets)
            
        Returns:
            Array of SHAP values
        """
        import shap
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Sample if needed
        n_samples = len(X_array)
        if n_samples > self.max_samples or sample_frac < 1.0:
            n_use = min(int(n_samples * sample_frac), self.max_samples)
            idx = np.random.choice(n_samples, size=n_use, replace=False)
            idx = np.sort(idx)  # Maintain order for time series
            X_sample = X_array[idx]
        else:
            X_sample = X_array
            idx = np.arange(n_samples)
        
        # Create explainer
        if self.explainer is None:
            self.explainer = get_shap_explainer(self.model, X_sample, self.model_type)
        
        # Handle expected_value which may be a list for multi-class problems
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            # For binary classification, use positive class
            self.expected_value_ = expected_value[1] if len(expected_value) == 2 else expected_value
        else:
            self.expected_value_ = expected_value
        
        # Compute SHAP values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = self.explainer.shap_values(X_sample)
        
        # Handle binary classification (returns list of 2 arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        self.shap_values_ = shap_values
        self.X_sample_ = X_sample
        self.sample_idx_ = idx
        
        return shap_values
    
    def get_global_importance(self) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.shap_values_ is None:
            raise ValueError("Must call compute_shap_values() first")
        
        mean_abs_shap = np.abs(self.shap_values_).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = importance_df['importance'] / total_importance * 100
        else:
            importance_df['importance_pct'] = 0.0
        
        return importance_df.reset_index(drop=True)
    
    def get_local_shap_values(self, sample_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get local SHAP values for specific samples.
        
        Args:
            sample_indices: Indices of samples to get SHAP values for.
                           If None, returns all computed samples.
                           
        Returns:
            DataFrame with SHAP values for each sample and feature
        """
        if self.shap_values_ is None:
            raise ValueError("Must call compute_shap_values() first")
        
        if sample_indices is not None:
            # Map to positions in our sample, filtering out indices not in sample_idx_
            positions = []
            valid_indices = []
            for idx in sample_indices:
                matches = np.where(self.sample_idx_ == idx)[0]
                if len(matches) > 0:
                    positions.append(matches[0])
                    valid_indices.append(idx)
            
            if len(positions) == 0:
                # Return empty dataframe if no valid indices
                df = pd.DataFrame(columns=self.feature_names + ['sample_index'])
                return df
            
            shap_subset = self.shap_values_[positions]
            df = pd.DataFrame(shap_subset, columns=self.feature_names)
            df['sample_index'] = valid_indices
        else:
            df = pd.DataFrame(self.shap_values_, columns=self.feature_names)
            df['sample_index'] = list(self.sample_idx_)
        
        return df
    
    def plot_summary(
        self,
        output_path: Optional[str] = None,
        max_display: int = 20,
        plot_type: str = 'dot'
    ) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            output_path: Path to save the plot
            max_display: Maximum features to display
            plot_type: 'dot', 'bar', or 'violin'
        """
        import shap
        
        if self.shap_values_ is None:
            raise ValueError("Must call compute_shap_values() first")
        
        plt.figure(figsize=(12, 10))
        
        shap.summary_plot(
            self.shap_values_,
            self.X_sample_,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )
        
        plt.title("SHAP Feature Importance Summary", fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.close('all')
    
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = 'auto',
        output_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            feature: Feature name to plot
            interaction_feature: Feature to color by ('auto' for automatic)
            output_path: Path to save the plot
        """
        import shap
        
        if self.shap_values_ is None:
            raise ValueError("Must call compute_shap_values() first")
        
        feature_idx = self.feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        
        if interaction_feature == 'auto':
            shap.dependence_plot(
                feature_idx,
                self.shap_values_,
                self.X_sample_,
                feature_names=self.feature_names,
                show=False
            )
        else:
            interaction_idx = self.feature_names.index(interaction_feature) if interaction_feature else None
            shap.dependence_plot(
                feature_idx,
                self.shap_values_,
                self.X_sample_,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False
            )
        
        plt.title(f"SHAP Dependence: {feature}", fontsize=12)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.close('all')
    
    def plot_top_features_dependence(
        self,
        n_features: int = 5,
        output_dir: str = './'
    ) -> None:
        """
        Create dependence plots for top N features.
        
        Args:
            n_features: Number of top features to plot
            output_dir: Directory to save plots
        """
        importance = self.get_global_importance()
        top_features = importance.head(n_features)['feature'].tolist()
        
        os.makedirs(output_dir, exist_ok=True)
        
        for feature in top_features:
            output_path = os.path.join(output_dir, f"shap_dependence_{feature}.png")
            self.plot_dependence(feature, output_path=output_path)
    
    def plot_bar_importance(
        self,
        output_path: Optional[str] = None,
        max_display: int = 20
    ) -> None:
        """
        Create bar plot of global SHAP importance.
        
        Args:
            output_path: Path to save the plot
            max_display: Maximum features to display
        """
        importance = self.get_global_importance()
        top_features = importance.head(max_display)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values[::-1])
        plt.yticks(range(len(top_features)), top_features['feature'].values[::-1])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Global Feature Importance (SHAP)', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.close('all')


class RegimeSHAPAnalyzer:
    """
    SHAP analysis by market regime.
    
    Analyzes feature importance separately for different market conditions:
    - Trend vs Range
    - High Volatility vs Low Volatility
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        model_type: str = 'tree',
        max_samples_per_regime: int = 500
    ):
        """
        Initialize regime-specific SHAP analyzer.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_type: 'tree', 'linear', 'kernel', or 'auto'
            max_samples_per_regime: Maximum samples per regime
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.max_samples_per_regime = max_samples_per_regime
        self.regime_results_ = {}
    
    def analyze_by_regime(
        self,
        X: pd.DataFrame,
        regime_labels: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Compute SHAP values for each regime.

        Splits are driven by regime_combined string values so the analysis
        automatically handles both legacy TREND/RANGE and direction-aware
        UPTREND/DOWNTREND/RANGE modes without any hardcoded numeric checks.

        Args:
            X: Feature matrix (pd.DataFrame with DatetimeIndex)
            regime_labels: DataFrame from generate_regime_labels()

        Returns:
            Dictionary keyed by regime name with SHAP analysis results
        """
        results = {}
        aligned = regime_labels.reindex(X.index)

        # Detect direction-aware mode from regime_trend values
        direction_aware = (
            'regime_trend' in aligned.columns and
            (aligned['regime_trend'] == -1).any()
        )

        # --- Direction split (aggregated) ---
        if 'regime_trend' in aligned.columns:
            if direction_aware:
                for val, label in [(1, 'Uptrend'), (-1, 'Downtrend'), (0, 'Range')]:
                    key = label.lower()
                    mask = aligned['regime_trend'] == val
                    results[key] = self._analyze_subset(X[mask], label)
            else:
                for val, label in [(1, 'Trend'), (0, 'Range')]:
                    key = label.lower()
                    mask = aligned['regime_trend'] == val
                    results[key] = self._analyze_subset(X[mask], label)

        # --- Volatility split (aggregated) ---
        if 'regime_volatility' in aligned.columns:
            for val, key, label in [
                (1,  'high_vol',  'High Volatility'),
                (0,  'med_vol',   'Medium Volatility'),
                (-1, 'low_vol',   'Low Volatility'),
            ]:
                mask = aligned['regime_volatility'] == val
                results[key] = self._analyze_subset(X[mask], label)

        # --- Per-combined-regime split (most granular) ---
        if 'regime_combined' in aligned.columns:
            for regime_val in sorted(aligned['regime_combined'].dropna().unique()):
                key = regime_val.lower()          # e.g. 'uptrend_high_vol'
                mask = aligned['regime_combined'] == regime_val
                results[key] = self._analyze_subset(X[mask], regime_val)

        self.regime_results_ = results
        return results
    
    def _analyze_subset(
        self,
        X_subset: pd.DataFrame,
        regime_name: str
    ) -> Dict:
        """
        Analyze SHAP for a subset of data.
        
        Args:
            X_subset: Feature matrix for this regime
            regime_name: Name of the regime
            
        Returns:
            Dictionary with SHAP analysis results
        """
        if len(X_subset) < 10:
            return {
                'regime_name': regime_name,
                'n_samples': len(X_subset),
                'error': 'Insufficient samples for analysis'
            }
        
        # Sample if needed
        if len(X_subset) > self.max_samples_per_regime:
            X_sample = X_subset.sample(n=self.max_samples_per_regime, random_state=42)
        else:
            X_sample = X_subset
        
        # Create analyzer and compute
        analyzer = SHAPAnalyzer(
            self.model,
            self.feature_names,
            self.model_type,
            self.max_samples_per_regime
        )
        
        try:
            analyzer.compute_shap_values(X_sample)
            importance = analyzer.get_global_importance()
            
            return {
                'regime_name': regime_name,
                'n_samples': len(X_subset),
                'n_analyzed': len(X_sample),
                'importance': importance,
                'shap_values': analyzer.shap_values_,
                'X_sample': analyzer.X_sample_,
                'analyzer': analyzer
            }
        except Exception as e:
            return {
                'regime_name': regime_name,
                'n_samples': len(X_subset),
                'error': str(e)
            }
    
    def get_regime_comparison(
        self,
        regime_keys: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare feature importance across regimes.

        Args:
            regime_keys: Subset of regime keys to include. None = all regimes.

        Returns:
            DataFrame with one importance column per regime, sorted by
            variance across regimes (most regime-sensitive features first).
        """
        if not self.regime_results_:
            raise ValueError("Must call analyze_by_regime() first")

        keys = regime_keys if regime_keys is not None else list(self.regime_results_.keys())
        comparison_data: Dict[str, list] = {'feature': self.feature_names}

        for key in keys:
            result = self.regime_results_.get(key, {})
            if 'importance' in result:
                imp = result['importance'].set_index('feature')['importance']
                comparison_data[f'{key}_importance'] = imp.reindex(self.feature_names).fillna(0).values

        comparison_df = pd.DataFrame(comparison_data)

        imp_cols = [c for c in comparison_df.columns if c.endswith('_importance')]
        if len(imp_cols) > 1:
            comparison_df['importance_variance'] = comparison_df[imp_cols].var(axis=1)
            comparison_df = comparison_df.sort_values('importance_variance', ascending=False)

        return comparison_df
    
    def plot_regime_comparison(
        self,
        output_path: Optional[str] = None,
        max_display: int = 15
    ) -> None:
        """
        Create two-panel comparison plot: aggregated regimes (top) and
        per-combined-regime (bottom), so neither chart is overcrowded.

        Args:
            output_path: Path to save the plot (suffix _aggregated / _combined added)
            max_display: Maximum features to display per panel
        """
        combined_keys = {k for k in self.regime_results_ if _is_combined_regime_key(k)}
        aggregated_keys = [k for k in self.regime_results_ if k not in combined_keys]

        for keys, suffix, title in [
            (aggregated_keys,       'aggregated', 'Feature Importance — Aggregated Regimes'),
            (sorted(combined_keys), 'combined',   'Feature Importance — Per Combined Regime'),
        ]:
            if not keys:
                continue

            comparison = self.get_regime_comparison(regime_keys=keys)
            if comparison.empty:
                continue

            imp_cols = [c for c in comparison.columns if c.endswith('_importance')]
            top_features = comparison.head(max_display)

            fig, ax = plt.subplots(figsize=(14, max(8, len(top_features) * 0.5)))
            x = np.arange(len(top_features))
            width = 0.8 / max(len(imp_cols), 1)

            for i, col in enumerate(imp_cols):
                label = col.replace('_importance', '').replace('_', ' ').title()
                offset = (i - len(imp_cols) / 2 + 0.5) * width
                ax.barh(x + offset, top_features[col].values, width, label=label)

            ax.set_yticks(x)
            ax.set_yticklabels(top_features['feature'].values)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title(title, fontsize=13)
            ax.legend(loc='lower right', fontsize=8)
            ax.invert_yaxis()
            plt.tight_layout()

            if output_path:
                base, ext = os.path.splitext(output_path)
                path = f"{base}_{suffix}{ext}"
                plt.savefig(path, dpi=150, bbox_inches='tight')

            plt.close('all')
    
    def plot_all_regime_summaries(
        self,
        output_dir: str,
        max_display: int = 15
    ) -> None:
        """
        Create SHAP summary plots for each regime.
        Combined regimes go into a 'combined/' subfolder to keep the
        top-level directory uncluttered.

        Args:
            output_dir: Directory to save plots
            max_display: Maximum features to display per plot
        """
        combined_keys = {k for k in self.regime_results_ if _is_combined_regime_key(k)}

        combined_dir = os.path.join(output_dir, 'combined')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(combined_dir, exist_ok=True)

        for regime_key, result in self.regime_results_.items():
            if 'analyzer' not in result:
                continue
            save_dir = combined_dir if regime_key in combined_keys else output_dir
            output_path = os.path.join(save_dir, f"shap_summary_{regime_key}.png")
            try:
                result['analyzer'].plot_summary(output_path=output_path, max_display=max_display)
            except Exception as e:
                print(f"Could not create summary plot for {regime_key}: {e}")
    
    def plot_regime_beeswarms(
        self,
        output_path: Optional[str] = None,
        max_display: int = 15,
        regime_keys: Optional[List[str]] = None,
        ncols: int = 2,
    ) -> None:
        """
        Multi-panel beeswarm plot: one subplot per regime in a single figure.

        Uses shap.plots.beeswarm(ax=) so each regime is embedded in its own
        subplot for easy side-by-side comparison. Features are sorted by their
        mean |SHAP| within each regime (standard beeswarm behaviour), so the
        order can differ across panels — highlighting which features change
        importance between regimes.

        Requires SHAP ≥ 0.44 (ax= parameter support).

        Args:
            output_path: Path to save the combined figure
            max_display: Max features per beeswarm panel
            regime_keys: Regimes to include (None = all with valid analyzers)
            ncols: Number of columns in the subplot grid
        """
        import shap as shap_lib

        all_keys = regime_keys if regime_keys is not None else list(self.regime_results_.keys())
        valid = [k for k in all_keys if 'analyzer' in self.regime_results_.get(k, {})]

        if not valid:
            print("No valid regime analyzers for beeswarm plot")
            return

        nrows = (len(valid) + ncols - 1) // ncols
        row_h = max(4.0, max_display * 0.38)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(10 * ncols, row_h * nrows),
            squeeze=False,
        )
        axes_flat = list(axes.flat)

        for i, k in enumerate(valid):
            result = self.regime_results_[k]
            analyzer = result['analyzer']
            n_samples = result.get('n_samples', '?')

            base = analyzer.expected_value_
            if base is None:
                base_scalar = 0.0
            elif isinstance(base, (list, np.ndarray)):
                base_scalar = float(np.asarray(base).flat[0])
            else:
                base_scalar = float(base)
            base_values = np.full(len(analyzer.shap_values_), base_scalar)

            explanation = shap_lib.Explanation(
                values=analyzer.shap_values_,
                base_values=base_values,
                data=analyzer.X_sample_,
                feature_names=analyzer.feature_names,
            )

            ax = axes_flat[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import inspect, io
                _supports_ax = 'ax' in inspect.signature(shap_lib.plots.beeswarm).parameters
                if _supports_ax:
                    shap_lib.plots.beeswarm(
                        explanation,
                        max_display=max_display,
                        ax=ax,
                        show=False,
                        color_bar=True,
                        plot_size=None,
                    )
                else:
                    # shap < ~0.46: beeswarm creates its own figure; capture and embed
                    tmp_fig = plt.figure()
                    shap_lib.plots.beeswarm(explanation, max_display=max_display, show=False)
                    buf = io.BytesIO()
                    tmp_fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    plt.close(tmp_fig)
                    ax.imshow(plt.imread(buf))
                    ax.axis('off')

            label = k.replace('_', ' ').upper()
            ax.set_title(f"{label}  (n={n_samples})", fontsize=9)

        for j in range(len(valid), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("SHAP Beeswarm by Regime", fontsize=13, y=1.01)
        plt.tight_layout()

        if output_path:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        plt.close('all')


def _is_combined_regime_key(k: str) -> bool:
    """Return True if k is a fine-grained combined regime key (e.g. 'uptrend_high_vol')."""
    return (
        any(v in k for v in ['high_vol', 'med_vol', 'low_vol'])
        and any(d in k for d in ['trend', 'range', 'uptrend', 'downtrend'])
    )


def plot_scope_comparison_by_regime(
    slow_regime_results: Dict,
    fast_regime_results: Dict,
    output_path: Optional[str] = None,
    max_display: int = 8,
    regime_keys: Optional[List[str]] = None,
    title_suffix: str = '',
) -> None:
    """
    Side-by-side comparison of slow vs fast model feature importance per regime.

    Each row is one regime; left column = slow model, right column = fast model.
    Slow and fast have different feature sets so each panel shows its own feature names.

    Args:
        slow_regime_results: regime_results dict from the slow model RegimeSHAPAnalyzer
        fast_regime_results: regime_results dict from the fast model RegimeSHAPAnalyzer
        output_path: Path to save the plot
        max_display: Max features to show per panel
        regime_keys: Regimes to include (None = all with valid data in either scope)
        title_suffix: Appended to the figure title
    """
    all_keys = sorted(set(slow_regime_results.keys()) | set(fast_regime_results.keys()))
    if regime_keys is not None:
        all_keys = [k for k in all_keys if k in regime_keys]

    valid_keys = [
        k for k in all_keys
        if 'importance' in slow_regime_results.get(k, {}) or 'importance' in fast_regime_results.get(k, {})
    ]

    if not valid_keys:
        print("No valid regime results for scope comparison")
        return

    n_regimes = len(valid_keys)
    cell_height = max(0.5, max_display * 0.35)
    fig, axes = plt.subplots(
        n_regimes, 2,
        figsize=(16, n_regimes * cell_height + 1.5),
        squeeze=False,
    )

    colors = ['steelblue', 'darkorange']
    for row, regime_key in enumerate(valid_keys):
        for col, (scope_label, regime_results) in enumerate([
            ('Slow', slow_regime_results),
            ('Fast', fast_regime_results),
        ]):
            ax = axes[row, col]
            result = regime_results.get(regime_key, {})

            if 'importance' in result:
                imp = result['importance'].head(max_display)
                y_pos = np.arange(len(imp))
                ax.barh(y_pos, imp['importance'].values[::-1], color=colors[col])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(imp['feature'].values[::-1], fontsize=7)
                n = result.get('n_samples', '?')
                ax.set_title(f"{regime_key.upper()} — {scope_label} (n={n})", fontsize=8)
            else:
                error = result.get('error', 'No data')
                ax.text(0.5, 0.5, error, ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(f"{regime_key.upper()} — {scope_label}", fontsize=8)

            ax.set_xlabel('Mean |SHAP|', fontsize=7)
            ax.tick_params(axis='both', labelsize=7)

    title = f'SHAP: Slow vs Fast by Regime{title_suffix}'
    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Scope comparison plot saved to: {output_path}")

    plt.close('all')


def run_full_shap_analysis(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
    output_dir: str,
    regime_labels: Optional[pd.DataFrame] = None,
    model_name: str = 'model',
    max_samples: int = 1000
) -> Dict:
    """
    Run complete SHAP analysis pipeline.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        output_dir: Directory to save outputs
        regime_labels: Optional regime labels for regime-specific analysis
        model_name: Name for output files
        max_samples: Maximum samples for SHAP calculation
        
    Returns:
        Dictionary with all analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {'model_name': model_name}
    
    print("\n" + "=" * 70)
    print(f"RUNNING SHAP ANALYSIS FOR: {model_name}")
    print("=" * 70)
    
    # 1. Global SHAP analysis
    print("\n[1/4] Computing global SHAP values...")
    analyzer = SHAPAnalyzer(model, feature_names, max_samples=max_samples)
    analyzer.compute_shap_values(X)
    
    global_importance = analyzer.get_global_importance()
    results['global_importance'] = global_importance
    
    # Save importance to CSV
    importance_path = os.path.join(output_dir, f"shap_importance_{model_name}.csv")
    global_importance.to_csv(importance_path, index=False)
    print(f"  Saved global importance to: {importance_path}")
    
    # 2. Summary plot
    print("\n[2/4] Creating summary plots...")
    summary_path = os.path.join(output_dir, f"shap_summary_{model_name}.png")
    analyzer.plot_summary(output_path=summary_path)
    
    bar_path = os.path.join(output_dir, f"shap_bar_{model_name}.png")
    analyzer.plot_bar_importance(output_path=bar_path)
    
    # 3. Dependence plots for top features
    print("\n[3/4] Creating dependence plots for top features...")
    dep_dir = os.path.join(output_dir, "dependence_plots")
    analyzer.plot_top_features_dependence(n_features=5, output_dir=dep_dir)
    
    # 4. Regime-specific analysis
    if regime_labels is not None:
        print("\n[4/4] Running regime-specific analysis...")
        regime_analyzer = RegimeSHAPAnalyzer(model, feature_names, max_samples_per_regime=500)
        regime_results = regime_analyzer.analyze_by_regime(X, regime_labels)
        results['regime_analysis'] = regime_results
        
        # Regime comparison plot
        comparison_path = os.path.join(output_dir, f"shap_regime_comparison_{model_name}.png")
        regime_analyzer.plot_regime_comparison(output_path=comparison_path)
        
        # Individual regime summaries (separate files, one per regime)
        regime_dir = os.path.join(output_dir, "regime_plots")
        regime_analyzer.plot_all_regime_summaries(regime_dir)

        # Multi-panel beeswarm: aggregated regimes in one figure
        aggregated_keys = [k for k in regime_results if not _is_combined_regime_key(k)]
        if aggregated_keys:
            beeswarm_path = os.path.join(regime_dir, f"shap_beeswarm_by_regime_{model_name}.png")
            regime_analyzer.plot_regime_beeswarms(
                output_path=beeswarm_path,
                max_display=15,
                regime_keys=aggregated_keys,
            )

        # Multi-panel beeswarm: combined regimes (uptrend_high_vol etc.)
        combined_keys = sorted(k for k in regime_results if _is_combined_regime_key(k))
        if combined_keys:
            beeswarm_combined_path = os.path.join(
                regime_dir, 'combined', f"shap_beeswarm_combined_{model_name}.png"
            )
            regime_analyzer.plot_regime_beeswarms(
                output_path=beeswarm_combined_path,
                max_display=15,
                regime_keys=combined_keys,
                ncols=3,
            )

        # Save regime comparison
        comparison_df = regime_analyzer.get_regime_comparison()
        comparison_path = os.path.join(output_dir, f"shap_regime_comparison_{model_name}.csv")
        comparison_df.to_csv(comparison_path, index=False)
    else:
        print("\n[4/4] Skipping regime analysis (no regime labels provided)")
    
    print("\n" + "=" * 70)
    print(f"SHAP ANALYSIS COMPLETE: {model_name}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    # Example usage / test
    import sys
    import os
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y = (0.5 * X['feature_0'] + 0.3 * X['feature_1'] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Test SHAP analyzer
    print("Testing SHAP Analyzer...")
    analyzer = SHAPAnalyzer(model, list(X.columns), model_type='tree', max_samples=200)
    analyzer.compute_shap_values(X)
    
    importance = analyzer.get_global_importance()
    print("\nTop 5 Features by SHAP importance:")
    print(importance.head())
    
    print("\nSHAP analysis test completed successfully!")
