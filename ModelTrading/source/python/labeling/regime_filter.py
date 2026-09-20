"""
Regime Filtering Module

Shared utilities for filtering training data by market regime.
Used by train.py, advanced_train.py, and iterative_training.py.

Available regimes (from get_regime_masks):
  Trend/Range:
    - trend: ADX > threshold OR efficiency > threshold
    - range: Not trend
  Volatility:
    - high_volatility: Volatility percentile >= 70th
    - medium_volatility: Volatility percentile between 30th and 70th
    - low_volatility: Volatility percentile <= 30th
  Combined (trend × volatility):
    - trend_high_vol, trend_med_vol, trend_low_vol
    - range_high_vol, range_med_vol, range_low_vol
"""

import argparse
import pandas as pd

try:
    from ModelTrading.source.python.labeling.regime import (
        generate_regime_labels,
        get_regime_masks,
        print_regime_statistics,
    )
except ImportError:
    from labeling.regime import (
        generate_regime_labels,
        get_regime_masks,
        print_regime_statistics,
    )

# All valid regime names that can be used for filtering
VALID_REGIMES = [
    # Individual regimes
    'trend', 'range', 'uptrend', 'downtrend',
    'high_volatility', 'medium_volatility', 'low_volatility',
    # Combined regimes (direction × volatility)
    'uptrend_high_vol', 'uptrend_med_vol', 'uptrend_low_vol',
    'downtrend_high_vol', 'downtrend_med_vol', 'downtrend_low_vol',
    'range_high_vol', 'range_med_vol', 'range_low_vol',
    # Backward-compat aliases (uptrend + downtrend combined)
    'trend_high_vol', 'trend_med_vol', 'trend_low_vol',
]


def add_regime_filter_args(parser: argparse.ArgumentParser) -> None:
    """
    Add regime-filter CLI arguments to an argparse parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        '--regime-filter',
        action='store_true',
        help='Enable regime filtering: train only on data from a specific market regime',
    )
    parser.add_argument(
        '--regime-type',
        type=str,
        default=None,
        choices=VALID_REGIMES,
        help='Market regime to keep when --regime-filter is enabled. '
             f'Choices: {", ".join(VALID_REGIMES)}',
    )


def apply_regime_filter(
    X: pd.DataFrame,
    labels: dict,
    ohlc: pd.DataFrame,
    regime_labels: pd.DataFrame,
    regime_type: str,
    extra_series: dict | None = None,
) -> tuple:
    """
    Filter training data to keep only rows matching a specific market regime.

    Args:
        X: Feature matrix (rows = bars, cols = features)
        labels: Dict of label Series keyed by name (e.g. long_fast, short_fast, …)
        ohlc: OHLC DataFrame aligned to X
        regime_labels: DataFrame produced by generate_regime_labels (same index as X)
        regime_type: One of VALID_REGIMES
        extra_series: Optional dict of additional Series/DataFrames to filter
                      (e.g. {'y_reg': y_reg_series}). Returned filtered in same dict.

    Returns:
        (X_filtered, labels_filtered, ohlc_filtered, extra_filtered)
        where extra_filtered is the dict with filtered values (empty dict if None passed).

    Raises:
        ValueError: If regime_type is invalid or yields an empty dataset.
    """
    if regime_type not in VALID_REGIMES:
        raise ValueError(
            f"Invalid regime_type '{regime_type}'. "
            f"Valid choices: {VALID_REGIMES}"
        )

    masks = get_regime_masks(regime_labels)
    if regime_type not in masks:
        raise ValueError(
            f"Regime mask '{regime_type}' not found. "
            f"Available masks: {list(masks.keys())}. "
            f"Check that regime_labels contains the required columns."
        )

    mask = masks[regime_type]
    # Align mask to X index (regime_labels may have extra/fewer rows)
    mask = mask.reindex(X.index, fill_value=False)

    n_before = len(X)
    n_kept = mask.sum()

    if n_kept == 0:
        raise ValueError(
            f"Regime filter '{regime_type}' removed ALL rows. "
            f"Cannot train on an empty dataset."
        )

    X_filtered = X.loc[mask].copy()
    labels_filtered = {k: v.loc[mask].copy() for k, v in labels.items()}
    # ohlc is optional: the fast training scope carries no OHLC frame of its own.
    ohlc_filtered = ohlc.loc[mask].copy() if ohlc is not None else None

    extra_filtered = {}
    if extra_series:
        for k, v in extra_series.items():
            if isinstance(v, (pd.Series, pd.DataFrame)):
                extra_filtered[k] = v.loc[mask].copy()
            else:
                extra_filtered[k] = v  # pass through non-pandas objects

    print(f"\n{'='*60}")
    print(f"REGIME FILTER: {regime_type.upper()}")
    print(f"{'='*60}")
    print(f"  Rows before filter: {n_before:,}")
    print(f"  Rows after filter:  {n_kept:,} ({n_kept/n_before:.1%})")
    print(f"  Rows removed:       {n_before - n_kept:,} ({(n_before - n_kept)/n_before:.1%})")
    print(f"{'='*60}\n")

    return X_filtered, labels_filtered, ohlc_filtered, extra_filtered


def validate_regime_args(args: argparse.Namespace) -> None:
    """
    Validate regime filter arguments and raise clear errors.

    Args:
        args: Parsed CLI arguments (must have regime_filter and regime_type)

    Raises:
        SystemExit: If --regime-filter is used without --regime-type
    """
    if getattr(args, 'regime_filter', False) and not getattr(args, 'regime_type', None):
        raise SystemExit(
            "ERROR: --regime-filter requires --regime-type. "
            f"Choose one of: {', '.join(VALID_REGIMES)}"
        )
