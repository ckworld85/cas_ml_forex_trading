"""
Regime Labels Generation Module

This module generates regime labels for forex trading data:
- Trend/Range regimes based on price movement and ADX-like metrics
- High/Low Volatility regimes based on ATR and Bollinger Band width

These labels can be used for regime-specific SHAP analysis and model conditioning.
"""

import warnings
import numpy as np
import pandas as pd
import ta
try:
    from features.indicators import calculate_trend_strength, calculate_price_efficiency, calculate_volatility_percentile, calculate_price_direction
except ImportError:
    from ModelTrading.source.python.features.indicators import calculate_trend_strength, calculate_price_efficiency, calculate_volatility_percentile, calculate_price_direction


def generate_regime_labels(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,
    volatility_high_percentile: float = 70.0,
    volatility_low_percentile: float = 30.0,
    efficiency_threshold: float = 0.5,
    adx_period: int = 14,
    volatility_period: int = 14,
    volatility_lookback: int = 500,
    direction_period: int = 50,
    direction_aware: bool = False,
) -> pd.DataFrame:
    """
    Generate regime labels for the given OHLC data.

    Trend detection uses ADX OR price efficiency. Direction (up/down) is determined
    by close relative to a rolling SMA, splitting TREND into UPTREND and DOWNTREND.
    This avoids mixing opposite-direction bars into a single 'TREND' bucket, which
    would wash long and short labels out to ~50% and destroy the training signal.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        adx_threshold: ADX threshold for trend classification (above = trend)
        volatility_high_percentile: Percentile above which volatility is considered high
        volatility_low_percentile: Percentile below which volatility is considered low
        efficiency_threshold: Price efficiency threshold for trend classification
        adx_period: Period for ADX calculation
        volatility_period: Period for ATR calculation
        volatility_lookback: Lookback for volatility percentile (default 500 bars,
                            which is ~5 days for M15 data or ~2 years for daily data)
        direction_period: SMA period for up/down trend direction (default 200 bars)
        direction_aware: When True, regime_trend uses {1=uptrend, -1=downtrend, 0=range}
            and regime_combined uses UPTREND/DOWNTREND prefixes, enabling direction-aware
            label generation in generate_regime_conditional_labels(). When False (default),
            regime_trend collapses to the original {1=trend, 0=range} and regime_combined
            uses TREND/RANGE prefixes — backward-compatible with the original behaviour.

    Returns:
        DataFrame with regime labels:
            - regime_trend: 1/−1/0 (direction_aware=True) or 1/0 (direction_aware=False)
            - regime_trend_label: 'UPTREND'/'DOWNTREND'/'RANGE' or 'TREND'/'RANGE'
            - regime_volatility: 1 (high), 0 (medium), -1 (low)
            - regime_combined: e.g. 'UPTREND_HIGH_VOL' or 'TREND_HIGH_VOL'
    """
    result = pd.DataFrame(index=df.index)

    # Calculate trend strength (ADX)
    adx = calculate_trend_strength(df, period=adx_period)

    # Calculate price efficiency
    efficiency = calculate_price_efficiency(df, period=adx_period)

    # Calculate volatility percentile
    vol_percentile = calculate_volatility_percentile(
        df,
        period=volatility_period,
        lookback=volatility_lookback
    )

    # Trend detection: ADX or efficiency indicates directional movement
    is_trend = (adx > adx_threshold) | (efficiency > efficiency_threshold)

    # Direction: close above SMA = uptrend, below = downtrend
    is_upward = calculate_price_direction(df, period=direction_period)

    is_uptrend   = is_trend & is_upward
    is_downtrend = is_trend & ~is_upward

    if direction_aware:
        result['regime_trend'] = np.where(is_uptrend, 1, np.where(is_downtrend, -1, 0)).astype(int)
        result['regime_trend_label'] = np.where(
            is_uptrend, 'UPTREND', np.where(is_downtrend, 'DOWNTREND', 'RANGE')
        )
    else:
        # Collapse uptrend + downtrend → TREND (original binary behaviour)
        result['regime_trend'] = (is_uptrend | is_downtrend).astype(int)
        result['regime_trend_label'] = np.where(is_uptrend | is_downtrend, 'TREND', 'RANGE')
    
    # Volatility regime: based on percentile
    result['regime_volatility'] = np.where(
        vol_percentile >= volatility_high_percentile,
        1,  # High volatility
        np.where(
            vol_percentile <= volatility_low_percentile,
            -1,  # Low volatility
            0   # Medium volatility
        )
    )
    result['regime_volatility_label'] = np.where(
        vol_percentile >= volatility_high_percentile,
        'HIGH_VOL',
        np.where(
            vol_percentile <= volatility_low_percentile,
            'LOW_VOL',
            'MED_VOL'
        )
    )
    
    # Combined regime label
    result['regime_combined'] = (
        result['regime_trend_label'] + '_' + result['regime_volatility_label']
    )
    
    # Store raw metrics for analysis
    result['adx'] = adx
    result['price_efficiency'] = efficiency
    result['volatility_percentile'] = vol_percentile
    
    return result


def generate_regime_labels_from_scores(
    scores: pd.DataFrame,
    trend_threshold: float = 0.15,
    volatility_high_percentile: float = 70.0,
    volatility_low_percentile: float = 30.0,
    volatility_lookback: int = 500,
    direction_aware: bool = False,
) -> pd.DataFrame:
    """
    Build regime labels from the ML regime model instead of the ADX/PE rule.

    Emits exactly the same column contract as generate_regime_labels() so every
    downstream consumer (regime_conditional labelling, regime filtering,
    per-regime CV metrics, backtest breakdown) works unchanged.

    This exists so the ML regime arm can be labelled by its *own* regime
    definition. Labelling both arms with the rule-based regime makes the
    comparison structurally unfair: generate_regime_conditional_labels() picks
    the labelling scheme per bar from regime_trend, so the rule's own inputs
    (daily_adx, daily_price_efficiency, daily_regime_trend) partly encode the
    recipe their labels were produced with.

    Args:
        scores: DataFrame indexed like the daily bars with columns
            'rgm_trend_score' (∈[-1,1]) and 'rgm_vol_score' (∈[0,1]), as
            written by data/update_regime_model_data.py. Values must be
            UNSHIFTED — matching generate_regime_labels(), which also reads
            current-bar indicators.
        trend_threshold: |rgm_trend_score| above which a bar counts as trending.
        volatility_high_percentile: percentile above which volatility is HIGH.
        volatility_low_percentile: percentile below which volatility is LOW.
        volatility_lookback: rolling window for the volatility percentile rank.
            Mirrors generate_regime_labels() so both arms bucket volatility by
            the same *statistical* definition (top ~30% of recent history),
            which keeps the class balance comparable between arms.
        direction_aware: see generate_regime_labels().

    Returns:
        DataFrame with regime_trend, regime_trend_label, regime_volatility,
        regime_volatility_label, regime_combined, plus the raw scores and the
        derived volatility percentile for analysis.
    """
    required = {'rgm_trend_score', 'rgm_vol_score'}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(
            f"scores is missing required column(s) {sorted(missing)}. "
            "Run data/update_regime_model_data.py to produce data/regime_daily.csv."
        )

    result = pd.DataFrame(index=scores.index)

    trend_score = scores['rgm_trend_score'].astype(float)
    vol_score = scores['rgm_vol_score'].astype(float)

    # Rolling percentile rank of the volatility score, mirroring
    # calculate_volatility_percentile()'s causal lookback.
    min_periods = max(2, volatility_lookback // 10)
    vol_percentile = vol_score.rolling(
        window=volatility_lookback, min_periods=min_periods
    ).rank(pct=True) * 100.0

    is_trend = trend_score.abs() > trend_threshold
    is_uptrend = is_trend & (trend_score > 0)
    is_downtrend = is_trend & (trend_score < 0)

    if direction_aware:
        result['regime_trend'] = np.where(is_uptrend, 1, np.where(is_downtrend, -1, 0)).astype(int)
        result['regime_trend_label'] = np.where(
            is_uptrend, 'UPTREND', np.where(is_downtrend, 'DOWNTREND', 'RANGE')
        )
    else:
        result['regime_trend'] = (is_uptrend | is_downtrend).astype(int)
        result['regime_trend_label'] = np.where(is_uptrend | is_downtrend, 'TREND', 'RANGE')

    result['regime_volatility'] = np.where(
        vol_percentile >= volatility_high_percentile,
        1,
        np.where(vol_percentile <= volatility_low_percentile, -1, 0)
    )
    result['regime_volatility_label'] = np.where(
        vol_percentile >= volatility_high_percentile,
        'HIGH_VOL',
        np.where(vol_percentile <= volatility_low_percentile, 'LOW_VOL', 'MED_VOL')
    )

    result['regime_combined'] = (
        result['regime_trend_label'] + '_' + result['regime_volatility_label']
    )

    # Raw metrics for analysis (mirrors the adx/price_efficiency/volatility_percentile
    # columns of the rule-based variant).
    result['rgm_trend_score'] = trend_score
    result['rgm_vol_score'] = vol_score
    result['volatility_percentile'] = vol_percentile

    return result


def add_regime_labels_to_features(
    df_features: pd.DataFrame,
    df_ohlc: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Add regime labels to an existing features DataFrame.
    
    Args:
        df_features: DataFrame with calculated features
        df_ohlc: DataFrame with OHLC data (must have same index as df_features)
        **kwargs: Arguments passed to generate_regime_labels
        
    Returns:
        DataFrame with features plus regime labels
    """
    # Ensure OHLC has the required columns
    ohlc_cols = ['high', 'low', 'close']
    
    # Handle prefixed column names
    if 'm15_high' in df_ohlc.columns:
        ohlc_for_regime = df_ohlc[['m15_high', 'm15_low', 'm15_close']].copy()
        ohlc_for_regime.columns = ohlc_cols
    elif all(col in df_ohlc.columns for col in ohlc_cols):
        ohlc_for_regime = df_ohlc[ohlc_cols].copy()
    else:
        raise ValueError(f"OHLC data must have columns: {ohlc_cols}")
    
    # Generate regime labels
    regime_labels = generate_regime_labels(ohlc_for_regime, **kwargs)
    
    # Align to features index
    regime_labels = regime_labels.reindex(df_features.index)
    
    # Combine
    result = pd.concat([df_features, regime_labels], axis=1)
    
    return result


def get_regime_masks(df: pd.DataFrame) -> dict:
    """
    Get boolean masks for each regime type.
    
    Args:
        df: DataFrame with regime labels (from generate_regime_labels)
        
    Returns:
        Dictionary of regime masks including:
            - trend, range (from regime_trend)
            - high_volatility, medium_volatility, low_volatility (from regime_volatility)
            - Combined masks (from regime_combined), e.g.:
              trend_high_vol, trend_med_vol, trend_low_vol,
              range_high_vol, range_med_vol, range_low_vol
    """
    masks = {}
    
    if 'regime_trend' in df.columns:
        masks['uptrend'] = df['regime_trend'] == 1
        masks['downtrend'] = df['regime_trend'] == -1
        masks['trend'] = df['regime_trend'] != 0   # uptrend OR downtrend (backward compat)
        masks['range'] = df['regime_trend'] == 0
        
    if 'regime_volatility' in df.columns:
        masks['high_volatility'] = df['regime_volatility'] == 1
        masks['low_volatility'] = df['regime_volatility'] == -1
        masks['medium_volatility'] = df['regime_volatility'] == 0

    if 'regime_combined' in df.columns:
        masks['uptrend_high_vol'] = df['regime_combined'] == 'UPTREND_HIGH_VOL'
        masks['uptrend_med_vol'] = df['regime_combined'] == 'UPTREND_MED_VOL'
        masks['uptrend_low_vol'] = df['regime_combined'] == 'UPTREND_LOW_VOL'
        masks['downtrend_high_vol'] = df['regime_combined'] == 'DOWNTREND_HIGH_VOL'
        masks['downtrend_med_vol'] = df['regime_combined'] == 'DOWNTREND_MED_VOL'
        masks['downtrend_low_vol'] = df['regime_combined'] == 'DOWNTREND_LOW_VOL'
        masks['range_high_vol'] = df['regime_combined'] == 'RANGE_HIGH_VOL'
        masks['range_med_vol'] = df['regime_combined'] == 'RANGE_MED_VOL'
        masks['range_low_vol'] = df['regime_combined'] == 'RANGE_LOW_VOL'
        # backward-compat aliases aggregating both trend directions
        masks['trend_high_vol'] = masks['uptrend_high_vol'] | masks['downtrend_high_vol']
        masks['trend_med_vol']  = masks['uptrend_med_vol']  | masks['downtrend_med_vol']
        masks['trend_low_vol']  = masks['uptrend_low_vol']  | masks['downtrend_low_vol']
        
    return masks


def print_regime_statistics(df: pd.DataFrame) -> None:
    """
    Print statistics about regime distribution in the data.
    
    Args:
        df: DataFrame with regime labels
    """
    print("\n" + "=" * 60)
    print("REGIME STATISTICS")
    print("=" * 60)
    
    total = len(df)
    
    if 'regime_trend' in df.columns:
        # regime_trend uses one of two encodings (see generate_regime_labels):
        #   direction_aware=True  -> {1=uptrend, -1=downtrend, 0=range}
        #   direction_aware=False -> {1=trend,   0=range}  (no -1 ever present)
        # Detect which one is in use so we don't report a structural Downtrend=0.
        direction_aware = (df['regime_trend'] == -1).any()
        print(f"\nTrend/Range Distribution:")
        if direction_aware:
            uptrend_count   = (df['regime_trend'] == 1).sum()
            downtrend_count = (df['regime_trend'] == -1).sum()
            range_count     = (df['regime_trend'] == 0).sum()
            print(f"  Uptrend:   {uptrend_count:,} ({uptrend_count/total:.1%})")
            print(f"  Downtrend: {downtrend_count:,} ({downtrend_count/total:.1%})")
            print(f"  Range:     {range_count:,} ({range_count/total:.1%})")
        else:
            trend_count = (df['regime_trend'] == 1).sum()
            range_count = (df['regime_trend'] == 0).sum()
            print(f"  Trend:     {trend_count:,} ({trend_count/total:.1%})")
            print(f"  Range:     {range_count:,} ({range_count/total:.1%})")
    
    if 'regime_volatility' in df.columns:
        high_vol = (df['regime_volatility'] == 1).sum()
        med_vol = (df['regime_volatility'] == 0).sum()
        low_vol = (df['regime_volatility'] == -1).sum()
        print(f"\nVolatility Distribution:")
        print(f"  High:   {high_vol:,} ({high_vol/total:.1%})")
        print(f"  Medium: {med_vol:,} ({med_vol/total:.1%})")
        print(f"  Low:    {low_vol:,} ({low_vol/total:.1%})")
    
    if 'regime_combined' in df.columns:
        print(f"\nCombined Regime Distribution:")
        counts = df['regime_combined'].value_counts()
        for regime, count in counts.items():
            print(f"  {regime}: {count:,} ({count/total:.1%})")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage / test
    import sys
    import os
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import ModelTrading.config.directories as dir_config
    import ModelTrading.source.python.utils.csv as csv
    
    # Load sample data
    df = csv.load_csv(
        os.path.join(dir_config.DATA_DIR, "eurusd_m15.csv"),
        filter_weekends_flag=False
    )
    
    # Generate regime labels
    regimes = generate_regime_labels(df)
    
    # Print statistics
    print_regime_statistics(regimes)
    
    # Show sample
    print("\nSample regime labels:")
    print(regimes.dropna().head(10))
