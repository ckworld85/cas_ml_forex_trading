"""
Dynamic Label Generation Module

Generates trading labels with adaptive targets that scale with market volatility.
Supports three modes:
- static: Fixed pip targets (backward compatible with original train.py)
- atr_scaled: Targets scale with ATR (Average True Range)
- percentile: Targets based on historical price move percentiles
- regime_conditional: Trend bars use static target/stop labels; range bars use
  mean-reversion labels (Bollinger Band touch → return to mean)
- trend_only: Trend bars use static target/stop labels; range bars get label=0
  (no entry signal — the model learns to avoid range conditions entirely)

The module also supports:
- Regime-aware adjustments when regime labels are provided
- Label denoising to suppress isolated counter-trend signals via majority-vote filter
  with asymmetric aggression in trending regimes
"""

import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Dict, Tuple

import ModelTrading.source.python.utils.forex as forex

@dataclass
class LabelConfig:
    """Configuration for label generation."""
    mode: str = 'static'  # 'static', 'atr_scaled', 'percentile'

    # Static mode parameters (backward compatibility)
    pip_target: int = 100
    stop_pips: int = 35

    # ATR-scaled mode parameters
    atr_period: int = 14
    atr_smoothing_period: int = 1920     # rolling mean over ATR to stabilize targets
                                         # 1920 M15 bars ≈ 1 month; prevents per-bar flickering
    atr_target_multiplier: float = 2.5   # target = smoothed_ATR * multiplier
    atr_stop_multiplier: float = 0.875   # stop = smoothed_ATR * multiplier (maintains ~2.86 R:R)
    slow_hysteresis_multiplier: float = 0.0  # volatility-relative arming move for slow labels; 0 disables

    # Percentile mode parameters
    percentile_target: float = 75.0      # target = X percentile of historical moves
    percentile_lookback: int = 1000      # bars for percentile calculation

    # Clipping bounds (to prevent extreme values)
    min_target_pips: float = 30.0
    max_target_pips: float = 200.0
    min_stop_pips: float = 30.0
    max_stop_pips: float = 40.0

    # Regime adjustments (multipliers applied to base ATR target)
    regime_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'TREND_HIGH_VOL': 1.2,    # More room in trending volatile markets
        'TREND_MED_VOL': 1.0,
        'TREND_LOW_VOL': 0.8,
        'RANGE_HIGH_VOL': 1.1,
        'RANGE_MED_VOL': 0.9,
        'RANGE_LOW_VOL': 0.7      # Tighter targets in ranging low-vol
    })

    # Horizons (in bars)
    horizon_min: int = 144  # 36h for M15
    horizon_max: int = 288  # 72h for M15

    # Fast model: MFE-before-MAE (entry quality)
    # "Does price move X pips in my favor before moving X pips against me?"
    mfe_threshold_pips: float = 10.0  # symmetric threshold for favorable/adverse excursion
    mfe_horizon: int = 16  # bars to look ahead (4h for M15) - tighter window for entry timing

    # Regime-conditional mode: Mean-Reversion parameters (for range regime bars)
    # Used by generate_mean_reversion_labels() and generate_regime_conditional_labels()
    mean_rev_bb_period: int = 20          # Bollinger Band period
    mean_rev_bb_std: float = 2.0          # Standard deviations for band width
    mean_rev_stop_pips: float = 30.0      # Stop if price moves this many pips further away from band
    mean_rev_horizon: int = 144           # Bars to wait for mean reversion (defaults to horizon_min)

    # Label denoising parameters
    # Suppresses isolated counter-trend labels using a local majority-vote filter
    denoise_enabled: bool = False
    # Window size as fraction of the label horizon (auto-scaled)
    # Fast window = mfe_horizon * denoise_window_fraction
    # Slow window = horizon_max * denoise_window_fraction
    denoise_window_fraction: float = 0.5
    # Threshold: suppress a label if the opposite direction's rolling mean exceeds this
    denoise_dominance_threshold: float = 0.65
    # Asymmetric denoising: multiplier applied to lower the threshold for counter-trend
    # labels in trending regimes (e.g. 0.8 means threshold becomes 0.65*0.8=0.52)
    denoise_trend_aggression: float = 0.8


def calculate_rolling_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR for each bar using shift(1) to prevent lookahead.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR calculation period

    Returns:
        Series with ATR values (shifted to prevent lookahead)
    """
    # Use shifted values to prevent lookahead bias
    high = df['high'].shift(1)
    low = df['low'].shift(1)
    close = df['close'].shift(1)
    prev_close = df['close'].shift(2)

    # True Range components
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()

    return atr


def calculate_dynamic_targets(
    df: pd.DataFrame,
    config: LabelConfig,
    regime_labels: Optional[pd.DataFrame] = None,
    pip_value: Optional[float] = None,
    symbol: str = "EURUSD"
) -> pd.DataFrame:
    """
    Calculate per-bar targets and stops based on ATR or percentile.

    Args:
        df: DataFrame with OHLC data
        config: LabelConfig with mode and parameters
        regime_labels: Optional DataFrame with 'regime_combined' column
        pip_value: Pip value for the symbol (calculated if not provided)
        symbol: Trading symbol (used if pip_value not provided)

    Returns:
        DataFrame with columns:
            - atr: raw ATR value
            - target_price_delta: target move in price units
            - stop_price_delta: stop move in price units
            - target_pips: target in pips
            - stop_pips: stop in pips
            - regime_multiplier: multiplier applied based on regime
    """
    if pip_value is None:
        pip_value = forex.pip_value_for_symbol(symbol)

    result = pd.DataFrame(index=df.index)

    if config.mode == 'static':
        # Static mode - same targets for all bars
        result['target_pips'] = float(config.pip_target)
        result['stop_pips'] = float(config.stop_pips)
        result['target_price_delta'] = config.pip_target * pip_value
        result['stop_price_delta'] = config.stop_pips * pip_value
        result['hysteresis_pips'] = 0.0
        result['hysteresis_price_delta'] = 0.0
        result['atr'] = np.nan
        result['regime_multiplier'] = 1.0

    elif config.mode == 'atr_scaled':
        # ATR-scaled mode
        atr = calculate_rolling_atr(df, config.atr_period)

        # Smooth ATR over a longer window so label boundaries change slowly
        # (weekly/monthly granularity) rather than flickering every 15 minutes
        if config.atr_smoothing_period > 1:
            atr = atr.rolling(
                window=config.atr_smoothing_period,
                min_periods=config.atr_period
            ).mean()

        result['atr'] = atr

        # Base targets from smoothed ATR
        base_target = atr * config.atr_target_multiplier
        base_stop = atr * config.atr_stop_multiplier

        # Apply regime adjustments if available
        if regime_labels is not None and 'regime_combined' in regime_labels.columns:
            # Align regime labels to df index
            regime_aligned = regime_labels['regime_combined'].reindex(df.index)
            regime_multipliers = regime_aligned.map(config.regime_adjustments).fillna(1.0)
            result['regime_multiplier'] = regime_multipliers
            base_target = base_target * regime_multipliers
            # Keep stop proportional to maintain R:R
            base_stop = base_stop * regime_multipliers
        else:
            result['regime_multiplier'] = 1.0

        # Convert to pips
        result['target_pips'] = base_target / pip_value
        result['stop_pips'] = base_stop / pip_value

        # Apply clipping
        result['target_pips'] = result['target_pips'].clip(
            config.min_target_pips, config.max_target_pips
        )
        result['stop_pips'] = result['stop_pips'].clip(
            config.min_stop_pips, config.max_stop_pips
        )

        # Convert back to price delta
        result['target_price_delta'] = result['target_pips'] * pip_value
        result['stop_price_delta'] = result['stop_pips'] * pip_value
        result['hysteresis_pips'] = 0.0
        result['hysteresis_price_delta'] = 0.0

    elif config.mode == 'percentile':
        # Percentile-based targets
        # Calculate rolling percentile of absolute price moves over horizon
        abs_moves = df['close'].diff().abs()

        # Sum of moves over horizon_min gives expected total move
        rolling_move = abs_moves.rolling(
            window=config.percentile_lookback,
            min_periods=config.percentile_lookback // 2
        ).apply(lambda x: np.percentile(x, config.percentile_target) * config.horizon_min)

        result['target_price_delta'] = rolling_move
        # Maintain same R:R as static mode
        rr_ratio = config.pip_target / config.stop_pips
        result['stop_price_delta'] = result['target_price_delta'] / rr_ratio

        result['target_pips'] = result['target_price_delta'] / pip_value
        result['stop_pips'] = result['stop_price_delta'] / pip_value

        # Apply clipping
        result['target_pips'] = result['target_pips'].clip(
            config.min_target_pips, config.max_target_pips
        )
        result['stop_pips'] = result['stop_pips'].clip(
            config.min_stop_pips, config.max_stop_pips
        )

        # Update price deltas after clipping
        result['target_price_delta'] = result['target_pips'] * pip_value
        result['stop_price_delta'] = result['stop_pips'] * pip_value
        result['hysteresis_pips'] = 0.0
        result['hysteresis_price_delta'] = 0.0

        result['atr'] = np.nan
        result['regime_multiplier'] = 1.0

    elif config.mode == 'regime_conditional':
        # Trend bars use static pip targets; range bars compute their own BB-based targets.
        # We provide static targets here so generate_dynamic_labels() works on trend bars.
        result['target_pips'] = float(config.pip_target)
        result['stop_pips'] = float(config.stop_pips)
        result['target_price_delta'] = config.pip_target * pip_value
        result['stop_price_delta'] = config.stop_pips * pip_value
        result['hysteresis_pips'] = 0.0
        result['hysteresis_price_delta'] = 0.0
        result['atr'] = np.nan
        result['regime_multiplier'] = 1.0

    else:
        raise ValueError(
            f"Unknown mode: {config.mode}. "
            "Must be 'static', 'atr_scaled', 'percentile', or 'regime_conditional'."
        )

    return result


def calculate_volatility_scaled_targets(
    df: pd.DataFrame,
    config: LabelConfig,
    volatility_series: pd.Series,
    pip_value: Optional[float] = None,
    symbol: str = "EURUSD",
    clip_bounds: bool = True,
) -> pd.DataFrame:
    """
    Build dynamic TP/SL targets from an externally supplied volatility series.

    This is used when volatility must come from another timeframe (e.g. Daily),
    then be aligned to the working timeframe (e.g. M15).

    Target model:
        target_price_delta = close * atr_target_multiplier * volatility
        stop_price_delta   = close * atr_stop_multiplier   * volatility

    Args:
        df: DataFrame with at least 'close' column and datetime index.
        config: LabelConfig; uses atr_target_multiplier / atr_stop_multiplier
            and clipping bounds.
        volatility_series: Volatility values indexed by timestamp. Should be
            pre-shifted by the caller to prevent lookahead when needed.
        pip_value: Pip value for the symbol (auto-computed when omitted).
        symbol: Trading symbol used when pip_value is omitted.
        clip_bounds: If True, clip target/stop pips to LabelConfig bounds.

    Returns:
        DataFrame with the same schema as calculate_dynamic_targets().
    """
    if pip_value is None:
        pip_value = forex.pip_value_for_symbol(symbol)

    if 'close' not in df.columns:
        raise ValueError("df must contain a 'close' column")
    if not isinstance(volatility_series, pd.Series):
        raise TypeError("volatility_series must be a pandas Series")

    result = pd.DataFrame(index=df.index)

    vol_aligned = volatility_series.reindex(df.index).ffill()
    close = df['close'].astype(float)

    result['atr'] = vol_aligned
    result['regime_multiplier'] = 1.0

    result['target_price_delta'] = close * config.atr_target_multiplier * vol_aligned
    result['stop_price_delta'] = close * config.atr_stop_multiplier * vol_aligned
    result['hysteresis_price_delta'] = close * config.slow_hysteresis_multiplier * vol_aligned

    result['target_pips'] = result['target_price_delta'] / pip_value
    result['stop_pips'] = result['stop_price_delta'] / pip_value
    result['hysteresis_pips'] = result['hysteresis_price_delta'] / pip_value

    if clip_bounds:
        result['target_pips'] = result['target_pips'].clip(
            config.min_target_pips, config.max_target_pips
        )
        result['stop_pips'] = result['stop_pips'].clip(
            config.min_stop_pips, config.max_stop_pips
        )
        result['target_price_delta'] = result['target_pips'] * pip_value
        result['stop_price_delta'] = result['stop_pips'] * pip_value
        # Hysteresis is an arming threshold, not a TP/SL barrier, so it stays unclipped.

    return result


def generate_dynamic_labels(
    df: pd.DataFrame,
    targets_df: pd.DataFrame,
    config: LabelConfig,
    symbol: str = "EURUSD",
    verbose: bool = False
) -> Dict:
    """
    Generate labels using per-bar dynamic targets/stops.

    This replaces the static label generation loops in train.py.

    Args:
        df: DataFrame with OHLC data ('high', 'low', 'close' columns)
        targets_df: DataFrame from calculate_dynamic_targets()
        config: LabelConfig with horizon parameters
        symbol: Trading symbol
        verbose: Print progress

    Returns:
        dict with:
            - long_fast, short_fast, long_slow, short_slow: binary classification targets
            - reg: regression target (unchanged - uses horizon_min)
            - metadata: DataFrame with per-bar target/stop values used
    """
    pip_value = forex.pip_value_for_symbol(symbol)

    df = df.sort_index()
    n = len(df)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # Per-bar targets and stops (for slow models — pip target)
    target_deltas = targets_df['target_price_delta'].values
    stop_deltas = targets_df['stop_price_delta'].values
    if 'hysteresis_price_delta' in targets_df.columns:
        hysteresis_deltas = targets_df['hysteresis_price_delta'].fillna(0.0).values
    else:
        hysteresis_deltas = np.zeros(n, dtype=float)

    # MFE-before-MAE threshold (for fast models — entry quality)
    mfe_delta = config.mfe_threshold_pips * pip_value
    mfe_horizon = config.mfe_horizon

    # Initialize result arrays
    hit_target_long_fast = np.zeros(n, dtype=int)
    hit_target_short_fast = np.zeros(n, dtype=int)
    hit_target_long_slow = np.zeros(n, dtype=int)
    hit_target_short_slow = np.zeros(n, dtype=int)

    # Bar at which each label was RESOLVED (a barrier was touched, or the vertical
    # horizon ran out). The races below already know it — they break on it — it was
    # simply never recorded. It is what makes the overlap between samples measurable:
    # a label resolved 5 bars later shares its outcome window with 5 neighbours, one
    # that runs the full horizon shares it with `horizon_max`. Without it, sample
    # uniqueness can only be assumed constant, which it is not.
    t1_long_fast = np.full(n, -1, dtype=int)
    t1_short_fast = np.full(n, -1, dtype=int)
    t1_long_slow = np.full(n, -1, dtype=int)
    t1_short_slow = np.full(n, -1, dtype=int)

    horizon_max = config.horizon_max

    if verbose:
        print(f"Generating dynamic labels for {n} bars...")
        print(f"  Slow models: pip target within {horizon_max} bars")
        print(f"  Fast models: MFE-before-MAE ({config.mfe_threshold_pips} pips, {mfe_horizon} bars)")
        print_every = n // 10

    for i in range(n):
        if verbose and i > 0 and i % print_every == 0:
            print(f"  Progress: {i}/{n} ({i/n:.0%})")

        start = i + 1
        if start >= n:
            continue

        entry_price = closes[i]

        # === SLOW LABELS: pip target within horizon_max ===
        if not np.isnan(target_deltas[i]) and not np.isnan(stop_deltas[i]):
            min_return_target = target_deltas[i]
            min_return_stop = stop_deltas[i]
            hysteresis_delta = float(hysteresis_deltas[i]) if i < len(hysteresis_deltas) else 0.0
            end_slow = min(i + horizon_max + 1, n)

            # Long slow
            armed_long = hysteresis_delta <= 0.0
            t1_long_slow[i] = end_slow - 1
            for j in range(start, end_slow):
                if not armed_long:
                    favorable_move = highs[j] - entry_price
                    adverse_move = entry_price - lows[j]
                    if max(favorable_move, adverse_move) >= hysteresis_delta:
                        armed_long = True
                    continue
                if lows[j] - entry_price <= -min_return_stop:
                    t1_long_slow[i] = j
                    break  # stop hit first
                if highs[j] - entry_price >= min_return_target:
                    hit_target_long_slow[i] = 1
                    t1_long_slow[i] = j
                    break

            # Short slow
            armed_short = hysteresis_delta <= 0.0
            t1_short_slow[i] = end_slow - 1
            for j in range(start, end_slow):
                if not armed_short:
                    favorable_move = entry_price - lows[j]
                    adverse_move = highs[j] - entry_price
                    if max(favorable_move, adverse_move) >= hysteresis_delta:
                        armed_short = True
                    continue
                if highs[j] - entry_price >= min_return_stop:
                    t1_short_slow[i] = j
                    break  # stop hit first
                if entry_price - lows[j] >= min_return_target:
                    hit_target_short_slow[i] = 1
                    t1_short_slow[i] = j
                    break

        # === FAST LABELS: MFE before MAE (entry quality) ===
        # "Does price move mfe_threshold pips in my favor before moving
        #  mfe_threshold pips against me, within mfe_horizon bars?"
        end_fast = min(i + mfe_horizon + 1, n)

        # Long fast: favorable = price goes up, adverse = price goes down
        t1_long_fast[i] = end_fast - 1
        for j in range(start, end_fast):
            if lows[j] - entry_price <= -mfe_delta:
                t1_long_fast[i] = j
                break  # adverse excursion first → label 0
            if highs[j] - entry_price >= mfe_delta:
                hit_target_long_fast[i] = 1
                t1_long_fast[i] = j
                break  # favorable excursion first → label 1

        # Short fast: favorable = price goes down, adverse = price goes up
        t1_short_fast[i] = end_fast - 1
        for j in range(start, end_fast):
            if highs[j] - entry_price >= mfe_delta:
                t1_short_fast[i] = j
                break  # adverse excursion first → label 0
            if entry_price - lows[j] >= mfe_delta:
                hit_target_short_fast[i] = 1
                t1_short_fast[i] = j
                break  # favorable excursion first → label 1

    # Regression target (unchanged)
    y_reg = (df["close"].shift(-config.horizon_min) / df["close"] - 1).astype(float)

    # Resolution bar per model, carried on the metadata frame so no caller signature and
    # no `labels` key contract changes (the label dict is validated key-by-key downstream).
    #
    # Stored as a TIMESTAMP, not a row position. Positions do not survive the trimming and
    # reindexing the frame goes through before it is persisted: measured 2026-08-29, the
    # persisted label_targets.parquet held t1 values up to 110,918 on a 102,423-row index,
    # which silently turned a 384-bar horizon into an 8,500-bar one and inflated every
    # uniqueness figure derived from it. A timestamp cannot be misaligned by a reindex.
    # NaT means "never resolved" — the bar is too close to the end of the data.
    #
    # Added IN PLACE, deliberately: `generate_labels` keeps its own reference to this same
    # frame and returns it alongside `labels`, so copying here would leave the caller with
    # a frame that has no t1 columns — which is exactly the silent failure this comment
    # exists to prevent (observed 2026-08-29: every model reported "no t1 column").
    _idx = df.index
    for key, arr in (('long_fast', t1_long_fast), ('short_fast', t1_short_fast),
                     ('long_slow', t1_long_slow), ('short_slow', t1_short_slow)):
        safe = np.clip(arr, 0, n - 1)
        stamps = pd.Series(_idx[safe], index=_idx)
        stamps[np.asarray(arr) < 0] = pd.NaT
        targets_df[f't1_{key}'] = stamps

    labels = {
        'long_fast': pd.Series(hit_target_long_fast, index=df.index),
        'short_fast': pd.Series(hit_target_short_fast, index=df.index),
        'long_slow': pd.Series(hit_target_long_slow, index=df.index),
        'short_slow': pd.Series(hit_target_short_slow, index=df.index),
        'reg': y_reg,
        'metadata': targets_df
    }

    return mask_fast_labels_by_slow(labels)


def generate_mean_reversion_labels(
    df: pd.DataFrame,
    config: LabelConfig,
    symbol: str = "EURUSD",
) -> Dict:
    """
    Generate mean-reversion labels for ranging market regimes.

    Entry signal: current bar's close crosses outside the Bollinger Band
    (bands are computed from the previous bb_period bars to avoid lookahead).

    - Long entry:  close < lower_bb → success when highs reach bb_middle before stop
    - Short entry: close > upper_bb → success when lows reach bb_middle before stop
    - Stop: price moves mean_rev_stop_pips further away from the entry close

    Bars without a BB touch receive NaN labels (no valid entry signal).

    Args:
        df: DataFrame with OHLC data ('open', 'high', 'low', 'close' columns)
        config: LabelConfig with mean_rev_* parameters
        symbol: Trading symbol for pip value calculation

    Returns:
        dict with keys:
            - long_slow, short_slow: binary mean-reversion success labels (NaN = no signal)
            - long_fast, short_fast: MFE-before-MAE entry quality labels (NaN = no signal)
            - reg: forward return over mean_rev_horizon bars (NaN = no signal)
            - metadata: empty DataFrame (for interface consistency with generate_dynamic_labels)
    """
    pip_value = forex.pip_value_for_symbol(symbol)

    df = df.sort_index()
    n = len(df)

    # Bollinger Bands — shift(1) so bands at bar i use close[i-period .. i-1] only
    close_shifted = df['close'].shift(1)
    bb_middle = close_shifted.rolling(
        window=config.mean_rev_bb_period,
        min_periods=config.mean_rev_bb_period
    ).mean()
    bb_std_series = close_shifted.rolling(
        window=config.mean_rev_bb_period,
        min_periods=config.mean_rev_bb_period
    ).std()
    bb_upper = bb_middle + config.mean_rev_bb_std * bb_std_series
    bb_lower = bb_middle - config.mean_rev_bb_std * bb_std_series

    # Entry signals: current close vs. shifted bands
    long_entry = (df['close'] < bb_lower) & bb_lower.notna()
    short_entry = (df['close'] > bb_upper) & bb_upper.notna()

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    bb_middle_arr = bb_middle.values
    long_entry_arr = long_entry.values
    short_entry_arr = short_entry.values

    stop_delta = config.mean_rev_stop_pips * pip_value
    mfe_delta = config.mfe_threshold_pips * pip_value
    horizon = config.mean_rev_horizon
    mfe_horizon = config.mfe_horizon

    # Initialize with NaN — bars without a BB touch get no label
    long_slow = np.full(n, np.nan)
    short_slow = np.full(n, np.nan)
    long_fast = np.full(n, np.nan)
    short_fast = np.full(n, np.nan)

    for i in range(n):
        entry_close = closes[i]
        start = i + 1
        if start >= n:
            continue

        # === LONG: close below lower BB ===
        if long_entry_arr[i]:
            target_level = bb_middle_arr[i]
            # Skip if BB middle is NaN (shouldn't happen if bb_lower is valid, but defensive check)
            if np.isnan(target_level):
                continue

            end_slow = min(i + horizon + 1, n)
            long_slow[i] = 0  # default: did not revert
            for j in range(start, end_slow):
                if lows[j] < entry_close - stop_delta:  # adverse: price fell further
                    break
                if highs[j] >= target_level:  # success: reached middle line
                    long_slow[i] = 1
                    break

            end_fast = min(i + mfe_horizon + 1, n)
            long_fast[i] = 0
            for j in range(start, end_fast):
                if lows[j] - entry_close <= -mfe_delta:
                    break  # adverse excursion first
                if highs[j] - entry_close >= mfe_delta:
                    long_fast[i] = 1
                    break

        # === SHORT: close above upper BB ===
        if short_entry_arr[i]:
            target_level = bb_middle_arr[i]
            # Skip if BB middle is NaN (shouldn't happen if bb_upper is valid, but defensive check)
            if np.isnan(target_level):
                continue

            end_slow = min(i + horizon + 1, n)
            short_slow[i] = 0
            for j in range(start, end_slow):
                if highs[j] > entry_close + stop_delta:  # adverse: price rose further
                    break
                if lows[j] <= target_level:  # success: reached middle line
                    short_slow[i] = 1
                    break

            end_fast = min(i + mfe_horizon + 1, n)
            short_fast[i] = 0
            for j in range(start, end_fast):
                if highs[j] - entry_close >= mfe_delta:
                    break  # adverse excursion first
                if entry_close - lows[j] >= mfe_delta:
                    short_fast[i] = 1
                    break

    # Regression target: forward return over mean_rev_horizon bars, only for signal bars
    y_reg_full = (df['close'].shift(-horizon) / df['close'] - 1).astype(float)
    y_reg = pd.Series(np.nan, index=df.index, dtype=float)
    has_signal = long_entry | short_entry
    y_reg[has_signal] = y_reg_full[has_signal]

    labels = {
        'long_fast': pd.Series(long_fast, index=df.index),
        'short_fast': pd.Series(short_fast, index=df.index),
        'long_slow': pd.Series(long_slow, index=df.index),
        'short_slow': pd.Series(short_slow, index=df.index),
        'reg': y_reg,
        'metadata': pd.DataFrame(index=df.index),
    }

    return mask_fast_labels_by_slow(labels)


def generate_regime_conditional_labels(
    df: pd.DataFrame,
    targets_df: pd.DataFrame,
    config: LabelConfig,
    regime_labels: pd.DataFrame,
    symbol: str = "EURUSD",
    verbose: bool = False,
) -> Dict:
    """
    Generate labels conditioned on the detected market regime.

    - Trend regime bars (regime_trend == 1): standard target/stop labels
      via generate_dynamic_labels() — the strategy that achieves AUC ~0.75.
    - Range regime bars (regime_trend == 0): Bollinger Band mean-reversion labels
      via generate_mean_reversion_labels() — suited for oscillating markets.

    A single model trained on these combined labels learns both trend-following
    and mean-reversion entry patterns.

    Args:
        df: DataFrame with OHLC data
        targets_df: DataFrame from calculate_dynamic_targets() (used for trend bars)
        config: LabelConfig with both standard and mean_rev_* parameters
        regime_labels: DataFrame with 'regime_trend' column (1=trend, 0=range),
            as produced by generate_regime_labels()
        symbol: Trading symbol
        verbose: Print per-regime label distribution

    Returns:
        dict with same structure as generate_dynamic_labels():
            long_fast, short_fast, long_slow, short_slow, reg, metadata
    """
    regime_shape = regime_labels.shape if regime_labels is not None else None

    if regime_labels is None or 'regime_trend' not in regime_labels.columns:
        raise ValueError(
            "regime_conditional mode requires a regime_labels DataFrame with a "
            "'regime_trend' column. Generate it with generate_regime_labels() first."
        )

    # Align regime to df index.
    # direction_aware mode:  regime_trend ∈ {1=uptrend, -1=downtrend, 0=range}
    # legacy mode:           regime_trend ∈ {1=trend,   0=range}
    trend_regime = regime_labels['regime_trend'].reindex(df.index).fillna(0)
    direction_aware = (trend_regime == -1).any()

    uptrend_mask   = trend_regime == 1
    downtrend_mask = trend_regime == -1
    range_mask     = trend_regime == 0
    n_up    = int(uptrend_mask.sum())
    n_down  = int(downtrend_mask.sum())
    n_range = int(range_mask.sum())

    if verbose:
        total = max(len(df), 1)
        print(f"\nRegime-conditional label generation:")
        if direction_aware:
            print(f"  Uptrend bars:   {n_up} ({n_up / total:.1%}) — long target/stop labels")
            print(f"  Downtrend bars: {n_down} ({n_down / total:.1%}) — short target/stop labels")
        else:
            print(f"  Trend bars: {n_up} ({n_up / total:.1%}) — target/stop labels (both directions)")
        print(f"  Range bars: {n_range} ({n_range / total:.1%}) — mean-reversion (BB) labels")

    # Generate both label sets on the full df so forward scans use all future data
    trend_labels = generate_dynamic_labels(df, targets_df, config, symbol, verbose=False)
    range_labels = generate_mean_reversion_labels(df, config, symbol)

    result = {}
    for key in ['long_fast', 'short_fast', 'long_slow', 'short_slow', 'reg']:
        merged = pd.Series(np.nan, index=df.index, dtype=float)

        if direction_aware:
            # Direction-aware: long only in uptrend, short only in downtrend
            if key in ('long_slow', 'long_fast'):
                merged[uptrend_mask]   = trend_labels[key][uptrend_mask]
                merged[downtrend_mask] = 0.0
            elif key in ('short_slow', 'short_fast'):
                merged[downtrend_mask] = trend_labels[key][downtrend_mask]
                merged[uptrend_mask]   = 0.0
            else:  # reg
                merged[uptrend_mask | downtrend_mask] = trend_labels[key][uptrend_mask | downtrend_mask]
        else:
            # Legacy: all trend bars get both long and short target/stop labels
            merged[uptrend_mask] = trend_labels[key][uptrend_mask]

        merged[range_mask] = range_labels[key][range_mask]

        # Range bars without a BB-touch signal are NaN — fill with 0 (no valid entry).
        # Trend NaNs near data end stay NaN (excluded from training).
        range_no_signal = range_mask & merged.isna()
        merged[range_no_signal] = 0.0

        result[key] = merged

    result['metadata'] = trend_labels['metadata']

    if verbose:
        regime_iter = (
            [('Uptrend', uptrend_mask), ('Downtrend', downtrend_mask), ('Range', range_mask)]
            if direction_aware else
            [('Trend', uptrend_mask), ('Range', range_mask)]
        )
        for regime_name, mask in regime_iter:
            print(f"\n  {regime_name} regime label rates:")
            for key in ['long_slow', 'short_slow', 'long_fast', 'short_fast']:
                valid = result[key][mask].dropna()
                if len(valid) > 0:
                    print(f"    {key}: {valid.mean():.2%}"
                          f" ({int(valid.sum())}/{len(valid)})")

    return result


def generate_trend_only_labels(
    df: pd.DataFrame,
    targets_df: pd.DataFrame,
    config: LabelConfig,
    regime_labels: pd.DataFrame,
    symbol: str = "EURUSD",
    verbose: bool = False,
) -> Dict:
    """
    Generate labels for trend bars only; range bars receive label=0 (no entry).

    - Trend regime bars: standard target/stop labels via generate_dynamic_labels()
    - Range regime bars: all labels set to 0 (model learns to avoid range conditions)

    Unlike regime_conditional, no mean-reversion BB logic is applied to range bars.
    The model is trained to predict trend entries; it sees range bars as non-events.

    Args:
        df: DataFrame with OHLC data
        targets_df: DataFrame from calculate_dynamic_targets()
        config: LabelConfig (static targets used for trend bars)
        regime_labels: DataFrame with 'regime_trend' column (1=trend, 0=range)
        symbol: Trading symbol
        verbose: Print per-regime label distribution

    Returns:
        dict with keys: long_fast, short_fast, long_slow, short_slow, reg, metadata
    """
    if regime_labels is None or 'regime_trend' not in regime_labels.columns:
        raise ValueError(
            "trend_only mode requires a regime_labels DataFrame with a "
            "'regime_trend' column. Generate it with generate_regime_labels() first."
        )

    trend_regime = regime_labels['regime_trend'].reindex(df.index).fillna(0)
    direction_aware = (trend_regime == -1).any()

    uptrend_mask   = trend_regime == 1
    downtrend_mask = trend_regime == -1
    range_mask     = trend_regime == 0
    n_up    = int(uptrend_mask.sum())
    n_down  = int(downtrend_mask.sum())
    n_range = int(range_mask.sum())

    if verbose:
        total = max(len(df), 1)
        print(f"\nTrend-only label generation:")
        if direction_aware:
            print(f"  Uptrend bars:   {n_up} ({n_up / total:.1%}) — long target/stop labels")
            print(f"  Downtrend bars: {n_down} ({n_down / total:.1%}) — short target/stop labels")
        else:
            print(f"  Trend bars: {n_up} ({n_up / total:.1%}) — target/stop labels (both directions)")
        print(f"  Range bars: {n_range} ({n_range / total:.1%}) — label=0 (no entry)")

    trend_labels = generate_dynamic_labels(df, targets_df, config, symbol, verbose=False)

    result = {}
    for key in ['long_fast', 'short_fast', 'long_slow', 'short_slow', 'reg']:
        merged = pd.Series(0.0, index=df.index, dtype=float)

        if direction_aware:
            if key in ('long_slow', 'long_fast'):
                merged[uptrend_mask]   = trend_labels[key][uptrend_mask]
                merged[downtrend_mask] = 0.0
            elif key in ('short_slow', 'short_fast'):
                merged[downtrend_mask] = trend_labels[key][downtrend_mask]
                merged[uptrend_mask]   = 0.0
            else:  # reg
                merged[uptrend_mask | downtrend_mask] = trend_labels[key][uptrend_mask | downtrend_mask]
        else:
            merged[uptrend_mask] = trend_labels[key][uptrend_mask]
        # range bars stay 0 — NaN trend bars near data end become 0 here too;
        # keep NaN so they are excluded from training the same way as other modes
        trend_nan = uptrend_mask & trend_labels[key].isna()
        if direction_aware:
            trend_nan = (uptrend_mask | downtrend_mask) & trend_labels[key].isna()
        merged[trend_nan] = np.nan

        result[key] = merged

    result['metadata'] = trend_labels['metadata']

    if verbose:
        regime_iter = (
            [('Uptrend', uptrend_mask), ('Downtrend', downtrend_mask)]
            if direction_aware else
            [('Trend', uptrend_mask)]
        )
        for regime_name, mask in regime_iter:
            print(f"\n  {regime_name} regime label rates:")
            for key in ['long_slow', 'short_slow', 'long_fast', 'short_fast']:
                valid = result[key][mask].dropna()
                if len(valid) > 0:
                    print(f"    {key}: {valid.mean():.2%}"
                          f" ({int(valid.sum())}/{len(valid)})")

    return result


def denoise_labels(
    labels: Dict,
    config: LabelConfig,
    regime_labels: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Suppress isolated counter-trend labels using a local majority-vote filter.

    For each label pair (long/short x fast/slow), a label is suppressed (set to 0)
    if the *opposite* direction dominates a surrounding window. In trending regimes,
    counter-trend labels are suppressed more aggressively (asymmetric denoising).

    The window sizes are auto-computed from the label horizons:
      - fast window = mfe_horizon * denoise_window_fraction
      - slow window = horizon_max * denoise_window_fraction

    Args:
        labels: Dict from generate_dynamic_labels() with keys:
            long_fast, short_fast, long_slow, short_slow, reg, metadata
        config: LabelConfig with denoise parameters
        regime_labels: Optional DataFrame with 'regime_trend' column (1=trend, 0=range).
            When provided, enables asymmetric denoising in trending phases.
        verbose: Print denoising statistics

    Returns:
        Tuple of:
            - labels: Dict with denoised label series (same structure as input)
            - denoise_stats: Dict with per-label suppression counts
    """
    if not config.denoise_enabled:
        return labels, {}

    # Auto-compute window sizes from horizons
    fast_window = max(4, int(config.mfe_horizon * config.denoise_window_fraction))
    slow_window = max(4, int(config.horizon_max * config.denoise_window_fraction))

    threshold = config.denoise_dominance_threshold
    trend_aggression = config.denoise_trend_aggression

    if verbose:
        print(f"\nLabel denoising:")
        print(f"  Fast window: +/-{fast_window} bars "
              f"(auto from mfe_horizon={config.mfe_horizon} x {config.denoise_window_fraction})")
        print(f"  Slow window: +/-{slow_window} bars "
              f"(auto from horizon_max={config.horizon_max} x {config.denoise_window_fraction})")
        print(f"  Dominance threshold: {threshold:.0%}")
        if regime_labels is not None:
            print(f"  Asymmetric trend aggression: {trend_aggression} "
                  f"(effective counter-trend threshold: {threshold * trend_aggression:.0%})")
        else:
            print(f"  Asymmetric denoising: DISABLED (no regime labels provided)")

    # Build regime trend mask if available (True = trending)
    is_trend = None
    if regime_labels is not None and 'regime_trend' in regime_labels.columns:
        is_trend = regime_labels['regime_trend'].reindex(
            labels['long_fast'].index
        ).fillna(0).astype(bool)

    # Determine trend direction where trending: use regression target sign
    # Positive reg = uptrend (long is with-trend, short is counter-trend)
    # Negative reg = downtrend (short is with-trend, long is counter-trend)
    trend_direction = None
    if is_trend is not None and 'reg' in labels:
        # Smooth the regression target over a window to get stable trend direction
        smoothing_window = max(fast_window, slow_window)
        reg_smooth = labels['reg'].rolling(
            window=smoothing_window, min_periods=1, center=True
        ).mean()
        trend_direction = np.sign(reg_smooth)

    # Define label pairs: (label_key, opposite_key, window, speed_name)
    pairs = [
        ('long_fast',  'short_fast', fast_window, 'fast'),
        ('short_fast', 'long_fast',  fast_window, 'fast'),
        ('long_slow',  'short_slow', slow_window, 'slow'),
        ('short_slow', 'long_slow',  slow_window, 'slow'),
    ]

    denoise_stats = {}

    for label_key, opposite_key, window, speed_name in pairs:
        if label_key not in labels or opposite_key not in labels:
            continue

        label_series = labels[label_key].copy()
        opposite_series = labels[opposite_key]

        original_count = int(label_series.sum())

        # Rolling mean of the opposite label (centered window = look both sides)
        # This is safe because both sides are training labels, not predictions
        full_window = 2 * window + 1
        opposite_dominance = opposite_series.rolling(
            window=full_window, min_periods=window, center=True
        ).mean()

        # Base suppression mask: label is 1 AND opposite dominates
        suppress_mask = (label_series == 1) & (opposite_dominance >= threshold)

        # Asymmetric denoising: lower threshold for counter-trend labels
        if is_trend is not None and trend_direction is not None:
            # long labels are counter-trend in downtrends (trend_direction == -1)
            # short labels are counter-trend in uptrends (trend_direction == +1)
            if 'long' in label_key:
                is_counter_trend = is_trend & (trend_direction < 0)
            else:  # 'short' in label_key
                is_counter_trend = is_trend & (trend_direction > 0)

            # Apply more aggressive threshold for counter-trend labels
            aggressive_threshold = threshold * trend_aggression
            aggressive_suppress = (
                (label_series == 1) &
                is_counter_trend &
                (opposite_dominance >= aggressive_threshold)
            )
            suppress_mask = suppress_mask | aggressive_suppress

        # Apply suppression
        label_series[suppress_mask] = 0
        labels[label_key] = label_series

        suppressed_count = original_count - int(label_series.sum())
        denoise_stats[label_key] = {
            'original': original_count,
            'suppressed': suppressed_count,
            'remaining': original_count - suppressed_count,
            'suppression_rate': suppressed_count / original_count if original_count > 0 else 0.0
        }

    if verbose:
        print_denoise_statistics(denoise_stats)

    return labels, denoise_stats


def print_denoise_statistics(denoise_stats: Dict) -> None:
    """
    Print denoising statistics.

    Args:
        denoise_stats: Dict from denoise_labels() with per-label suppression info
    """
    if not denoise_stats:
        return

    print(f"\nDenoising results:")
    display_names = {
        'long_fast': 'Long fast',
        'short_fast': 'Short fast',
        'long_slow': 'Long slow',
        'short_slow': 'Short slow',
    }
    for label_name, stats in denoise_stats.items():
        display = display_names.get(label_name, label_name)
        print(f"  {display}: {stats['original']} -> {stats['remaining']} "
              f"(suppressed {stats['suppressed']}, {stats['suppression_rate']:.1%})")


def validate_label_distribution(
    labels: Dict,
    expected_min: float = 0.03,
    expected_max: float = 0.15,
    period_name: str = "all"
) -> Dict:
    """
    Validate that label distribution falls within expected ranges.

    Args:
        labels: Dict from generate_dynamic_labels()
        expected_min: Minimum expected label rate
        expected_max: Maximum expected label rate
        period_name: Name for reporting

    Returns:
        dict with validation results and warnings
    """
    results = {
        'valid': True,
        'warnings': [],
        'stats': {}
    }

    for label_name in ['long_fast', 'short_fast', 'long_slow', 'short_slow']:
        if label_name not in labels:
            continue

        label_series = labels[label_name]
        # Remove NaN before calculating stats
        valid_labels = label_series.dropna()
        if len(valid_labels) == 0:
            continue

        mean_rate = valid_labels.mean()

        results['stats'][label_name] = {
            'count': int(valid_labels.sum()),
            'total': len(valid_labels),
            'rate': float(mean_rate)
        }

        if mean_rate < expected_min:
            results['warnings'].append(
                f"{label_name} rate ({mean_rate:.2%}) below minimum ({expected_min:.2%}) "
                f"in period '{period_name}' - targets may be too aggressive"
            )
            results['valid'] = False
        elif mean_rate > expected_max:
            results['warnings'].append(
                f"{label_name} rate ({mean_rate:.2%}) above maximum ({expected_max:.2%}) "
                f"in period '{period_name}' - targets may be too conservative"
            )
            results['valid'] = False

    return results


def print_label_distribution(
    labels: Dict,
    expected_min: float = 0.03,
    expected_max: float = 0.15
) -> None:
    """
    Print label distribution in the same format as train.py.

    Args:
        labels: Dict from generate_dynamic_labels()
        expected_min: Minimum expected label rate
        expected_max: Maximum expected label rate
    """
    print(f"\nLabel distribution (dynamic targets):")

    for label_name, display_name in [
        ('long_fast', 'MFE before MAE long (fast)'),
        ('short_fast', 'MFE before MAE short (fast)'),
        ('long_slow', 'Hit pip target long (slow)'),
        ('short_slow', 'Hit pip target short (slow)')
    ]:
        if label_name not in labels:
            continue

        series = labels[label_name]
        valid = series.dropna()
        count = int(valid.sum())
        total = len(valid)
        rate = valid.mean() if len(valid) > 0 else 0

        warning_low = '(WARNING: POTENTIALLY TOO LOW!)' if rate < expected_min else ''
        warning_high = '(WARNING: POTENTIALLY TOO HIGH!)' if rate > expected_max else ''

        print(f"{display_name}: {count} / {total} = {rate:.2%} {warning_low}{warning_high}")


def print_target_statistics(targets_df: pd.DataFrame) -> None:
    """
    Print statistics about dynamic targets.

    Args:
        targets_df: DataFrame from calculate_dynamic_targets()
    """
    print("\nDynamic target statistics:")

    valid = targets_df.dropna()

    if 'target_pips' in valid.columns:
        print(f"  Target pips: mean={valid['target_pips'].mean():.1f}, "
              f"std={valid['target_pips'].std():.1f}, "
              f"min={valid['target_pips'].min():.1f}, "
              f"max={valid['target_pips'].max():.1f}")

    if 'stop_pips' in valid.columns:
        print(f"  Stop pips: mean={valid['stop_pips'].mean():.1f}, "
              f"std={valid['stop_pips'].std():.1f}, "
              f"min={valid['stop_pips'].min():.1f}, "
              f"max={valid['stop_pips'].max():.1f}")

    if 'hysteresis_pips' in valid.columns and valid['hysteresis_pips'].notna().any():
        hyst = valid['hysteresis_pips']
        if float(hyst.max()) > 0.0:
            print(f"  Hysteresis pips: mean={hyst.mean():.1f}, "
                  f"std={hyst.std():.1f}, "
                  f"min={hyst.min():.1f}, "
                  f"max={hyst.max():.1f}")

    if 'atr' in valid.columns and not valid['atr'].isna().all():
        print(f"  ATR: mean={valid['atr'].mean():.6f}, "
              f"std={valid['atr'].std():.6f}")

    if 'regime_multiplier' in valid.columns:
        regime_mult = valid['regime_multiplier']
        if regime_mult.nunique() > 1:
            print(f"  Regime multipliers: min={regime_mult.min():.2f}, "
                  f"max={regime_mult.max():.2f}")


def analyze_weekly_labels(
    labels: Dict,
    df_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Analyze label distribution by week.

    Args:
        labels: Dict from generate_dynamic_labels()
        df_index: DatetimeIndex for the labels

    Returns:
        DataFrame with weekly statistics
    """
    results = []

    # Create a DataFrame with all labels
    label_df = pd.DataFrame({
        'long_fast': labels['long_fast'],
        'short_fast': labels['short_fast'],
        'long_slow': labels['long_slow'],
        'short_slow': labels['short_slow']
    }, index=df_index)

    # Group by year and week
    label_df['year'] = label_df.index.year
    label_df['week'] = label_df.index.isocalendar().week

    for (year, week), group in label_df.groupby(['year', 'week']):
        valid = group.dropna()
        if len(valid) == 0:
            continue

        results.append({
            'year': year,
            'week': week,
            'total_bars': len(valid),
            'long_count': int(valid['long_slow'].sum()),
            'short_count': int(valid['short_slow'].sum()),
            'total_labels': int(valid['long_slow'].sum() + valid['short_slow'].sum()),
            'long_rate': valid['long_slow'].mean(),
            'short_rate': valid['short_slow'].mean()
        })

    return pd.DataFrame(results)


def print_weekly_label_summary(weekly_df: pd.DataFrame, outlier_pct: float = 0.05) -> None:
    """
    Print aggregate weekly-label statistics — no per-week enumeration.

    Reports the distribution of total labels per week (mean/median/std/min/max
    plus low/high percentiles), the long vs short split, and the count of
    zero-label weeks. `outlier_pct` only controls which percentiles are shown.
    """
    if weekly_df.empty:
        print("\nWeekly label distribution: no weeks with any data.")
        return

    zero_label_weeks = int((weekly_df['total_labels'] == 0).sum())
    nonzero = weekly_df[weekly_df['total_labels'] > 0]

    print("\nWeekly label distribution:")
    if nonzero.empty:
        print(f"  {len(weekly_df)} weeks observed, all with zero labels.")
    else:
        totals = nonzero['total_labels']
        low_thr = totals.quantile(outlier_pct)
        high_thr = totals.quantile(1.0 - outlier_pct)
        print(
            f"  {len(weekly_df)} weeks ({len(nonzero)} non-zero) | "
            f"mean={totals.mean():.1f} median={totals.median():.0f} "
            f"std={totals.std():.1f} min={int(totals.min())} max={int(totals.max())} "
            f"p{int(outlier_pct*100)}={low_thr:.0f} p{int((1-outlier_pct)*100)}={high_thr:.0f}"
        )

        long_total = int(nonzero['long_count'].sum())
        short_total = int(nonzero['short_count'].sum())
        label_total = long_total + short_total
        if label_total > 0:
            print(
                f"  long={long_total} ({long_total/label_total:.1%}) "
                f"short={short_total} ({short_total/label_total:.1%}) "
                f"total={label_total}"
            )

    if zero_label_weeks > 0:
        print(f"\nWARNING: {zero_label_weeks} weeks with zero labels!")


def auto_tune_atr_multiplier(
    df: pd.DataFrame,
    config: LabelConfig,
    target_rate: float = 0.08,
    tolerance: float = 0.02,
    max_iterations: int = 10,
    symbol: str = "EURUSD"
) -> float:
    """
    Auto-tune ATR multiplier to achieve target label rate.

    Uses binary search to find optimal multiplier.

    Args:
        df: DataFrame with OHLC data
        config: Base LabelConfig
        target_rate: Target label rate (default 8%)
        tolerance: Acceptable deviation from target
        max_iterations: Maximum search iterations
        symbol: Trading symbol

    Returns:
        Optimal ATR target multiplier
    """
    pip_value = forex.pip_value_for_symbol(symbol)

    low_mult = 1.0
    high_mult = 5.0

    for iteration in range(max_iterations):
        mid_mult = (low_mult + high_mult) / 2
        test_config = replace(config, mode='atr_scaled', atr_target_multiplier=mid_mult)

        targets_df = calculate_dynamic_targets(df, test_config, pip_value=pip_value)
        labels = generate_dynamic_labels(df, targets_df, test_config, symbol)

        current_rate = labels['long_slow'].dropna().mean()

        if abs(current_rate - target_rate) < tolerance:
            print(f"Auto-tune converged at multiplier={mid_mult:.3f}, rate={current_rate:.2%}")
            return mid_mult
        elif current_rate < target_rate:
            high_mult = mid_mult  # Lower multiplier = easier targets = higher rate
        else:
            low_mult = mid_mult   # Higher multiplier = harder targets = lower rate

    print(f"Auto-tune reached max iterations, using multiplier={mid_mult:.3f}, rate={current_rate:.2%}")
    return mid_mult


def mask_fast_labels_by_slow(labels: Dict) -> Dict:
    """
    Remove fast labels unless the matching slow label is present.

    For each bar:
      - long_fast is kept only where long_slow == 1
      - short_fast is kept only where short_slow == 1

    Bars without a matching slow label are forced to 0.
    """
    if not isinstance(labels, dict):
        raise TypeError("labels must be a dict")

    for fast_key, slow_key in [('long_fast', 'long_slow'), ('short_fast', 'short_slow')]:
        if fast_key not in labels or slow_key not in labels:
            continue

        fast_series = labels[fast_key]
        slow_series = labels[slow_key]

        if not isinstance(fast_series, pd.Series):
            fast_series = pd.Series(fast_series, index=slow_series.index)
        if not isinstance(slow_series, pd.Series):
            slow_series = pd.Series(slow_series, index=fast_series.index)

        aligned = pd.Series(0, index=fast_series.index, dtype=float)
        keep_mask = slow_series == 1
        aligned.loc[keep_mask] = fast_series.loc[keep_mask].fillna(0).astype(float)
        labels[fast_key] = aligned

    return labels


if __name__ == "__main__":
    # Test the module
    import ModelTrading.config.directories as dir_config
    import ModelTrading.source.python.utils.csv as csv_utils
    from ModelTrading.source.python.labeling.regime import generate_regime_labels

    print("=" * 80)
    print("DYNAMIC LABELS MODULE TEST")
    print("=" * 80)

    # Load sample data
    data_path = os.path.join(dir_config.DATA_DIR, "eurusd_m15.csv")
    df = csv_utils.load_csv(data_path, filter_weekends_flag=False)
    print(f"\nLoaded {len(df)} bars from {data_path}")

    # Use last year of data for testing
    df = df.last('365D')
    print(f"Using last year: {len(df)} bars")

    # Generate regime labels
    print("\nGenerating regime labels...")
    regime_df = generate_regime_labels(df)

    # Test static mode
    print("\n" + "=" * 60)
    print("Testing STATIC mode")
    print("=" * 60)

    config_static = LabelConfig(mode='static')
    targets_static = calculate_dynamic_targets(df, config_static)
    labels_static = generate_dynamic_labels(df, targets_static, config_static, verbose=True)
    print_label_distribution(labels_static)

    # Test ATR-scaled mode without regime
    print("\n" + "=" * 60)
    print("Testing ATR_SCALED mode (no regime)")
    print("=" * 60)

    config_atr = LabelConfig(mode='atr_scaled')
    targets_atr = calculate_dynamic_targets(df, config_atr)
    labels_atr = generate_dynamic_labels(df, targets_atr, config_atr, verbose=True)
    print_target_statistics(targets_atr)
    print_label_distribution(labels_atr)

    # Test ATR-scaled mode with regime
    print("\n" + "=" * 60)
    print("Testing ATR_SCALED mode (with regime adjustments)")
    print("=" * 60)

    targets_atr_regime = calculate_dynamic_targets(df, config_atr, regime_labels=regime_df)
    labels_atr_regime = generate_dynamic_labels(df, targets_atr_regime, config_atr, verbose=True)
    print_target_statistics(targets_atr_regime)
    print_label_distribution(labels_atr_regime)

    # Compare weekly labels
    print("\n" + "=" * 60)
    print("Weekly label comparison (static vs dynamic)")
    print("=" * 60)

    weekly_static = analyze_weekly_labels(labels_static, df.index)
    weekly_dynamic = analyze_weekly_labels(labels_atr_regime, df.index)

    print("\nStatic mode - weeks with zero labels:")
    zero_weeks_static = weekly_static[weekly_static['total_labels'] == 0]
    print(f"  Count: {len(zero_weeks_static)}")

    print("\nDynamic mode - weeks with zero labels:")
    zero_weeks_dynamic = weekly_dynamic[weekly_dynamic['total_labels'] == 0]
    print(f"  Count: {len(zero_weeks_dynamic)}")

    print("\nDynamic mode weekly summary:")
    print(weekly_dynamic.tail(20).to_string(index=False))

    # Test denoising (symmetric, no regime)
    print("\n" + "=" * 60)
    print("Testing LABEL DENOISING (symmetric, no regime)")
    print("=" * 60)

    config_denoise = LabelConfig(mode='static', denoise_enabled=True)
    targets_denoise = calculate_dynamic_targets(df, config_denoise)
    labels_denoise = generate_dynamic_labels(df, targets_denoise, config_denoise, verbose=True)
    print("\nBefore denoising:")
    print_label_distribution(labels_denoise)
    labels_denoise, stats_sym = denoise_labels(labels_denoise, config_denoise)
    print("\nAfter denoising (symmetric):")
    print_label_distribution(labels_denoise)

    # Test denoising (asymmetric, with regime)
    print("\n" + "=" * 60)
    print("Testing LABEL DENOISING (asymmetric, with regime)")
    print("=" * 60)

    labels_denoise2 = generate_dynamic_labels(df, targets_denoise, config_denoise, verbose=False)
    print("\nBefore denoising:")
    print_label_distribution(labels_denoise2)
    labels_denoise2, stats_asym = denoise_labels(labels_denoise2, config_denoise, regime_labels=regime_df)
    print("\nAfter denoising (asymmetric with regime):")
    print_label_distribution(labels_denoise2)

    # Mask fast labels by slow label presence
    print("\n" + "=" * 60)
    print("Masking fast labels by slow label presence")
    print("=" * 60)

    labels_masked = mask_fast_labels_by_slow(labels_denoise2)

    print("\nMasked labels distribution:")
    print_label_distribution(labels_masked)
