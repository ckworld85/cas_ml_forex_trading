"""
Lookahead Label Generation Module

Generates regime-stable slow labels based on percentile rank of forward returns.

Instead of asking "does price hit a fixed pip target?", this asks:
  "Is this bar's forward return in the top/bottom X% of recent history,
   AND did price never draw down more than stop_pips against the trade?"

This is regime-stable by construction:
  - High-vol periods (2022): top 20% requires large moves — but they happen
  - Low-vol periods (2024): top 20% requires smaller moves — those still happen
  - Label rate is always ~20% regardless of volatility regime (before SL filter)

Label logic:
  long_slow  = 1  if 240-bar forward return > rolling p80 of trailing window
                   AND min(low[t+1:t+horizon+1]) - close[t] > -stop_delta
  short_slow = 1  if 240-bar forward return < rolling p20 of trailing window
                   AND max(high[t+1:t+horizon+1]) - close[t] < +stop_delta

No lookahead bias in features:
  - Labels use future data (standard for supervised learning)
  - Percentile thresholds use only backward-looking historical forward returns
  - Features at training and inference time use only past data (unchanged)
"""

import numpy as np
import pandas as pd


def generate_lookahead_slow_labels(
    df: pd.DataFrame,
    horizon: int = 240,
    top_pct: float = 10.0,
    lookback_window: int = 2880,
    stop_pips: float = 35.0,
    min_pip_target: float = 40.0,
    pip_value: float = 0.0001,
    verbose: bool = True,
) -> dict:
    """
    Generate slow model labels using percentile rank of forward returns,
    filtered by a stop-loss condition and a minimum pip target.

    A label is only generated if ALL three conditions hold:
      1. Forward return is in top/bottom `top_pct`% of recent history
      2. No adverse excursion > `stop_pips` during the horizon
      3. The rolling percentile threshold itself is >= `min_pip_target` pips
         (if the market is so quiet that top-X% only means 20 pips, no label is issued)

    Args:
        df:              M15 DataFrame with 'high', 'low', 'close' columns and DatetimeIndex.
        horizon:         Bars to look forward for return calculation (default 240 = 60h).
        top_pct:         Percentage threshold for long/short labelling (default 10 → top/bottom 10%).
        lookback_window: Bars in the backward-looking reference window for percentile calculation
                         (default 2880 ≈ 30 trading days of M15 bars).
        stop_pips:       Max allowed adverse excursion in pips during the horizon (default 35).
        min_pip_target:  Minimum pip value the percentile threshold must reach (default 40).
                         Bars where top_thresh < min_pip_target get label 0.
        pip_value:       Price per pip (default 0.0001 for EUR/USD).
        verbose:         Print summary statistics.

    Returns:
        dict with keys:
            'long_slow'       : pd.Series[int] — 1 if all three conditions hold
            'short_slow'      : pd.Series[int] — 1 if all three conditions hold
            'fwd_ret'         : pd.Series[float] — raw forward return (for diagnostics)
            'top_thresh'      : pd.Series[float] — rolling top threshold used
            'bot_thresh'      : pd.Series[float] — rolling bottom threshold used
            'no_sl_long'      : pd.Series[bool] — True where SL not hit for longs
            'no_sl_short'     : pd.Series[bool] — True where SL not hit for shorts
            'thresh_above_min': pd.Series[bool] — True where threshold >= min_pip_target
    """
    close = df['close']
    low   = df['low']
    high  = df['high']

    stop_delta = stop_pips * pip_value

    # --- Forward return (uses future data — this IS the label, not a feature) ---
    fwd_ret = close.shift(-horizon) / close - 1

    # --- Rolling backward-looking percentile thresholds ---
    # shift(1) ensures the threshold at bar i uses returns from bars [i-lookback, i-1]
    # so we never use the current bar's own forward return to define its own threshold.
    top_q = 1.0 - top_pct / 100.0   # e.g. 0.80 for top 20%
    bot_q = top_pct / 100.0          # e.g. 0.20 for bottom 20%

    fwd_ret_lagged = fwd_ret.shift(1)  # shift so current bar is excluded from its own threshold

    top_thresh = fwd_ret_lagged.rolling(
        window=lookback_window,
        min_periods=lookback_window // 4,
    ).quantile(top_q)

    bot_thresh = fwd_ret_lagged.rolling(
        window=lookback_window,
        min_periods=lookback_window // 4,
    ).quantile(bot_q)

    # --- Stop-loss filter: no adverse excursion > stop_pips within horizon ---
    # Compute forward min(low) and max(high) over [t+1, t+horizon] vectorially.
    # Reverse → backward rolling → reverse back → shift(-1) gives forward window
    # excluding the entry bar itself.
    min_low_fwd  = low.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    max_high_fwd = high.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)

    no_sl_long  = (min_low_fwd  - close) > -stop_delta
    no_sl_short = (max_high_fwd - close) <  stop_delta

    # --- Minimum pip target filter: threshold must be worth trading ---
    # If the rolling p80 threshold is < min_pip_target pips, the market is too quiet
    # and no label is issued even if the bar is technically in the top X%.
    min_delta = min_pip_target * pip_value
    thresh_above_min = (top_thresh >= min_delta) & (bot_thresh <= -min_delta)

    # --- Binary labels: percentile condition AND no SL hit AND threshold >= minimum ---
    long_slow  = ((fwd_ret > top_thresh) & no_sl_long  & thresh_above_min).astype(int)
    short_slow = ((fwd_ret < bot_thresh) & no_sl_short & thresh_above_min).astype(int)

    # Mask bars where forward return or threshold is NaN (end of data / warmup)
    valid_mask = fwd_ret.notna() & top_thresh.notna()
    long_slow[~valid_mask]  = 0
    short_slow[~valid_mask] = 0

    if verbose:
        n_valid = valid_mask.sum()
        lr = long_slow[valid_mask].mean()
        sr = short_slow[valid_mask].mean()
        # Label rates at each filter stage (for diagnostics)
        lr_pct   = ((fwd_ret > top_thresh)[valid_mask]).mean()
        sr_pct   = ((fwd_ret < bot_thresh)[valid_mask]).mean()
        lr_sl    = ((fwd_ret > top_thresh) & no_sl_long)[valid_mask].mean()
        sr_sl    = ((fwd_ret < bot_thresh) & no_sl_short)[valid_mask].mean()
        n_quiet  = (~thresh_above_min[valid_mask]).sum()
        med_top  = top_thresh.dropna().median() * 10000   # pips
        med_bot  = bot_thresh.dropna().median() * 10000
        print(f"  Lookahead labels ({horizon}-bar horizon, top/bot {top_pct:.0f}%, "
              f"lookback {lookback_window} bars, SL {stop_pips:.0f} pips, min target {min_pip_target:.0f} pips):")
        print(f"    Valid bars:        {n_valid:,}  (quiet/filtered: {n_quiet:,} = {n_quiet/n_valid:.1%})")
        print(f"    Long  -- pct: {lr_pct:.1%}  -> after SL: {lr_sl:.1%}  -> after min-target: {lr:.1%}")
        print(f"    Short -- pct: {sr_pct:.1%}  -> after SL: {sr_sl:.1%}  -> after min-target: {sr:.1%}")
        print(f"    Median top threshold: {med_top:+.1f} pips")
        print(f"    Median bot threshold: {med_bot:+.1f} pips")

    return {
        'long_slow':        long_slow,
        'short_slow':       short_slow,
        'fwd_ret':          fwd_ret,
        'top_thresh':       top_thresh,
        'bot_thresh':       bot_thresh,
        'no_sl_long':       no_sl_long,
        'no_sl_short':      no_sl_short,
        'thresh_above_min': thresh_above_min,
    }


def threshold_pips_by_year(df: pd.DataFrame, horizon: int = 240,
                            top_pct: float = 20.0,
                            lookback_window: int = 2880,
                            stop_pips: float = 35.0) -> pd.DataFrame:
    """
    Diagnostic helper: show per-year median TP/SL equivalent thresholds and SL filter impact.

    Useful for understanding what forward move the lookahead label represents
    in different volatility regimes and how many labels are removed by the SL filter.

    Args:
        df:              M15 DataFrame with 'high', 'low', 'close'.
        horizon:         Forward horizon in bars.
        top_pct:         Percentile threshold.
        lookback_window: Rolling lookback for percentile.
        stop_pips:       Stop-loss filter in pips.

    Returns:
        DataFrame with columns: year, top_thresh_pips, bot_thresh_pips,
                                 label_rate_long, label_rate_short, sl_filter_long, sl_filter_short
    """
    result = generate_lookahead_slow_labels(df, horizon, top_pct, lookback_window,
                                            stop_pips=stop_pips, verbose=False)
    top_thresh_pips = result['top_thresh'] * 10000
    bot_thresh_pips = result['bot_thresh'] * 10000

    fwd_ret    = result['fwd_ret']
    top_thresh = result['top_thresh']
    bot_thresh = result['bot_thresh']

    rows = []
    for year in sorted(df.index.year.unique()):
        mask = df.index.year == year
        raw_long  = ((fwd_ret > top_thresh) & fwd_ret.notna() & top_thresh.notna())[mask].mean()
        raw_short = ((fwd_ret < bot_thresh) & fwd_ret.notna() & bot_thresh.notna())[mask].mean()
        rows.append({
            'year': year,
            'top_thresh_pips':  top_thresh_pips[mask].median(),
            'bot_thresh_pips':  bot_thresh_pips[mask].median(),
            'label_rate_long':  result['long_slow'][mask].mean(),
            'label_rate_short': result['short_slow'][mask].mean(),
            'sl_filter_long':   raw_long  - result['long_slow'][mask].mean(),
            'sl_filter_short':  raw_short - result['short_slow'][mask].mean(),
            'n_bars': mask.sum(),
        })
    return pd.DataFrame(rows)
