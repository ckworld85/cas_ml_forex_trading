"""
Window Cascade Label Generation

Two-model labeling architecture for structured trend entries:

Model 1 (Setup/Slow): Labels bars inside an *opportunity window* — a regime-gated
  trend episode where a 75-pip MFE is achievable. These bars represent the context
  in which a trade should be considered.

Model 2 (Timing/Fast): Within each setup window, labels the first bar that shows a
  First Higher Low (FHL) pullback entry pattern — a disciplined trend-continuation
  entry where price has dipped and then formed the first recovery bar.

Both models are trained separately for long and short directions. Labels are
validated by a forward MFE scan to ensure the labeled bar actually represents a
profitable opportunity.

Usage:
    from ModelTrading.source.python.labeling.window_labels import (
        compute_regime_windows,
        generate_setup_labels,
        generate_timing_labels_fhl,
        generate_timing_labels_rebound,
    )
"""

import warnings
import numpy as np
import pandas as pd
import ta

import ModelTrading.source.python.utils.forex as forex


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ATR via exponential moving average of True Range (no shift — raw indicator)."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


def _compute_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ADX(14) using the ta library. Returns raw array (no shift applied here)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adx_indicator = ta.trend.ADXIndicator(
            high=df['high'], low=df['low'], close=df['close'], window=period
        )
        return adx_indicator.adx().values


def _compute_price_efficiency(df: pd.DataFrame, period: int = 20) -> np.ndarray:
    """Price efficiency ratio: |net_move| / sum_of_abs_moves over `period` bars."""
    close = df['close']
    net = close.diff(period).abs()
    gross = close.diff().abs().rolling(period).sum()
    pe = np.where(gross > 0, net / gross, 0.0)
    return pe


def _compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(close).rolling(period, min_periods=period // 2).mean().values


def _mfe_pips(high: np.ndarray, entry_close: float, start_idx: int, end_idx: int,
              pip_value: float) -> float:
    """Maximum Favorable Excursion in pips from entry_close over bars [start_idx, end_idx)."""
    if end_idx <= start_idx:
        return 0.0
    return (float(np.max(high[start_idx:end_idx])) - entry_close) / pip_value


def _mfe_pips_short(low: np.ndarray, entry_close: float, start_idx: int, end_idx: int,
                    pip_value: float) -> float:
    """MFE in pips for a short position."""
    if end_idx <= start_idx:
        return 0.0
    return (entry_close - float(np.min(low[start_idx:end_idx]))) / pip_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_regime_windows(
    df: pd.DataFrame,
    adx_start_thresh: float = 25.0,
    adx_stop_thresh: float = 30.0,
    pe_thresh: float = 0.50,
    swing_lookback: int = 24,
    window_bars: int = 4,
    direction: str = 'long',
    sma200_period: int = 200,
    sma50_period: int = 50,
) -> pd.DataFrame:
    """
    Identify regime opportunity windows for setup label generation.

    A window opens when ALL of:
      - ADX(14) > adx_start_thresh OR price_efficiency > pe_thresh  (regime active, shifted 1)
      - close > SMA50 (long) or close < SMA50 (short)
      - close > SMA200 (long) or close < SMA200 (short)  — macro direction filter

    The window closes at the first bar where any of:
      - close < SMA50 (long) or close > SMA50 (short)  — structural break
      - ADX(14) drops below adx_start_thresh  — trend strength faded
    capped at window_bars bars from the trigger.

    Args:
        df: M15 OHLC DataFrame (DatetimeIndex, chronological order).
        adx_start_thresh: ADX threshold for regime activation (default 25).
        pe_thresh: Price efficiency threshold for regime activation (default 0.50).
        swing_lookback: Rolling bars for swing high/low (default 24 bars = 6h).
        window_bars: Maximum window duration in bars (default 12 = 3h).
        direction: 'long' or 'short'.
        sma200_period: SMA period for macro direction filter (default 200).
        sma50_period: SMA period for trend MA and window-end condition (default 50).

    Returns:
        DataFrame with same index as df and columns:
            - in_window: bool — True if bar is inside an opportunity window
            - window_id: int — sequential window ID (0 outside windows)
            - window_start: int — integer position of window trigger bar (-1 outside)
    """
    if direction not in ('long', 'short'):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    df = df.sort_index()
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Indicators (not shifted — shift applied explicitly below)
    adx_raw = _compute_adx(df)
    pe_raw = _compute_price_efficiency(df)
    sma50 = _compute_sma(close, sma50_period)
    sma200 = _compute_sma(close, sma200_period)

    # Regime (shifted 1 bar to avoid lookahead)
    regime_raw = ((adx_raw > adx_start_thresh) & (adx_raw < adx_stop_thresh)) | (pe_raw > pe_thresh)
    regime = np.roll(regime_raw, 1).astype(bool)
    regime[0] = False

    # Shifted ADX for window-end check (close window when trend strength fades)
    adx_shifted = np.roll(adx_raw, 1)
    adx_shifted[0] = np.nan

    # Macro direction (shifted 1 bar)
    is_uptrend = np.roll(close > sma200, 1)
    is_uptrend[0] = False

    # Swing reference (shifted 1 bar — rolling max/min of the PREVIOUS 24 bars)
    if direction == 'long':
        swing_ref = pd.Series(high).rolling(swing_lookback).max().shift(1).values
    else:
        swing_ref = pd.Series(low).rolling(swing_lookback).min().shift(1).values

    # Output arrays
    in_window = np.zeros(n, dtype=bool)
    window_id_arr = np.zeros(n, dtype=int)
    window_start_arr = np.full(n, -1, dtype=int)

    warmup = max(sma200_period, swing_lookback, 50)
    current_window_id = 0

    i = warmup
    while i < n:
        if not regime[i]:
            i += 1
            continue

        # Macro direction check
        if direction == 'long' and not is_uptrend[i]:
            i += 1
            continue
        if direction == 'short' and is_uptrend[i]:
            i += 1
            continue

        # Above/below SMA50
        if direction == 'long' and close[i] <= sma50[i]:
            i += 1
            continue
        if direction == 'short' and close[i] >= sma50[i]:
            i += 1
            continue

        # Valid trigger at bar i — mark window
        current_window_id += 1
        wend = min(i + window_bars, n)
        for j in range(i, wend):
            # Early-exit conditions checked BEFORE marking bar j (only after the trigger bar)
            if j > i:
                # Structural break: SMA50 violated
                if direction == 'long' and close[j] < sma50[j]:
                    break
                if direction == 'short' and close[j] > sma50[j]:
                    break
                # Trend strength faded: ADX dropped below start threshold or
                # spiked above stop threshold (avoid labeling blow-off top bars)
                if not np.isnan(adx_shifted[j]) and (adx_shifted[j] < adx_start_thresh or adx_shifted[j] > adx_stop_thresh):
                    break
            in_window[j] = True
            window_id_arr[j] = current_window_id
            window_start_arr[j] = i

        # Advance past this window to avoid overlap
        i = j + 1

    return pd.DataFrame({
        'in_window': in_window,
        'window_id': window_id_arr,
        'window_start': window_start_arr,
    }, index=df.index)


def generate_setup_labels(
    df: pd.DataFrame,
    windows: pd.DataFrame,
    mfe_pips: float = 75.0,
    atr_mult: float = 2.5,
    horizon_max: int = 288,
    direction: str = 'long',
    symbol: str = 'EURUSD',
) -> pd.Series:
    """
    Generate setup_label = 1 for bars inside validated opportunity windows.

    A window is validated if, from its trigger bar, price achieves either:
      - MFE >= mfe_pips within horizon_max bars, OR
      - MFE >= atr_mult × ATR(14) at the trigger bar

    Bars inside non-validated windows receive label 0.

    Args:
        df: M15 OHLC DataFrame.
        windows: Output of compute_regime_windows().
        mfe_pips: Minimum MFE in pips for validation (default 75).
        atr_mult: ATR multiple as alternative MFE threshold (default 2.5).
        horizon_max: Maximum forward scan in bars (default 288 = 72h).
        direction: 'long' or 'short'.
        symbol: Trading symbol for pip value lookup.

    Returns:
        Series of 0/1 integers with same index as df.
    """
    pip_value = forex.pip_value_for_symbol(symbol)
    df = df.sort_index()
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    atr = _compute_atr(df)

    in_window = windows['in_window'].reindex(df.index).fillna(False).values
    window_id = windows['window_id'].reindex(df.index).fillna(0).astype(int).values
    window_start = windows['window_start'].reindex(df.index).fillna(-1).astype(int).values

    # Pre-validate each window (keyed by window_id)
    valid_windows: set = set()
    seen_windows: set = set()

    for i in range(n):
        wid = window_id[i]
        ws = window_start[i]
        if wid == 0 or wid in seen_windows:
            continue
        seen_windows.add(wid)

        entry_close = close[ws]
        end_idx = min(ws + horizon_max + 1, n)
        atr_thresh = atr[ws] * atr_mult / pip_value  # convert to pips

        if direction == 'long':
            mfe = _mfe_pips(high, entry_close, ws + 1, end_idx, pip_value)
        else:
            mfe = _mfe_pips_short(low, entry_close, ws + 1, end_idx, pip_value)

        if mfe >= mfe_pips or mfe >= atr_thresh:
            valid_windows.add(wid)

    labels = np.zeros(n, dtype=int)
    for i in range(n):
        if in_window[i] and window_id[i] in valid_windows:
            labels[i] = 1

    return pd.Series(labels, index=df.index, name='setup_label')


def generate_timing_labels_fhl(
    df: pd.DataFrame,
    setup_labels: pd.Series,
    windows: pd.DataFrame,
    mfe_pips: float = 75.0,
    horizon_max: int = 288,
    direction: str = 'long',
    symbol: str = 'EURUSD',
    sma50_period: int = 50,
) -> pd.Series:
    """
    Generate timing_label = 1 using the First Higher Low (FHL) entry strategy.

    Within each validated setup window:
      1. Confirms at least one pullback bar (close < previous close).
      2. Labels the FIRST subsequent bar that:
         - close > close[j-1]  (first recovery close)
         - close > open        (bullish bar)
         - (close - open) / pip_value <= 35  (risk cap: body <= 35 pips = SL at open)
         - close > SMA50 (long) or close < SMA50 (short)  (trend still intact)
      3. Validates the labeled bar with MFE >= mfe_pips forward scan.

    Only the FIRST qualifying bar per window is labeled (disciplined, not every recovery).

    Args:
        df: M15 OHLC DataFrame.
        setup_labels: Output of generate_setup_labels() — only bars with label=1 are searched.
        windows: Output of compute_regime_windows().
        mfe_pips: MFE forward validation threshold in pips (default 75).
        horizon_max: Maximum forward bars for MFE scan (default 288).
        direction: 'long' or 'short'.
        symbol: Trading symbol for pip value lookup.
        sma50_period: SMA period for trend integrity check (default 50).

    Returns:
        Series of 0/1 integers with same index as df.
    """
    pip_value = forex.pip_value_for_symbol(symbol)
    df = df.sort_index()
    n = len(df)
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values

    sma50 = _compute_sma(close, sma50_period)

    setup = setup_labels.reindex(df.index).fillna(0).astype(int).values
    window_id = windows['window_id'].reindex(df.index).fillna(0).astype(int).values
    window_start = windows['window_start'].reindex(df.index).fillna(-1).astype(int).values

    labels = np.zeros(n, dtype=int)

    # Process each window
    seen_windows: set = set()
    for i in range(n):
        wid = window_id[i]
        ws = window_start[i]
        if wid == 0 or wid in seen_windows:
            continue
        if setup[i] != 1:
            continue
        seen_windows.add(wid)

        # Find end of this window
        wend = i
        while wend + 1 < n and window_id[wend + 1] == wid:
            wend += 1

        # Scan within window for FHL pattern
        pullback_seen = False
        for j in range(ws, wend + 1):
            if setup[j] != 1:
                break
            if j == 0:
                continue

            # Track pullback (any bar that closes lower than its predecessor)
            if close[j] < close[j - 1]:
                pullback_seen = True
                continue

            if not pullback_seen:
                continue

            # First recovery bar: close > prev close
            if close[j] <= close[j - 1]:
                continue

            # Must be a bullish bar
            if close[j] <= open_[j]:
                continue

            # Risk cap: body <= 35 pips (SL at bar open)
            body_pips = (close[j] - open_[j]) / pip_value
            if direction == 'short':
                body_pips = (open_[j] - close[j]) / pip_value
            if body_pips > 35.0:
                continue

            # Trend still intact at this bar
            if direction == 'long' and close[j] < sma50[j]:
                break
            if direction == 'short' and close[j] > sma50[j]:
                break

            # Forward MFE validation
            end_idx = min(j + horizon_max + 1, n)
            if direction == 'long':
                mfe = _mfe_pips(high, close[j], j + 1, end_idx, pip_value)
            else:
                mfe = _mfe_pips_short(low, close[j], j + 1, end_idx, pip_value)

            if mfe >= mfe_pips:
                labels[j] = 1
                break  # only first qualifying bar per window

    return pd.Series(labels, index=df.index, name='timing_label')


def generate_timing_labels_rebound(
    df: pd.DataFrame,
    setup_labels: pd.Series,
    windows: pd.DataFrame,
    pullback_atr: float = 2.0,
    body_atr_min: float = 0.3,
    rsi_thresh: float = 40.0,
    max_per_window: int = 2,
    gap_bars: int = 4,
    mfe_pips: float = 75.0,
    horizon_max: int = 288,
    direction: str = 'long',
    symbol: str = 'EURUSD',
    sma50_period: int = 50,
    atr_period: int = 14,
    rsi_period: int = 14,
) -> pd.Series:
    """
    Generate timing_label = 1 using a relaxed rebound entry strategy (fallback).

    Within each validated setup window, labels bars where:
      - Pullback: (close - SMA50) / ATR <= pullback_atr AND close > SMA50 (long)
      - Rebound candle: close > open AND (close - open) / ATR >= body_atr_min
      - RSI(14) > rsi_thresh
      - Risk cap: (close - open) / pip_value <= 35 pips

    Up to max_per_window bars per window are labeled, with at least gap_bars between them.
    Each candidate is validated with a forward MFE scan.

    This is the fallback strategy. Use generate_timing_labels_fhl() as primary.

    Args:
        df: M15 OHLC DataFrame.
        setup_labels: Output of generate_setup_labels().
        windows: Output of compute_regime_windows().
        pullback_atr: Maximum (close - SMA50) / ATR for pullback detection (default 2.0).
        body_atr_min: Minimum (close - open) / ATR for rebound candle (default 0.3).
        rsi_thresh: Minimum RSI for momentum confirmation (default 40.0).
        max_per_window: Maximum labels per window (default 2).
        gap_bars: Minimum bars between labels in the same window (default 4).
        mfe_pips: Forward MFE validation threshold (default 75).
        horizon_max: Maximum forward bars for MFE scan (default 288).
        direction: 'long' or 'short'.
        symbol: Trading symbol for pip value lookup.
        sma50_period: SMA period for pullback reference (default 50).
        atr_period: ATR period (default 14).
        rsi_period: RSI period (default 14).

    Returns:
        Series of 0/1 integers with same index as df.
    """
    pip_value = forex.pip_value_for_symbol(symbol)
    df = df.sort_index()
    n = len(df)
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values

    sma50 = _compute_sma(close, sma50_period)
    atr = _compute_atr(df, atr_period)

    # RSI
    delta = np.diff(close, prepend=close[0])
    avg_gain = pd.Series(np.where(delta > 0, delta, 0.0)).ewm(span=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(np.where(delta < 0, -delta, 0.0)).ewm(span=rsi_period, adjust=False).mean().values
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-10))

    setup = setup_labels.reindex(df.index).fillna(0).astype(int).values
    window_id = windows['window_id'].reindex(df.index).fillna(0).astype(int).values
    window_start = windows['window_start'].reindex(df.index).fillna(-1).astype(int).values

    labels = np.zeros(n, dtype=int)
    seen_windows: set = set()

    for i in range(n):
        wid = window_id[i]
        ws = window_start[i]
        if wid == 0 or wid in seen_windows:
            continue
        if setup[i] != 1:
            continue
        seen_windows.add(wid)

        wend = i
        while wend + 1 < n and window_id[wend + 1] == wid:
            wend += 1

        count_in_window = 0
        last_label_j = -gap_bars - 1

        for j in range(ws, wend + 1):
            if setup[j] != 1:
                break
            if count_in_window >= max_per_window:
                break
            if j - last_label_j < gap_bars:
                continue

            # Pullback criterion
            dist_sma50 = (close[j] - sma50[j]) / (atr[j] + 1e-10)
            if direction == 'long':
                if dist_sma50 > pullback_atr or close[j] <= sma50[j]:
                    continue
            else:
                dist_sma50 = (sma50[j] - close[j]) / (atr[j] + 1e-10)
                if dist_sma50 > pullback_atr or close[j] >= sma50[j]:
                    continue

            # RSI
            if rsi[j] <= rsi_thresh:
                continue

            # Rebound candle
            if direction == 'long':
                body = close[j] - open_[j]
            else:
                body = open_[j] - close[j]
            if body <= 0:
                continue
            if body / (atr[j] + 1e-10) < body_atr_min:
                continue
            if body / pip_value > 35.0:
                continue

            # Forward MFE validation
            end_idx = min(j + horizon_max + 1, n)
            if direction == 'long':
                mfe = _mfe_pips(high, close[j], j + 1, end_idx, pip_value)
            else:
                mfe = _mfe_pips_short(low, close[j], j + 1, end_idx, pip_value)

            if mfe >= mfe_pips:
                labels[j] = 1
                count_in_window += 1
                last_label_j = j

    return pd.Series(labels, index=df.index, name='timing_label')


def print_window_label_stats(
    setup_labels: pd.Series,
    timing_labels: pd.Series,
    direction: str,
    min_positive_count: int = 300,
) -> None:
    """
    Print label distribution statistics and warn if count is below threshold.

    Args:
        setup_labels: Output of generate_setup_labels().
        timing_labels: Output of generate_timing_labels_fhl() or generate_timing_labels_rebound().
        direction: 'long' or 'short' (for display).
        min_positive_count: Minimum expected positive count; warns if below (default 300).
    """
    n_total = len(setup_labels)
    n_setup = int(setup_labels.sum())
    n_timing = int(timing_labels.sum())

    print(f"\nWindow Cascade Labels ({direction}):")
    print(f"  Total bars:          {n_total:>8,}")
    print(f"  setup_label = 1:     {n_setup:>8,}  ({n_setup / n_total:.3%})")
    print(f"  timing_label = 1:    {n_timing:>8,}  ({n_timing / max(n_setup, 1):.3%} of setup bars)")

    if n_timing < min_positive_count:
        print(f"\n  WARNING: timing_label count {n_timing} < {min_positive_count}. "
              f"Consider switching to rebound strategy or relaxing entry criteria.")
    if n_setup < min_positive_count:
        print(f"\n  WARNING: setup_label count {n_setup} < {min_positive_count}. "
              f"Regime conditions may be too restrictive for this date range.")
