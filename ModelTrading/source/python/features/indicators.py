import os
import sys
import warnings
import numpy as np
import pandas as pd
import ta
from .config import get_feature_config
import ModelTrading.config.directories as dir_config
import ModelTrading.source.python.utils.datahandling as datahandling
import ModelTrading.source.python.utils.forex as forex

## is the shift(1) correct here?

# ---------------------------------------------------------------------------
# ATR normalisation windows.
#
# The ATR that normalises a feature is computed over the feature's own base
# window (SMA(80) -> ATR(80)); a feature built from several windows uses the
# LARGEST of them (sma_30_cross_over_sma_80 -> ATR(80)). Features whose
# numerator is a single bar (atr_pips, atr_volatility_ratio, hl_range_atr,
# oc_range_atr) have no base window of their own and keep the default period
# as the "typical bar range" reference.
#
# The underlying `ta` indicators are built with library defaults, so their
# windows are spelled out here:
_DEFAULT_ATR_WINDOW = 14     # ta.volatility.AverageTrueRange default
_BB_WINDOW = 20              # ta.volatility.BollingerBands default window
_MACD_SLOW_WINDOW = 26       # ta.trend.MACD slow EMA
_ICHIMOKU_KIJUN_WINDOW = 26  # ta.trend.IchimokuIndicator base (kijun) line
_ICHIMOKU_SPAN_B_WINDOW = 52 # ta.trend.IchimokuIndicator senkou span B

# ---------------------------------------------------------------------------
# ATR debug — fires once per process, only for the daily timeframe.
# Set _ATR_DEBUG_TARGET to the daily bar date you want to inspect.
# In backtest:      the target date must exist in the full df index.
# In feature_server: the last daily bar sent by Java must be on/near that date.
# Output goes to stderr so it never interferes with stdout JSON.
# ---------------------------------------------------------------------------
_ATR_DEBUG_TARGET = pd.Timestamp("2026-01-20")
_ATR_DEBUG_DONE   = True  # Set to False to enable debug logging for the next matching bar (only for daily timeframe)


def _log_atr_window(atr_series: pd.Series, timeframe: str, df_len: int, vp_lookback: int, min_periods: int) -> None:
    """Log the ATR window used for volatility_percentile at the debug target bar."""
    global _ATR_DEBUG_DONE
    if _ATR_DEBUG_DONE or timeframe != "daily":
        return

    target = _ATR_DEBUG_TARGET

    # Locate the target bar: exact match in index (backtest) or last bar nearby (feature_server)
    idx = atr_series.index
    exact = [ts for ts in idx if pd.Timestamp(ts).date() == target.date()]
    if exact:
        bar_ts = exact[-1]
    elif abs((pd.Timestamp(idx[-1]) - target).days) <= 2:
        bar_ts = idx[-1]          # feature_server: last bar is the relevant bar
    else:
        return

    _ATR_DEBUG_DONE = True

    bar_pos     = idx.get_loc(bar_ts)
    start_pos   = max(0, bar_pos - vp_lookback + 1)
    window      = atr_series.iloc[start_pos : bar_pos + 1].dropna()
    current_atr = atr_series.loc[bar_ts]

    if pd.isna(current_atr):
        print(f"[ATR DEBUG] ATR is NaN at {bar_ts} — cannot log percentile", file=sys.stderr)
        return

    current_atr = float(current_atr)
    percentile  = float((current_atr <= window).mean() * 100)
    source      = f"feature_server (df_len={df_len})" if df_len < 2000 else f"backtest (df_len={df_len})"

    out = sys.stderr
    sep = "=" * 80
    print(f"\n{sep}", file=out)
    print(f"ATR DEBUG [{source}]  |  target bar: {bar_ts}", file=out)
    print(sep, file=out)
    print(f"  df total bars       : {df_len}", file=out)
    print(f"  ATR valid values    : {atr_series.notna().sum()}", file=out)
    print(f"  window bars used    : {len(window)} / {vp_lookback}  (min_periods={min_periods})", file=out)
    print(f"  current ATR         : {current_atr:.10f}", file=out)
    print(f"  ATR >= current      : {(window >= current_atr).sum()}", file=out)
    print(f"  percentile result   : {percentile:.6f}", file=out)
    print(f"\n  Full ATR window ({len(window)} bars, oldest → newest):", file=out)
    print(f"  {'i':>4}  {'date':<24}  {'ATR':>14}  note", file=out)
    print(f"  {'-'*4}  {'-'*24}  {'-'*14}  ----", file=out)
    for i, (ts, val) in enumerate(window.items()):
        note = "<-- current (ranked)" if ts == bar_ts else ""
        print(f"  {i:>4}  {str(ts):<24}  {val:.10f}  {note}", file=out)
    print(sep, file=out, flush=True)

def calculate_trend_strength(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate trend strength similar to ADX (Average Directional Index).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period for calculation
        
    Returns:
        Series with ADX-like values (0-100 scale)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        adx = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        )
        return adx.adx()


def calculate_volatility_percentile(df: pd.DataFrame, period: int = 20, lookback: int = 252) -> pd.Series:
    """
    Calculate current volatility as a percentile of historical volatility.
    
    Args:
        df: DataFrame with OHLC data
        period: ATR calculation period
        lookback: Historical lookback for percentile calculation
        
    Returns:
        Series with volatility percentile (0-100)
    """
    atr = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period
    ).average_true_range()
    
    # Calculate rolling percentile with adaptive min_periods
    # Use at least half the lookback or the period, whichever is larger
    min_periods = max(period, min(lookback // 2, 50))
    
    # Percentile RANK: the share of window values at or below the current one, so 100
    # means "the most volatile bar in the window". The comparison used to run the other
    # way round, which inverted the scale and swapped the HIGH_VOL / LOW_VOL regime
    # labels — measured on daily EUR/USD, bars tagged HIGH_VOL averaged a 55-pip range
    # against 93 pips for LOW_VOL. Keep the explicit count rather than
    # rolling().rank(pct=True): MLFeatureCalculator.java mirrors this loop, and rank()
    # resolves ties by averaging, which Java would have to reproduce exactly.
    vol_percentile = atr.rolling(window=lookback, min_periods=min_periods).apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100,
        raw=False
    )
    
    return vol_percentile


def calculate_price_direction(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """
    Determine price direction as close relative to its rolling SMA.

    Returns True where close > SMA (upward direction), False otherwise.
    Used to split direction-agnostic trend bars into UPTREND / DOWNTREND.

    Args:
        df: DataFrame with 'close' column
        period: SMA period (default 200 bars)

    Returns:
        Boolean Series — True = upward, False = downward
    """
    sma = df['close'].rolling(window=period, min_periods=period // 2).mean()
    return df['close'] > sma


def calculate_price_efficiency(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate price efficiency ratio (trend vs noise).
    
    High efficiency = strong trend, low efficiency = range/noise.
    
    Args:
        df: DataFrame with 'close' column
        period: Lookback period
        
    Returns:
        Series with efficiency ratio (0-1)
    """
    # Net price change over period
    net_change = abs(df['close'] - df['close'].shift(period))
    
    # Sum of absolute price changes
    abs_changes = abs(df['close'].diff()).rolling(window=period).sum()
    
    # Efficiency ratio
    efficiency = np.divide(
        net_change,
        abs_changes,
        out=np.zeros_like(net_change.values, dtype=float),
        where=abs_changes > 0
    )
    
    return pd.Series(efficiency, index=df.index)

def calculate_rolling_slope(close: pd.Series, window: int) -> pd.Series:
    """
    OLS regression slope of price on bar position over a rolling window.

    Fits close = a + b * k (k = 0..window-1) over each window and returns b in
    price units per bar. Exact closed form via the centred design vector:
    b = sum((k - mean(k)) * y_k) / sum((k - mean(k))^2) — no cancellation of
    large cumsums, and a NaN anywhere in a window propagates to that window's
    result (same behaviour as rolling(min_periods=window)).

    Args:
        close: price series (pass the already-shifted close for features)
        window: regression window length in bars

    Returns:
        Series of slopes, NaN for the first window-1 bars
    """
    if window < 2:
        raise ValueError(f"trend_slope window must be >= 2, got {window}")
    out = np.full(len(close), np.nan)
    if len(close) >= window:
        x = np.arange(window, dtype=np.float64)
        x_centered = x - x.mean()
        x_var = float((x_centered ** 2).sum())
        y = close.to_numpy(dtype=np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(y, window)
        out[window - 1:] = windows @ x_centered / x_var
    return pd.Series(out, index=close.index)


def calculate_run_length(x: pd.Series) -> pd.Series:
    """
    Signed run length of the sign of ``x``: +k after k consecutive positive
    values, -k after k consecutive negative values. A zero or NaN breaks the
    run (its own run of sign 0 / NaN). Vectorised via the groupby-cumsum idiom
    (see bars_since_flip); a NaN input yields NaN output.

    Args:
        x: series whose SIGN defines the run (pass e.g. close.diff() — already
           shifted by the caller like every feature input)
    """
    sgn = pd.Series(np.sign(x.to_numpy(dtype=np.float64)), index=x.index)
    grp = (sgn != sgn.shift()).cumsum()
    run = sgn.groupby(grp).cumcount() + 1
    return (run * sgn).astype(np.float64)


def calculate_rolling_percentile_rank(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """
    Rolling percentile rank of the LATEST value within its own trailing window,
    scaled to [0, 100].

    Uses rolling().rank(pct=True) — the volume_percentile_ idiom. NOTE: the
    Java-mirrored features (volatility_percentile, adx_percentile) use the
    explicit ``(x <= x.iloc[-1]).mean()`` loop instead, which differs only in
    tie handling; if a feature built on this helper ever goes live through
    MLFeatureCalculator.java, mirror THAT convention there.
    """
    if min_periods is None:
        min_periods = max(2, window // 2)
    return s.rolling(window, min_periods=min_periods).rank(pct=True) * 100.0


def calculate_wick_fractions(open_: pd.Series, high: pd.Series, low: pd.Series,
                             close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Upper and lower wick as fractions of the bar range, each in [0, 1].

    upper = (high - max(open, close)) / (high - low)
    lower = (min(open, close) - low)  / (high - low)

    A zero-range bar carries no wick information -> 0 for both (mirrors the
    bar_body_ratio convention). NaN inputs stay NaN.
    """
    bar_range = high - low
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bot = pd.concat([open_, close], axis=1).min(axis=1)
    upper = ((high - body_top) / bar_range.where(bar_range > 0)).fillna(0.0)
    lower = ((body_bot - low) / bar_range.where(bar_range > 0)).fillna(0.0)
    nan_mask = open_.isna() | high.isna() | low.isna() | close.isna()
    upper[nan_mask] = np.nan
    lower[nan_mask] = np.nan
    return upper, lower


def gap_masked_log_returns(close: pd.Series, max_gap: pd.Timedelta | None = None) -> pd.Series:
    """
    Bar-to-bar log returns with returns that CROSS a session gap masked NaN.

    A weekend/holiday gap return inside a rolling "day window" would register
    as a fabricated jump and pollute every realized moment built on it (A8).
    The gap threshold defaults to 1.5x the median index spacing, so weekends,
    holidays and the year-end 30-minute sessions are all masked while the
    regular grid passes untouched. Non-positive price ratios also yield NaN.
    """
    ratio = close / close.shift(1)
    r = pd.Series(np.log(ratio.where(ratio > 0)), index=close.index)
    if len(close.index) > 2:
        delta = close.index.to_series().diff()
        if max_gap is None:
            base = delta.median()
            max_gap = base * 1.5 if pd.notna(base) else None
        if max_gap is not None:
            r = r.mask(delta > max_gap)
    return r


def realized_moments_frame(r: pd.Series, window: int, min_frac: float = 0.75) -> pd.DataFrame:
    """
    Rolling realized moments of gap-masked returns over ``window`` bars.

    Columns (all NaN when fewer than ``min_frac * window`` valid returns):

      - ``rsv_diff``:   (RSV+ - RSV-) / RV in [-1,1] — the vol-normalised
                        signed jump variation (Patton/Sheppard 2015; the
                        up-share of RV is the affine twin (1+rsv_diff)/2)
      - ``rskew``:      sqrt(n) * sum r^3 / RV^1.5 (Amaya et al. 2015)
      - ``rkurt``:      n * sum r^4 / RV^2
      - ``jump_share``: max(0, RV_mean - BV_mean) / RV_mean in [0,1], with
                        BV = (pi/2) * mean(|r_t||r_{t-1}|) (bipower variation;
                        a pair with a masked leg is excluded)
      - ``dvol``:       sqrt(mean of downside r^2) — downside semivolatility
    """
    w = int(window)
    minp = max(4, int(round(w * float(min_frac))))
    valid = r.notna()
    r2 = r * r
    up2 = r2.where(r > 0, 0.0).where(valid)
    dn2 = r2.where(r < 0, 0.0).where(valid)
    n = valid.astype(float).rolling(w, min_periods=1).sum()
    n = n.where(n >= minp)
    s2 = r2.rolling(w, min_periods=minp).sum()
    s2u = up2.rolling(w, min_periods=minp).sum()
    s2d = dn2.rolling(w, min_periods=minp).sum()
    s3 = (r2 * r).rolling(w, min_periods=minp).sum()
    s4 = (r2 * r2).rolling(w, min_periods=minp).sum()
    s2_pos = s2.where(s2 > 0)
    rvm = s2 / n
    pair = r.abs() * r.abs().shift(1)
    bvm = pair.rolling(w, min_periods=max(3, minp - 1)).mean() * (np.pi / 2.0)
    return pd.DataFrame({
        'rsv_diff': (s2u - s2d) / s2_pos,
        'rskew': np.sqrt(n) * s3 / s2_pos.pow(1.5),
        'rkurt': n * s4 / (s2_pos * s2_pos),
        'jump_share': ((rvm - bvm) / rvm.where(rvm > 0)).clip(lower=0.0),
        'dvol': np.sqrt(s2d / n),
    }, index=r.index)


def calculate_rolling_wick_asymmetry(open_: pd.Series, high: pd.Series, low: pd.Series,
                                     close: pd.Series, window: int,
                                     min_frac: float = 0.75) -> pd.Series:
    """
    Rolling mean of (lower - upper wick fraction) over ``window`` bars: a
    dense candle path-asymmetry readout (persistent lower wicks = dip buying).
    Reuses calculate_wick_fractions — a rolling AVERAGE of the primitive, not
    an A5-style event conjunction.
    """
    upper, lower = calculate_wick_fractions(open_, high, low, close)
    minp = max(4, int(round(int(window) * float(min_frac))))
    return (lower - upper).rolling(int(window), min_periods=minp).mean()


def calculate_jump_asymmetry(r: pd.Series, window: int, z_thresh: float = 2.0,
                             min_frac: float = 0.75) -> pd.Series:
    """
    Signed tail-count imbalance: (#(r > z*sigma) - #(r < -z*sigma)) / n over
    ``window`` bars, sigma = the trailing window's return std read one bar
    back (a bar never sets its own threshold). Count-based, so it measures
    tail-event asymmetry rather than variance asymmetry (distinct from
    rsv_diff by construction).
    """
    w = int(window)
    minp = max(4, int(round(w * float(min_frac))))
    sd = r.rolling(w, min_periods=minp).std().shift(1)
    valid = r.notna() & sd.notna() & (sd > 0)
    up = ((r > z_thresh * sd) & valid).astype(float).where(valid)
    dn = ((r < -z_thresh * sd) & valid).astype(float).where(valid)
    n = valid.astype(float).rolling(w, min_periods=1).sum()
    n = n.where(n >= minp)
    return (up.rolling(w, min_periods=minp).sum()
            - dn.rolling(w, min_periods=minp).sum()) / n


def calculate_leverage_corr(r: pd.Series, window: int, min_frac: float = 0.75) -> pd.Series:
    """
    Intraday leverage effect: Pearson corr(r_{t-1}, |r_t|) over ``window``
    bars, computed from rolling sums restricted to bars where BOTH legs are
    valid (a gap-masked leg drops the pair), so the estimate is exact and
    deterministic rather than dependent on pandas' pairwise-NaN handling.
    """
    w = int(window)
    minp = max(4, int(round(w * float(min_frac))))
    a = r.shift(1)
    b = r.abs()
    valid = a.notna() & b.notna()
    a = a.where(valid)
    b = b.where(valid)
    n = valid.astype(float).rolling(w, min_periods=1).sum()
    n = n.where(n >= minp)
    sa = a.rolling(w, min_periods=minp).sum()
    sb = b.rolling(w, min_periods=minp).sum()
    sab = (a * b).rolling(w, min_periods=minp).sum()
    saa = (a * a).rolling(w, min_periods=minp).sum()
    sbb = (b * b).rolling(w, min_periods=minp).sum()
    cov = n * sab - sa * sb
    var = (n * saa - sa * sa) * (n * sbb - sb * sb)
    return cov / np.sqrt(var.where(var > 0))


def compute_realized_moments(df: pd.DataFrame, names: list, min_frac: float = 0.75,
                             jump_z: float = 2.0, cache: dict | None = None) -> pd.DataFrame:
    """
    Raw (UNSHIFTED) A8 realized-moment member series for bare ``names``.

    The value at bar t uses data through bar t's close; ``add_features``
    shifts the result by ``lag`` (the tfm/rgm end-shift precedent), the
    audits read it unshifted against outcomes that start at bar t's close.
    Used by BOTH the indicator dispatch and the audit builders in
    analytics/realized_moments_catalog.py, so they cannot drift apart.
    ``cache`` memoises the return series and per-window moment frames across
    calls (pass the same dict for every feature of one frame).
    """
    if cache is None:
        cache = {}

    def _returns() -> pd.Series:
        if 'r' not in cache:
            cache['r'] = gap_masked_log_returns(df['close'])
        return cache['r']

    def _frame(w: int) -> pd.DataFrame:
        key = ('frame', int(w))
        if key not in cache:
            cache[key] = realized_moments_frame(_returns(), int(w), min_frac)
        return cache[key]

    def _causal_z(s: pd.Series, w: int) -> pd.Series:
        past = s.shift(1)
        mu = past.rolling(w, min_periods=w // 2).mean()
        sd = past.rolling(w, min_periods=w // 2).std()
        return (s - mu) / sd.where(sd > 0)

    _CHG_BASE = {'rsv_chg': 'rsv_diff', 'rskew_chg': 'rskew', 'jump_chg': 'jump_share'}
    _Z_BASE = {'rsv_z': 'rsv_diff', 'rskew_z': 'rskew', 'jump_z': 'jump_share',
               'dvol_z': 'dvol'}
    _PCT_BASE = {'rsv_pct': 'rsv_diff', 'rskew_pct': 'rskew'}

    out = {}
    for name in names:
        parsed = _parse_realized_moment(name)
        if parsed is None:
            raise ValueError(f"'{name}' is not a realized-moments member")
        kind, ints = parsed['kind'], parsed['ints']
        if kind in ('rsv_diff', 'rskew', 'rkurt', 'jump_share'):
            s = _frame(ints[0])[kind]
        elif kind in _CHG_BASE:
            w, k = ints
            s = _frame(w)[_CHG_BASE[kind]].diff(k)
        elif kind in _Z_BASE:
            w, z = ints
            s = _causal_z(_frame(w)[_Z_BASE[kind]], z)
        elif kind in _PCT_BASE:
            w, z = ints
            s = calculate_rolling_percentile_rank(_frame(w)[_PCT_BASE[kind]], z)
        elif kind == 'rskew_ts':
            w1, w2 = ints
            s = _frame(w1)['rskew'] - _frame(w2)['rskew']
        elif kind == 'wick_asym':
            s = calculate_rolling_wick_asymmetry(df['open'], df['high'], df['low'],
                                                 df['close'], ints[0], min_frac)
        elif kind == 'jump_asym':
            s = calculate_jump_asymmetry(_returns(), ints[0], z_thresh=jump_z,
                                         min_frac=min_frac)
        elif kind == 'levcorr':
            s = calculate_leverage_corr(_returns(), ints[0], min_frac)
        else:  # pragma: no cover — the parser and this table share one roster
            raise ValueError(f"unhandled realized-moments kind '{kind}'")
        out[name] = s.astype(np.float64)
    return pd.DataFrame(out, index=df.index)


def _parse_int_suffix(feat: str, prefix: str, count: int):
    """
    Parse ``<prefix><i1>_<i2>..._<iN>`` -> (i1, .., iN) or None if the name
    does not match. Convention: the LAST integer is the largest window (the
    config warm-up regex reads only the trailing ``_(\\d+)``).
    """
    if not feat.startswith(prefix):
        return None
    parts = feat.removeprefix(prefix).split('_')
    if len(parts) != count or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def _parse_calendar_direction(feat: str):
    """
    Parse an A6 calendar-direction feature name (bare, without timeframe
    prefix) into ``{'kind': ..., 'clock': 'fed'|'ezb'|None, 'ints': tuple}``,
    or None when the name is not from that family. Roster:
    analytics/calendar_direction_catalog.py (docs/preregistration.md A6).
    """
    if feat == 'cb_convergence':
        return {'kind': 'convergence', 'clock': None, 'ints': ()}
    if feat == 'cb_cycle_diff':
        return {'kind': 'cycle_diff', 'clock': None, 'ints': ()}
    if (ws := _parse_int_suffix(feat, 'cb_news_density_', 1)):
        return {'kind': 'news_density', 'clock': None, 'ints': ws}
    if (ws := _parse_int_suffix(feat, 'nfp_friday_momo_', 1)):
        return {'kind': 'nfp_friday_momo', 'clock': None, 'ints': ws}
    for clock in ('fed', 'ezb'):
        if not feat.startswith(clock + '_'):
            continue
        rest = feat.removeprefix(clock + '_')
        if rest in ('proximity', 'recency', 'cycle_pos', 'prox_intraday',
                    'pre_ramp', 'post_ramp'):
            return {'kind': rest, 'clock': clock, 'ints': ()}
        for kind, n_ints in (('pre_momo', 1), ('day_momo', 1), ('post_cont', 1),
                             ('pre_squeeze', 2), ('post_event_cont', 1)):
            if (ws := _parse_int_suffix(rest, kind + '_', n_ints)):
                return {'kind': kind, 'clock': clock, 'ints': ws}
    return None


def _parse_session_anchor(feat: str):
    """
    Parse an A7 session-anchor feature name (bare, without timeframe prefix)
    into ``{'kind': ..., 'ints': tuple}``, or None when the name is not from
    that family. Roster: analytics/session_anchor_catalog.py
    (docs/preregistration.md A7).
    """
    if feat in ('asia_pos', 'day_range_pos', 'fix_proximity', 'tom_pos',
                'week_pos'):
        return {'kind': feat, 'ints': ()}
    for kind, n_ints in (('hour_seasonal', 2), ('dowhour_seasonal', 2),
                         ('slot_seasonal', 2), ('dow_seasonal', 2),
                         ('day_ret_z', 1), ('asia_break_dir', 1),
                         ('asia_ret', 1), ('london_first_hour', 1),
                         ('overnight_gap', 1), ('fix_momo', 1),
                         ('fix_tom_momo', 1), ('ny_open_momo', 1),
                         ('friday_unwind', 1)):
        if (ws := _parse_int_suffix(feat, kind + '_', n_ints)):
            return {'kind': kind, 'ints': ws}
    return None


def _parse_realized_moment(feat: str):
    """
    Parse an A8 realized-moments feature name (bare, without timeframe
    prefix) into ``{'kind': ..., 'ints': tuple}``, or None when the name is
    not from that family. Roster: analytics/realized_moments_catalog.py
    (docs/preregistration.md A8). Windows are M15 bars; the exact int count
    per kind keeps prefixes unambiguous (rskew_ts_96_1920 never matches the
    1-int 'rskew' case).
    """
    for kind, n_ints in (('rsv_diff', 1), ('rskew', 1), ('rkurt', 1),
                         ('jump_share', 1), ('wick_asym', 1), ('jump_asym', 1),
                         ('levcorr', 1),
                         ('rsv_chg', 2), ('rskew_chg', 2), ('jump_chg', 2),
                         ('rsv_z', 2), ('rskew_z', 2), ('jump_z', 2),
                         ('dvol_z', 2), ('rsv_pct', 2), ('rskew_pct', 2),
                         ('rskew_ts', 2)):
        if (ws := _parse_int_suffix(feat, kind + '_', n_ints)):
            return {'kind': kind, 'ints': ws}
    return None


def _parse_volume_spread(feat: str):
    """
    Parse an A9 volume-spread feature name (bare, without timeframe prefix)
    into ``{'kind': ..., 'ints': tuple}``, or None when the name is not from
    that family. Roster: analytics/volume_spread_catalog.py
    (docs/preregistration.md A9). Windows are M15 bars; slot-trailing
    integers count same-slot OCCURRENCES (~ trading days). The three legacy
    kinds at the end (volume_percentile / effort_result /
    vol_price_divergence) are matched by their own earlier cases in
    add_features — they are listed here only so the audit builders can
    compute their raw (lag-0) mirrors through one shared function.
    """
    for kind, n_ints in (('vz_ma', 2), ('vz', 1),
                         ('vimb_chg', 2), ('vimb_z', 2), ('vimb', 1),
                         ('vconf', 2), ('vpdiv', 1), ('vwret', 2),
                         ('amihud_z', 2), ('amihud_chg', 2),
                         ('sz_ma', 2), ('sz', 1), ('sexp', 2),
                         ('spread_ratio', 2), ('roll_z', 2), ('sconf', 2),
                         ('vrange_imb', 1),
                         ('vshare_ldn', 1), ('vshare_asia', 1),
                         ('break_vol', 2), ('jump_vol', 2),
                         ('volume_percentile', 1), ('effort_result', 1),
                         ('vol_price_divergence', 1)):
        if (ws := _parse_int_suffix(feat, kind + '_', n_ints)):
            return {'kind': kind, 'ints': ws}
    return None


def _parse_cross_pair(feat: str):
    """
    Parse an A10 cross-pair feature name (bare, without timeframe prefix)
    into ``{'kind': ..., 'ints': tuple}``, or None when the name is not from
    that family. Roster: analytics/cross_pair_catalog.py
    (docs/preregistration.md A10). Windows are M15 bars (4 = 1h, 16 = 4h,
    32 = 8h, 96 = 1d, 480 = 5d, 1920 = 20d); computation lives in
    features/cross_pair_intraday.py (the pair CSVs need the generation-aware
    timezone correction documented there).
    """
    for kind, n_ints in (('tri_gbp_chg_z', 2), ('tri_jpy_chg_z', 2),
                         ('tri_gbp_z', 1), ('tri_jpy_z', 1),
                         ('eurx_z', 2), ('usdb', 1), ('usdz', 2),
                         ('usd_disp', 2),
                         ('llpred_gbp', 1), ('llpred_jpy', 1),
                         ('llpred_eurx', 1),
                         ('corrbk_gbp', 2), ('corrbk_jpy', 2),
                         ('rvz_gbp', 2), ('rvz_jpy', 2),
                         ('xcons', 2), ('risk_z', 2)):
        if (ws := _parse_int_suffix(feat, kind + '_', n_ints)):
            return {'kind': kind, 'ints': ws}
    return None


def _parse_timing_execution(feat: str):
    """
    Match a B6 timing/execution feature name (bare, without timeframe
    prefix). Fixed names, windows in ``parameters:`` (tx_*) — the mtf/session
    bare-name precedent. Roster: analytics/timing_execution_catalog.py;
    computation: features/timing_execution.py.
    """
    from ModelTrading.source.python.features import timing_execution as _tx
    return feat if _tx.is_timing_execution_feature(feat) else None


def compute_volume_spread_members(df: pd.DataFrame, names: list, min_frac: float = 0.75,
                                  slot_min_frac: float = 0.4, jump_z: float = 2.0,
                                  break_ma: int = 4, cache: dict | None = None) -> pd.DataFrame:
    """
    Raw (UNSHIFTED) A9 volume-spread member series for bare ``names``.

    The value at bar t uses data through bar t's close; ``add_features``
    shifts the result by ``lag`` (the A8 end-shift precedent), the audits
    read it unshifted against outcomes that start at bar t's close. Used by
    BOTH the indicator dispatch and the audit builders in
    analytics/volume_spread_catalog.py, so they cannot drift apart.

    Inputs and their masks (measured 2026-09-07, frozen in A9):

      - tick volume: zero-volume bars (0.90 %) -> NaN in log space; the level
        drifts ~14x across years, so every quantity is a same-slot z, a
        share, or a ratio — never a level or cumulative sum.
      - spread = 2*(ask_close - mid) in pips: stalled-feed bars (mid == close,
        2.33 %) and negative rows -> NaN, never floored (a feature floor
        would inject a false constant; utils/costs.py floors because a FILL
        needs a number).
      - the volume/spread seasonality baseline is the causal trailing
        SAME-SLOT mean/std (London-local 15-min time-of-day slot, group-shift
        1 so a bar never enters its own baseline — the A7 DST precedent).
      - returns crossing session gaps are masked (gap_masked_log_returns).

    A frame without a ``volume``/``mid`` column (the live Java payload)
    degrades the affected members to NaN instead of raising.
    """
    if cache is None:
        cache = {}
    _PIP = 1e-4  # EUR/USD pip; cancels in every z/ratio, kept for readability

    def _nan() -> pd.Series:
        return pd.Series(np.nan, index=df.index)

    def _minp(w: int) -> int:
        return max(2, int(round(w * float(min_frac))))

    def _r() -> pd.Series:
        if 'r' not in cache:
            cache['r'] = gap_masked_log_returns(df['close'])
        return cache['r']

    def _vol():
        if 'vol' not in cache:
            if 'volume' not in df.columns:
                cache['vol'] = None
            else:
                v = df['volume'].astype(float)
                cache['vol'] = v.where(v > 0)
        return cache['vol']

    def _spr():
        if 'spr' not in cache:
            if 'mid' not in df.columns:
                cache['spr'] = None
            else:
                s = 2.0 * (df['close'].astype(float) - df['mid'].astype(float)) / _PIP
                cache['spr'] = s.where(s > 0)
        return cache['spr']

    def _slot() -> pd.Series:
        if 'slot' not in cache:
            idx = df.index
            utc = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
            loc = utc.tz_convert('Europe/London')
            cache['slot'] = pd.Series(np.asarray(loc.hour) * 60 + np.asarray(loc.minute),
                                      index=idx)
        return cache['slot']

    def _slotz(x: pd.Series, n: int) -> pd.Series:
        minp = min(int(n), max(10, int(round(n * float(slot_min_frac)))))
        grouped = x.groupby(_slot())
        mu = grouped.transform(
            lambda s: s.shift(1).rolling(n, min_periods=minp).mean())
        sd = grouped.transform(
            lambda s: s.shift(1).rolling(n, min_periods=minp).std())
        return (x - mu) / sd.where(sd > 0)

    def _vz(n: int) -> pd.Series:
        key = ('vz', n)
        if key not in cache:
            vol = _vol()
            cache[key] = _slotz(np.log(vol), n) if vol is not None else _nan()
        return cache[key]

    def _sz(n: int) -> pd.Series:
        key = ('sz', n)
        if key not in cache:
            spr = _spr()
            cache[key] = _slotz(np.log(spr), n) if spr is not None else _nan()
        return cache[key]

    def _causal_z(s: pd.Series, w: int) -> pd.Series:
        past = s.shift(1)
        mu = past.rolling(w, min_periods=w // 2).mean()
        sd = past.rolling(w, min_periods=w // 2).std()
        return (s - mu) / sd.where(sd > 0)

    def _ret_sd1(z: int) -> pd.Series:
        """Trailing z-bar return std, read one bar back."""
        key = ('sd1', z)
        if key not in cache:
            cache[key] = _r().rolling(z, min_periods=_minp(z)).std().shift(1)
        return cache[key]

    def _move_z(k: int, z: int) -> pd.Series:
        key = ('move_z', k, z)
        if key not in cache:
            sd = _ret_sd1(z)
            s = _r().rolling(k, min_periods=_minp(k)).sum()
            cache[key] = s / (sd.where(sd > 0) * np.sqrt(k))
        return cache[key]

    def _vimb(w: int) -> pd.Series:
        key = ('vimb', w)
        if key not in cache:
            vol = _vol()
            if vol is None:
                cache[key] = _nan()
            else:
                r = _r()
                valid = r.notna() & vol.notna()
                sv = (np.sign(r) * vol).where(valid)
                num = sv.rolling(w, min_periods=_minp(w)).sum()
                den = vol.where(valid).rolling(w, min_periods=_minp(w)).sum()
                cache[key] = num / den.where(den > 0)
        return cache[key]

    def _amihud_log(w: int) -> pd.Series:
        key = ('amihud_log', w)
        if key not in cache:
            vol = _vol()
            if vol is None:
                cache[key] = _nan()
            else:
                r = _r()
                il = (r.abs() / vol).where(r.notna() & vol.notna())
                a = il.rolling(w, min_periods=_minp(w)).mean()
                cache[key] = pd.Series(np.log(a.where(a > 0)), index=df.index)
        return cache[key]

    def _spread_mean(w: int) -> pd.Series:
        key = ('sm', w)
        if key not in cache:
            spr = _spr()
            cache[key] = (spr.rolling(w, min_periods=_minp(w)).mean()
                          if spr is not None else _nan())
        return cache[key]

    def _roll(w: int) -> pd.Series:
        """Roll (1984) effective-spread estimator over w bars, in pips:
        2*sqrt(max(-cov(dp_t, dp_{t-1}), 0)) with gap-masked price diffs and
        pair-valid rolling sums (the calculate_leverage_corr machinery)."""
        key = ('roll', w)
        if key not in cache:
            d = df['close'].astype(float).diff().where(_r().notna())
            a = d.shift(1)
            b = d
            valid = a.notna() & b.notna()
            a = a.where(valid)
            b = b.where(valid)
            minp = _minp(w)
            n = valid.astype(float).rolling(w, min_periods=1).sum()
            n = n.where(n >= minp)
            sa = a.rolling(w, min_periods=minp).sum()
            sb = b.rolling(w, min_periods=minp).sum()
            sab = (a * b).rolling(w, min_periods=minp).sum()
            cov = sab / n - (sa / n) * (sb / n)
            cache[key] = 2.0 * np.sqrt((-cov).clip(lower=0.0)) / _PIP
        return cache[key]

    def _vrange_imb(w: int) -> pd.Series:
        key = ('vrange_imb', w)
        if key not in cache:
            vol = _vol()
            if vol is None:
                cache[key] = _nan()
            else:
                minp = _minp(w)
                hi = df['high'].rolling(w, min_periods=minp).max()
                lo = df['low'].rolling(w, min_periods=minp).min()
                rng = (hi - lo).where(hi > lo)
                chpos = (df['close'] - lo) / rng
                valid = vol.notna() & chpos.notna()
                side = ((chpos >= 0.75).astype(float)
                        - (chpos <= 0.25).astype(float))
                num = (vol * side).where(valid).rolling(w, min_periods=minp).sum()
                den = vol.where(valid).rolling(w, min_periods=minp).sum()
                cache[key] = num / den.where(den > 0)
        return cache[key]

    def _vshare(kind: str, n: int) -> pd.Series:
        """Previous COMPLETED UTC day's session-window volume share minus the
        trailing n-day mean of that share (read one further day back, so the
        baseline never contains the compared day)."""
        vol = _vol()
        if vol is None:
            return _nan()
        date = pd.Series(df.index.normalize(), index=df.index)
        if kind == 'vshare_ldn':
            slot = _slot()  # London-local minutes: 08:00-17:00 local
            mask = (slot >= 8 * 60) & (slot < 17 * 60)
        else:  # vshare_asia — 00:00-07:00 UTC (Tokyo has no DST)
            mask = pd.Series(df.index.hour < 7, index=df.index)
        tot = vol.groupby(date).sum(min_count=1)
        win = vol.where(mask).groupby(date).sum(min_count=1)
        share = win / tot.where(tot > 0)
        minp = min(int(n), max(5, int(n) // 2))
        dev = share.shift(1) - share.shift(2).rolling(n, min_periods=minp).mean()
        return pd.Series(dev.reindex(date.to_numpy()).to_numpy(), index=df.index)

    def _break_vol(w: int, n: int) -> pd.Series:
        """Fresh Donchian-w close-break direction x mean vz of the last
        break_ma bars. The naked break is A5-rejected and never scored alone —
        the volume surprise is the payload; 0 when no break fired."""
        hi = df['high'].shift(1).rolling(w).max()
        lo = df['low'].shift(1).rolling(w).min()
        c = df['close']
        direction = (c > hi).astype(float) - (c < lo).astype(float)
        m = max(2, int(break_ma))
        vzma = _vz(n).rolling(m, min_periods=max(2, m - 1)).mean()
        score = (direction * vzma).where(direction != 0, 0.0)
        return score.where(hi.notna() & lo.notna() & vzma.notna() & c.notna())

    def _jump_vol(w: int, n: int) -> pd.Series:
        """sign(r) x vz on bars whose |return| exceeds jump_z trailing sigmas
        (sigma read one bar back — a bar never sets its own threshold), 0
        otherwise."""
        r = _r()
        sd = r.rolling(w, min_periods=_minp(w)).std().shift(1)
        vz = _vz(n)
        score = (np.sign(r) * vz).where(r.abs() > float(jump_z) * sd, 0.0)
        return score.where(r.notna() & sd.notna() & (sd > 0) & vz.notna())

    def _raw_volume():
        """UNMASKED tick volume — the legacy cases include zero-volume bars."""
        return df['volume'].astype(float) if 'volume' in df.columns else None

    def _atr_raw(w: int) -> pd.Series:
        key = ('atr', w)
        if key not in cache:
            if len(df) < w:
                cache[key] = _nan()
            else:
                cache[key] = ta.volatility.AverageTrueRange(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=w).average_true_range()
        return cache[key]

    out = {}
    for name in names:
        parsed = _parse_volume_spread(name)
        if parsed is None:
            raise ValueError(f"'{name}' is not a volume-spread member")
        kind, ints = parsed['kind'], parsed['ints']
        if kind == 'vz':
            s = _vz(ints[0])
        elif kind == 'vz_ma':
            k, n = ints
            s = _vz(n).rolling(k, min_periods=_minp(k)).mean()
        elif kind == 'vimb':
            s = _vimb(ints[0])
        elif kind == 'vimb_chg':
            w, k = ints
            s = _vimb(w).diff(k)
        elif kind == 'vimb_z':
            w, z = ints
            s = _causal_z(_vimb(w), z)
        elif kind == 'vconf':
            k, n = ints
            s = _move_z(k, 1920) * _vz(n).rolling(k, min_periods=_minp(k)).mean()
        elif kind == 'vpdiv':
            w = ints[0]
            r = _r()
            num = r.rolling(w, min_periods=_minp(w)).sum()
            den = r.abs().rolling(w, min_periods=_minp(w)).sum()
            s = _vimb(w) - num / den.where(den > 0)
        elif kind == 'vwret':
            w, z = ints
            vol = _vol()
            if vol is None:
                s = _nan()
            else:
                r = _r()
                valid = r.notna() & vol.notna()
                minp = _minp(w)
                pv = (vol * r).where(valid).rolling(w, min_periods=minp).sum()
                vv = vol.where(valid).rolling(w, min_periods=minp).sum()
                ew = r.where(valid).rolling(w, min_periods=minp).mean()
                sd = _ret_sd1(z)
                s = (pv / vv.where(vv > 0) - ew) / sd.where(sd > 0)
        elif kind == 'amihud_z':
            w, z = ints
            s = _causal_z(_amihud_log(w), z)
        elif kind == 'amihud_chg':
            w, k = ints
            s = _amihud_log(w).diff(k)
        elif kind == 'sz':
            s = _sz(ints[0])
        elif kind == 'sz_ma':
            k, n = ints
            s = _sz(n).rolling(k, min_periods=_minp(k)).mean()
        elif kind == 'sexp':
            k, w = ints
            base = _spread_mean(w)
            s = _spread_mean(k) / base.where(base > 0) - 1.0
        elif kind == 'spread_ratio':
            w, z = ints
            base = _spread_mean(z)
            s = _spread_mean(w) / base.where(base > 0) - 1.0
        elif kind == 'roll_z':
            w, z = ints
            s = _causal_z(_roll(w), z)
        elif kind == 'sconf':
            k, n = ints
            s = _move_z(k, 1920) * _sz(n).rolling(k, min_periods=_minp(k)).mean()
        elif kind == 'vrange_imb':
            s = _vrange_imb(ints[0])
        elif kind in ('vshare_ldn', 'vshare_asia'):
            s = _vshare(kind, ints[0])
        elif kind == 'break_vol':
            s = _break_vol(*ints)
        elif kind == 'jump_vol':
            s = _jump_vol(*ints)
        # --- raw (lag-0) mirrors of the legacy tick-volume cases, for the
        # audit builders; inside add_features these names are matched by
        # their own earlier cases, which are identical at lag 0 (pinned by
        # tests/test_volume_spread_features.py).
        elif kind == 'volume_percentile':
            v = _raw_volume()
            w = ints[0]
            s = (v.rolling(w, min_periods=max(2, w // 4)).rank(pct=True) * 100.0
                 if v is not None else _nan())
        elif kind == 'effort_result':
            v = _raw_volume()
            w = ints[0]
            if v is None:
                s = _nan()
            else:
                vol_ratio = v / v.rolling(w, min_periods=max(2, w // 4)).mean()
                range_ratio = (df['high'] - df['low']) / _atr_raw(w)
                s = (np.log(vol_ratio.where(vol_ratio > 0))
                     - np.log(range_ratio.where(range_ratio > 0)))
        elif kind == 'vol_price_divergence':
            v = _raw_volume()
            w = ints[0]
            if v is None:
                s = _nan()
            else:
                vol_pct = v.rolling(w, min_periods=max(2, w // 4)).rank(pct=True)
                mom = df['close'] - df['close'].shift(w)
                s = np.sign(mom) * (2.0 * vol_pct - 1.0)
        else:  # pragma: no cover — the parser and this table share one roster
            raise ValueError(f"unhandled volume-spread kind '{kind}'")
        vals = np.asarray(s, dtype=np.float64)
        # +/-inf (log of a warm-up-zero ATR ratio etc.) is an undefined value;
        # add_features normalises it at frame level, the audit builders read
        # this function directly — so normalise here.
        vals[~np.isfinite(vals)] = np.nan
        out[name] = pd.Series(vals, index=df.index)
    return pd.DataFrame(out, index=df.index)


def calculate_mtf_trend_agreement(slopes: pd.DataFrame, scale: float) -> pd.Series:
    """
    Trend-agreement score across timeframes from ATR-normalised slopes: [-1, 1].

    mean(tanh(slope_i / scale)) over the source columns. tanh equalises the
    per-timeframe magnitudes BEFORE averaging, so one timeframe with a large
    slope cannot outvote two disagreeing ones — without it the score would just
    be the mean slope, dominated by whichever timeframe runs hottest. All
    timeframes strongly up -> ~+1, strongly down -> ~-1, mixed signs -> ~0.

    skipna=False: while any source is still in warm-up the score is NaN rather
    than silently becoming a 2-of-3 average with a different meaning.

    Args:
        slopes: one column per timeframe, ATR-normalised slope (price drift per
                bar in units of that timeframe's ATR)
        scale: tanh scale — an |slope| equal to `scale` maps to tanh(1) = 0.76
    """
    squashed = np.tanh(slopes.astype(np.float64) / scale)
    return squashed.mean(axis=1, skipna=False)


def _atr_normalized_trend_slope(df: pd.DataFrame, window: int, lag: int, atr) -> pd.Series:
    """trend_slope_<w>: OLS slope over w bars divided by ATR (drift per bar in ATR units)."""
    atr_s = atr if isinstance(atr, pd.Series) else pd.Series(atr, index=df.index)
    atr_s = atr_s.replace(0.0, np.nan)
    return calculate_rolling_slope(df['close'].shift(lag), window) / atr_s


def add_features(df, timeframe=None, apply_shift=True, compute_timesfm=False, compute_regime=False):
    df = df.copy()
    config = get_feature_config()
    helper_features = config.get_helper_features(prefix=timeframe)
    required_features = config.get_usedInModel_features(prefix=timeframe)
    externalSourced_features = config.get_externalSourced_features(prefix=timeframe)
    lag = 1 if apply_shift else 0
    params = config.get_parameters()
    
    # Calculate all features

    ## Precompute reusable indicators
    bb = None
    sma = {}
    ema = {}
    ichi = None
    atr = {}
    macd = None
    stoch = None
    adx_indicator = None
    price_efficiency = None

    def _atr(w: int) -> pd.Series:
        """ATR over `w` bars from the shifted OHLC, created on demand and cached.

        Callers pass the window that matches the feature's own base window
        (see the module-level ATR window notes); declaring `atr_<w>` helpers in
        the config is optional — a missing window is simply computed here.
        """
        if w not in atr:
            if len(df) < w:
                # ta's AverageTrueRange raises IndexError when the frame is
                # shorter than the window (short live histories). Degrade to
                # NaN like every rolling indicator does during warm-up.
                atr[w] = pd.Series(np.nan, index=df.index)
            else:
                atr[w] = ta.volatility.AverageTrueRange(
                    high=df['high'].shift(lag), low=df['low'].shift(lag),
                    close=df['close'].shift(lag), window=w,
                ).average_true_range()
        return atr[w]

    # ------------------------------------------------------------------
    # Event-sequence family helpers (pre-registration A5). Every candidate
    # is a SIGNED event score: 0.0 = event absent, NaN = inputs undefined.
    # ------------------------------------------------------------------
    _shifted_cols: dict = {}

    def _s(col: str) -> pd.Series:
        """Memoised df[col].shift(lag)."""
        if col not in _shifted_cols:
            _shifted_cols[col] = df[col].shift(lag)
        return _shifted_cols[col]

    def _p(name: str, default):
        """Parameter with per-timeframe override: `<name>_<timeframe>` wins."""
        if timeframe is not None and f'{name}_{timeframe}' in params:
            return params[f'{name}_{timeframe}']
        return params.get(name, default)

    def _event(score: pd.Series, cond: pd.Series) -> pd.Series:
        """Gate a signed score by an event condition: score where the event
        fired, 0.0 where it did not, NaN wherever score or condition is
        undefined (warm-up, missing inputs)."""
        out = score.where(cond.fillna(False), 0.0)
        return out.where(score.notna() & cond.notna())

    _donchian_cache: dict = {}

    def _donchian(w: int):
        """(w-bar rolling high, low) of the SHIFTED bars, excluding the bar
        being scored (extra .shift(1) so a close can break its own channel)."""
        if w not in _donchian_cache:
            _donchian_cache[w] = (
                _s('high').shift(1).rolling(w).max(),
                _s('low').shift(1).rolling(w).min(),
            )
        return _donchian_cache[w]

    def _impulse_score(n: int, w: int) -> pd.Series:
        """n-bar signed body sum / ATR(w), gated on directional efficiency
        |sum bodies| / sum ranges >= impulse_min_efficiency."""
        body = _s('close') - _s('open')
        rng = _s('high') - _s('low')
        body_sum = body.rolling(n).sum()
        range_sum = rng.rolling(n).sum()
        eff = body_sum.abs() / range_sum.where(range_sum > 0)
        score = (body_sum / _atr(w).replace(0.0, np.nan)) * eff
        return _event(score, eff >= _p('impulse_min_efficiency', 0.6))

    def _break_carry(m: int, w: int):
        """Donchian(w) break levels carried forward up to m bars (the
        break-retest / failed-break machinery). Returns (up_level, dn_level,
        warmup_nan) where levels are the broken channel edge as of the bar
        AFTER the break, NaN when no break happened within the last m bars."""
        hi, lo = _donchian(w)
        c_ = _s('close')
        up_level = hi.where(c_ > hi).ffill(limit=m).shift(1)
        dn_level = lo.where(c_ < lo).ffill(limit=m).shift(1)
        return up_level, dn_level, hi.isna() | c_.isna()

    # Tick volume, shifted like every other input.
    #
    # Dukascopy delivers TICK volume (number of price updates per bar), not traded
    # volume — FX has no consolidated tape. Its LEVEL drifts by roughly 9x across
    # 2005-2026 with the feed's liquidity-provider mix, so only measures relative to
    # recent history are usable. Never use raw levels or cumulative sums: OBV /
    # AccDist / VolumePriceTrend are non-stationary on this data by construction.
    #
    # NOTE: the live payload from Java ({t,o,h,l,c}) carries no volume, so these
    # features are TRAINING-ONLY until PythonFeatureClient sends it. They resolve to
    # NaN rather than raising, so live inference degrades instead of crashing.
    vol_shifted = df['volume'].shift(lag).astype(float) if 'volume' in df.columns else None

    # Resolve TimesFM features once if any horizon-suffixed tfm_* feature is
    # requested. The horizon is encoded in the feature-name suffix (e.g.
    # tfm_mean_144). Training loads them from the precomputed CSV written by
    # data/update_timesfm_data.py; live inference (compute_timesfm=True)
    # recomputes the forecast from the bars Java sends.
    from ModelTrading.source.python.features.timesfm_features import (
        get_timesfm_features as _get_tfm,
        is_tfm_feature as _is_tfm,
    )
    _tfm_tf = timeframe if timeframe else 'm15'
    _tfm_names = [
        (f.removeprefix(f"{timeframe}_") if timeframe else f)
        for f in (set(helper_features) | set(required_features))
        if _is_tfm(f.removeprefix(f"{timeframe}_") if timeframe else f)
    ]
    tfm_cache = None
    if _tfm_names:
        tfm_cache = _get_tfm(df, _tfm_tf, _tfm_names, params, compute=compute_timesfm)

    # Calendar countdown features (until_*): timestamp-derived from the
    # pre-published FXStreet event schedule, no shift — see calendar_events.py.
    from ModelTrading.source.python.features.calendar_events import (
        EVENT_NAME_BY_FEATURE as _cal_names,
        days_since_last_event as _days_since_event,
        days_until_next_event as _days_until_event,
        high_impact_event_count as _news_counts,
        hours_until_next_event as _hours_until_event,
        is_calendar_feature as _is_cal,
        scheduled_event_density as _sched_density,
    )

    # --- A6 calendar-direction machinery (docs/preregistration.md A6) ---
    # Schedule clocks carry NO shift (pre-published calendar, the documented
    # exception); every price component below goes through _s()/_atr()
    # (shift(1)) or is anchored to strictly completed past days.
    _cal_ev_name = {'fed': _cal_names['until_fed_ir_decision'],
                    'ezb': _cal_names['until_ezb_ir_decision']}
    _cal_dir_cache: dict = {}

    def _cal_clock(kind: str, clock: str) -> pd.Series:
        """Cached per-frame clocks: 'until'/'since' in calendar days, 'hours'
        to the published decision time. NaN outside calendar coverage."""
        key = (kind, clock)
        if key not in _cal_dir_cache:
            if kind == 'until':
                s = _days_until_event(df.index, f'until_{clock}_ir_decision')
            elif kind == 'since':
                s = _days_since_event(df.index, _cal_ev_name[clock])
            else:
                s = _hours_until_event(df.index, _cal_ev_name[clock])
            _cal_dir_cache[key] = s.astype(float)
        return _cal_dir_cache[key]

    def _cal_proximity(clock: str) -> pd.Series:
        return 1.0 / (1.0 + _cal_clock('until', clock))

    def _cal_recency(clock: str) -> pd.Series:
        return 1.0 / (1.0 + _cal_clock('since', clock))

    def _cal_cycle_pos(clock: str) -> pd.Series:
        """Position in the inter-meeting cycle: 0 = just after a decision,
        1 = the decision is now (decision day: since = until = 0)."""
        since, until = _cal_clock('since', clock), _cal_clock('until', clock)
        denom = since + until
        pos = since / denom.where(denom > 0)
        return pos.mask((denom == 0) & since.notna() & until.notna(), 1.0)

    def _cal_window(kind: str, clock: str, w: int) -> pd.Series:
        """Window gate: 'pre' = 1..w calendar days ahead, 'day' = decision
        today, 'post' = 1..w days past. NaN outside coverage (news_impulse
        precedent — unknown coverage must not read as 'no event')."""
        c = _cal_clock('until' if kind in ('pre', 'day') else 'since', clock)
        cond = (c == 0) if kind == 'day' else c.between(1, w)
        return cond.where(c.notna())

    def _cal_ret_atr(k: int) -> pd.Series:
        """Trailing k-bar move of the SHIFTED closes in ATR(k) units."""
        c_ = _s('close')
        return (c_ - c_.shift(k)) / _atr(k).replace(0.0, np.nan)

    def _cal_day_ret_z(sd: int) -> pd.Series:
        """Per-calendar-day log return (first open -> last close of RAW bars
        of that day), z-scored against its trailing sd-day std. Only ever
        read for strictly PAST days (the post_event_cont gate is since >= 1),
        which is coarser than shift(1)."""
        key = ('day_ret_z', sd)
        if key not in _cal_dir_cache:
            dates = df.index.normalize()
            day_open = df['open'].groupby(dates).first()
            day_close = df['close'].groupby(dates).last()
            day_ret = np.log(day_close / day_open.where(day_open > 0))
            std = day_ret.shift(1).rolling(sd, min_periods=max(5, sd // 2)).std()
            _cal_dir_cache[key] = day_ret / std.where(std > 0)
        return _cal_dir_cache[key]

    def _cal_post_event_cont(clock: str, sd: int) -> pd.Series:
        """The decision day's own z-scored return, applied on T+1..T+w."""
        since = _cal_clock('since', clock)
        z_by_day = _cal_day_ret_z(sd)
        decision_dates = df.index.normalize() - pd.to_timedelta(
            since.to_numpy(), unit='D')  # NaN clock -> NaT -> NaN z
        z = pd.Series(z_by_day.reindex(decision_dates).to_numpy(), index=df.index)
        return _event(z, _cal_window('post', clock, _p('cb_window_days', 3)))

    def _cal_pre_squeeze(clock: str, k: int, w: int) -> pd.Series:
        """Pre-event compression (k-bar range width in its lowest w-bar
        percentile) -> signed nascent resolution move."""
        width = _s('high').rolling(k).max() - _s('low').rolling(k).min()
        width_pct = calculate_rolling_percentile_rank(width, w)
        gate = _cal_window('pre', clock, _p('cb_window_days', 3))
        cond = (gate.fillna(False) & (width_pct <= _p('squeeze_max_pctile', 20.0)))
        cond = cond.where(gate.notna() & width_pct.notna())
        return _event(_cal_ret_atr(k), cond)

    def _calendar_direction(cal: dict) -> pd.Series:
        kind, clock, ints = cal['kind'], cal['clock'], cal['ints']
        w = _p('cb_window_days', 3)
        if kind == 'proximity':
            return _cal_proximity(clock)
        if kind == 'recency':
            return _cal_recency(clock)
        if kind == 'cycle_pos':
            return _cal_cycle_pos(clock)
        if kind == 'prox_intraday':
            return 1.0 / (1.0 + _cal_clock('hours', clock) / 24.0)
        if kind == 'convergence':
            return _cal_proximity('fed') - _cal_proximity('ezb')
        if kind == 'cycle_diff':
            return _cal_cycle_pos('fed') - _cal_cycle_pos('ezb')
        if kind == 'news_density':
            return _sched_density(df.index, window_days=ints[0]).astype(float)
        if kind == 'nfp_friday_momo':
            # First trading Friday of the month — a TIMESTAMP proxy for the
            # NFP anchor (the FXStreet export carries no NFP entries; the BLS
            # schedule deviates from this rule in a minority of months, which
            # biases the proxy toward null — recorded in A6).
            cond = pd.Series((df.index.dayofweek == 4) & (df.index.day <= 7),
                             index=df.index)
            return _event(_cal_ret_atr(ints[0]), cond)
        if kind == 'pre_ramp':
            return _event(_cal_proximity(clock), _cal_window('pre', clock, w))
        if kind == 'post_ramp':
            return _event(_cal_recency(clock), _cal_window('post', clock, w))
        if kind == 'pre_momo':
            return _event(_cal_ret_atr(ints[0]), _cal_window('pre', clock, w))
        if kind == 'day_momo':
            return _event(_cal_ret_atr(ints[0]), _cal_window('day', clock, w))
        if kind == 'post_cont':
            return _event(_cal_ret_atr(ints[0]), _cal_window('post', clock, w))
        if kind == 'pre_squeeze':
            return _cal_pre_squeeze(clock, *ints)
        if kind == 'post_event_cont':
            return _cal_post_event_cont(clock, ints[0])
        raise ValueError(f"unhandled calendar-direction kind '{kind}'")

    # --- A7 session-anchor machinery (docs/preregistration.md A7) ---
    # Timestamp quantities (clocks, sessions, week/month position) carry no
    # shift — the documented hour_sin class. Every price component is _s()/
    # _atr() (shift(1)) or an aggregate of COMPLETED strictly-earlier bars
    # (Asia range, previous day's last close), gated so its first reader sits
    # at or after the completing bar's close. DST: London/NY quantities in
    # their LOCAL clocks (transitions fall on weekends — no bars), the Asia
    # window in UTC (Tokyo has no DST).
    _sa_cache: dict = {}

    # A8 realized-moments memo: shares the gap-masked return series and the
    # per-window moment frames across all family members of this frame.
    _rm_cache: dict = {}

    # A9 volume-spread memo: shares the masked volume/spread series, the
    # London-slot baseline z-scores and the imbalance frames.
    _vs_cache: dict = {}

    # A10 cross-pair memo: shares the aligned pair closes, endpoint diffs and
    # per-pair z-scores across all family members of this frame.
    _xp_cache: dict = {}

    # B6 timing/execution memo: shares the true-range/ATR series, the spread
    # and slot machinery and the percentile components across all members.
    _tx_cache: dict = {}

    def _sa(key, builder):
        if key not in _sa_cache:
            _sa_cache[key] = builder()
        return _sa_cache[key]

    def _sa_local(tz: str) -> pd.DatetimeIndex:
        def build():
            idx = df.index
            utc = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
            return utc.tz_convert(tz)
        return _sa(('local', tz), build)

    def _sa_hourfrac(tz: str) -> pd.Series:
        def build():
            loc = _sa_local(tz)
            return pd.Series(np.asarray(loc.hour) + np.asarray(loc.minute) / 60.0,
                             index=df.index)
        return _sa(('hourfrac', tz), build)

    def _sa_dow(tz: str) -> pd.Series:
        return _sa(('dow', tz),
                   lambda: pd.Series(np.asarray(_sa_local(tz).dayofweek),
                                     index=df.index))

    def _sa_date() -> pd.Series:
        """The UTC calendar day of each bar (the data's day)."""
        return _sa(('date',), lambda: pd.Series(df.index.normalize(), index=df.index))

    def _sa_seasonal(slot_keys, k: int, w: int, group_shift: int) -> pd.Series:
        """Trailing per-slot Sharpe of the next-k-bar move of the SHIFTED closes.

        r_start[t] = c1[t+k] - c1[t] is the next-k move starting at t (in
        shifted-close space); the per-slot rolling mean/std reads it only
        `group_shift` same-slot occurrences back. Causality is enforced twice:
        the group shift pushes the newest included move into a past occurrence,
        and the positional guard masks any bar whose newest included move has
        not COMPLETED yet (prev_pos + k > pos — possible on holiday-shortened
        days where same-slot spacing collapses below k).
        """
        c1 = _s('close')
        r_start = c1.diff(k).shift(-k)
        minp = min(w, max(20, int(w * float(_p('seasonal_min_frac', 0.3333333333)))))
        keys = slot_keys if isinstance(slot_keys, list) else [slot_keys]
        grouped = r_start.groupby(keys)
        mean = grouped.transform(
            lambda s: s.shift(group_shift).rolling(w, min_periods=minp).mean())
        std = grouped.transform(
            lambda s: s.shift(group_shift).rolling(w, min_periods=minp).std())
        score = mean / std.where(std > 0)
        pos = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        prev_pos = pos.groupby(keys).shift(group_shift)
        return score.where(prev_pos + k <= pos)

    def _sa_asia_agg(col: str, how: str) -> pd.Series:
        """Per-day aggregate over the completed Asia window (00:00..06:45 UTC),
        broadcast to the day's bars and readable only from 07:00 UTC on — the
        last Asia bar closes at 07:00, so every reader sits at/after that
        close (the completed-window equivalent of shift(1))."""
        asia = df[col].where(pd.Series(df.index.hour < 7, index=df.index))
        agg = asia.groupby(_sa_date()).transform(how)
        return agg.where(pd.Series(df.index.hour >= 7, index=df.index))

    def _sa_prev_day_last_close() -> pd.Series:
        def build():
            date = _sa_date()
            by_day = df['close'].groupby(date).last()
            prev = by_day.shift(1)
            return pd.Series(prev.reindex(date.to_numpy()).to_numpy(), index=df.index)
        return _sa(('prev_day_last_close',), build)

    def _sa_ret_atr(k: int) -> pd.Series:
        c1 = _s('close')
        return c1.diff(k) / _atr(k).replace(0.0, np.nan)

    def _sa_tom_counter() -> pd.Series:
        """Turn-of-month counter by PURE weekday arithmetic (np.busday, no
        holiday calendar): -W..-1 = last W weekdays of the month, +1..+W =
        first W weekdays, 0 outside. Holidays make the true trading calendar
        deviate occasionally, which biases toward null (A6 proxy precedent)."""
        def build():
            W = int(_p('tom_window_days', 2))
            days = pd.DatetimeIndex(_sa_date().unique())
            d64 = days.to_numpy().astype('datetime64[D]')
            per = days.to_period('M')
            start64 = per.start_time.to_numpy().astype('datetime64[D]')
            next64 = (per + 1).start_time.to_numpy().astype('datetime64[D]')
            k_end = np.busday_count(d64, next64)      # busdays in [d, next month)
            k_start = np.busday_count(start64, d64)   # busdays before d this month
            val = np.zeros(len(days))
            is_bd = np.is_busday(d64)
            val[is_bd & (k_end >= 1) & (k_end <= W)] = -k_end[is_bd & (k_end >= 1) & (k_end <= W)]
            sel = is_bd & (k_start < W) & (val == 0)
            val[sel] = k_start[sel] + 1
            by_day = pd.Series(val, index=days)
            return pd.Series(by_day.reindex(_sa_date().to_numpy()).to_numpy(),
                             index=df.index)
        return _sa(('tom_counter',), build)

    def _session_anchor(sa: dict) -> pd.Series:
        kind, ints = sa['kind'], sa['ints']
        if kind == 'hour_seasonal':
            k, w = ints
            return _sa_seasonal(_sa_hourfrac('Europe/London'), k, w, group_shift=1)
        if kind == 'dowhour_seasonal':
            k, w = ints
            return _sa_seasonal([_sa_dow('Europe/London'),
                                 _sa_hourfrac('Europe/London')], k, w, group_shift=1)
        if kind == 'slot_seasonal':
            # 4h layer: the UTC bar grid (it does not move with DST — a
            # London-clock slot would split bars). group_shift 1 = yesterday's
            # same slot; the positional guard covers shortened days.
            k, w = ints
            return _sa_seasonal(pd.Series(df.index.hour, index=df.index),
                                k, w, group_shift=1)
        if kind == 'dow_seasonal':
            # 6 same-dow bars per day -> shift 6 = last week's same position.
            k, w = ints
            return _sa_seasonal(_sa_dow('UTC'), k, w, group_shift=6)
        if kind == 'day_ret_z':
            # Day move so far (incl. the overnight gap: anchored at the
            # previous day's last close), z-scored against the trailing
            # same-UTC-time-of-day distribution. dr[t] only reads data <= t,
            # so the per-slot baseline needs no completion guard.
            (w,) = ints
            dr = _s('close') - _sa_prev_day_last_close()
            minp = min(w, max(20, int(w * float(_p('seasonal_min_frac', 0.3333333333)))))
            grouped = dr.groupby([_sa_hourfrac('UTC')])
            mean = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=minp).mean())
            std = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=minp).std())
            return (dr - mean) / std.where(std > 0)
        if kind == 'asia_pos':
            ah, al = _sa_asia_agg('high', 'max'), _sa_asia_agg('low', 'min')
            half = (ah - al) / 2.0
            return (_s('close') - (ah + al) / 2.0) / half.where(half > 0)
        if kind == 'asia_break_dir':
            (w,) = ints
            ah, al = _sa_asia_agg('high', 'max'), _sa_asia_agg('low', 'min')
            c1 = _s('close')
            atr_s = _atr(w).replace(0.0, np.nan)
            score = pd.Series(0.0, index=df.index)
            score = score.mask(c1 > ah, (c1 - ah) / atr_s)
            score = score.mask(c1 < al, (c1 - al) / atr_s)
            return score.where(c1.notna() & ah.notna() & al.notna() & atr_s.notna())
        if kind == 'asia_ret':
            (w,) = ints
            first_o = _sa_asia_agg('open', 'first')
            last_c = _sa_asia_agg('close', 'last')
            return (last_c - first_o) / _atr(w).replace(0.0, np.nan)
        if kind == 'london_first_hour':
            # The London 08:00-09:00 move, readable from 09:00 London on (the
            # 08:45 bar closes at 09:00) and carried through the day.
            (w,) = ints
            lf = _sa_hourfrac('Europe/London')
            in_fh = (lf >= 8.0) & (lf < 9.0)
            fh = df['close'].where(in_fh).groupby(_sa_date()).transform('last') \
                - df['open'].where(in_fh).groupby(_sa_date()).transform('first')
            return (fh / _atr(w).replace(0.0, np.nan)).where(lf >= 9.0)
        if kind == 'overnight_gap':
            # First close of the day vs the previous day's last close, NaN on
            # the day's first bar (its own close has not printed yet).
            (w,) = ints
            date = _sa_date()
            first_c = df['close'].groupby(date).transform('first')
            gap = (first_c - _sa_prev_day_last_close()) / _atr(w).replace(0.0, np.nan)
            intraday_pos = pd.Series(date.groupby(date).cumcount().to_numpy(),
                                     index=df.index)
            return gap.where(intraday_pos >= 1)
        if kind == 'day_range_pos':
            # Running position of the shifted close in the day's range SO FAR,
            # built from shifted bars masked to the same day (the first bar of
            # a day must not inherit the previous day's last bar).
            date = _sa_date()
            same_day = date.eq(date.shift(1))
            h1 = _s('high').where(same_day)
            l1 = _s('low').where(same_day)
            dh = h1.groupby(date).cummax()
            dl = l1.groupby(date).cummin()
            rng = dh - dl
            return (2.0 * (_s('close') - dl) / rng.where(rng > 0) - 1.0)
        if kind == 'fix_proximity':
            hu = (16.0 - _sa_hourfrac('Europe/London')) % 24.0
            return 1.0 / (1.0 + hu)
        if kind == 'fix_momo':
            (k,) = ints
            lf = _sa_hourfrac('Europe/London')
            return _event(_sa_ret_atr(k), (lf >= 15.0) & (lf < 16.0))
        if kind == 'fix_tom_momo':
            (k,) = ints
            lf = _sa_hourfrac('Europe/London')
            gate = (lf >= 14.0) & (lf < 16.0) & (_sa_tom_counter() < 0)
            return _event(_sa_ret_atr(k), gate)
        if kind == 'tom_pos':
            return _sa_tom_counter()
        if kind == 'week_pos':
            dow = _sa_dow('UTC')
            wp = (dow + _sa_hourfrac('UTC') / 24.0) / 5.0
            return wp.where(dow < 5)
        if kind == 'ny_open_momo':
            (k,) = ints
            nyh = _sa_hourfrac('America/New_York')
            return _event(_sa_ret_atr(k), (nyh >= 8.0) & (nyh < 9.0))
        if kind == 'friday_unwind':
            (k,) = ints
            lf = _sa_hourfrac('Europe/London')
            gate = (_sa_dow('Europe/London') == 4) & (lf >= 16.0)
            return _event(_sa_ret_atr(k), gate)
        raise ValueError(f"unhandled session-anchor kind '{kind}'")

    # Resolve ML regime-model features once if any rgm_* feature is requested.
    # Training loads them from the precomputed CSV written by
    # data/update_regime_model_data.py; live inference (compute_regime=True)
    # loads the persisted fitted model and infers causally from the bars Java
    # sends. Mirrors the TimesFM resolution above.
    from ModelTrading.source.python.features.regime_model import (
        get_regime_features as _get_rgm,
        is_rgm_feature as _is_rgm,
    )
    _rgm_tf = timeframe if timeframe else 'm15'
    _rgm_names = [
        (f.removeprefix(f"{timeframe}_") if timeframe else f)
        for f in (set(helper_features) | set(required_features))
        if _is_rgm(f.removeprefix(f"{timeframe}_") if timeframe else f)
    ]
    rgm_cache = None
    if _rgm_names:
        rgm_cache = _get_rgm(df, _rgm_tf, _rgm_names, params, compute=compute_regime)

    for feat in helper_features:
        feat = feat if timeframe is None else feat.removeprefix(f"{timeframe}_")

        if feat in df.columns:
            continue  # Already exists
        else: 
            match feat:
                case "mid":
                    df["mid"] = (df["high"].shift(lag) + df["low"].shift(lag)) / 2
                case feat if feat.startswith('sma_') and 'cross_over' not in feat and 'slope_acceleration' not in feat and (w := feat.removeprefix('sma_')).isdigit():
                    w = int(w)
                    if w not in sma:
                        sma[w] = ta.trend.SMAIndicator(close=df['close'].shift(lag), window=w).sma_indicator()
                case feat if feat.startswith("ema_") and (w := feat.removeprefix('ema_')).isdigit():
                    w = int(w)
                    if w not in ema:
                        ema[w] = ta.trend.EMAIndicator(close=df['close'].shift(lag), window=w).ema_indicator()
                case "bb_upper" | "bb_middle" | "bb_lower" | "bb_width" | "bb_percent" | "bb_position" | "bb_squeeze" | "bb_deviation":
                    if bb is None:
                        bb = ta.volatility.BollingerBands(close=df['close'].shift(lag))
                case "ichimoku_conv" | "ichimoku_base" | "ichimoku_a" | "ichimoku_b":
                    if ichi is None:
                        ichi = ta.trend.IchimokuIndicator(high=df['high'].shift(lag), low=df['low'].shift(lag))
                case "atr":
                    # Back-compat helper name: the default-window ATR.
                    _atr(_DEFAULT_ATR_WINDOW)
                case feat if feat.startswith('atr_') and (w := feat.removeprefix('atr_')).isdigit():
                    _atr(int(w))
                case "macd" | "macd_signal" | "macd_diff":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                case "adx_base":
                    adx_period = params.get('adx_period', 14)
                    adx_indicator = ta.trend.ADXIndicator(
                        high=df['high'].shift(lag),
                        low=df['low'].shift(lag),
                        close=df['close'].shift(lag),
                        window=adx_period,
                        fillna=True
                    )
                case "price_efficiency_base":
                    pe_period = params.get('price_efficiency_period', 20)
                    shifted_close = df['close'].shift(lag)
                    net_change = abs(shifted_close - shifted_close.shift(pe_period))
                    abs_changes = abs(shifted_close.diff()).rolling(window=pe_period).sum()
                    # Use .values to work with numpy arrays; fill NaN in both
                    # numerator and denominator before dividing to prevent NaN leakage
                    net_vals = net_change.values.copy()
                    abs_vals = abs_changes.values.copy()
                    result = np.zeros_like(net_vals, dtype=float)
                    valid = (abs_vals > 0) & np.isfinite(net_vals) & np.isfinite(abs_vals)
                    np.divide(net_vals, abs_vals, out=result, where=valid)
                    # Positions where inputs are NaN should remain NaN (not zero)
                    nan_mask = np.isnan(net_vals) | np.isnan(abs_vals)
                    result[nan_mask] = np.nan
                    price_efficiency = result
                case feat if feat.startswith('trend_slope_') and (w := feat.removeprefix('trend_slope_')).isdigit():
                    # As helper so mtf_trend_agreement can source a per-timeframe
                    # slope that is not itself a model input.
                    df[feat] = _atr_normalized_trend_slope(df, int(w), lag, _atr(int(w)))
                case feat if (ws := _parse_int_suffix(feat, 'range_pctile_', 2)):
                    # Percentile rank of the c-bar range width over w bars,
                    # [0,100]. Helper primitive for the mtf squeeze conjunction.
                    c_w, w = ws
                    width = _s('high').rolling(c_w).max() - _s('low').rolling(c_w).min()
                    df[feat] = calculate_rolling_percentile_rank(width, w)
                case feat if (ws := _parse_int_suffix(feat, 'atr_ratio_', 2)):
                    # Fast-vs-slow ATR ratio minus 1: >0 = volatility expanding.
                    # Helper primitive for the mtf volatility-shift conjunction.
                    f_w, w = ws
                    df[feat] = _atr(f_w) / _atr(w).replace(0.0, np.nan) - 1.0
                case feat if _is_rgm(feat):
                    # role: helper — the score reaches the execution layer
                    # (backtest --regime-risk via the regime_scores sidecar)
                    # without ever being a model input.
                    if rgm_cache is not None and feat in rgm_cache.columns:
                        df[feat] = rgm_cache[feat].shift(lag)
                    else:
                        df[feat] = np.nan
                case _:
                    if (timeframe + "_" + feat if timeframe else feat) not in externalSourced_features:
                        print(f"WARNING: Helper Feature '{feat}' calculation not implemented!")

    # Now calculate required features
    for feat in required_features:
        feat = feat if timeframe is None else feat.removeprefix(f"{timeframe}_")
        
        if feat in df.columns:
            continue  # Already exists
        else: 
            match feat:
                case "o":
                    df["o"] = df["open"].shift(lag)
                case "h":
                    df["h"] = df["high"].shift(lag)
                case "l":
                    df["l"] = df["low"].shift(lag)
                case "c":
                    df["c"] = df["close"].shift(lag)
                case "v":
                    df["v"] = df["volume"].shift(lag)
                case "mid":
                    df["mid"] = (df["high"].shift(lag) + df["low"].shift(lag)) / 2
                case "regime_trend":
                    adx = calculate_trend_strength(df, period=params.get('adx_period', 14))
                    # Calculate price efficiency
                    efficiency = calculate_price_efficiency(df, period=params.get('price_efficiency_period', 20))
                    # Trend detection: ADX or efficiency indicates directional movement
                    df["regime_trend"] = (adx > params.get('adx_threshold', 14)) | (efficiency > params.get('price_efficiency_threshold', 0.5))
                case "regime_range":
                    adx = calculate_trend_strength(df, period=params.get('adx_period', 14))
                    # Calculate price efficiency
                    efficiency = calculate_price_efficiency(df, period=params.get('price_efficiency_period', 20))
                    # Trend detection: ADX or efficiency indicates directional movement
                    df["regime_range"] = (adx <= params.get('adx_threshold', 14)) & (efficiency <= params.get('price_efficiency_threshold', 0.5))
                case "uptrend":
                    df["uptrend"] = calculate_price_direction(df, period=params.get('direction_period', 50))
                case "downtrend":
                    df["downtrend"] = ~calculate_price_direction(df, period=params.get('direction_period', 50))
                case feat if feat.startswith('sma_') and (w := feat.removeprefix('sma_')).isdigit():
                    w = int(w)
                    if w not in sma:
                        sma[w] = ta.trend.SMAIndicator(close=df['close'].shift(lag),window=w).sma_indicator()
                    df[f'sma_{w}'] = sma[w]
                case feat if feat.startswith("ema_") and (w := feat.removeprefix('ema_')).isdigit():
                    w = int(w)
                    if w not in ema:
                        ema[w] = ta.trend.EMAIndicator(close=df['close'].shift(lag),window=w).ema_indicator()
                    df[f'ema_{w}'] = ema[w]
                case "atr":
                    df['atr'] = _atr(_DEFAULT_ATR_WINDOW)
                case "atr_pips":
                    pip_value = forex.pip_value_for_symbol(params.get('symbol', 'EURUSD'))
                    df['atr_pips'] = _atr(_DEFAULT_ATR_WINDOW) / pip_value
                case "bb_upper":
                    df['bb_upper'] = bb.bollinger_hband()
                case "bb_middle":   
                    df['bb_middle'] = bb.bollinger_mavg()
                case "bb_lower":
                    df['bb_lower'] = bb.bollinger_lband()
                case "bb_width":
                    df['bb_width'] = bb.bollinger_wband()
                case "bb_percent":
                    df['bb_percent'] = bb.bollinger_pband()
                case "bb_position":
                    df['bb_position'] = np.divide((df['close'].shift(lag) - bb.bollinger_lband()),(bb.bollinger_hband() - bb.bollinger_lband()),where=(bb.bollinger_hband() != bb.bollinger_lband()),out=np.zeros_like(df['close'].shift(lag), dtype=float),)
                case "bb_squeeze":
                    df['bb_squeeze'] = bb.bollinger_wband() / df['close'].shift(lag)
                case "bb_deviation":
                    df['bb_deviation'] = (df['close'].shift(lag) - bb.bollinger_mavg()) / _atr(_BB_WINDOW)
                case "abs_bb_deviation":
                    df['abs_bb_deviation'] = abs(df['close'].shift(lag) - bb.bollinger_mavg()) / _atr(_BB_WINDOW)
                case "atr_volatility_ratio":
                    df['atr_volatility_ratio'] = (_atr(_DEFAULT_ATR_WINDOW) / df['close'].shift(lag) * 100)
                case "ichimoku_conv":
                    df['ichimoku_conv'] = ichi.ichimoku_conversion_line()
                case "ichimoku_base":
                    df['ichimoku_base'] = ichi.ichimoku_base_line()
                case "ichimoku_a":
                    df['ichimoku_a'] = ichi.ichimoku_a()
                case "ichimoku_b":
                    df['ichimoku_b'] = ichi.ichimoku_b()
                case "price_above_cloud":
                    df['price_above_cloud'] = ((df['close'].shift(lag) > ichi.ichimoku_a()) & (df['close'].shift(lag) > ichi.ichimoku_b())).astype(int)
                case "cloud_thickness":
                    df['cloud_thickness'] = abs(ichi.ichimoku_a() - ichi.ichimoku_b()) / _atr(_ICHIMOKU_SPAN_B_WINDOW)
                case "tenkan_kijun_diff":
                    df['tenkan_kijun_diff'] = (ichi.ichimoku_conversion_line() - ichi.ichimoku_base_line()) / _atr(_ICHIMOKU_KIJUN_WINDOW)
                case "cloud_color":
                    df['cloud_color'] = np.where(ichi.ichimoku_a() > ichi.ichimoku_b(), 1, -1)
                case "distance_to_cloud":
                    df_temp = pd.DataFrame({
                        'ichimoku_a': ichi.ichimoku_a(),
                        'ichimoku_b': ichi.ichimoku_b()
                    })
                    cloud_top = df_temp[['ichimoku_a', 'ichimoku_b']].max(axis=1)
                    cloud_bottom = df_temp[['ichimoku_a', 'ichimoku_b']].min(axis=1)
                    df['distance_to_cloud'] = np.where(
                        df['close'].shift(lag) > cloud_top,
                        (df['close'].shift(lag) - cloud_top) / _atr(_ICHIMOKU_SPAN_B_WINDOW),  # Above cloud
                        np.where(
                            df['close'].shift(lag) < cloud_bottom,
                            (cloud_bottom - df['close'].shift(lag)) / _atr(_ICHIMOKU_SPAN_B_WINDOW),  # Below cloud
                            0  # Inside cloud
                        )
                    )
                case "rsi":
                    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'].shift(lag)).rsi()
                case "stoch_k":
                    if stoch is None:
                        stoch = ta.momentum.StochasticOscillator(high=df['high'].shift(lag), low=df['low'].shift(lag), close=df['close'].shift(lag), window=14, smooth_window=1)
                    df['stoch_k'] = stoch.stoch()
                case "stoch_d":
                    if stoch is None:
                        stoch = ta.momentum.StochasticOscillator(high=df['high'].shift(lag), low=df['low'].shift(lag), close=df['close'].shift(lag), window=14, smooth_window=1)
                    if 'stoch_k' not in df.columns:
                        df['stoch_k'] = stoch.stoch()
                    #df['stoch_d'] = stoch.stoch_signal()
                    # %D = 3-period SMA of %K (simple rolling mean)
                    # aligned the implementation in python with the one in Java, because ta.stoch_signal() uses a different (unknown) method
                    df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=3).mean()
                case "macd":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                    df['macd'] = macd.macd()
                case "macd_signal":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                    df['macd_signal'] = macd.macd_signal()
                case "macd_diff":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                    df['macd_diff'] = macd.macd_diff()
                case "macd_atr":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                    df['macd_atr'] = macd.macd() / _atr(_MACD_SLOW_WINDOW)
                case "abs_macd_atr":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                    df['abs_macd_atr'] = abs(macd.macd() / _atr(_MACD_SLOW_WINDOW))
                case "macd_signal_atr":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                    df['macd_signal_atr'] = macd.macd_signal() / _atr(_MACD_SLOW_WINDOW)
                case "macd_diff_atr":
                    if macd is None:
                        macd = ta.trend.MACD(close=df['close'].shift(lag))
                    df['macd_diff_atr'] = macd.macd_diff() / _atr(_MACD_SLOW_WINDOW)
                case feat if feat.startswith('signed_price_vs_sma_') and (w := feat.removeprefix('signed_price_vs_sma_')).isdigit():
                    # Signed distance from the SMA in matched-window ATR units.
                    # Distinct from price_vs_sma_* (percent of the SMA level).
                    w = int(w)
                    if w not in sma:
                        sma[w] = ta.trend.SMAIndicator(close=df['close'].shift(lag), window=w).sma_indicator()
                    df[feat] = (df['close'].shift(lag) - sma[w]) / _atr(w)
                case feat if feat.startswith('price_vs_sma_') and (w := feat.removeprefix('price_vs_sma_')).isdigit():
                    w = int(w)
                    if w in sma:
                        df[f'price_vs_sma_{w}'] = (
                            df['close'].shift(lag) - sma[w]
                    ) / sma[w] * 100
                case feat if feat.startswith('abs_price_vs_sma_') and (w := feat.removeprefix('abs_price_vs_sma_')).isdigit():
                    w = int(w)
                    if w in sma:
                        df[f'abs_price_vs_sma_{w}'] = abs(((
                            df['close'].shift(lag) - sma[w]
                    ) / sma[w] * 100))
                case feat if feat.startswith('sma_') and 'cross_over' in feat and (w1 := feat.removeprefix('sma_').split('_cross_over')[0]).isdigit() and (w2 := feat.split('_cross_over_sma_')[-1]).isdigit():
                    w1 = int(w1)
                    w2 = int(w2)
                    ma_col1 = f'sma_{w1}'
                    ma_col2 = f'sma_{w2}'
                    if w1 in sma and w2 in sma:
                        # Normalize crossover distance by the ATR of the LARGER window
                        cross = (sma[w1] - sma[w2]) / _atr(max(w1, w2))
                        df[feat] = cross
                case feat if feat.startswith('sma_slope_') and (w := feat.removeprefix('sma_slope_')).isdigit():
                    w = int(w)
                    ma_col = f'sma_{w}'
                    if w in sma:
                        slope = (sma[w] - sma[w].shift(params.get(f'{ma_col}_slope_diff', 5))) / _atr(w)  # Normalize slope by matched-window ATR
                        df[feat] = slope
                case feat if feat.startswith('trend_slope_') and (w := feat.removeprefix('trend_slope_')).isdigit():
                    # Linear-regression slope of close over w bars, / ATR(w): price
                    # drift per bar in units of this timeframe's typical bar range.
                    # Dimensionless per-bar quantity, so comparable across
                    # timeframes — the input to mtf_trend_agreement.
                    df[feat] = _atr_normalized_trend_slope(df, int(w), lag, _atr(int(w)))
                case feat if (feat.startswith('momentum')
                              and not feat.startswith('momentum_accel_')
                              and feat != 'momentum_consistency_5'):
                    # Plain n-bar momentum (momentum_short/medium/long/xlong).
                    # The guard matters: without it this case swallows
                    # momentum_accel_<w> and momentum_consistency_5, which
                    # silently became a 10-bar momentum instead of their own
                    # formulas (their cases below were unreachable).
                    momentum_period = params.get(feat, 10)
                    df[f'{feat}'] = df['close'].shift(lag) / df['close'].shift(1 + momentum_period) - 1
                case "adx":
                    df['adx'] = adx_indicator.adx()
                case "adx_pos" | "adx_neg":
                    # The DIRECTIONAL components ADX throws away. ADX is
                    # 100 * SMA(|+DI - -DI| / (+DI + -DI)): it measures trend STRENGTH and
                    # discards the sign by construction. daily_adx is one of the largest
                    # gain contributors in the slow models, so the model demonstrably uses
                    # this indicator — these two give it back the direction. Bounded
                    # [0, 100], stationary, and the indicator object already exists.
                    if adx_indicator is None:
                        adx_indicator = ta.trend.ADXIndicator(
                            high=df['high'].shift(lag), low=df['low'].shift(lag),
                            close=df['close'].shift(lag),
                            window=params.get('adx_period', 14), fillna=True)
                    df[feat] = (adx_indicator.adx_pos() if feat == 'adx_pos'
                                else adx_indicator.adx_neg())
                case "cci":
                    # Self-normalised by mean absolute deviation, so approximately
                    # stationary without further treatment.
                    df['cci'] = ta.trend.CCIIndicator(
                        high=df['high'].shift(lag), low=df['low'].shift(lag),
                        close=df['close'].shift(lag),
                        window=params.get('cci_period', 20), fillna=True).cci()
                case feat if feat.startswith('channel_pos_'):
                    # Donchian position: where the close sits in the n-bar high/low range.
                    # Bounded [0, 1] by construction. A different normalisation from
                    # bb_percent, which divides by the Bollinger width instead.
                    cp_period = int(feat.rsplit('_', 1)[1])
                    hi = df['high'].shift(lag).rolling(cp_period).max()
                    lo = df['low'].shift(lag).rolling(cp_period).min()
                    width = (hi - lo).replace(0.0, np.nan)
                    df[feat] = (df['close'].shift(lag) - lo) / width
                case "hl_range_atr" | "oc_range_atr":
                    # Bar range in ATR units. The RAW ranges are non-stationary: EUR/USD
                    # traded 0.95-1.60 over 2005-2026 and its volatility moved by a factor
                    # of ~3, so a split on a raw range means different things in different
                    # eras. Dividing by ATR is the project's established idiom (see
                    # bb_deviation, distance_to_cloud). oc_range is SIGNED — it is the
                    # bar's own direction, which is the part worth keeping.
                    # Single-bar numerator -> default ATR as the typical-range
                    # reference (a window-1 ATR IS the bar's own true range and
                    # would bound the ratio by construction).
                    atr_s = _atr(_DEFAULT_ATR_WINDOW).replace(0.0, np.nan)
                    if feat == 'hl_range_atr':
                        df[feat] = (df['high'].shift(lag) - df['low'].shift(lag)) / atr_s
                    else:
                        df[feat] = (df['close'].shift(lag) - df['open'].shift(lag)) / atr_s
                case feat if feat.startswith('log_return_atr_'):
                    # n-bar log return in ATR units. The raw log return is stationary in
                    # mean but heteroskedastic — its volatility differs by a factor of ~2
                    # between calm and stressed eras, the same defect the rate-differential
                    # changes had. Scaling by ATR removes it.
                    lr_period = int(feat.rsplit('_', 1)[1])
                    # Floor the ATR window at the default: for lr_period=1 a
                    # matched window-1 ATR is the bar's own true range, which
                    # bounds |return|/ATR to ~[-1,1] and destroys the
                    # volatility-normalisation meaning of this feature.
                    atr_s = _atr(max(lr_period, _DEFAULT_ATR_WINDOW))
                    shifted = df['close'].shift(lag)
                    df[feat] = (np.log(shifted / shifted.shift(lr_period))
                                * shifted / atr_s.replace(0.0, np.nan))
                case "adx_slope":
                    adx_series = adx_indicator.adx()
                    df['adx_slope'] = adx_series - adx_series.shift(params.get('adx_slope_diff', 5))
                case "price_efficiency":
                    df['price_efficiency'] = price_efficiency
                case "volatility_percentile":
                    vp_period = params.get('volatility_percentile_period', 14)
                    vp_lookback = params.get('volatility_percentile_lookback', 500)
                    # ATR window = vp_period (Java's calculateVolatilityPercentile
                    # already passed the configured period to its ATR).
                    atr_series = _atr(vp_period)
                    # min_periods matches regime_labels.py: ensures enough history for meaningful percentile
                    min_periods = max(vp_period, min(vp_lookback // 2, 50))
                    def _vol_percentile(x):
                        # Share of the window at or below the current ATR: 100 = most
                        # volatile bar in the window. See calculate_volatility_percentile
                        # for why the comparison direction matters and is spelled out.
                        current = x.iloc[-1]
                        if np.isnan(current):
                            return np.nan
                        return (x <= current).mean() * 100
                    df['volatility_percentile'] = atr_series.rolling(
                        window=vp_lookback, min_periods=min_periods
                    ).apply(_vol_percentile, raw=False)
                    _log_atr_window(atr_series, timeframe, len(df), vp_lookback, min_periods)
                case "adx_percentile":
                    adx_period = params.get('adx_percentile_period', 14)
                    adx_lookback = params.get('adx_percentile_lookback', 500)

                    # adx_indicator is only built by the "adx_base" helper. Requesting
                    # adx_percentile without it used to raise AttributeError on None —
                    # invisible so far because the shipped config always enables an ADX
                    # feature alongside. Build it here the same way the ATR branch does.
                    if adx_indicator is None:
                        adx_indicator = ta.trend.ADXIndicator(
                            high=df['high'].shift(lag),
                            low=df['low'].shift(lag),
                            close=df['close'].shift(lag),
                            window=params.get('adx_period', 14),
                            fillna=True
                        )
                    adx_series = adx_indicator.adx()
                    # min_periods: enough history for meaningful percentile
                    min_periods = max(adx_period, min(adx_lookback // 2, 50))
                    def _adx_percentile(x):
                        # Share of the window at or below the current ADX: 100 = the
                        # strongest trend in the window. Was inverted like the volatility
                        # percentile — harmless for a tree (a monotone flip only reverses
                        # its splits) but it made the feature mean the opposite of its name.
                        current = x.iloc[-1]
                        if np.isnan(current):
                            return np.nan
                        return (x <= current).mean() * 100
                    df['adx_percentile'] = adx_series.rolling(
                        window=adx_lookback, min_periods=min_periods
                    ).apply(_adx_percentile, raw=False)
                case "day_of_week":
                    # Monday=0 … Friday=4, normalized to [0, 1]
                    df[feat] = (df.index.dayofweek / 4.0).astype(np.float32)
                case "month_sin":
                    df[feat] = np.sin(2 * np.pi * df.index.month / 12).astype(np.float32)
                case "month_cos":
                    df[feat] = np.cos(2 * np.pi * df.index.month / 12).astype(np.float32)
                case "week_of_year_sin":
                    weeks = df.index.isocalendar().week.to_numpy(dtype=float)
                    df[feat] = np.sin(2 * np.pi * weeks / 53).astype(np.float32)
                case "week_of_year_cos":
                    weeks = df.index.isocalendar().week.to_numpy(dtype=float)
                    df[feat] = np.cos(2 * np.pi * weeks / 53).astype(np.float32)
                case "quarter":
                    # Q1=0, Q2=0.333, Q3=0.667, Q4=1.0
                    df[feat] = ((df.index.quarter - 1) / 3.0).astype(np.float32)
                case "hour_sin":
                    # Includes minutes for smooth M15 resolution (96 distinct values/day)
                    hours = df.index.hour + df.index.minute / 60.0
                    df[feat] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
                case "hour_cos":
                    hours = df.index.hour + df.index.minute / 60.0
                    df[feat] = np.cos(2 * np.pi * hours / 24).astype(np.float32)
                case "is_london_session":
                    df[feat] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(np.float32)
                case "is_ny_session":
                    df[feat] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(np.float32)
                case "is_asia_session":
                    df[feat] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(np.float32)
                case "is_session_overlap":
                    # London + NY overlap: 13:00–16:00 UTC
                    df[feat] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(np.float32)
                case "close_vs_bar_range":
                    bar_range = df['high'].shift(lag) - df['low'].shift(lag)
                    df[feat] = np.where(
                        bar_range > 0,
                        (df['close'].shift(lag) - df['low'].shift(lag)) / bar_range,
                        0.5
                    ).astype(np.float32)
                case "bar_body_ratio":
                    bar_range = df['high'].shift(lag) - df['low'].shift(lag)
                    df[feat] = np.where(
                        bar_range > 0,
                        np.abs(df['close'].shift(lag) - df['open'].shift(lag)) / bar_range,
                        0.0
                    ).astype(np.float32)
                case "return_3_atr":
                    # 3-bar move -> matched ATR(3) (same convention as
                    # momentum_accel_<w>: the name's window sets the ATR window).
                    df[feat] = ((df['close'].shift(lag) - df['close'].shift(lag + 3)) / _atr(3)).astype(np.float32)
                case "momentum_consistency_5":
                    bull = (df['close'].shift(lag) > df['open'].shift(lag)).astype(float)
                    df[feat] = bull.rolling(5).mean().astype(np.float32)

                # ------------------------------------------------------------------
                # Event-sequence direction family (pre-registration A5).
                # Every feature is a SIGNED CONTINUOUS event score: 0.0 = the event
                # did not fire, sign = direction, magnitude = event strength in ATR
                # units. Windows come from the name (largest window LAST — the
                # warm-up regex reads only the trailing _<n>); thresholds from
                # parameters: with optional per-timeframe override (_p).
                # ------------------------------------------------------------------
                case feat if (ws := _parse_int_suffix(feat, 'impulse_burst_', 2)):
                    # Impulse strength: n-bar body sum x directional efficiency.
                    n, w = ws
                    df[feat] = _impulse_score(n, w)
                case feat if (ws := _parse_int_suffix(feat, 'impulse_london_', 2)):
                    # Impulse conjunction with the London session. The session
                    # clock is the bar's own timestamp (known at bar time, like
                    # hour_sin) — deliberately unshifted.
                    n, w = ws
                    in_london = pd.Series((df.index.hour >= 8) & (df.index.hour < 16),
                                          index=df.index)
                    df[feat] = _impulse_score(n, w).where(in_london, 0.0)
                case feat if (ws := _parse_int_suffix(feat, 'breakout_close_', 2)):
                    # Fresh Donchian(w) close-break: no break in the prior m bars,
                    # score = signed penetration depth / ATR(w).
                    m, w = ws
                    hi, lo = _donchian(w)
                    c_ = _s('close')
                    atr_s = _atr(w).replace(0.0, np.nan)
                    up_break = c_ > hi
                    dn_break = c_ < lo
                    prior_up = up_break.astype(float).shift(1).rolling(m, min_periods=1).max()
                    prior_dn = dn_break.astype(float).shift(1).rolling(m, min_periods=1).max()
                    score = pd.Series(0.0, index=df.index)
                    score = score.mask(up_break & (prior_up == 0), (c_ - hi) / atr_s)
                    score = score.mask(dn_break & (prior_dn == 0), (c_ - lo) / atr_s)
                    df[feat] = score.where(c_.notna() & hi.notna() & lo.notna() & atr_s.notna())
                case feat if (ws := _parse_int_suffix(feat, 'volspike_body_', 1)):
                    # Volatility spike + dominant body: range >= k·ATR and the
                    # body fills most of it -> signed range/ATR.
                    (w,) = ws
                    rng = _s('high') - _s('low')
                    body = _s('close') - _s('open')
                    spike = rng / _atr(w).replace(0.0, np.nan)
                    body_frac = body.abs() / rng.where(rng > 0)
                    cond = ((spike >= _p('volspike_min_ratio', 2.0))
                            & (body_frac >= _p('volspike_min_body_frac', 0.6)))
                    df[feat] = _event(np.sign(body) * spike, cond)
                case feat if (ws := _parse_int_suffix(feat, 'squeeze_breakout_', 2)):
                    # Compression -> expansion: the c-bar range width sat in its
                    # lowest percentile band and the close breaks outside it.
                    c_w, w = ws
                    hi_c = _s('high').rolling(c_w).max()
                    lo_c = _s('low').rolling(c_w).min()
                    width_pct = calculate_rolling_percentile_rank(hi_c - lo_c, w)
                    compressed_prev = width_pct.shift(1) <= _p('squeeze_max_pctile', 20.0)
                    hi_prev = hi_c.shift(1)
                    lo_prev = lo_c.shift(1)
                    c_ = _s('close')
                    atr_s = _atr(w).replace(0.0, np.nan)
                    score = pd.Series(0.0, index=df.index)
                    score = score.mask(compressed_prev & (c_ > hi_prev), (c_ - hi_prev) / atr_s)
                    score = score.mask(compressed_prev & (c_ < lo_prev), (c_ - lo_prev) / atr_s)
                    df[feat] = score.where(c_.notna() & width_pct.shift(1).notna() & atr_s.notna())
                case feat if (ws := _parse_int_suffix(feat, 'hh_thrust_', 1)):
                    # Structure thrust: high-high (low-low) increment in its top
                    # percentile band, confirmed by a same-direction body.
                    (w,) = ws
                    d_hh = _s('high').diff()
                    d_ll = _s('low').diff()
                    hh_pct = calculate_rolling_percentile_rank(d_hh, w)
                    ll_pct = calculate_rolling_percentile_rank(-d_ll, w)
                    body = _s('close') - _s('open')
                    atr_s = _atr(w).replace(0.0, np.nan)
                    thr = _p('hh_thrust_min_pctile', 80.0)
                    score = pd.Series(0.0, index=df.index)
                    score = score.mask((hh_pct >= thr) & (body > 0), d_hh / atr_s)
                    score = score.mask((ll_pct >= thr) & (body < 0), d_ll / atr_s)
                    df[feat] = score.where(d_hh.notna() & hh_pct.notna() & ll_pct.notna() & atr_s.notna())
                case feat if (ws := _parse_int_suffix(feat, 'run_momentum_', 2)):
                    # Directional count >= r consecutive same-direction closes ->
                    # the r-bar move in ATR units.
                    r, w = ws
                    c_ = _s('close')
                    run = calculate_run_length(c_.diff())
                    mom = (c_ - c_.shift(r)) / _atr(w).replace(0.0, np.nan)
                    df[feat] = _event(mom, run.abs() >= r)
                case feat if (ws := _parse_int_suffix(feat, 'break_retest_', 2)):
                    # Break -> retest -> resumption: Donchian(w) break within the
                    # last m bars, price comes back to touch the broken level and
                    # closes beyond it again in break direction.
                    m, w = ws
                    up_level, dn_level, warm = _break_carry(m, w)
                    c_ = _s('close')
                    l_ = _s('low')
                    h_ = _s('high')
                    atr_s = _atr(w).replace(0.0, np.nan)
                    frac = _p('retest_atr_frac', 0.25)
                    up_evt = up_level.notna() & (l_ <= up_level + frac * atr_s) & (c_ > up_level)
                    dn_evt = dn_level.notna() & (h_ >= dn_level - frac * atr_s) & (c_ < dn_level)
                    score = pd.Series(0.0, index=df.index)
                    score = score.mask(up_evt, (c_ - up_level) / atr_s)
                    score = score.mask(dn_evt, (c_ - dn_level) / atr_s)
                    df[feat] = score.where(~warm & atr_s.notna())
                case feat if (ws := _parse_int_suffix(feat, 'failed_break_reversal_', 2)):
                    # Failed breakout: break within the last m bars, close is back
                    # INSIDE the channel -> signed against the break direction.
                    m, w = ws
                    up_level, dn_level, warm = _break_carry(m, w)
                    c_ = _s('close')
                    atr_s = _atr(w).replace(0.0, np.nan)
                    up_fail = up_level.notna() & (c_ < up_level)
                    dn_fail = dn_level.notna() & (c_ > dn_level)
                    score = pd.Series(0.0, index=df.index)
                    score = score.mask(up_fail, (c_ - up_level) / atr_s)
                    score = score.mask(dn_fail, (c_ - dn_level) / atr_s)
                    df[feat] = score.where(~warm & atr_s.notna())
                case feat if (ws := _parse_int_suffix(feat, 'micro_burst_', 2)):
                    # Microstructure burst: near-unanimous body direction over n
                    # bars with a total move of at least 1 ATR.
                    n, w = ws
                    body = _s('close') - _s('open')
                    body_valid = body.notna()
                    pos = (body > 0).astype(float).where(body_valid).rolling(n).sum()
                    neg = (body < 0).astype(float).where(body_valid).rolling(n).sum()
                    burst = body.rolling(n).sum() / _atr(w).replace(0.0, np.nan)
                    cond = (((pos >= n - 1) | (neg >= n - 1))
                            & (burst.abs() >= _p('micro_burst_min_atr', 1.0)))
                    df[feat] = _event(burst, cond)
                case feat if (ws := _parse_int_suffix(feat, 'volshift_bias_', 2)):
                    # Volatility shift + trend bias: fast ATR expanding vs slow
                    # ATR, direction from the w-bar regression slope.
                    f_w, w = ws
                    ratio = _atr(f_w) / _atr(w).replace(0.0, np.nan) - 1.0
                    slope = calculate_rolling_slope(_s('close'), w)
                    df[feat] = _event(ratio * np.sign(slope),
                                      (ratio >= _p('volshift_min', 0.25)) & slope.notna())
                case feat if (ws := _parse_int_suffix(feat, 'range_imbalance_', 1)):
                    # Directional imbalance of bar RANGES over w bars, [-1, 1],
                    # gated on |imbalance| >= threshold.
                    (w,) = ws
                    body = _s('close') - _s('open')
                    rng = _s('high') - _s('low')
                    up_rng = rng.where(body > 0, 0.0).where(body.notna() & rng.notna())
                    dn_rng = rng.where(body < 0, 0.0).where(body.notna() & rng.notna())
                    tot = rng.rolling(w).sum()
                    imb = (up_rng.rolling(w).sum() - dn_rng.rolling(w).sum()) / tot.where(tot > 0)
                    df[feat] = _event(imb, imb.abs() >= _p('imbalance_min', 0.5))
                case feat if (ws := _parse_int_suffix(feat, 'tick_imbalance_', 1)):
                    # Volume-weighted directional imbalance. TRAINING-ONLY: the
                    # live Java payload carries no volume -> NaN.
                    (w,) = ws
                    if vol_shifted is None:
                        df[feat] = np.nan
                    else:
                        body = _s('close') - _s('open')
                        mp = max(2, w // 4)
                        signed_vol = np.sign(body) * vol_shifted
                        tot = vol_shifted.rolling(w, min_periods=mp).sum()
                        imb = signed_vol.rolling(w, min_periods=mp).sum() / tot.where(tot > 0)
                        df[feat] = _event(imb, imb.abs() >= _p('imbalance_min', 0.5))
                case feat if (ws := _parse_int_suffix(feat, 'spike_wick_reversal_', 1)):
                    # Volatility spike + dominant wick -> fade the wick side.
                    (w,) = ws
                    upper, lower = calculate_wick_fractions(_s('open'), _s('high'),
                                                            _s('low'), _s('close'))
                    spike = (_s('high') - _s('low')) / _atr(w).replace(0.0, np.nan)
                    wick_thr = _p('wick_min_frac', 0.6)
                    spike_thr = _p('spike_min_ratio', 2.0)
                    score = pd.Series(0.0, index=df.index)
                    score = score.mask((spike >= spike_thr) & (upper >= wick_thr), -spike)
                    score = score.mask((spike >= spike_thr) & (lower >= wick_thr), spike)
                    df[feat] = score.where(spike.notna() & upper.notna() & lower.notna())
                case feat if (ws := _parse_int_suffix(feat, 'compression_spread_', 2)):
                    # Compression + spread widening. TRAINING-ONLY: needs the
                    # provider 'mid' column (absent from the live payload).
                    c_w, w = ws
                    if 'mid' not in df.columns:
                        df[feat] = np.nan
                    else:
                        spread = (2.0 * (df['close'] - df['mid'])).shift(lag)
                        spread_std = spread.rolling(w).std()
                        spread_z = (spread - spread.rolling(w).mean()) / spread_std.where(spread_std > 0)
                        width = _s('high').rolling(c_w).max() - _s('low').rolling(c_w).min()
                        width_pct = calculate_rolling_percentile_rank(width, w)
                        body = _s('close') - _s('open')
                        cond = ((width_pct <= _p('squeeze_max_pctile', 20.0))
                                & (spread_z >= _p('compression_spread_min_z', 1.0)))
                        df[feat] = _event(np.sign(body) * spread_z, cond)
                case feat if (ws := _parse_int_suffix(feat, 'jump_reversion_', 1)):
                    # Jump -> mean reversion: fade a 1-bar move >= k·ATR.
                    (w,) = ws
                    jump = _s('close').diff() / _atr(w).replace(0.0, np.nan)
                    df[feat] = _event(-jump, jump.abs() >= _p('jump_min_atr', 3.0))
                case feat if (ws := _parse_int_suffix(feat, 'jump_burst_follow_', 2)):
                    # Jump -> burst: a jump n bars ago followed through by
                    # near-unanimous same-direction bodies -> ride it.
                    n, w = ws
                    c_ = _s('close')
                    atr_s = _atr(w).replace(0.0, np.nan)
                    jump = c_.diff() / atr_s
                    jump_n_ago = jump.where(jump.abs() >= _p('jump_min_atr', 3.0)).shift(n)
                    body = _s('close') - _s('open')
                    body_valid = body.notna()
                    pos = (body > 0).astype(float).where(body_valid).rolling(n).sum()
                    neg = (body < 0).astype(float).where(body_valid).rolling(n).sum()
                    mom = (c_ - c_.shift(n)) / atr_s
                    score = pd.Series(0.0, index=df.index)
                    score = score.mask((jump_n_ago > 0) & (pos >= n - 1), mom)
                    score = score.mask((jump_n_ago < 0) & (neg >= n - 1), mom)
                    df[feat] = score.where(mom.notna() & jump.notna())
                case feat if (ws := _parse_int_suffix(feat, 'news_impulse_', 1)):
                    # Macro news cluster + impulse: the PREVIOUS bar's day had
                    # >= news_cluster_min scheduled HIGH-impact EUR/USD events
                    # and that bar closed with a directional impulse. The
                    # schedule is pre-published (calendar_events precedent); the
                    # count is shifted with the bar so the whole event is a
                    # property of the completed bar. NaN outside calendar
                    # coverage.
                    (w,) = ws
                    counts = _news_counts(df.index).shift(lag)
                    body = _s('close') - _s('open')
                    rng = _s('high') - _s('low')
                    impulse = np.sign(body) * (rng / _atr(w).replace(0.0, np.nan))
                    # Keep the count's NaN (outside calendar coverage): a bare
                    # comparison would collapse NaN to False and fabricate a
                    # "no event" claim where the calendar is simply unknown.
                    cluster = (counts >= _p('news_cluster_min', 2)).where(counts.notna())
                    df[feat] = _event(impulse, cluster)
                case feat if (ws := _parse_int_suffix(feat, 'range_pctile_', 2)):
                    # Same as the helper case (a config may declare it as model).
                    c_w, w = ws
                    width = _s('high').rolling(c_w).max() - _s('low').rolling(c_w).min()
                    df[feat] = calculate_rolling_percentile_rank(width, w)
                case feat if (ws := _parse_int_suffix(feat, 'atr_ratio_', 2)):
                    f_w, w = ws
                    df[feat] = _atr(f_w) / _atr(w).replace(0.0, np.nan) - 1.0

                # --- tick volume (see vol_shifted above for the drift caveat) ---
                case feat if feat.startswith('volume_percentile_') and (w := feat.removeprefix('volume_percentile_')).isdigit():
                    # Rolling percentile rank of tick volume, [0,100]. Bounded and
                    # rank-based, so immune to both the feed drift and the heavy
                    # right skew of raw volume.
                    w = int(w)
                    if vol_shifted is None:
                        df[feat] = np.nan
                    else:
                        df[feat] = (vol_shifted.rolling(w, min_periods=max(2, w // 4))
                                    .rank(pct=True) * 100.0).astype(np.float32)

                case feat if feat.startswith('effort_result_') and (w := feat.removeprefix('effort_result_')).isdigit():
                    # Wyckoff "effort vs result": the activity NOT explained by the
                    # resulting move. >0 = much turnover for little range (absorption,
                    # trend meeting resistance); <0 = large range on thin volume
                    # (frictionless continuation). Relative volume and relative range
                    # correlate only ~0.47, so this residual carries information ATR
                    # does not. Log-difference keeps it symmetric around 0.
                    w = int(w)
                    if vol_shifted is None:
                        df[feat] = np.nan
                    else:
                        vol_ratio = vol_shifted / vol_shifted.rolling(w, min_periods=max(2, w // 4)).mean()
                        range_ratio = (df['high'].shift(lag) - df['low'].shift(lag)) / _atr(w)
                        # .where() masks the zero-volume / zero-range bars (~0.9%) to
                        # NaN instead of producing -inf from log(0).
                        df[feat] = (np.log(vol_ratio.where(vol_ratio > 0))
                                    - np.log(range_ratio.where(range_ratio > 0))).astype(np.float32)

                case feat if feat.startswith('vwap_deviation_') and (w := feat.removeprefix('vwap_deviation_')).isdigit():
                    # Signed distance from the volume-weighted average price, in ATR
                    # units. Distinct from price_vs_sma_*: anchored where volume
                    # actually traded rather than where time passed — VWAP is the
                    # institutional execution benchmark.
                    w = int(w)
                    if vol_shifted is None:
                        df[feat] = np.nan
                    else:
                        mp = max(2, w // 4)
                        typical = (df['high'].shift(lag) + df['low'].shift(lag) + df['close'].shift(lag)) / 3.0
                        pv = (typical * vol_shifted).rolling(w, min_periods=mp).sum()
                        vv = vol_shifted.rolling(w, min_periods=mp).sum()
                        vwap = pv / vv.where(vv > 0)
                        # Scale by the MEAN ATR over the same w-bar window, not the
                        # instantaneous ATR: the numerator is a w-bar deviation, so the
                        # denominator must span the same horizon. Instantaneous ATR
                        # collapses to ~1e-5 on dead bars (holidays, 21:00 rollover),
                        # which inflated this ratio past |500| on 0.26% of bars.
                        atr_window = _atr(w).rolling(w, min_periods=mp).mean()
                        df[feat] = ((df['close'].shift(lag) - vwap)
                                    / atr_window.where(atr_window > 0)).astype(np.float32)

                case feat if feat.startswith('cmf_') and (w := feat.removeprefix('cmf_')).isdigit():
                    # Chaikin Money Flow: volume-weighted mean of the money-flow
                    # multiplier ((C-L)-(H-C))/(H-L). Bounded to [-1,1] by
                    # construction and therefore immune to the volume drift — it is a
                    # weighted average of a bounded quantity, not a cumulative sum.
                    w = int(w)
                    if vol_shifted is None:
                        df[feat] = np.nan
                    else:
                        mp = max(2, w // 4)
                        h = df['high'].shift(lag)
                        l = df['low'].shift(lag)
                        c = df['close'].shift(lag)
                        bar_range = h - l
                        # H==L bars carry no directional information -> neutral 0.
                        mfm = (((c - l) - (h - c)) / bar_range.where(bar_range > 0)).fillna(0.0)
                        mfv = (mfm * vol_shifted).rolling(w, min_periods=mp).sum()
                        vv = vol_shifted.rolling(w, min_periods=mp).sum()
                        df[feat] = (mfv / vv.where(vv > 0)).astype(np.float32)

                case feat if feat.startswith('vol_price_divergence_') and (w := feat.removeprefix('vol_price_divergence_')).isdigit():
                    # Is the w-bar move backed by participation? [-1,1].
                    # >0 = move on above-median volume (supported);
                    # <0 = move on thin volume (suspect, classic exhaustion tell).
                    w = int(w)
                    if vol_shifted is None:
                        df[feat] = np.nan
                    else:
                        vol_pct = vol_shifted.rolling(w, min_periods=max(2, w // 4)).rank(pct=True)
                        mom = df['close'].shift(lag) - df['close'].shift(lag + w)
                        df[feat] = (np.sign(mom) * (2.0 * vol_pct - 1.0)).astype(np.float32)
                case feat if feat.startswith('momentum_accel_') and (w := feat.removeprefix('momentum_accel_')).isdigit():
                    # Trend acceleration / deceleration: change in the w-bar return
                    # between the current window and the prior window, in ATR units.
                    # >0 = momentum building; <0 = momentum rolling over (exhaustion).
                    # Signed (direction-aware): in an uptrend a negative value flags
                    # a stalling advance, in a downtrend a stalling decline.
                    w = int(w)
                    c = df['close'].shift(lag)
                    roc_now  = c - c.shift(w)
                    roc_prev = c.shift(w) - c.shift(2 * w)
                    df[feat] = ((roc_now - roc_prev) / _atr(w)).astype(np.float32)
                case "bb_edge_impact":        
                    df['bb_edge_impact'] = np.divide((df['close'].shift(lag) - df['bb_lower']),(df['bb_upper'] - df['bb_lower']),where=(df['bb_upper'] != df['bb_lower']),out=np.zeros_like(df['close'].shift(lag), dtype=float),) * bb.bollinger_wband()
                case feat if feat.startswith('pullback_') and (w := feat.removeprefix('pullback_')).isdigit():
                    w = int(w)
                    sma_period = w * 4
                    if sma_period not in sma:
                        sma[sma_period] = ta.trend.SMAIndicator(close=df['close'].shift(lag), window=sma_period).sma_indicator()
                    # Trend direction from shifted close vs SMA: 1 = up, -1 = down.
                    trend = np.where(df['close'].shift(lag) > sma[sma_period], 1, -1)

                    # Swing extremes over the lookback window. Shift by lag so the
                    # window ends at the last *completed* bar (no current-bar leak).
                    swing_high = df['high'].shift(lag).rolling(window=w).max()
                    swing_low = df['low'].shift(lag).rolling(window=w).min()

                    # Pullback depth in ATR units, measured in the trend direction:
                    #   uptrend   -> (swing_high - close) / atr  (distance below the high)
                    #   downtrend -> (close - swing_low)  / atr  (distance above the low)
                    df[f'{feat}'] = np.where(
                        trend == 1,
                        (swing_high - df['close'].shift(lag)) / _atr(w),
                        (df['close'].shift(lag) - swing_low) / _atr(w)
                    )
                case "meta_pullback_score": ## TODO dirty implementation - needs refactoring if useful
                    sma_period = 200
                    # 1. Daily Trend-Richtung bestimmen
                    # Wir nutzen den Daily Close im Vergleich zum Daily SMA 50 oder 200
                    daily_dir = np.where(df['close'].shift(lag) > sma[sma_period], 1, -1)

                    # 2. Meta-Feature berechnen
                    # Multiplikation der Richtung mit der 4h-Pullback-Tiefe
                    df['meta_pullback_score'] = daily_dir * df['pullback_20']

                    # 3. Säubern: Nur positive Werte behalten (Pullbacks IN Trendrichtung)
                    df['meta_pullback_score'] = df['meta_pullback_score'].clip(lower=0)
                case "slope_efficiency": ## TODO dirty implementation - needs refactoring if useful
                    # Parameter
                    window = 12
                    lag = 1

                    # --- Temporäre Variablen (werden nicht im df gespeichert) ---
                    # 1. Den verschobenen Preis und eine Zeit-Reihe als Basis
                    _y = df["close"].shift(lag)
                    _x = pd.Series(np.arange(len(df)), index=df.index)

                    # 2. Rollierende Korrelation (r)
                    _r = _y.rolling(window=window).corr(_x)

                    # 3. Effizienz (R-Squared) und Richtung (Slope-Vorzeichen)
                    _efficiency = _r ** 2
                    _direction = np.sign(_r) # Korrelations-Vorzeichen entspricht dem Slope-Vorzeichen

                    # --- Finales Feature direkt ins DF schreiben ---
                    # Ergibt Werte von -1.0 (perfekter Short) bis +1.0 (perfekter Long)
                    df[f'{feat}'] = _efficiency * _direction
                case feat if feat.startswith('sma_') and 'slope_acceleration' in feat and (w1 := feat.removeprefix('sma_').split('_slope_acceleration')[0]).isdigit() and (w2 := feat.split('_slope_acceleration_sma_')[-1]).isdigit():
                    w1 = int(w1)
                    w2 = int(w2)
                    df[f'{feat}'] = (sma[w1] - sma[w2]) / _atr(max(w1, w2))  # Normalize by the larger window's ATR
                case "y":
                    df["y"] = np.nan
                case "bars_since_flip":
                    sma_period = 200
                    # 1. Temporäre Trend-Series erstellen (nicht im df gespeichert)
                    tmp_trend = pd.Series(np.where(df['close'].shift(lag) > sma[sma_period], 1, -1), index=df.index)

                    # 2. Flips identifizieren
                    tmp_flip = tmp_trend.diff().fillna(0) != 0

                    # 3. Das finale Feature direkt zuweisen
                    df[f'{feat}'] = tmp_trend.groupby(tmp_flip.cumsum()).cumcount()
                case feat if (sa := _parse_session_anchor(feat)) is not None:
                    # A7 session-anchor family: causal intraday seasonality,
                    # session-anchored states (Asia range, London first hour,
                    # overnight gap), fix/turn-of-month/weekday anchors.
                    # Roster: analytics/session_anchor_catalog.py.
                    df[feat] = _session_anchor(sa).astype(np.float32)
                case feat if _parse_realized_moment(feat) is not None:
                    # A8 realized-moments family: signed semivariance,
                    # realized skew/kurtosis, bipower jump share, path
                    # asymmetry — from gap-masked M15 log-returns, raw values
                    # end-shifted by `lag` (the tfm/rgm precedent). Roster:
                    # analytics/realized_moments_catalog.py.
                    df[feat] = compute_realized_moments(
                        df, [feat], min_frac=float(_p('rm_min_frac', 0.75)),
                        jump_z=float(_p('rm_jump_z', 2.0)),
                        cache=_rm_cache)[feat].shift(lag).astype(np.float32)
                case feat if _parse_volume_spread(feat) is not None:
                    # A9 volume-spread family: deseasonalised tick-volume
                    # surprise, signed volume imbalance, Amihud, spread
                    # state/expansion and volume-conditioned events — raw
                    # values end-shifted by `lag` (the A8 precedent).
                    # TRAINING-ONLY: a frame without volume/mid (the live
                    # Java payload; mid also absent unless loaded with
                    # keep_mid=True) degrades to NaN. Roster:
                    # analytics/volume_spread_catalog.py. The legacy
                    # volume_percentile/effort_result/vol_price_divergence
                    # names never reach this case (their own cases above
                    # match first).
                    df[feat] = compute_volume_spread_members(
                        df, [feat], min_frac=float(_p('vs_min_frac', 0.75)),
                        slot_min_frac=float(_p('vs_slot_min_frac', 0.4)),
                        jump_z=float(_p('vs_jump_z', 2.0)),
                        break_ma=int(_p('vs_break_ma', 4)),
                        cache=_vs_cache)[feat].shift(lag).astype(np.float32)
                case feat if _parse_timing_execution(feat) is not None:
                    # B6 timing/execution family: volatility shift, spread
                    # regime, liquidity impulse, microstructure break, maker
                    # queue proxy + adaptive SL/TP/trailing/cost-floor
                    # distances. Raw values end-shifted by `lag` (the A8
                    # precedent). Spread/volume members are TRAINING-ONLY
                    # (live payload has no volume/mid) and degrade to NaN /
                    # their price-only fallbacks — see
                    # features/timing_execution.py. Roster:
                    # analytics/timing_execution_catalog.py.
                    from ModelTrading.source.python.features import (
                        timing_execution as _tx_mod,
                    )
                    df[feat] = _tx_mod.compute_timing_execution_members(
                        df, [feat], params=params,
                        cache=_tx_cache)[feat].shift(lag).astype(np.float32)
                case feat if _parse_cross_pair(feat) is not None:
                    # A10 cross-pair family: triangle residuals, intraday USD
                    # breadth/strength, cross-only EUR strength, lead-lag
                    # betas, correlation breakdown, vol spillover — from the
                    # nine additional pairs' M15 closes on corrected UTC
                    # stamps (generation-aware Europe/Berlin fix, see
                    # features/cross_pair_intraday.py). Raw values end-shifted
                    # by `lag` (the A8 precedent). TRAINING-ONLY: the live
                    # payload carries no pair data — members degrade to NaN
                    # if the pair CSVs are absent. Roster:
                    # analytics/cross_pair_catalog.py.
                    from ModelTrading.source.python.features import (
                        cross_pair_intraday as _xp_mod,
                    )
                    df[feat] = _xp_mod.compute_cross_pair_members(
                        df, [feat], min_frac=float(_p('xp_min_frac', 0.75)),
                        cache=_xp_cache)[feat].shift(lag).astype(np.float32)
                case feat if (cal := _parse_calendar_direction(feat)) is not None:
                    # A6 calendar-direction family: dense schedule clocks and
                    # sparse event-window scores around FED/ECB decisions.
                    # Roster: analytics/calendar_direction_catalog.py.
                    df[feat] = _calendar_direction(cal).astype(np.float32)
                case feat if _is_cal(feat):
                    # Days until the next scheduled central-bank decision
                    # (0 = today). Timestamp-derived like hour_sin — the meeting
                    # calendar is published over a year in advance, so applying
                    # the shift(1) here would only make the countdown wrong by
                    # one day. NaN beyond the calendar's last known event.
                    df[feat] = _days_until_event(df.index, feat)
                case feat if _is_tfm(feat):
                    if tfm_cache is not None and feat in tfm_cache.columns:
                        df[feat] = tfm_cache[feat].shift(lag)
                    else:
                        df[feat] = np.nan
                case feat if _is_rgm(feat):
                    if rgm_cache is not None and feat in rgm_cache.columns:
                        df[feat] = rgm_cache[feat].shift(lag)
                    else:
                        df[feat] = np.nan
                case _:
                    if (timeframe + "_" + feat if timeframe else feat) not in externalSourced_features:
                        print(f"WARNING: Model Feature '{feat}' calculation not implemented!")

    # ---------------------------------------------------------------------------
    # Loop 3: External features (COT/VIX/Yields)
    # All features in this loop are listed in features.yaml with usedInModel: true
    # but are NOT computed from OHLC — they come from 
    # external CSV files loaded via external_data.get_external_df().
    # ---------------------------------------------------------------------------

    _ext_df = None  # lazily loaded on first external feature request

    for feat in externalSourced_features:
        feat = feat if timeframe is None else feat.removeprefix(f"{timeframe}_")

        # --- External features: COT / VIX / yields ---
        # Loaded from CSV files via external_data module (TTL-cached for live trading).
        # Lookahead-bias prevention is handled inside get_external_df() via shift(1)
        # on the sparse (daily/weekly) index before forward-filling onto df.index.
        if _ext_df is None:
            try:
                from ModelTrading.source.python.features import external_data as _ext_mod
                _ext_df = _ext_mod.get_external_df(df.index)
            except Exception as exc:
                print(
                    f"WARNING [indicators]: Could not load external data: {exc}",
                    file=sys.stderr,
                )
                _ext_df = False  # sentinel: don't retry

        if _ext_df is not False and _ext_df is not None and feat in _ext_df.columns:
            df[feat] = _ext_df[feat].values.astype(np.float32)
        else:
            print(
                f"WARNING [indicators]: External feature '{feat}' not available — "
                "column will remain absent. Ensure external CSVs exist "
                "(run data/update_external_data.py).",
                file=sys.stderr,
            )

    df = datahandling.remove_duplicates(df, name=f"features_{timeframe if timeframe else 'base'}")

    # Normalize ±inf to NaN. ATR-normalized features (bb_deviation, distance_to_cloud,
    # pullback, momentum_accel, ...) divide by ATR, which the `ta` library fills with 0
    # during warmup rather than NaN — producing inf that survives the downstream
    # leading-NaN trim and crashes the scaler. NaN is the correct representation of an
    # undefined value and is handled by the existing first-valid-row trim.
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.astype(np.float32)

    return df


def add_cross_timeframe_features(combined: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Compute features that need columns from MORE THAN ONE timeframe (mtf_* prefix).

    Runs AFTER the per-timeframe frames are aligned onto the M15 index and
    concatenated (advanced_train.calculate_features / train.py / feature_server).
    The source columns already carry their own shift/alignment, so no additional
    shift is applied here: a value at bar t is a pure function of other feature
    values at the same bar and adds no lookahead beyond its sources.

    strict: behaviour when a feature's source columns are missing from the
    config/frame. Training passes True — an all-NaN model feature is guaranteed
    to kill the downstream data alignment ("No rows without NaN values found"),
    so fail HERE with the actual cause instead. The live feature server keeps
    the default False and degrades to NaN rather than crashing mid-session.

    Currently implemented:
        mtf_trend_agreement — mean(tanh(slope/scale)) over the ATR-normalised
        trend slopes named in parameters.mtf_trend_slope_sources.
    """
    config = get_feature_config()
    mtf_features = config.get_usedInModel_features(prefix='mtf_')
    if not mtf_features:
        return combined
    params = config.get_parameters()

    def _resolve_sources(feat: str, param_key: str):
        """Return the source columns for an mtf conjunction, or None after the
        standard missing-source handling (strict raise / NaN degrade)."""
        sources = params.get(param_key, []) or []
        missing = [c for c in sources if c not in combined.columns]
        if not sources or missing:
            msg = (
                f"{feat} is enabled but its source columns are "
                + (f"missing from the feature frame: {missing}" if sources
                   else f"not configured (parameters.{param_key} is unset)")
                + ". Add/enable the per-timeframe source features (model or "
                f"helper role) and the {param_key} parameter in the active "
                "features yaml."
            )
            if strict:
                raise ValueError(f"[indicators] {msg}")
            print(f"WARNING [indicators]: {msg} Feature set to NaN.", file=sys.stderr)
            combined[feat] = np.float32(np.nan)
            return None
        return sources

    for feat in mtf_features:
        if feat in combined.columns:
            continue
        match feat:
            case "mtf_trend_agreement":
                sources = params.get('mtf_trend_slope_sources', []) or []
                scale = float(params.get('mtf_agreement_scale', 0.05))
                missing = [c for c in sources if c not in combined.columns]
                if not sources or missing:
                    msg = (
                        "mtf_trend_agreement is enabled but its source columns are "
                        + (f"missing from the feature frame: {missing}" if sources
                           else "not configured (parameters.mtf_trend_slope_sources is unset)")
                        + ". Add/enable the per-timeframe trend_slope features "
                        "(model or helper role) and the mtf_trend_slope_sources / "
                        "mtf_agreement_scale parameters in the active features yaml."
                    )
                    if strict:
                        raise ValueError(f"[indicators] {msg}")
                    print(f"WARNING [indicators]: {msg} Feature set to NaN.", file=sys.stderr)
                    combined[feat] = np.float32(np.nan)
                    continue
                combined[feat] = calculate_mtf_trend_agreement(
                    combined[sources], scale
                ).astype(np.float32)
            case "mtf_breakout_h4_bias":
                # Event conjunction (A5): the M15 breakout score, kept only when
                # its sign agrees with the 4h trend slope. Sources carry their
                # own shift; 0.0 = no aligned breakout, NaN while any source is
                # in warm-up.
                sources = _resolve_sources(feat, 'mtf_breakout_h4_bias_sources')
                if sources is None:
                    continue
                breakout, slope = combined[sources[0]], combined[sources[1]]
                aligned = np.sign(breakout) == np.sign(slope)
                score = breakout.where(aligned, 0.0)
                combined[feat] = score.where(breakout.notna() & slope.notna()).astype(np.float32)
            case "mtf_daily_squeeze_h4_expand":
                # Daily compression + 4h expansion: the 4h breakout score, kept
                # only while the daily range percentile sits in its lowest band.
                sources = _resolve_sources(feat, 'mtf_daily_squeeze_h4_expand_sources')
                if sources is None:
                    continue
                breakout, daily_pct = combined[sources[0]], combined[sources[1]]
                squeezed = daily_pct <= float(params.get('mtf_squeeze_max_pctile', 20.0))
                score = breakout.where(squeezed, 0.0)
                combined[feat] = score.where(breakout.notna() & daily_pct.notna()).astype(np.float32)
            case "mtf_daily_volshift_h4_bias":
                # Daily volatility shift + 4h trend bias: the daily ATR-ratio
                # magnitude, signed by the 4h slope, only while expanding.
                sources = _resolve_sources(feat, 'mtf_daily_volshift_h4_bias_sources')
                if sources is None:
                    continue
                ratio, slope = combined[sources[0]], combined[sources[1]]
                shifting = ratio >= float(params.get('volshift_min', 0.25))
                score = (ratio * np.sign(slope)).where(shifting, 0.0)
                combined[feat] = score.where(ratio.notna() & slope.notna()).astype(np.float32)
            case _:
                if strict:
                    raise ValueError(f"[indicators] Cross-timeframe feature '{feat}' calculation not implemented!")
                print(f"WARNING: Cross-timeframe feature '{feat}' calculation not implemented!")

    return combined


def save_feature_importance(model, feature_names, model_name, output_dir=None):
    import csv

    importance = model.get_score(importance_type='gain')

    importance_named = {}
    for key, value in importance.items():
        if key.startswith('f') and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(feature_names):
                importance_named[feature_names[idx]] = value
            else:
                importance_named[key] = value
        else:
            importance_named[key] = value

    sorted_imp = sorted(importance_named.items(), key=lambda x: x[1], reverse=True)

    # Use provided output_dir or default to FEATURE_MAP_DIR
    if output_dir is None:
        output_dir = dir_config.FEATURE_MAP_DIR

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'feature_importance_{model_name}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Gain'])
        writer.writerows(sorted_imp)
    print(f"Feature importance saved to {csv_path}")
    
    print(f"\n=== Top 50 Features for {model_name} ===")
    # Use relative importance (%) so regressor gains are readable
    total_gain = sum(v for _, v in sorted_imp) if sorted_imp else 1.0
    for i, (feat, gain) in enumerate(sorted_imp[:50], 1):
        timeframe = "DAILY" if feat.startswith("daily_") else "4HOURS" if feat.startswith("4hours_") else "M15"
        pct = (gain / total_gain * 100) if total_gain > 0 else 0.0
        print(f"{i:2d}. {feat:25s} | Gain: {pct:6.2f}% | {timeframe}")
    print("=" * 60 + "\n")
    
    return sorted_imp