"""
Timing & execution family (B6, added 2026-09-12) — the computation.

Five TIMING components (when is the market entering a state worth acting in)
and five EXECUTION components (how an order should be placed once a signal
exists), computed at M15 from the same frame training uses:

  Timing:
    vol_ratio         ATR(fast)/ATR(slow) — volatility-shift level
    vol_shift_signal  robust causal z of log(vol_ratio) (median/IQR, trailing)
    range_ratio       fast-window range / slow-window range (compression state)
    micro_break       signed close-break of the trailing Donchian channel / ATR
    spread_norm       log(spread / trailing same-slot median spread)
    spread_regime     [0,1] causal percentile rank of spread_norm
    liq_impulse_score volume surprise minus spread surprise (slot-z space)
    liq_impulse       [0,1] causal percentile rank of liq_impulse_score
    queue_good        [0,1] maker queue-position proxy (calm spread + calm vol)

  Execution:
    maker_ok          [0,1] maker-execution suitability = queue_good x (1 - impulse)
    sl_pips           adaptive stop distance  = clip(k_sl    x range(slow) pips)
    tp_pips           adaptive target distance= clip(k_tp    x range(slow) pips)
    trail_pips        volatility trailing stop= clip(k_trail x range(fast) pips)
    cost_floor_pips   round-trip cost floor: spread + 2x slippage + commission

The SL/TP/trailing distances scale with the trailing HIGH-LOW RANGE of the
slow/fast window (not the mean per-bar true range — a stop for a trade held
up to 96h must scale with a day's movement, not a 15-minute bar's). The
default multipliers put the median EUR/USD values near the measured-viable
35/35 geometry (A15: every TP >= 70 arm at SL 35 loses reliably); the clips
are risk limits, not signal thresholds.

Design rules (the A8/A9 conventions):

- STRICTLY CAUSAL: the raw value at bar t uses data through bar t's close and
  nothing later; ``add_features`` end-shifts the series by ``lag`` (the A8
  precedent), so a model/consumer at bar t sees only completed-bar data.
- ROLLING ONLY: every baseline is a trailing window (rolling median/IQR,
  trailing same-slot statistics with group-shift(1), rolling percentile
  ranks). No global statistic enters any member.
- NO TUNED THRESHOLDS: every member is a continuous score, ratio, z or
  percentile — there is no binary cut anywhere in the timing block. The only
  constants are the execution CLIPS (sl/tp/trail bounds), which are risk
  limits, not signal thresholds.
- Spread/volume members follow A9 exactly: spread = 2*(ask_close - mid) in
  pips, stalled-feed bars (mid == close) and zero-volume bars masked NaN,
  never floored; volume/spread surprises are causal same-slot z-scores
  (London-local 15-min slot, group-shift 1 — the A7 DST precedent).
- DEGRADATION: a frame without ``mid``/``volume`` (the live Java payload,
  and the default training load without keep_mid=True) degrades the affected
  members instead of raising: spread_norm/spread_regime/liq_* -> NaN,
  queue_good falls back to its volatility component, maker_ok falls back to
  the price-based impulse proxy (percentile of vol_shift_signal), and
  cost_floor_pips falls back to ``tx_fallback_spread_pips``. cost_floor is
  the one member that FILLS missing spread with the fallback — it prices a
  fill, and a fill needs a number (the utils/costs.py floor rationale);
  every feature-like member keeps NaN.
- Window parameters deliberately avoid the warm-up regex suffixes
  (``_window``/``_period``/...): the family ships ``enabled: false`` in every
  config and the parameters block is GLOBAL, so a counted 1920-bar key would
  inflate every run's warm-up horizon (documented under-provisioning, the
  A7/A9 precedent — leading NaNs fall to the complete-case filter).

Roster/screening wiring: analytics/timing_execution_catalog.py.
Tests: tests/test_timing_execution.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PIP = 1e-4  # EUR/USD pip

# Bare member names (configs carry them with the m15_ prefix).
TIMING_MEMBERS: tuple[str, ...] = (
    'vol_ratio', 'vol_shift_signal', 'range_ratio', 'micro_break',
    'spread_norm', 'spread_regime', 'liq_impulse_score', 'liq_impulse',
    'queue_good',
)
EXECUTION_MEMBERS: tuple[str, ...] = (
    'maker_ok', 'sl_pips', 'tp_pips', 'trail_pips', 'cost_floor_pips',
)
MEMBER_NAMES: tuple[str, ...] = TIMING_MEMBERS + EXECUTION_MEMBERS

# Mirrored into the parameters: block of every features*.yaml. Window keys
# avoid the warm-up regex suffixes ON PURPOSE (see module docstring).
PARAMETER_DEFAULTS: dict[str, object] = {
    'tx_vol_fast': 16,      # fast ATR window, M15 bars (4h)
    'tx_vol_slow': 96,      # slow ATR window, M15 bars (1d)
    'tx_shift_span': 480,   # robust-z window for vol_shift_signal (5d)
    'tx_pct_span': 1920,    # percentile-rank window (20d)
    'tx_break_span': 96,    # Donchian window for micro_break (1d)
    'tx_slot_days': 60,     # same-slot occurrences for the seasonal baselines
    'tx_min_frac': 0.75,    # min_periods = frac * window
    'tx_slot_min_frac': 0.4,  # min same-slot occurrences = frac * tx_slot_days
    # Execution clips/multipliers — risk limits, not signal thresholds.
    # Distances scale with the trailing high-low RANGE of the window.
    'tx_sl_range_mult': 0.5,
    'tx_sl_min_pips': 15.0,
    'tx_sl_max_pips': 60.0,
    'tx_tp_range_mult': 0.5,
    'tx_tp_min_pips': 15.0,
    'tx_tp_max_pips': 150.0,
    'tx_trail_range_mult': 0.5,
    'tx_trail_min_pips': 8.0,
    'tx_trail_max_pips': 40.0,
    # Cost model (the measured Dukascopy defaults from utils/costs.py).
    'tx_slippage_pips': 0.2,           # per leg; a round trip pays twice
    'tx_commission_per_million': 18.0,  # per 1M notional per side
    'tx_fallback_spread_pips': 0.4,    # measured EUR/USD M15 median
}


def is_timing_execution_feature(bare_name: str) -> bool:
    return bare_name in MEMBER_NAMES


def _param(params: dict | None, key: str):
    if params and key in params:
        return params[key]
    return PARAMETER_DEFAULTS[key]


def compute_timing_execution_members(df: pd.DataFrame, names: list,
                                     params: dict | None = None,
                                     cache: dict | None = None) -> pd.DataFrame:
    """
    Raw (UNSHIFTED) timing/execution member series for bare ``names``.

    The value at bar t uses data through bar t's close; ``add_features``
    shifts the result by ``lag``, the live feature server reads the last row
    directly (Java already sends only completed bars). One shared ``cache``
    lets one call compute all 14 members without recomputing the ATR/spread
    machinery per member.
    """
    if cache is None:
        cache = {}
    p = lambda k: _param(params, k)  # noqa: E731

    w_fast = int(p('tx_vol_fast'))
    w_slow = int(p('tx_vol_slow'))
    w_shift = int(p('tx_shift_span'))
    w_pct = int(p('tx_pct_span'))
    w_break = int(p('tx_break_span'))
    n_slot = int(p('tx_slot_days'))
    min_frac = float(p('tx_min_frac'))
    slot_min_frac = float(p('tx_slot_min_frac'))

    def _nan() -> pd.Series:
        return pd.Series(np.nan, index=df.index)

    def _minp(w: int) -> int:
        return max(2, int(round(w * min_frac)))

    # --- volatility machinery (true range; the weekend gap is REAL stop
    # placement risk, so TR is deliberately not gap-masked) ---------------
    def _tr() -> pd.Series:
        if 'tr' not in cache:
            prev_close = df['close'].shift(1)
            hi = pd.concat([df['high'], prev_close], axis=1).max(axis=1)
            lo = pd.concat([df['low'], prev_close], axis=1).min(axis=1)
            cache['tr'] = (hi - lo).astype(float)
        return cache['tr']

    def _atr(w: int) -> pd.Series:
        key = ('atr', w)
        if key not in cache:
            cache[key] = _tr().rolling(w, min_periods=_minp(w)).mean()
        return cache[key]

    def _pct01(s: pd.Series, w: int) -> pd.Series:
        """Causal trailing percentile rank of the latest value, in [0, 1]."""
        return s.rolling(w, min_periods=_minp(w) // 2 + 1).rank(pct=True)

    def _vol_ratio() -> pd.Series:
        if 'vol_ratio' not in cache:
            slow = _atr(w_slow)
            cache['vol_ratio'] = _atr(w_fast) / slow.where(slow > 0)
        return cache['vol_ratio']

    def _vol_shift() -> pd.Series:
        if 'vol_shift' not in cache:
            x = pd.Series(np.log(_vol_ratio().where(_vol_ratio() > 0)),
                          index=df.index)
            past = x.shift(1)
            minp = _minp(w_shift)
            med = past.rolling(w_shift, min_periods=minp).median()
            q75 = past.rolling(w_shift, min_periods=minp).quantile(0.75)
            q25 = past.rolling(w_shift, min_periods=minp).quantile(0.25)
            sd = (q75 - q25) / 1.349  # IQR -> sigma under normality
            cache['vol_shift'] = (x - med) / sd.where(sd > 0)
        return cache['vol_shift']

    # --- spread machinery (A9 conventions) --------------------------------
    def _spr() -> pd.Series | None:
        """Spread in pips: 2*(ask_close - mid); stalled/negative bars NaN."""
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

    def _slot_minp() -> int:
        return min(n_slot, max(10, int(round(n_slot * slot_min_frac))))

    def _slotz(x: pd.Series) -> pd.Series:
        minp = _slot_minp()
        grouped = x.groupby(_slot())
        mu = grouped.transform(lambda s: s.shift(1).rolling(n_slot, min_periods=minp).mean())
        sd = grouped.transform(lambda s: s.shift(1).rolling(n_slot, min_periods=minp).std())
        return (x - mu) / sd.where(sd > 0)

    def _spread_norm() -> pd.Series:
        if 'spread_norm' not in cache:
            spr = _spr()
            if spr is None:
                cache['spread_norm'] = _nan()
            else:
                minp = _slot_minp()
                med = spr.groupby(_slot()).transform(
                    lambda s: s.shift(1).rolling(n_slot, min_periods=minp).median())
                ratio = spr / med.where(med > 0)
                cache['spread_norm'] = pd.Series(np.log(ratio.where(ratio > 0)),
                                                 index=df.index)
        return cache['spread_norm']

    def _spread_regime() -> pd.Series:
        if 'spread_regime' not in cache:
            cache['spread_regime'] = _pct01(_spread_norm(), w_pct)
        return cache['spread_regime']

    # --- volume machinery --------------------------------------------------
    def _vol() -> pd.Series | None:
        if 'vol' not in cache:
            if 'volume' not in df.columns:
                cache['vol'] = None
            else:
                v = df['volume'].astype(float)
                cache['vol'] = v.where(v > 0)
        return cache['vol']

    def _liq_score() -> pd.Series:
        """Volume surprise minus spread surprise (both slot-z). High = burst
        of activity while the spread tightens = a liquidity impulse; without
        a spread column the volume surprise alone carries the member."""
        if 'liq_score' not in cache:
            vol = _vol()
            if vol is None:
                cache['liq_score'] = _nan()
            else:
                vz = _slotz(pd.Series(np.log(vol), index=df.index))
                spr = _spr()
                if spr is None:
                    cache['liq_score'] = vz
                else:
                    sz = _slotz(pd.Series(np.log(spr), index=df.index))
                    cache['liq_score'] = vz - sz
        return cache['liq_score']

    def _liq_impulse() -> pd.Series:
        if 'liq_impulse' not in cache:
            cache['liq_impulse'] = _pct01(_liq_score(), w_pct)
        return cache['liq_impulse']

    # --- structure ----------------------------------------------------------
    def _range_ratio() -> pd.Series:
        if 'range_ratio' not in cache:
            hi_f = df['high'].rolling(w_fast, min_periods=_minp(w_fast)).max()
            lo_f = df['low'].rolling(w_fast, min_periods=_minp(w_fast)).min()
            hi_s = df['high'].rolling(w_slow, min_periods=_minp(w_slow)).max()
            lo_s = df['low'].rolling(w_slow, min_periods=_minp(w_slow)).min()
            rng_s = (hi_s - lo_s).where(hi_s > lo_s)
            cache['range_ratio'] = (hi_f - lo_f) / rng_s
        return cache['range_ratio']

    def _micro_break() -> pd.Series:
        """Signed break-of-structure: how far the close sits OUTSIDE the
        trailing Donchian(w_break) channel (which excludes the bar itself),
        in ATR units; 0 inside the channel."""
        if 'micro_break' not in cache:
            hi = df['high'].shift(1).rolling(w_break, min_periods=_minp(w_break)).max()
            lo = df['low'].shift(1).rolling(w_break, min_periods=_minp(w_break)).min()
            atr = _atr(w_break)
            c = df['close']
            up = ((c - hi) / atr.where(atr > 0)).clip(lower=0.0)
            dn = ((c - lo) / atr.where(atr > 0)).clip(upper=0.0)
            s = up + dn
            cache['micro_break'] = s.where(hi.notna() & lo.notna()
                                           & atr.notna() & (atr > 0))
        return cache['micro_break']

    # --- queue / maker -------------------------------------------------------
    def _vol_pct() -> pd.Series:
        if 'vol_pct' not in cache:
            cache['vol_pct'] = _pct01(_vol_ratio(), w_pct)
        return cache['vol_pct']

    def _queue_good() -> pd.Series:
        """[0,1] queue-position proxy: calm short-horizon volatility and (when
        the spread is observable) a spread at the tight end of its regime —
        the state in which a resting limit order keeps and advances its queue
        position instead of being repriced."""
        if 'queue_good' not in cache:
            calm_vol = 1.0 - _vol_pct()
            if _spr() is None:
                cache['queue_good'] = calm_vol
            else:
                calm_spread = 1.0 - _spread_regime()
                cache['queue_good'] = 0.5 * (calm_vol + calm_spread)
        return cache['queue_good']

    def _impulse_pct() -> pd.Series:
        """[0,1] 'activity burst right now': the liquidity impulse when tick
        volume is observable, else the price-based proxy (percentile of the
        volatility-shift signal)."""
        if 'impulse_pct' not in cache:
            if _vol() is not None:
                cache['impulse_pct'] = _liq_impulse()
            else:
                cache['impulse_pct'] = _pct01(_vol_shift(), w_pct)
        return cache['impulse_pct']

    def _maker_ok() -> pd.Series:
        if 'maker_ok' not in cache:
            cache['maker_ok'] = _queue_good() * (1.0 - _impulse_pct())
        return cache['maker_ok']

    # --- execution distances ------------------------------------------------
    def _range_pips(w: int) -> pd.Series:
        """Trailing w-bar high-low range in pips (the movement scale a stop
        or target for that holding horizon must answer to)."""
        key = ('range_pips', w)
        if key not in cache:
            hi = df['high'].rolling(w, min_periods=_minp(w)).max()
            lo = df['low'].rolling(w, min_periods=_minp(w)).min()
            cache[key] = (hi - lo).where(hi > lo) / _PIP
        return cache[key]

    def _clip(s: pd.Series, lo_key: str, hi_key: str) -> pd.Series:
        return s.clip(lower=float(p(lo_key)), upper=float(p(hi_key)))

    def _cost_floor() -> pd.Series:
        """Round-trip cost floor in pips: spread (trailing median, fallback
        constant when unobservable — a fill needs a number) + two slippage
        legs + round-trip commission converted to pips at the current price
        (2 * comm/1M * price / pip; the EUR/USD currency conversion cancels,
        see utils/costs.py)."""
        spr = _spr()
        fallback = float(p('tx_fallback_spread_pips'))
        if spr is None:
            spread_est = pd.Series(fallback, index=df.index)
        else:
            spread_est = spr.rolling(w_pct, min_periods=_minp(w_pct) // 2 + 1).median()
            spread_est = spread_est.fillna(fallback)
        commission = (2.0 * float(p('tx_commission_per_million')) / 1e6
                      * df['close'].astype(float) / _PIP)
        return spread_est + 2.0 * float(p('tx_slippage_pips')) + commission

    builders = {
        'vol_ratio': _vol_ratio,
        'vol_shift_signal': _vol_shift,
        'range_ratio': _range_ratio,
        'micro_break': _micro_break,
        'spread_norm': _spread_norm,
        'spread_regime': _spread_regime,
        'liq_impulse_score': _liq_score,
        'liq_impulse': _liq_impulse,
        'queue_good': _queue_good,
        'maker_ok': _maker_ok,
        'sl_pips': lambda: _clip(float(p('tx_sl_range_mult')) * _range_pips(w_slow),
                                 'tx_sl_min_pips', 'tx_sl_max_pips'),
        'tp_pips': lambda: _clip(float(p('tx_tp_range_mult')) * _range_pips(w_slow),
                                 'tx_tp_min_pips', 'tx_tp_max_pips'),
        'trail_pips': lambda: _clip(float(p('tx_trail_range_mult')) * _range_pips(w_fast),
                                    'tx_trail_min_pips', 'tx_trail_max_pips'),
        'cost_floor_pips': _cost_floor,
    }

    out = {}
    for name in names:
        if name not in builders:
            raise ValueError(f"'{name}' is not a timing/execution member")
        vals = np.asarray(builders[name](), dtype=np.float64)
        vals[~np.isfinite(vals)] = np.nan
        out[name] = pd.Series(vals, index=df.index)
    return pd.DataFrame(out, index=df.index)


def execution_snapshot(df_m15: pd.DataFrame, params: dict | None = None) -> dict:
    """
    The live feature-server payload: every member's value at the LAST bar of
    ``df_m15`` (Java sends completed bars only, so no extra shift), NaN
    mapped to None so the response stays strict JSON.
    """
    frame = compute_timing_execution_members(df_m15, list(MEMBER_NAMES),
                                             params=params)
    last = frame.iloc[-1]
    return {name: (None if pd.isna(last[name]) else float(last[name]))
            for name in MEMBER_NAMES}
