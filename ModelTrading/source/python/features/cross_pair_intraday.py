"""
Intraday cross-pair lead-lag & triangle residuals (pre-registration A10).

WHAT THIS IS, AND WHY THE CLOSED DAILY FAMILY DOES NOT COVER IT
---------------------------------------------------------------
The daily currency-strength family (A4, features/cross_asset.py) is closed:
0/240 audit cells, and its frozen replication cell flipped sign out of sample.
That family asked "which currency is strong this quarter" — a level ranking
over 60-250 day lookbacks. This module asks a different question at a
different cadence: which leg moved in the LAST 1-8 HOURS, and has EUR/USD
caught up. Its core object, the triangle residual
(log EURUSD - log EURGBP - log GBPUSD, and the EURJPY/USDJPY twin), is a pure
microstructure quantity that has no daily counterpart at all.

THE TIMEZONE DEFECT, DISCOVERED 2026-09-07 AND HANDLED HERE
-----------------------------------------------------------
The non-EURUSD M15 exports are NOT stamped in UTC. The pre-2026 generation
(naive timestamps) is stamped in **Europe/Berlin local time**: measured on
2014+ bars, the cross-correlation of 1-bar returns EURUSD vs GBPUSD peaks at
lag -4 bars (+1 h) in January and lag -8 (+2 h) in July with a lag-0
correlation of ~0.03 — the same export-timezone artefact documented for the
delivered daily files (01:00/02:00 grids), now measured at M15. The 2026
refill generation ('Z'-suffixed rows from 2025-12-31T23:00Z) is true UTC; the
generations abut without overlap at 2026-01-01 00:00 Europe/Berlin.

``load_pair_m15`` therefore normalises GENERATION-AWARE: tz-aware rows are
taken as UTC, naive rows are localised Europe/Berlin and converted
(DST-ambiguous/nonexistent stamps dropped — they fall in the repeated/skipped
hour and cannot be assigned). After the correction the lag-0 correlation is
0.573 (Jan 0.552 / Jul 0.607) and every other lag reads <= |0.007|; the GBP
triangle residual tightens from +/-40 pips (p1/p99, artefact) to a median of
-0.76 pips with std 1.95 — an actual microstructure spread. A blanket
Berlin shift would double-shift the 2026 rows; a blanket UTC read is what
produced the artefact.

STALENESS POLICY: EXACT JOIN, NEVER FORWARD-FILL
------------------------------------------------
Pair bars are joined onto the EURUSD index at identical corrected-UTC stamps.
A missing pair bar yields NaN, full stop — at M15 a stale value is worthless
and no fill is ever attempted (the A4 staleness guard, sharpened). Endpoint
log-differences propagate NaN from either endpoint, which segment-bounds every
window across the ~13 four-to-eleven-day export holes per pair (clustered on
the late-October DST week) by construction. EUR/USD bar selection is never
touched: the EURUSD frame passed in defines the grid.

CAUSALITY
---------
Raw member values at bar t use data through bar t's close and are end-shifted
by ``lag`` inside ``indicators.add_features`` (the A8 precedent); the audits
read the unshifted values against outcomes that start at that close. EURUSD
itself never enters a member, with three named, pre-registered exceptions
(docs/preregistration.md A10): the triangle residuals (the residual measures
precisely EURUSD's deviation from its cross-implied price), the lead-lag beta
(past EURUSD returns are the regressand of a causal estimation read one bar
back; the bar-t payload is the OTHER pair's move), and the correlation-
breakdown state (a direction-symmetric second moment). The USD/EUR composites
(usdb/usdz/eurx/usd_disp/xcons) exclude EURUSD by construction — the A4
``exclude_self`` rule carried over, so no DXY-style self-reference can form.

Return-type members use ENDPOINT log-differences (weekend repricing across
pairs is genuine cross-pair information); RV, beta and correlation inputs use
gap-masked 1-bar returns (``indicators.gap_masked_log_returns`` — a gap return
inside a variance or regression estimate is a fabricated jump, the A8 rule).

TRAINING-ONLY: the live Java payload carries EURUSD bars only; the nine pair
CSVs are not staged. All members resolve NaN in live inference; enabling any
member requires a live pair-data plan first (stage_jforex.ps1 allowlists,
feature-server wiring).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import ModelTrading.config.directories as dir_config
import ModelTrading.config.instruments as instruments

# Timezone of the naive-stamped export generation of every non-EURUSD pair
# (measured 2026-09-07, frozen in A10 — see module docstring).
PAIR_TZ = 'Europe/Berlin'

# The six USD pairs of the intraday breadth/strength composites and the two
# EUR crosses of the cross-only EUR strength. EURUSD is deliberately absent
# from both (the A4 exclude_self rule).
USD_PAIRS = ('GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDCAD')
EUR_CROSSES = ('EURGBP', 'EURJPY')
RISK_PAIR = 'AUDJPY'

TRIANGLES = {
    # residual = log(EURUSD) - sign_a*log(a) - sign_b*log(b)
    'gbp': (('EURGBP', 1.0), ('GBPUSD', 1.0)),
    'jpy': (('EURJPY', 1.0), ('USDJPY', -1.0)),
}

_pair_close_cache: dict[str, pd.Series] = {}


def _normalise_stamps(raw_time: pd.Series) -> pd.DatetimeIndex:
    """Generation-aware timestamp normalisation to naive UTC.

    tz-aware strings (the 2026 refill generation, 'Z'/offset suffix) are UTC;
    naive strings (the pre-2026 generation) are Europe/Berlin local time.
    DST-ambiguous and nonexistent naive stamps become NaT (dropped by the
    caller) — they sit inside the repeated/skipped hour and cannot be
    assigned to a UTC instant.
    """
    s = raw_time.astype(str)
    aware = s.str.contains('Z') | s.str.contains(r'\+', regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    if aware.any():
        ts = pd.to_datetime(s[aware], format='ISO8601', utc=True)
        out[aware] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    naive = ~aware
    if naive.any():
        ts = pd.to_datetime(s[naive], format='ISO8601')
        loc = ts.dt.tz_localize(PAIR_TZ, ambiguous='NaT', nonexistent='NaT')
        out[naive] = loc.dt.tz_convert('UTC').dt.tz_localize(None)
    return pd.DatetimeIndex(out)


def load_pair_m15(symbol: str, data_dir=None) -> pd.Series:
    """Close series of one pair's M15 CSV on corrected UTC stamps.

    Deliberately NOT csv_utils.load_csv: that loader parses the whole file
    with utc=True, which reads the naive Berlin-stamped generation as UTC —
    the exact defect this module exists to correct. No weekend filter is
    applied here; the EURUSD index the members are aligned onto already
    carries the project's bar selection.
    """
    symbol = symbol.upper()
    if symbol in _pair_close_cache:
        return _pair_close_cache[symbol]
    path = instruments.csv_path(symbol, 'm15', data_dir)
    raw = pd.read_csv(path, usecols=['time', 'close'])
    idx = _normalise_stamps(raw['time'])
    close = pd.Series(raw['close'].to_numpy(dtype=np.float64), index=idx)
    close = close[close.index.notna()].sort_index()
    close = close[~close.index.duplicated(keep='last')]
    _pair_close_cache[symbol] = close
    return close


def clear_pair_cache() -> None:
    _pair_close_cache.clear()


def compute_cross_pair_members(df: pd.DataFrame, names: list,
                               min_frac: float = 0.75,
                               pair_frames: dict | None = None,
                               cache: dict | None = None) -> pd.DataFrame:
    """Raw (UNSHIFTED) A10 cross-pair member series for bare ``names``.

    The value at bar t uses data through bar t's close (pair bars joined at
    identical corrected-UTC stamps complete exactly when EURUSD's bar does);
    ``add_features`` shifts the result by ``lag``, the audits read it
    unshifted against outcomes starting at that close. Used by BOTH the
    indicator dispatch and the audit builders in
    analytics/cross_pair_catalog.py, so they cannot drift apart.

    Args:
        df: the EURUSD frame (defines the grid; only 'close' is read).
        pair_frames: optional {symbol: close Series or frame with 'close'}
            override for tests; default loads the provider CSVs
            generation-aware via ``load_pair_m15``.
    """
    if cache is None:
        cache = {}
    from ModelTrading.source.python.features import indicators as _ind

    eur_close = df['close'].astype(np.float64)

    def _minp(w: int) -> int:
        return max(2, int(round(w * float(min_frac))))

    def _pair_close(sym: str) -> pd.Series:
        key = ('close', sym)
        if key not in cache:
            if pair_frames is not None:
                src = pair_frames[sym]
                s = src['close'] if isinstance(src, pd.DataFrame) else src
                s = s.astype(np.float64)
            else:
                s = load_pair_m15(sym)
            cache[key] = s.reindex(df.index)  # exact join, never ffill
        return cache[key]

    def _log(sym: str) -> pd.Series:
        key = ('log', sym)
        if key not in cache:
            c = _pair_close(sym)
            cache[key] = pd.Series(np.log(c.where(c > 0)), index=df.index)
        return cache[key]

    def _dw(sym: str, w: int) -> pd.Series:
        """Endpoint log-difference over w bars (NaN if either endpoint is)."""
        key = ('dw', sym, w)
        if key not in cache:
            cache[key] = _log(sym).diff(w)
        return cache[key]

    def _ret1(sym: str) -> pd.Series:
        """Gap-masked 1-bar log return of the aligned pair close."""
        key = ('ret1', sym)
        if key not in cache:
            cache[key] = _ind.gap_masked_log_returns(_pair_close(sym))
        return cache[key]

    def _eur_ret1() -> pd.Series:
        if 'eur_ret1' not in cache:
            cache['eur_ret1'] = _ind.gap_masked_log_returns(eur_close)
        return cache['eur_ret1']

    def _causal_z(s: pd.Series, w: int) -> pd.Series:
        past = s.shift(1)
        mu = past.rolling(w, min_periods=w // 2).mean()
        sd = past.rolling(w, min_periods=w // 2).std()
        return (s - mu) / sd.where(sd > 0)

    def _usd_sign(sym: str) -> float:
        base, quote = instruments.base_quote(sym)
        return 1.0 if base == 'USD' else -1.0

    def _resid(tri: str) -> pd.Series:
        key = ('resid', tri)
        if key not in cache:
            r = pd.Series(np.log(eur_close.where(eur_close > 0)), index=df.index)
            for sym, sign in TRIANGLES[tri]:
                r = r - sign * _log(sym)
            cache[key] = r
        return cache[key]

    def _usd_z(w: int, z: int) -> pd.DataFrame:
        """Per-pair causal z of the USD-positive Δw return, all 6 USD pairs."""
        key = ('usd_z', w, z)
        if key not in cache:
            cols = {sym: _causal_z(_dw(sym, w) * _usd_sign(sym), z)
                    for sym in USD_PAIRS}
            cache[key] = pd.DataFrame(cols, index=df.index)
        return cache[key]

    def _eurx_z(w: int, z: int) -> pd.Series:
        key = ('eurx_z', w, z)
        if key not in cache:
            # both crosses are EUR-base -> +Δlog is EUR-positive
            zs = pd.DataFrame({sym: _causal_z(_dw(sym, w), z)
                               for sym in EUR_CROSSES}, index=df.index)
            cache[key] = zs.mean(axis=1, skipna=False)
        return cache[key]

    def _usdz(w: int, z: int) -> pd.Series:
        key = ('usdz', w, z)
        if key not in cache:
            cache[key] = _usd_z(w, z).mean(axis=1, skipna=False)
        return cache[key]

    def _llpred(x: pd.Series, w: int) -> pd.Series:
        """Trailing lead-lag prediction: beta(EURUSD_t ~ x_{t-1}) read one bar
        back, times x's bar-t value, in trailing EURUSD return-sd units. The
        shift(1) on beta and sd keeps EURUSD's bar-t return OUT of the bar-t
        feature value — only x_t (the other pair's move) enters at t."""
        y = _eur_ret1()
        xl = x.shift(1)
        minp = _minp(w)
        cov = y.rolling(w, min_periods=minp).cov(xl)
        var = xl.rolling(w, min_periods=minp).var()
        beta = (cov / var.where(var > 0)).shift(1)
        sd1 = y.rolling(w, min_periods=minp).std().shift(1)
        return beta * x / sd1.where(sd1 > 0)

    _LL_X = {
        'llpred_gbp': lambda: _ret1('GBPUSD'),
        'llpred_jpy': lambda: _ret1('USDJPY'),
        'llpred_eurx': lambda: pd.DataFrame(
            {sym: _ret1(sym) for sym in EUR_CROSSES},
            index=df.index).mean(axis=1, skipna=False),
    }
    _CORR_PAIR = {'corrbk_gbp': 'GBPUSD', 'corrbk_jpy': 'USDJPY'}
    _RV_PAIR = {'rvz_gbp': 'GBPUSD', 'rvz_jpy': 'USDJPY'}

    out = {}
    for name in names:
        parsed = _ind._parse_cross_pair(name)
        if parsed is None:
            raise ValueError(f"'{name}' is not a cross-pair member")
        kind, ints = parsed['kind'], parsed['ints']
        if kind in ('tri_gbp_z', 'tri_jpy_z'):
            s = _causal_z(_resid(kind[4:7]), ints[0])
        elif kind in ('tri_gbp_chg_z', 'tri_jpy_chg_z'):
            w, z = ints
            s = _causal_z(_resid(kind[4:7]).diff(w), z)
        elif kind == 'eurx_z':
            s = _eurx_z(*ints)
        elif kind == 'usdb':
            w = ints[0]
            signs = pd.DataFrame(
                {sym: np.sign(_dw(sym, w) * _usd_sign(sym)) for sym in USD_PAIRS},
                index=df.index)
            s = signs.mean(axis=1, skipna=False)
        elif kind == 'usdz':
            s = _usdz(*ints)
        elif kind == 'usd_disp':
            s = _usd_z(*ints).std(axis=1, ddof=0, skipna=False)
        elif kind in _LL_X:
            s = _llpred(_LL_X[kind](), ints[0])
        elif kind in _CORR_PAIR:
            w, z = ints
            c = _eur_ret1().rolling(w, min_periods=_minp(w)).corr(
                _ret1(_CORR_PAIR[kind]))
            s = _ind.calculate_rolling_percentile_rank(c, z)
        elif kind in _RV_PAIR:
            w, z = ints
            r2 = _ret1(_RV_PAIR[kind]) ** 2
            rv = np.sqrt(r2.rolling(w, min_periods=_minp(w)).sum())
            s = _causal_z(pd.Series(np.log(rv.where(rv > 0)), index=df.index), z)
        elif kind == 'xcons':
            w, z = ints
            s = 0.5 * (_eurx_z(w, z) - _usdz(w, z))
        elif kind == 'risk_z':
            w, z = ints
            s = _causal_z(_dw(RISK_PAIR, w), z)
        else:  # pragma: no cover — the parser and this table share one roster
            raise ValueError(f"unhandled cross-pair kind '{kind}'")
        out[name] = s.astype(np.float64)
    return pd.DataFrame(out, index=df.index)
