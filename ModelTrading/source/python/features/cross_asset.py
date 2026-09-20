"""
Cross-asset currency strength — the one information family the audit never tested.

WHAT THIS IS, AND WHY IT IS NOT ALREADY REJECTED
------------------------------------------------
`analytics/information_audit.py` tested carry, COT, US yields, DXY, equity risk sentiment,
VIX, realized-volatility state and the rate differential. 2,130 tests, three learner
families, and every layer produced *fewer* nominally-significant cells than noise would.

What it did not test is a statement about EUR/USD formed from **other instruments' prices**.
That is a different object from everything in that list: it is price-derived, so it is not a
"non-price family", but it is not derivable from EUR/USD's own history either. A currency's
rank against seven others is information the target series does not contain.

THE ONE TRAP, NAMED UP FRONT: DXY
---------------------------------
DXY *is* a USD strength index and it was tested — best |IC| 0.073, p 0.272, and a half-year
sign stability of 0.14, the worst reading in the whole table. Rebuilding it and calling it new
would just re-measure that result.

The reason a rank can still be different: **DXY is 57.6 % EUR**. For a EUR/USD model it is
close to the mechanical inverse of the target itself and carries almost no independent
information. This module therefore defaults to `exclude_self=True`, which drops the traded
pair from the design matrix entirely. USD is then identified through GBPUSD/AUDUSD/NZDUSD/
USDJPY/USDCHF/USDCAD and EUR through EURJPY/EURGBP — the resulting spread is computed from
instruments that never touch the series being predicted. Without that exclusion the feature
would contain its own target and the audit would be meaningless.

A PAIR IS A DIFFERENCE
----------------------
    log-return(EURUSD) = strength(EUR) - strength(USD)

so N pairs over K currencies is a linear system `A c = r`, with `A[p, base] = +1` and
`A[p, quote] = -1`. Its null space is exactly `span(1)` whenever the pair graph is connected
(adding a constant to every currency changes no pair), so the system determines strengths only
*relative to each other*. `np.linalg.pinv` returns the minimum-norm solution, which is the one
orthogonal to the null space — i.e. `sum(c) = 0` falls out automatically, and the currencies
are measured against an equal-weighted basket of themselves. No explicit constraint row is
needed, and adding one would be a second way of saying the same thing.

Crosses are what make this overdetermined. With USD pairs only, the graph is a star centred on
USD: every currency hangs on exactly one edge, the solution is exact but carries no redundancy,
and USD-specific noise lands identically on all seven estimates. `EURJPY`, `EURGBP` and
`AUDJPY` create cycles, so the least-squares fit averages that noise out and the residual
becomes a usable quality check on the decomposition itself.

GAPS ARE SEGMENTED, NOT INTERPOLATED
------------------------------------
The provider export arrives in waves and currently has a ~1,050-day hole between 2023-02 and
2026-01. A 250-day momentum computed *across* that hole is not a slow signal, it is two
different regimes subtracted from each other. `segment_ids` cuts the index at gaps longer than
`max_gap_days` and every rolling window is confined to one segment; a window that would span a
boundary yields NaN rather than a number nobody can interpret.

CAUSALITY
---------
Values are written **unshifted**, exactly like `data/update_timesfm_data.py` and
`data/update_regime_model_data.py`. The `shift(1)` is applied downstream by
`features/external_data.get_external_df`, on the sparse daily index *before* the reindex onto
the dense bar index — so at any bar the value visible is the one formed on an earlier day.
Shifting here as well would lag the features by two days and nothing would report it.
"""

import numpy as np
import pandas as pd

import ModelTrading.config.instruments as instruments

# Momentum lookbacks in trading days. Deliberately few: the point of this family is the
# cross-sectional construction, not a wide horizon sweep.
DEFAULT_LOOKBACKS = (60, 250)

# A hole longer than this ends a segment. Two weeks is long enough to survive a holiday
# cluster and far short of the export gap this data actually has.
MAX_GAP_DAYS = 14

# Trailing window for turning dispersion into a bounded percentile, matching how
# `daily_volatility_percentile` and `vix_percentile` are built elsewhere in the pipeline.
DISPERSION_WINDOW = 20
DISPERSION_LOOKBACK = 500


def design_matrix(symbols):
    """`(A, currencies)` for the pair set: A[p, base] = +1, A[p, quote] = -1."""
    symbols = [s.upper() for s in symbols]
    currencies = instruments.currencies_of(symbols)
    pos = {c: i for i, c in enumerate(currencies)}
    A = np.zeros((len(symbols), len(currencies)))
    for i, s in enumerate(symbols):
        base, quote = instruments.base_quote(s)
        A[i, pos[base]] += 1.0
        A[i, pos[quote]] -= 1.0
    return A, currencies


def decompose(returns, symbols=None, check_rank=True):
    """Per-date currency strengths from pair log-returns.

    Args:
        returns: DataFrame (date x symbol) of pair log-returns. NaNs are not allowed —
            the caller restricts to complete cases so the design matrix is the same on
            every date and a changing universe can never pass unnoticed.
        symbols: column subset to use; defaults to every column.

    Returns:
        (currency_returns, residual_rms) — a DataFrame (date x currency) summing to zero
        across each row, and the per-date RMS of `A c - r`, which is the quality check the
        crosses make possible.
    """
    symbols = [s.upper() for s in (symbols if symbols is not None else returns.columns)]
    R = returns[symbols].to_numpy(dtype=float)
    if not np.isfinite(R).all():
        raise ValueError("decompose() needs complete cases; filter before calling")
    A, currencies = design_matrix(symbols)

    if check_rank:
        rank = np.linalg.matrix_rank(A)
        if rank != len(currencies) - 1:
            raise ValueError(
                f"pair graph is not connected: rank {rank} over {len(currencies)} "
                f"currencies, expected {len(currencies) - 1}. Some currency is not "
                f"reachable from the others and its strength is not identified."
            )

    # Minimum-norm least squares. The null space is span(1), so the solution is orthogonal
    # to it and each row sums to zero — the basket normalisation, for free.
    C = R @ np.linalg.pinv(A).T
    resid = R - C @ A.T
    rms = pd.Series(np.sqrt((resid ** 2).mean(axis=1)), index=returns.index, name='resid_rms')
    return pd.DataFrame(C, index=returns.index, columns=currencies), rms


def segment_ids(index, max_gap_days=MAX_GAP_DAYS):
    """Integer segment id per timestamp; a gap longer than `max_gap_days` starts a new one."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return pd.Series(dtype='int64', index=idx)
    gap = idx.to_series().diff().dt.days.fillna(0.0)
    return (gap > max_gap_days).cumsum().astype('int64')


def _segmented_diff(frame, lookback, segments):
    """`frame - frame.shift(lookback)` computed inside segments only.

    A window that would span a segment boundary yields NaN. Without this the ~1,050-day
    export hole turns a 250-day momentum into the difference between two unrelated regimes.
    """
    out = frame - frame.shift(lookback)
    same = segments == segments.shift(lookback)
    return out.where(same.reindex(frame.index).fillna(False), np.nan)


def currency_indices(currency_returns):
    """Cumulative strength index per currency. Level is arbitrary; only differences matter."""
    return currency_returns.cumsum()


def cross_sectional_rank(frame):
    """Row-wise percentile rank in [0, 1]. Bounded by construction, hence STATIONARY.

    `rank(pct=True)` gives (1/n .. 1); centring on the row mean of that grid maps it onto a
    symmetric scale so that "middle of the pack" is 0.5 regardless of how many currencies the
    universe happens to contain that day.
    """
    return frame.rank(axis=1, pct=True)


def cross_sectional_z(frame):
    """Row-wise z-score. Keeps the magnitude a rank throws away; unbounded, hence PARTIAL."""
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mu, axis=0).div(sd, axis=0)


def breadth(frame, currency):
    """Share of the other currencies `currency` is beating, in [0, 1].

    Rows with any NaN yield NaN, not 0. `lt()` treats NaN as False, so without the mask a
    warm-up row would silently read 0.0 — "beats nobody" — instead of "unknown", and the
    first `lookback` days of every segment would carry a fabricated extreme value.
    """
    others = frame.drop(columns=[currency])
    out = others.lt(frame[currency], axis=0).sum(axis=1) / float(others.shape[1])
    return out.where(frame.notna().all(axis=1), np.nan)


def _rolling_pct(series, window, min_periods):
    """Trailing percentile of the last value within its own history, in [0, 1]."""
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: float((x[:-1] <= x[-1]).mean()) if len(x) > 1 else np.nan, raw=True)


def build_features(returns, symbol=instruments.PRIMARY, lookbacks=DEFAULT_LOOKBACKS,
                   exclude_self=True, max_gap_days=MAX_GAP_DAYS):
    """The `daily_ccy_*` feature frame for one traded pair.

    Args:
        returns: DataFrame (date x symbol) of pair log-returns, complete cases only.
        symbol: the pair the features describe — its base and quote become the named legs.
        exclude_self: drop `symbol` from the design matrix. **Default True**: with it in, the
            feature contains its own target through DXY's failure mode and the audit that
            follows cannot mean anything.

    Returns unshifted daily values; `external_data.get_external_df` applies the shift.
    """
    symbol = symbol.upper()
    base, quote = instruments.base_quote(symbol)
    cols = [c for c in returns.columns if not (exclude_self and c.upper() == symbol)]
    if len(cols) < 3:
        raise ValueError(f"need at least 3 pairs to decompose, got {len(cols)}")

    ccy_ret, resid = decompose(returns[cols], cols)
    for leg in (base, quote):
        if leg not in ccy_ret.columns:
            raise ValueError(
                f"{leg} is not identified by the remaining pairs {cols}. With "
                f"exclude_self=True the traded pair is dropped, so both its legs must "
                f"still be reachable through other instruments."
            )

    idx = currency_indices(ccy_ret)
    seg = segment_ids(idx.index, max_gap_days)

    out = pd.DataFrame(index=idx.index)
    for lb in lookbacks:
        mom = _segmented_diff(idx, lb, seg)
        rank = cross_sectional_rank(mom)
        z = cross_sectional_z(mom)
        out[f'daily_ccy_rank_spread_{lb}'] = rank[base] - rank[quote]
        out[f'daily_ccy_spread_z_{lb}'] = z[base] - z[quote]
        out[f'daily_ccy_{base.lower()}_rank_{lb}'] = rank[base]
        out[f'daily_ccy_{quote.lower()}_rank_{lb}'] = rank[quote]
        out[f'daily_ccy_{quote.lower()}_breadth_{lb}'] = breadth(mom, quote)

    # Dispersion: how far apart the currencies are moving. Low = one common factor drives
    # everything, high = the currencies are being repriced against each other. A regime
    # statement neither the ADX/price-efficiency rule nor a single-instrument HMM can see,
    # because both only ever look at one series.
    disp = ccy_ret.std(axis=1, ddof=0).rolling(
        DISPERSION_WINDOW, min_periods=max(5, DISPERSION_WINDOW // 2)).mean()
    out['daily_ccy_dispersion_pct'] = _rolling_pct(
        disp, DISPERSION_LOOKBACK, min_periods=DISPERSION_LOOKBACK // 4)

    out.attrs['symbols_used'] = list(cols)
    out.attrs['currencies'] = list(ccy_ret.columns)
    out.attrs['resid_rms_mean'] = float(resid.mean())
    return out


def feature_names(symbol=instruments.PRIMARY, lookbacks=DEFAULT_LOOKBACKS):
    """The feature names `build_features` produces, without computing anything."""
    base, quote = instruments.base_quote(symbol.upper())
    names = []
    for lb in lookbacks:
        names += [f'daily_ccy_rank_spread_{lb}', f'daily_ccy_spread_z_{lb}',
                  f'daily_ccy_{base.lower()}_rank_{lb}', f'daily_ccy_{quote.lower()}_rank_{lb}',
                  f'daily_ccy_{quote.lower()}_breadth_{lb}']
    return names + ['daily_ccy_dispersion_pct']
