"""
Event-direction audit — model-free go/no-go for the A5 event-sequence family.

Answers ONE question, frozen in docs/preregistration.md A5 before this ran:
do candle event conjunctions/sequences carry a causal DIRECTIONAL edge INSIDE
trend / high-vol regimes? Direction is claimed only for composed events —
single features are helpers and are never scored here.

Why a new module: `information_audit.py` is daily-only by construction (it
loads eurusd_daily.csv, its FAMILIES are external columns, its conditioners
are its own causal-z terciles). This family lives at M15/4h/daily cadence and
its regime restriction is the PROJECT's rule-based regime
(labeling/regime.generate_regime_labels), so the machinery is rebuilt here on
top of the same primitives (_rank_corr, half_year_sign_stability,
benjamini_hochberg, the shift-then-mask null of conditional_ic).

Design points, all pre-registered:

- Candidates are SIGNED event scores (0 = no event). The decision statistic is
  the EVENT-CONDITIONAL Spearman IC: computed on bars where score != 0 AND the
  regime mask holds, against (a) the forward log-return and (b) the symmetric
  +/-1*ATR(14) barrier race (drift vs tradeability).
- Null: circular shift of the FULL score series; the event mask is rebuilt
  from the SHIFTED series and the regime mask applied afterwards — never
  subset-then-shift (the conditional_ic invariant, extended so the event
  definition travels with the draw).
- Regimes: rule-based labels computed on DAILY bars (vol lookback 500 is
  daily-calibrated), the label index shifted +24h to bar-close time and
  forward-filled onto the target index — day D's regime is only used after
  D's close (one day more conservative than the training convention). Bars
  where the volatility percentile is still in warm-up are excluded.
- Cells: the six TREND/RANGE x HIGH/MED/LOW_VOL combinations (the CLAUDE.md
  non-artifact grid) plus the aggregates TREND_all and HIGH_VOL_all. DECISION
  cells per the mandate: TREND_all, HIGH_VOL_all, TREND_HIGH_VOL — the others
  are reported and cannot produce a pass.
- BH per timeframe layer; accept = p_BH < 0.05 AND |IC| >= 0.03 AND half-year
  sign stability >= 0.60 in >= 1 decision cell on the forward return, with
  the same cell's barrier IC not collapsing (p_BH < 0.05 AND |IC| >= 0.03) —
  a forward-return-only pass is flagged drift_only and does not count.
- Benchmark control (H5 lesson): mean(sign(score) * outcome) on event bars is
  ranked against the same circular-shift draws — random bars of the SAME
  regime cell — never against zero or break-even.
- Development data only: the load is clamped below timeframes.HOLDOUT_START.

Usage
-----
    python -m ModelTrading.source.python.analytics.event_direction_audit --timeframe all
    python -m ModelTrading.source.python.analytics.event_direction_audit --timeframe m15 \\
        --exclude m15_impulse_burst_3_20,m15_tick_imbalance_20
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import ta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config  # noqa: E402
import ModelTrading.config.timeframes as timeframes  # noqa: E402
import ModelTrading.source.python.utils.csv as csv_utils  # noqa: E402
import ModelTrading.source.python.features.config as fconfig  # noqa: E402
from ModelTrading.source.python.analytics import calendar_direction_catalog as cal_catalog  # noqa: E402
from ModelTrading.source.python.analytics import cross_pair_catalog as xp_catalog  # noqa: E402
from ModelTrading.source.python.analytics import event_sequence_catalog as catalog  # noqa: E402
from ModelTrading.source.python.analytics import realized_moments_catalog as rm_catalog  # noqa: E402
from ModelTrading.source.python.analytics import session_anchor_catalog as sess_catalog  # noqa: E402
from ModelTrading.source.python.analytics import surprise_catalog as sp_catalog  # noqa: E402
from ModelTrading.source.python.analytics import core_feature_catalog as core_catalog  # noqa: E402
from ModelTrading.source.python.analytics import volume_spread_catalog as vs_catalog  # noqa: E402
from ModelTrading.source.python.analytics.information_audit import (  # noqa: E402
    _rank_corr,
    benjamini_hochberg,
    forward_return,
    half_year_sign_stability,
)
from ModelTrading.source.python.features import indicators  # noqa: E402
from ModelTrading.source.python.labeling.regime import generate_regime_labels  # noqa: E402

# Accept rule (pre-registered, mirrors H1)
P_BH_MAX = 0.05
IC_MIN = 0.03
SIGN_STABILITY_MIN = 0.60

DECISION_CELLS = ('TREND_all', 'HIGH_VOL_all', 'TREND_HIGH_VOL')
REPORT_CELLS = ('TREND_HIGH_VOL', 'TREND_MED_VOL', 'TREND_LOW_VOL',
                'RANGE_HIGH_VOL', 'RANGE_MED_VOL', 'RANGE_LOW_VOL')

TF_CONFIG = {
    'm15':    dict(csv='eurusd_m15.csv',    horizons=(16, 96, 384), n_shuffles=500,  min_events=100),
    '4hours': dict(csv='eurusd_4hours.csv', horizons=(6, 18, 24),   n_shuffles=1000, min_events=100),
    'daily':  dict(csv='eurusd_daily.csv',  horizons=(5, 10, 20),   n_shuffles=1000, min_events=50),
    'mtf':    dict(csv='eurusd_m15.csv',    horizons=(16, 96, 384), n_shuffles=500,  min_events=100),
}

ATR_WINDOW = 14


def yearly_sign_stability(x, y, index, min_obs=30):
    """Share of calendar YEARS whose IC has the same sign as the pooled IC.

    The A6 calendar family's replacement for half_year_sign_stability
    (pre-registered): the evidence unit is the scheduled event (~8-11 per
    clock-year), so a half-year holds ~4 events and its per-period IC is pure
    noise — the coarser period A5's limitation record demanded.
    """
    df = pd.DataFrame({'x': x, 'y': y}, index=index).dropna()
    if df.empty:
        return np.nan, 0
    pooled = df['x'].corr(df['y'], method='spearman')
    if not np.isfinite(pooled):
        return np.nan, 0
    signs = []
    for _, chunk in df.groupby(df.index.year):
        if len(chunk) < min_obs:
            continue
        c = chunk['x'].corr(chunk['y'], method='spearman')
        if np.isfinite(c):
            signs.append(np.sign(c) == np.sign(pooled))
    if not signs:
        return np.nan, 0
    return float(np.mean(signs)), len(signs)


def multiyear_sign_stability(x, y, index, min_obs=None):
    """Share of the three fixed A7 multi-year windows agreeing with the pooled
    IC sign, counting only windows with >= min_obs scored observations.

    The A7 session family's SHARPENED robustness gate (pre-registered): the
    clock carries ~5,500 day-clusters of power, so all three disjoint windows
    (2005-2011 / 2012-2018 / 2019-dev end) must qualify AND agree — enforced
    via stability_min=1.0 and min_periods=3 in apply_decision.
    """
    if min_obs is None:
        min_obs = sess_catalog.MULTIYEAR_MIN_OBS
    df = pd.DataFrame({'x': x, 'y': y}, index=index).dropna()
    if df.empty:
        return np.nan, 0
    pooled = df['x'].corr(df['y'], method='spearman')
    if not np.isfinite(pooled):
        return np.nan, 0
    signs = []
    for start, end in sess_catalog.MULTIYEAR_WINDOWS:
        chunk = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
        if len(chunk) < min_obs:
            continue
        c = chunk['x'].corr(chunk['y'], method='spearman')
        if np.isfinite(c):
            signs.append(np.sign(c) == np.sign(pooled))
    if not signs:
        return np.nan, 0
    return float(np.mean(signs)), len(signs)


# The audited families. 'sequence' is the frozen A5 configuration and stays
# bit-identical; 'calendar' (A6) adds the event-window conditioning cells,
# yearly sign stability and the distinct-days floor; 'session' (A7) adds the
# session conditioning cells and the sharpened multi-year sign gate (all
# three disjoint windows must qualify and agree).
CATALOGS = {
    'sequence': dict(module=catalog, decision_cells=DECISION_CELLS,
                     stability_fn=half_year_sign_stability, min_days=None,
                     extra_cells=None, stability_min=SIGN_STABILITY_MIN,
                     min_periods=None,
                     stability_desc='half-year stability>=0.60',
                     tfs=('daily', '4hours', 'm15', 'mtf'),
                     output_name='event_sequence_audit'),
    'calendar': dict(module=cal_catalog, decision_cells=cal_catalog.DECISION_CELLS,
                     stability_fn=yearly_sign_stability,
                     min_days=cal_catalog.MIN_DISTINCT_DAYS,
                     extra_cells=cal_catalog.calendar_window_cells,
                     stability_min=SIGN_STABILITY_MIN, min_periods=None,
                     stability_desc='calendar-year stability>=0.60',
                     tfs=('daily', '4hours', 'm15'),
                     output_name='calendar_direction_audit'),
    'session': dict(module=sess_catalog, decision_cells=sess_catalog.DECISION_CELLS,
                    stability_fn=multiyear_sign_stability,
                    min_days=sess_catalog.MIN_DISTINCT_DAYS,
                    extra_cells=sess_catalog.session_window_cells,
                    stability_min=sess_catalog.STABILITY_MIN,
                    min_periods=sess_catalog.MIN_PERIODS,
                    stability_desc='same sign in all 3 multi-year windows',
                    tfs=('4hours', 'm15'),
                    output_name='session_anchor_audit'),
    # A8: realized moments & path asymmetry — computed at M15, read at the
    # last completed M15 bar inside each 4h bar (the catalog's build_scores
    # hook). Daily cadence runs through information_audit.py --families rmom,
    # never through this tool (pre-registered split).
    'moments': dict(module=rm_catalog, decision_cells=rm_catalog.DECISION_CELLS,
                    stability_fn=half_year_sign_stability,
                    min_days=rm_catalog.MIN_DISTINCT_DAYS,
                    extra_cells=rm_catalog.moment_window_cells,
                    stability_min=SIGN_STABILITY_MIN, min_periods=None,
                    stability_desc='half-year stability>=0.60',
                    tfs=rm_catalog.AUDIT_TFS,
                    output_name='realized_moments_audit'),
    # A9: volume & spread microstructure — computed at M15 from the tick-
    # volume/mid columns (build_scores hook); the 4h layer carries all 29
    # candidates, the m15 layer only the pre-registered intraday subset
    # (vs_catalog.M15_AUDIT_MEMBERS).
    'volume': dict(module=vs_catalog, decision_cells=vs_catalog.DECISION_CELLS,
                   stability_fn=half_year_sign_stability,
                   min_days=vs_catalog.MIN_DISTINCT_DAYS,
                   extra_cells=vs_catalog.volume_window_cells,
                   stability_min=SIGN_STABILITY_MIN, min_periods=None,
                   stability_desc='half-year stability>=0.60',
                   tfs=vs_catalog.AUDIT_TFS,
                   output_name='volume_spread_audit'),
    # A10: intraday cross-pair lead-lag & triangle residuals — computed at
    # M15 from the nine additional pairs' corrected-UTC closes (build_scores
    # hook), full roster on both layers. The pre-registered development/
    # replication split (search <= xp_catalog.DEV_END, frozen cell scored
    # once from REPLICATION_START) is enforced by the CLI --start/--end of
    # the two runs, per A10.
    'crosspair': dict(module=xp_catalog, decision_cells=xp_catalog.DECISION_CELLS,
                      stability_fn=half_year_sign_stability,
                      min_days=xp_catalog.MIN_DISTINCT_DAYS,
                      extra_cells=xp_catalog.cross_pair_window_cells,
                      stability_min=SIGN_STABILITY_MIN, min_periods=None,
                      stability_desc='half-year stability>=0.60',
                      tfs=xp_catalog.AUDIT_TFS,
                      output_name='cross_pair_audit'),
    # Core-set inventory (2026-09-12): the catalogue features with no audit row
    # anywhere (feature store tier=untested, role=model), computed by the real
    # training pipeline and scored as dense series on the 4h layer — all three
    # timeframes on one grid. Inventory framing: no registered decision cells;
    # results feed the feature store's tier derivation.
    'core': dict(module=core_catalog, decision_cells=core_catalog.DECISION_CELLS,
                 stability_fn=half_year_sign_stability,
                 min_days=core_catalog.MIN_DISTINCT_DAYS,
                 extra_cells=None, stability_min=SIGN_STABILITY_MIN,
                 min_periods=None,
                 stability_desc='half-year stability>=0.60',
                 tfs=core_catalog.AUDIT_TFS,
                 output_name='core_feature_audit'),
    # A14: standardized macro surprises from the validated MT5 calendar export
    # (build_scores hook; operative first-print window only). Event-anchored,
    # regime-orthogonal by design -> the decision cell is ALL; yearly sign
    # stability (the pooled index carries ~200+ events per year).
    'surprise': dict(module=sp_catalog, decision_cells=sp_catalog.DECISION_CELLS,
                     stability_fn=yearly_sign_stability,
                     min_days=sp_catalog.MIN_DISTINCT_DAYS,
                     extra_cells=sp_catalog.surprise_window_cells,
                     stability_min=SIGN_STABILITY_MIN, min_periods=None,
                     stability_desc='calendar-year stability>=0.60',
                     tfs=sp_catalog.AUDIT_TFS,
                     output_name='surprise_audit'),
}


# ---------------------------------------------------------------------------
# Vectorised barrier race
# ---------------------------------------------------------------------------
def barrier_outcome_fast(high, low, close, horizon, atr, k=1.0, chunk=20000):
    """Vectorised drop-in for information_audit.barrier_outcome.

    Same semantics, pinned element-wise by tests/test_event_direction_audit.py:
    +1 the up barrier is touched first within `horizon` bars, -1 down first,
    0 neither (or both inside one bar — unresolvable at bar resolution), NaN
    for a non-finite/zero ATR and for the last `horizon` bars. The reference
    is an O(n*h) Python loop — unusable at 500k M15 bars x h=384; this builds
    chunked sliding windows over the forward highs/lows instead.
    """
    n = len(close)
    h_arr = high.to_numpy(dtype=float)
    l_arr = low.to_numpy(dtype=float)
    c_arr = close.to_numpy(dtype=float)
    a_arr = atr.to_numpy(dtype=float)
    H = int(horizon)
    out = np.full(n, np.nan)
    if n > H:
        # sw[i] = values[i : i+H]; bar i races over bars i+1 .. i+H -> sw[i+1].
        sw_h = np.lib.stride_tricks.sliding_window_view(h_arr, H)
        sw_l = np.lib.stride_tricks.sliding_window_view(l_arr, H)
        up = c_arr + k * a_arr
        dn = c_arr - k * a_arr
        sentinel = H + 1
        for s0 in range(0, n - H, chunk):
            e0 = min(s0 + chunk, n - H)
            win_h = sw_h[s0 + 1:e0 + 1]
            win_l = sw_l[s0 + 1:e0 + 1]
            hit_up = win_h >= up[s0:e0, None]
            hit_dn = win_l <= dn[s0:e0, None]
            any_up = hit_up.any(axis=1)
            any_dn = hit_dn.any(axis=1)
            ju = np.where(any_up, hit_up.argmax(axis=1), sentinel)
            jd = np.where(any_dn, hit_dn.argmax(axis=1), sentinel)
            res = np.zeros(e0 - s0)
            res[ju < jd] = 1.0
            res[jd < ju] = -1.0
            # ju == jd: both inside one bar -> 0; neither hit -> 0. Matches ref.
            out[s0:e0] = res
    out[~np.isfinite(a_arr) | (a_arr <= 0)] = np.nan
    if H > 0:
        out[max(n - H, 0):] = np.nan
    return pd.Series(out, index=close.index)


# ---------------------------------------------------------------------------
# Regime cells
# ---------------------------------------------------------------------------
def daily_regime_on_index(target_index, daily_df):
    """Rule-based daily regime labels, made causal for intraday use.

    generate_regime_labels on the daily bars, index shifted +24h to bar-CLOSE
    time, forward-filled onto `target_index`. Day D's regime therefore first
    applies to bars after D's close — one day more conservative than the
    training-label convention (advanced_train ffills unshifted), which is the
    right side to err on for an audit. Warm-up bars (volatility percentile
    still NaN) are left NaN and excluded from every cell.
    """
    regime = generate_regime_labels(daily_df)
    shifted = regime.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=24)
    aligned = shifted.reindex(target_index, method='ffill')
    valid = aligned['volatility_percentile'].notna() & aligned['adx'].notna()
    return aligned, valid.to_numpy(dtype=bool)


def build_cell_masks(regime_aligned, valid):
    """{cell_name: bool ndarray} for the six combined cells + the aggregates."""
    trend = (regime_aligned['regime_trend_label'] == 'TREND').to_numpy(dtype=bool)
    vol = regime_aligned['regime_volatility_label']
    cells = {
        'TREND_all': trend,
        'HIGH_VOL_all': (vol == 'HIGH_VOL').to_numpy(dtype=bool),
    }
    for cell in REPORT_CELLS:
        cells[cell] = (regime_aligned['regime_combined'] == cell).to_numpy(dtype=bool)
    return {name: mask & valid for name, mask in cells.items()}


# ---------------------------------------------------------------------------
# Core statistic
# ---------------------------------------------------------------------------
def audit_candidate(name, x, index, cells, outcomes, n_shuffles, min_events, rng,
                    verbose=True, decision_cells=DECISION_CELLS,
                    stability_fn=half_year_sign_stability):
    """All (horizon, cell, outcome) rows for one candidate.

    outcomes: {horizon: {outcome_name: float ndarray}}. The null draws one
    offset set per horizon and rolls the score ONCE per draw, evaluating every
    cell x outcome on that rolled series — the event mask is rebuilt from the
    shifted values, the regime mask stays fixed (shift-then-mask).
    Every scored row also carries n_days (distinct calendar days among its
    bars) — the A6 evidence-unit diagnostic.
    """
    rows = []
    n = len(x)
    finite_x = np.isfinite(x)
    event = (x != 0) & finite_x
    for horizon, by_outcome in outcomes.items():
        lo = max(int(horizon), 1)
        if n <= 2 * lo + 2:
            continue
        offsets = rng.integers(lo, n - lo, size=int(n_shuffles))
        obs = {}
        for cell_name, cmask in cells.items():
            base = cmask & event
            for oc_name, y in by_outcome.items():
                m = base & np.isfinite(y)
                n_ev = int(m.sum())
                key = (cell_name, oc_name)
                if n_ev < min_events:
                    obs[key] = dict(n_events=n_ev, skipped=True)
                    continue
                ic = _rank_corr(x[m], y[m])
                stab, n_periods = stability_fn(x[m], y[m], index[m])
                mean_signed = float(np.mean(np.sign(x[m]) * y[m]))
                n_days = int(pd.DatetimeIndex(index[m]).normalize().nunique())
                obs[key] = dict(n_events=n_ev, skipped=False, ic=ic, stab=stab,
                                n_periods=n_periods, mean_signed=mean_signed,
                                n_days=n_days, null_ic=[], null_ms=[])
        active = [k for k, v in obs.items() if not v['skipped']]
        if active:
            for o in offsets:
                xs = np.roll(x, int(o))
                ev_s = (xs != 0) & np.isfinite(xs)
                for cell_name, oc_name in active:
                    y = by_outcome[oc_name]
                    m = cells[cell_name] & ev_s & np.isfinite(y)
                    if m.sum() < min_events:
                        continue
                    c = _rank_corr(xs[m], y[m])
                    if np.isfinite(c):
                        obs[(cell_name, oc_name)]['null_ic'].append(abs(c))
                    obs[(cell_name, oc_name)]['null_ms'].append(
                        float(np.mean(np.sign(xs[m]) * y[m])))
        for (cell_name, oc_name), v in obs.items():
            row = dict(candidate=name, horizon=int(horizon), cell=cell_name,
                       outcome=oc_name, n_events=v['n_events'],
                       decision_cell=cell_name in decision_cells)
            if v['skipped']:
                row.update(ic=np.nan, p_raw=np.nan, sign_stability=np.nan,
                           n_half_years=0, mean_signed=np.nan, n_days=0,
                           bench_pctile=np.nan, verdict='skipped')
            else:
                null_ic = np.asarray(v['null_ic'])
                p = (float((np.sum(null_ic >= abs(v['ic'])) + 1) / (len(null_ic) + 1))
                     if len(null_ic) else np.nan)
                null_ms = np.asarray(v['null_ms'])
                bench = (float(np.mean(null_ms < v['mean_signed']) * 100)
                         if len(null_ms) else np.nan)
                row.update(ic=v['ic'], p_raw=p, sign_stability=v['stab'],
                           n_half_years=v['n_periods'], mean_signed=v['mean_signed'],
                           n_days=v['n_days'], bench_pctile=bench, verdict='scored')
            rows.append(row)
        if verbose:
            n_scored = sum(1 for k in obs if not obs[k]['skipped'])
            print(f"    h={horizon}: {n_scored} scored, {len(obs) - n_scored} skipped cells")
    return rows


def _pass_flags(table, min_days=None, stability_min=SIGN_STABILITY_MIN,
                min_periods=None):
    """The row-level accept criteria, shared by both decision protocols.

    Expects ``p_bh`` to be present. Returns (fwd_pass, bar_ok) boolean Series
    with EXACTLY the A5-A10 semantics — apply_decision stays bit-identical.
    """
    fwd_pass = ((table['outcome'] == 'fwd_return')
                & (table['p_bh'] < P_BH_MAX)
                & (table['ic'].abs() >= IC_MIN)
                & (table['sign_stability'] >= stability_min))
    if min_days is not None:
        fwd_pass &= table['n_days'] >= int(min_days)
    if min_periods is not None:
        fwd_pass &= table['n_half_years'] >= int(min_periods)
    bar_ok = ((table['outcome'] == 'barrier')
              & (table['p_bh'] < P_BH_MAX)
              & (table['ic'].abs() >= IC_MIN))
    return fwd_pass, bar_ok


def _candidate_decisions(table):
    """Per-candidate verdicts from a table carrying p_bh/fwd_pass/barrier_ok."""
    decisions = []
    for cand, grp in table.groupby('candidate'):
        dec = grp[grp['decision_cell']]
        fwd_cells = [tuple(r) for r in
                     dec.loc[dec['fwd_pass'], ['horizon', 'cell']].itertuples(index=False)]
        bar_cells = {tuple(r) for r in
                     dec.loc[dec['barrier_ok'], ['horizon', 'cell']].itertuples(index=False)}
        full = [hc for hc in fwd_cells if hc in bar_cells]
        drift = [hc for hc in fwd_cells if hc not in bar_cells]
        n_skip = int((grp['verdict'] == 'skipped').sum())
        best = dec[dec['outcome'] == 'fwd_return'].sort_values('p_bh')
        decisions.append(dict(
            candidate=cand,
            passes=bool(full),
            drift_only=bool(drift) and not full,
            passing_cells=str(full) if full else '',
            drift_cells=str(drift) if drift else '',
            n_cells_scored=int((grp['verdict'] == 'scored').sum()),
            n_cells_skipped=n_skip,
            best_decision_p_bh=float(best['p_bh'].iloc[0]) if len(best) else np.nan,
            best_decision_ic=float(best['ic'].iloc[0]) if len(best) else np.nan,
        ))
    return pd.DataFrame(decisions)


def apply_decision(table, min_days=None, stability_min=SIGN_STABILITY_MIN,
                   min_periods=None):
    """BH within the layer, then the pre-registered accept rule per candidate.

    Pass = at least one DECISION cell whose forward-return row clears
    p_BH < 0.05 AND |IC| >= 0.03 AND sign stability >= ``stability_min``, AND
    whose barrier row in the same (candidate, horizon, cell) clears
    p_BH < 0.05 AND |IC| >= 0.03. Forward-return-only passes are flagged
    drift_only. ``min_days`` (A6/A7) additionally requires the passing cell to
    rest on at least that many distinct calendar days; ``min_periods`` (A7)
    requires at least that many stability periods to have QUALIFIED (a
    candidate whose warm-up or sparsity empties a multi-year window cannot
    pass).
    """
    table = table.copy()
    table['p_bh'] = benjamini_hochberg(table['p_raw'].to_numpy())
    fwd_pass, bar_ok = _pass_flags(table, min_days=min_days,
                                   stability_min=stability_min,
                                   min_periods=min_periods)
    table['fwd_pass'] = fwd_pass
    table['barrier_ok'] = bar_ok
    return table, _candidate_decisions(table)


# ---------------------------------------------------------------------------
# A11 registered-cells protocol (--decision-cells)
# ---------------------------------------------------------------------------
# The A5-A10 design corrects every scored cell of a layer as one BH family
# (366-1,458 rows) with a permutation floor of 1/(draws+1). A single true cell
# can then never clear p_BH < 0.05 (docs/preregistration.md A11: ~15-30 floor
# cells needed SIMULTANEOUSLY). The registered protocol restores single-cell
# power: a short pre-registered list of (candidate, cell, horizon, outcome,
# expected sign) entries forms its OWN BH family (each fwd_return entry brings
# its barrier-guard row into the family); everything else stays reported but is
# flagged exploratory and cannot produce a pass. Recommended >= 5,000 draws so
# the p_raw floor (1/(draws+1)) sits well below 0.05/m.
DECISION_DRAWS_RECOMMENDED = 5000

_SIGN_TOKENS = {'+': 1, '-': -1, '+1': 1, '-1': -1, '1': 1,
                'pos': 1, 'neg': -1, 'positive': 1, 'negative': -1}


def _parse_sign(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (1, -1):
            return int(value)
        raise ValueError(f"expected_sign must be +1 or -1, got {value!r}")
    token = str(value).strip().lower()
    if token in _SIGN_TOKENS:
        return _SIGN_TOKENS[token]
    raise ValueError(f"expected_sign must be one of {sorted(_SIGN_TOKENS)}, got {value!r}")


def parse_decision_cells(spec):
    """Registered decision cells from a JSON/CSV file or an inline list.

    File: JSON (a list of objects, or {"cells": [...]}) or CSV, with fields
    candidate (alias: feature), cell, horizon, outcome, expected_sign (alias:
    sign; +/-, +1/-1, pos/neg). Inline: semicolon-separated
    'candidate:cell:horizon:outcome:sign' entries.
    Returns a list of normalised dicts; raises on malformed entries — a
    decision run must never silently drop a registered cell.
    """
    if os.path.isfile(spec):
        if spec.lower().endswith('.json'):
            with open(spec, encoding='utf-8') as f:
                data = json.load(f)
            raw = data['cells'] if isinstance(data, dict) else data
        else:
            raw = pd.read_csv(spec).to_dict('records')
    else:
        raw = []
        for part in str(spec).split(';'):
            part = part.strip()
            if not part:
                continue
            bits = part.split(':')
            if len(bits) != 5:
                raise ValueError(
                    f"inline decision cell needs candidate:cell:horizon:outcome:sign, got {part!r}")
            raw.append(dict(candidate=bits[0], cell=bits[1], horizon=bits[2],
                            outcome=bits[3], expected_sign=bits[4]))
    if not raw:
        raise ValueError(f"--decision-cells {spec!r} contains no entries")
    entries = []
    for r in raw:
        candidate = r.get('candidate', r.get('feature'))
        if not candidate:
            raise ValueError(f"decision cell without candidate/feature: {r!r}")
        outcome = str(r['outcome']).strip()
        if outcome not in ('fwd_return', 'barrier'):
            raise ValueError(f"outcome must be fwd_return or barrier, got {outcome!r}")
        entries.append(dict(candidate=str(candidate).strip(),
                            cell=str(r['cell']).strip(),
                            horizon=int(r['horizon']),
                            outcome=outcome,
                            expected_sign=_parse_sign(r.get('expected_sign', r.get('sign')))))
    return entries


def _row_key(table):
    return list(zip(table['candidate'], table['cell'],
                    table['horizon'].astype(int), table['outcome']))


def apply_decision_registered(table, registered, min_days=None,
                              stability_min=SIGN_STABILITY_MIN, min_periods=None):
    """The A11 protocol: registered rows form their OWN BH family.

    Family = the registered rows plus, for every registered fwd_return entry,
    the matching barrier row (auto-included guard). BH runs separately over
    the family and over the exploratory remainder. A registered entry passes
    iff its row clears p_BH < 0.05 AND |IC| >= 0.03 AND sign(IC) matches the
    registration AND the source family's stability gate; a fwd_return entry
    additionally needs its guard row at p_BH < 0.05 AND |IC| >= 0.03 with the
    same registered sign (else drift_only). Exploratory candidates keep the
    legacy per-candidate evaluation for reporting, with ``passes`` forced
    False (``would_pass_unregistered`` records what the legacy rule would have
    said within the exploratory family).

    Returns (table, exploratory_decision, registered_decision).
    """
    table = table.copy()
    reg_keys = {(e['candidate'], e['cell'], int(e['horizon']), e['outcome'])
                for e in registered}
    guard_keys = {(e['candidate'], e['cell'], int(e['horizon']), 'barrier')
                  for e in registered if e['outcome'] == 'fwd_return'}
    fam_keys = reg_keys | guard_keys
    keys = _row_key(table)
    in_family = np.array([k in fam_keys for k in keys], dtype=bool)
    table['registered'] = np.array([k in reg_keys for k in keys], dtype=bool)
    table['family'] = np.where(in_family, 'decision', 'exploratory')

    p = table['p_raw'].to_numpy(dtype=float)
    p_bh = np.full(len(table), np.nan)
    if in_family.any():
        p_bh[in_family] = benjamini_hochberg(p[in_family])
    if (~in_family).any():
        p_bh[~in_family] = benjamini_hochberg(p[~in_family])
    table['p_bh'] = p_bh
    fwd_pass, bar_ok = _pass_flags(table, min_days=min_days,
                                   stability_min=stability_min,
                                   min_periods=min_periods)
    table['fwd_pass'] = fwd_pass
    table['barrier_ok'] = bar_ok

    lookup = {}
    for i, k in enumerate(keys):
        lookup.setdefault(k, i)

    def _row_gates(i, expected_sign):
        r = table.iloc[i]
        scored = r['verdict'] == 'scored' and np.isfinite(r['p_raw'])
        ok_p = bool(scored and r['p_bh'] < P_BH_MAX)
        ok_ic = bool(scored and abs(r['ic']) >= IC_MIN)
        ok_sign = bool(scored and np.sign(r['ic']) == expected_sign)
        stab = r['sign_stability']
        ok_stab = bool(scored and np.isfinite(stab) and stab >= stability_min)
        if min_periods is not None:
            ok_stab &= bool(r['n_half_years'] >= int(min_periods))
        ok_days = True if min_days is None else bool(r['n_days'] >= int(min_days))
        return r, scored, ok_p, ok_ic, ok_sign, ok_stab, ok_days

    verdicts = []
    for e in registered:
        k = (e['candidate'], e['cell'], int(e['horizon']), e['outcome'])
        out = dict(candidate=e['candidate'], cell=e['cell'], horizon=int(e['horizon']),
                   outcome=e['outcome'], expected_sign=e['expected_sign'])
        i = lookup.get(k)
        if i is None:
            out.update(scored=False, passes=False, drift_only=False,
                       ic=np.nan, p_raw=np.nan, p_bh=np.nan, sign_stability=np.nan,
                       n_events=0, guard_ok=False, note='row not produced by this run')
            verdicts.append(out)
            continue
        r, scored, ok_p, ok_ic, ok_sign, ok_stab, ok_days = _row_gates(i, e['expected_sign'])
        entry_ok = scored and ok_p and ok_ic and ok_sign and ok_stab and ok_days
        guard_ok, guard_note = True, ''
        if e['outcome'] == 'fwd_return':
            gi = lookup.get((e['candidate'], e['cell'], int(e['horizon']), 'barrier'))
            if gi is None:
                guard_ok, guard_note = False, 'barrier guard row missing'
            else:
                g = table.iloc[gi]
                g_scored = g['verdict'] == 'scored' and np.isfinite(g['p_raw'])
                guard_ok = bool(g_scored and g['p_bh'] < P_BH_MAX
                                and abs(g['ic']) >= IC_MIN
                                and np.sign(g['ic']) == e['expected_sign'])
                if not g_scored:
                    guard_note = 'barrier guard not scored'
        note = '' if scored else 'cell skipped (below the event floor)'
        out.update(scored=bool(scored), passes=bool(entry_ok and guard_ok),
                   drift_only=bool(entry_ok and not guard_ok),
                   ic=float(r['ic']) if scored else np.nan,
                   p_raw=float(r['p_raw']) if scored else np.nan,
                   p_bh=float(r['p_bh']) if scored else np.nan,
                   sign_stability=float(r['sign_stability']) if scored else np.nan,
                   n_events=int(r['n_events']),
                   guard_ok=bool(guard_ok), note=note or guard_note)
        verdicts.append(out)
    registered_decision = pd.DataFrame(verdicts)

    exploratory_decision = _candidate_decisions(table[~in_family])
    if len(exploratory_decision):
        exploratory_decision['would_pass_unregistered'] = exploratory_decision['passes']
        exploratory_decision['would_drift_unregistered'] = exploratory_decision['drift_only']
        exploratory_decision['passes'] = False
        exploratory_decision['drift_only'] = False
        exploratory_decision['exploratory'] = True
    return table, exploratory_decision, registered_decision


# ---------------------------------------------------------------------------
# Layer assembly
# ---------------------------------------------------------------------------
def _load_ohlc(csv_name, start, end):
    return csv_utils.load_csv(os.path.join(dir_config.DATA_DIR, csv_name),
                              start_date=start, end_date=end, keep_mid=True)


def _clamp_end(end):
    """Never read past the sealed hold-out."""
    seal = pd.Timestamp(timeframes.HOLDOUT_START) - pd.Timedelta(days=1)
    if end is None:
        return seal.strftime('%Y-%m-%d')
    return min(pd.Timestamp(end), seal).strftime('%Y-%m-%d')


def build_layer_inputs(tf, start, end, exclude=(), cat=catalog):
    """(scores DataFrame [prefixed names], ohlc frame, index) for one layer."""
    cfg = TF_CONFIG[tf]
    if hasattr(cat, 'build_scores'):
        # A8: computation cadence (M15) differs from evaluation cadence — the
        # catalog aligns its own member series onto this layer's bar grid.
        ohlc = _load_ohlc(cfg['csv'], start, end)
        return cat.build_scores(tf, ohlc, exclude=exclude), ohlc
    if tf == 'mtf':
        from ModelTrading.source.python.analytics.redundancy_screen import build_combined_frame
        originals = cat.force_enable(fconfig)
        try:
            combined = build_combined_frame()
        finally:
            cat.restore(fconfig, originals)
        combined = combined.loc[(combined.index >= pd.Timestamp(start))
                                & (combined.index <= pd.Timestamp(end))]
        ohlc = _load_ohlc(cfg['csv'], start, end).reindex(combined.index)
        names = [n for n in cat.CANDIDATES['mtf']
                 if n in combined.columns and n not in exclude]
        scores = combined[names]
    else:
        ohlc = _load_ohlc(cfg['csv'], start, end)
        originals = cat.force_enable(fconfig, timeframe=tf)
        try:
            feats = indicators.add_features(ohlc.copy(), timeframe=tf, apply_shift=False)
        finally:
            cat.restore(fconfig, originals)
        names = [n for n in cat.CANDIDATES[tf] if n not in exclude]
        cols, out_names = [], []
        for n in names:
            bare = n.removeprefix(f'{tf}_')
            if bare in feats.columns:
                cols.append(feats[bare].astype(float))
                out_names.append(n)
            else:
                print(f"WARNING: {n} not computed — excluded from the layer")
        scores = pd.concat(cols, axis=1)
        scores.columns = out_names
    return scores, ohlc


def run_layer(tf, args):
    cfg = TF_CONFIG[tf]
    fam = CATALOGS[args.catalog]
    start = args.start
    end = _clamp_end(args.end)
    n_shuffles = args.n_shuffles or cfg['n_shuffles']
    min_events = args.min_events or cfg['min_events']
    exclude = set(args.exclude.split(',')) if args.exclude else set()

    print(f"\n=== layer {tf} ({args.catalog}): {start} .. {end}, "
          f"horizons {cfg['horizons']}, {n_shuffles} draws, min_events {min_events} ===")
    t0 = time.time()
    scores, ohlc = build_layer_inputs(tf, start, end, exclude, cat=fam['module'])
    print(f"  {len(scores.columns)} candidates on {len(scores):,} bars "
          f"({time.time() - t0:.0f}s)")

    daily_df = _load_ohlc('eurusd_daily.csv', None, end)
    regime_aligned, valid = daily_regime_on_index(scores.index, daily_df)
    cells = build_cell_masks(regime_aligned, valid)
    if fam['extra_cells'] is not None:
        # A6/A7: the event/session windows carry the claim — add them as
        # conditioning cells (each intersected with the regime-validity mask).
        window_cells = fam['extra_cells'](scores.index)
        cells.update({name: mask & valid for name, mask in window_cells.items()})
    for cell in fam['decision_cells']:
        print(f"  {cell}: {int(cells[cell].sum()):,} bars")

    atr = ta.volatility.AverageTrueRange(high=ohlc['high'], low=ohlc['low'],
                                         close=ohlc['close'], window=ATR_WINDOW
                                         ).average_true_range()
    outcomes = {}
    for h in cfg['horizons']:
        outcomes[int(h)] = {
            'fwd_return': forward_return(ohlc['close'], h).to_numpy(dtype=float),
            'barrier': barrier_outcome_fast(ohlc['high'], ohlc['low'], ohlc['close'],
                                            h, atr, k=args.barrier_k).to_numpy(dtype=float),
        }
    print(f"  outcomes ready ({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(args.seed)
    rows = []
    for name in scores.columns:
        print(f"  {name}")
        rows.extend(audit_candidate(name, scores[name].to_numpy(dtype=float),
                                    scores.index, cells, outcomes,
                                    n_shuffles, min_events, rng,
                                    decision_cells=fam['decision_cells'],
                                    stability_fn=fam['stability_fn']))
    table = pd.DataFrame(rows)
    if table.empty:
        print('  layer produced no rows')
        return table, pd.DataFrame(), {}, None
    registered = getattr(args, 'registered_entries', None)
    if registered:
        if n_shuffles < DECISION_DRAWS_RECOMMENDED:
            print(f"  WARNING: decision run at {n_shuffles} draws — "
                  f">= {DECISION_DRAWS_RECOMMENDED} recommended (p_raw floor "
                  f"{1.0 / (n_shuffles + 1):.5f})")
        table, decision, registered_decision = apply_decision_registered(
            table, registered, min_days=fam['min_days'],
            stability_min=fam['stability_min'], min_periods=fam['min_periods'])
    else:
        registered_decision = None
        table, decision = apply_decision(table, min_days=fam['min_days'],
                                         stability_min=fam['stability_min'],
                                         min_periods=fam['min_periods'])

    meta = dict(timeframe=tf, catalog=args.catalog, start=start, end=end,
                horizons=list(cfg['horizons']),
                n_shuffles=int(n_shuffles), min_events=int(min_events),
                barrier_k=args.barrier_k, atr_window=ATR_WINDOW, seed=args.seed,
                n_bars=int(len(scores)), candidates=list(scores.columns),
                excluded=sorted(exclude),
                n_cells_scored=int((table['verdict'] == 'scored').sum()),
                n_cells_skipped=int((table['verdict'] == 'skipped').sum()),
                n_pass=int(decision['passes'].sum()) if len(decision) else 0,
                runtime_s=round(time.time() - t0, 1))
    if registered is not None:
        meta.update(
            registered_cells=registered,
            n_registered=len(registered),
            n_registered_pass=int(registered_decision['passes'].sum()),
            n_registered_drift_only=int(registered_decision['drift_only'].sum()),
            decision_draws_recommended=DECISION_DRAWS_RECOMMENDED,
            decision_draws_warning=bool(n_shuffles < DECISION_DRAWS_RECOMMENDED))
    return table, decision, meta, registered_decision


def print_registered_report(tf, registered_decision):
    print(f"\nREGISTERED DECISION CELLS — layer {tf} (own BH family, "
          f"exploratory rows cannot pass)")
    cols = ['candidate', 'cell', 'horizon', 'outcome', 'expected_sign',
            'ic', 'p_raw', 'p_bh', 'passes', 'drift_only', 'note']
    print(registered_decision[cols].to_string(index=False))


def print_layer_report(tf, table, decision):
    print(f"\n{'=' * 100}\nEVENT-DIRECTION AUDIT — layer {tf}")
    scored = table[table['verdict'] == 'scored']
    print(f"{len(scored)} scored cells, {int((table['verdict'] == 'skipped').sum())} skipped "
          f"(below the event floor)")
    dec = scored[scored['decision_cell'] & (scored['outcome'] == 'fwd_return')]
    if len(dec):
        top = dec.reindex(dec['p_bh'].sort_values().index).head(10)
        print('\ntop decision cells (forward return):')
        print(f"{'candidate':<36} {'h':>4} {'cell':<16} {'n':>6} {'IC':>7} "
              f"{'p_raw':>7} {'p_BH':>7} {'stab':>5} {'bench%':>6}")
        for _, r in top.iterrows():
            print(f"{r['candidate']:<36} {r['horizon']:>4} {r['cell']:<16} "
                  f"{r['n_events']:>6} {r['ic']:>7.3f} {r['p_raw']:>7.3f} "
                  f"{r['p_bh']:>7.3f} "
                  f"{(r['sign_stability'] if np.isfinite(r['sign_stability']) else 0):>5.2f} "
                  f"{(r['bench_pctile'] if np.isfinite(r['bench_pctile']) else -1):>6.1f}")
    n_pass = int(decision['passes'].sum()) if len(decision) else 0
    n_drift = int(decision['drift_only'].sum()) if len(decision) else 0
    print(f"\nVERDICT layer {tf}: {n_pass} of {len(decision)} candidates pass the "
          f"pre-registered rule ({n_drift} drift_only).")
    if n_pass:
        print(decision[decision['passes']][['candidate', 'passing_cells']].to_string(index=False))
    print('=' * 100)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalog', default='sequence', choices=sorted(CATALOGS),
                   help="candidate family: 'sequence' (A5), 'calendar' (A6 — "
                        "event-window cells, yearly stability, distinct-days floor), "
                        "'session' (A7), 'moments' (A8 — M15-computed realized "
                        "moments read at 4h cadence), 'volume' (A9 — tick-volume/"
                        "spread microstructure, 4h layer full roster + m15 subset) "
                        "or 'crosspair' (A10 — cross-pair lead-lag/triangle "
                        "residuals; search <= DEV_END, frozen-cell replication "
                        "from REPLICATION_START via --start/--end)")
    p.add_argument('--timeframe', default='all',
                   choices=['m15', '4hours', 'daily', 'mtf', 'all'])
    p.add_argument('--start', default='2005-01-01')
    p.add_argument('--end', default=None,
                   help='clamped below the sealed hold-out (timeframes.HOLDOUT_START)')
    p.add_argument('--n-shuffles', type=int, default=None,
                   help='override the per-layer default (500 m15/mtf, 1000 4h/daily)')
    p.add_argument('--min-events', type=int, default=None,
                   help='override the per-layer event floor (100; 50 daily)')
    p.add_argument('--barrier-k', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--exclude', default=None,
                   help='comma-separated candidate names (e.g. the redundancy-screen rejects)')
    p.add_argument('--decision-cells', default=None,
                   help='A11 registered protocol: JSON/CSV file or inline '
                        "'candidate:cell:horizon:outcome:sign[;...]' list of "
                        'pre-registered cells. They form their OWN BH family '
                        '(fwd entries bring their barrier guard row); every '
                        'other scored row is flagged exploratory and cannot '
                        f'pass. Recommended >= {DECISION_DRAWS_RECOMMENDED} draws.')
    p.add_argument('--output-prefix', default=None,
                   help='default: docs/results/<family output name>')
    args = p.parse_args()
    args.registered_entries = (parse_decision_cells(args.decision_cells)
                               if args.decision_cells else None)

    fam = CATALOGS[args.catalog]
    if args.output_prefix is None:
        args.output_prefix = os.path.join(project_root, 'docs', 'results',
                                          fam['output_name'])
    tfs = list(fam['tfs']) if args.timeframe == 'all' else [args.timeframe]
    date_tag = pd.Timestamp.now().strftime('%Y-%m-%d')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=project_root, text=True).strip()
    except Exception:
        commit = 'unknown'

    for tf in tfs:
        table, decision, meta, registered_decision = run_layer(tf, args)
        if table.empty:
            continue
        print_layer_report(tf, table, decision)
        if registered_decision is not None:
            print_registered_report(tf, registered_decision)
        base = f"{args.output_prefix}_{tf}_{date_tag}"
        table.to_csv(f"{base}.csv", index=False)
        decision.to_csv(f"{base}_decision.csv", index=False)
        if registered_decision is not None:
            registered_decision.to_csv(f"{base}_registered_decision.csv", index=False)
        floor_note = (f" & n_days>={fam['min_days']}" if fam['min_days'] else "")
        meta.update(run_date=date_tag, commit=commit,
                    decision_cells=list(fam['decision_cells']),
                    accept_rule=f"p_BH<{P_BH_MAX} & |IC|>={IC_MIN} & "
                                f"{fam['stability_desc']}"
                                f"{floor_note} in a decision cell "
                                f"on fwd_return, barrier not collapsed",
                    regime='daily generate_regime_labels, +24h shift, ffill; '
                           'warm-up excluded')
        with open(f"{base}.meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f"written: {base}.csv / _decision.csv / .meta.json")


if __name__ == '__main__':
    main()
