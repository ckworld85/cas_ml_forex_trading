"""
Redundancy screen — is a candidate feature new information, or a feature we already have?

WHY THIS RUNS FIRST, BEFORE THE AUDIT AND LONG BEFORE ANY MODEL
---------------------------------------------------------------
Measured on the externally-reported paper features (2026-08-29): four of seven candidates —
carrying 2,089 of the paper's reported importance — correlated **0.84–0.94** with a single
feature already active here (`bb_deviation`). Trees spread splits across interchangeable
measurements of one construct, so the importance table showed four "important" features where
there was one. Auditing or training on such a candidate measures nothing; the screen is what
established the project's protocol: **build → redundancy → audit → only then a model arm**.

WHAT IS COMPARED
----------------
Candidates against the **enabled** `daily_*` model features from `features.yaml`, both
computed on the daily bars over their common index. The daily set is the right comparison for
a daily candidate: an M15 oscillator cannot be its double, a daily momentum can. |Spearman| is
reported alongside |Pearson| and the verdict uses the larger of the two — a candidate that is
a monotone transform of an active feature is just as redundant as a linear copy.

THE THRESHOLD
-------------
0.80, from `analytics/feature_pruning.py`, where |corr| >= 0.80 is the measured level at which
the active set itself collapses into clusters (30 features -> 19 clusters). A candidate above
it would land inside an existing cluster and add a vote, not information.

Read-only: computes, prints, optionally writes a CSV. Nothing decides anything here — the
verdict column says which candidates are worth the audit's multiple-testing budget.

Usage
-----
    python -m ModelTrading.source.python.analytics.redundancy_screen \\
        --candidates-csv ModelTrading/data/cross_asset_daily.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config  # noqa: E402
import ModelTrading.source.python.utils.csv as csv_utils  # noqa: E402
from ModelTrading.source.python.features.config import get_feature_config  # noqa: E402
from ModelTrading.source.python.features.indicators import add_features  # noqa: E402

REDUNDANCY_THRESHOLD = 0.80  # feature_pruning.py's measured cluster boundary


def active_daily_features(start=None, end=None, verbose=True):
    """The enabled daily model features, computed exactly as training computes them.

    Uses `indicators.add_features` on the daily CSV — not a re-implementation — so the
    screen compares against the features as the model actually sees them, shift included.
    """
    daily = csv_utils.load_csv(os.path.join(dir_config.DATA_DIR, 'eurusd_daily.csv'),
                               start_date=start, end_date=end)
    feats = add_features(daily, timeframe='daily')
    config = get_feature_config()
    # add_features emits BARE column names; the daily_ prefix is only attached later,
    # during multi-timeframe alignment. Strip it for matching, restore it for display.
    prefixed = config.get_usedInModel_features(prefix='daily')
    bare = {n[len('daily_'):]: n for n in prefixed}
    names = [b for b in bare if b in feats.columns]
    if verbose:
        print(f"active daily model features: {len(names)} of {len(prefixed)} configured")
        absent = [bare[b] for b in bare if b not in feats.columns]
        if absent:
            print(f"  not computed on this frame (external/regime sources): {absent}")
    out = feats[names].copy()
    out.columns = [bare[b] for b in names]
    return out


def _best_corr(c, active, min_overlap):
    """Worst offender of one candidate series against every active column."""
    best = dict(n_overlap=0, max_abs_pearson=np.nan,
                max_abs_spearman=np.nan, vs_pearson='', vs_spearman='')
    for act in active.columns:
        pair = pd.concat([c, active[act].astype(float)], axis=1).dropna()
        if len(pair) < min_overlap:
            continue
        pear = abs(pair.corr(method='pearson').iloc[0, 1])
        spear = abs(pair.corr(method='spearman').iloc[0, 1])
        best['n_overlap'] = max(best['n_overlap'], len(pair))
        if not np.isfinite(best['max_abs_pearson']) or pear > best['max_abs_pearson']:
            best['max_abs_pearson'], best['vs_pearson'] = pear, act
        if not np.isfinite(best['max_abs_spearman']) or spear > best['max_abs_spearman']:
            best['max_abs_spearman'], best['vs_spearman'] = spear, act
    pair_vals = [v for v in (best['max_abs_pearson'], best['max_abs_spearman'])
                 if np.isfinite(v)]
    best['max_abs_corr'] = max(pair_vals) if pair_vals else np.nan
    return best


def screen(candidates, active, threshold=REDUNDANCY_THRESHOLD, min_overlap=500):
    """Max |corr| of each candidate column against every active column.

    Returns one row per candidate: the worst offender under both Pearson and Spearman,
    the overlap the estimate rests on, and the verdict.
    """
    rows = []
    for cand in candidates.columns:
        best = dict(candidate=cand)
        best.update(_best_corr(candidates[cand].astype(float), active, min_overlap))
        worst = best['max_abs_corr']
        if best['n_overlap'] < min_overlap:
            best['verdict'] = 'INSUFFICIENT OVERLAP'
        elif np.isfinite(worst) and worst >= threshold:
            best['verdict'] = 'redundant'
        else:
            best['verdict'] = 'orthogonal'
        rows.append(best)
    return pd.DataFrame(rows)


def screen_event_conditional(candidates, active, threshold=REDUNDANCY_THRESHOLD,
                             min_overlap=300):
    """The same worst-offender scan, restricted to each candidate's EVENT bars.

    Event-coded scores are mostly 0.0, so a full-sample correlation is diluted
    toward zero by construction — a candidate can hide a 0.9 correlation on
    its event bars behind a 0.2 full-sample reading. This pass subsets to the
    non-zero bars first; both passes are label-free, so neither spends
    multiple-testing budget. Verdict thresholds mirror screen().
    """
    rows = []
    for cand in candidates.columns:
        c = candidates[cand].astype(float)
        c = c[(c != 0) & c.notna()]
        best = dict(candidate=cand, n_events=len(c))
        best.update(_best_corr(c, active, min_overlap))
        worst = best['max_abs_corr']
        if best['n_overlap'] < min_overlap:
            best['verdict'] = 'INSUFFICIENT OVERLAP'
        elif np.isfinite(worst) and worst >= threshold:
            best['verdict'] = 'redundant'
        else:
            best['verdict'] = 'orthogonal'
        rows.append(best)
    return pd.DataFrame(rows)


def print_report(table, threshold=REDUNDANCY_THRESHOLD):
    print('\n' + '=' * 100)
    print(f'REDUNDANCY SCREEN — candidates vs. the active daily features '
          f'(threshold |corr| >= {threshold})')
    print('=' * 100)
    head = (f"{'candidate':<34} {'n':>6} {'|pearson|':>10} {'vs':>26} "
            f"{'|spearman|':>11} {'verdict':>12}")
    print(head)
    print('-' * len(head))
    for _, r in table.sort_values('max_abs_corr', ascending=False).iterrows():
        print(f"{r['candidate']:<34} {r['n_overlap']:>6} {r['max_abs_pearson']:>10.2f} "
              f"{r['vs_pearson']:>26} {r['max_abs_spearman']:>11.2f} {r['verdict']:>12}")
    n_red = int((table['verdict'] == 'redundant').sum())
    n_ok = int((table['verdict'] == 'orthogonal').sum())
    print(f"\n{n_ok} orthogonal, {n_red} redundant, "
          f"{len(table) - n_ok - n_red} with insufficient overlap.")
    print("Only the orthogonal ones are worth the audit's multiple-testing budget; a")
    print("redundant one would enter feature selection as an extra vote for a cluster")
    print("that already exists (the paper-features lesson: 4 of 7 at 0.84-0.94 vs one")
    print("feature).")
    print('=' * 100)


def build_combined_frame(verbose=True):
    """The M15-aligned combined feature frame, built EXACTLY as training builds
    it — `advanced_train.calculate_features` (per-TF add_features with
    apply_shift=False, 4h index +4h / daily +24h to bar-close time, ffill onto
    M15, cross-TF features strict). Committed here so the previously ad-hoc
    M15 screen variant (trend-slope / calendar screens, reproduction record
    only in their meta.json) has a repository CLI.
    """
    from ModelTrading.source.python import advanced_train  # heavy import, lazy
    df_m15 = csv_utils.load_csv(os.path.join(dir_config.DATA_DIR, 'eurusd_m15.csv'),
                                keep_mid=True)
    df_4h = csv_utils.load_csv(os.path.join(dir_config.DATA_DIR, 'eurusd_4hours.csv'),
                               keep_mid=True)
    df_daily = csv_utils.load_csv(os.path.join(dir_config.DATA_DIR, 'eurusd_daily.csv'),
                                  keep_mid=True)
    if verbose:
        print(f"bars: m15 {len(df_m15):,}, 4h {len(df_4h):,}, daily {len(df_daily):,}")
    combined, _ = advanced_train.calculate_features(df_m15, df_4h, df_daily)
    return combined


def run_combined_frame_screen(args):
    """--combined-frame mode: screen a catalog family (A5 event sequences or
    the A6 calendar-direction roster) on the training-aligned combined frame,
    full-sample AND event-conditional."""
    import json
    import subprocess
    import ModelTrading.source.python.features.config as fconfig
    which = getattr(args, 'catalog', 'sequence')
    if which == 'calendar':
        from ModelTrading.source.python.analytics import calendar_direction_catalog as cat
    elif which == 'session':
        from ModelTrading.source.python.analytics import session_anchor_catalog as cat
    elif which == 'moments':
        from ModelTrading.source.python.analytics import realized_moments_catalog as cat
    elif which == 'volume':
        from ModelTrading.source.python.analytics import volume_spread_catalog as cat
    elif which == 'crosspair':
        from ModelTrading.source.python.analytics import cross_pair_catalog as cat
    elif which == 'timing':
        from ModelTrading.source.python.analytics import timing_execution_catalog as cat
    else:
        from ModelTrading.source.python.analytics import event_sequence_catalog as cat

    originals = cat.force_enable(fconfig)
    try:
        combined = build_combined_frame()
    finally:
        cat.restore(fconfig, originals)

    active_names = [n for n in originals[0]() if n in combined.columns]
    cand_names = [n for n in cat.all_candidates() if n in combined.columns]
    missing = [n for n in cat.all_candidates() if n not in combined.columns]
    if missing:
        print(f"WARNING: candidates not present in the combined frame: {missing}")

    if args.start:
        combined = combined[combined.index >= pd.to_datetime(args.start)]
    if args.end:
        combined = combined[combined.index <= pd.to_datetime(args.end)]
    strided = combined.iloc[::args.stride]
    print(f"combined frame: {len(combined):,} M15 bars "
          f"({combined.index.min()} .. {combined.index.max()}), "
          f"stride {args.stride} -> {len(strided):,} rows; "
          f"{len(active_names)} active features, {len(cand_names)} candidates")

    candidates = strided[cand_names]
    active = strided[active_names]

    table = screen(candidates, active, threshold=args.threshold,
                   min_overlap=args.min_overlap)
    print_report(table, args.threshold)

    evt = screen_event_conditional(candidates, active, threshold=args.threshold,
                                  min_overlap=args.event_min_overlap)
    print('\nEVENT-CONDITIONAL PASS (non-zero bars only — a sparse event score can '
          'hide redundancy behind its zeros):')
    print_report(evt, args.threshold)

    merged = table.merge(
        evt.rename(columns={c: f'evt_{c}' for c in evt.columns if c != 'candidate'}),
        on='candidate')
    both_scored = (merged['verdict'] != 'INSUFFICIENT OVERLAP') | \
                  (merged['evt_verdict'] != 'INSUFFICIENT OVERLAP')
    merged['final_verdict'] = np.where(
        (merged['verdict'] == 'redundant') | (merged['evt_verdict'] == 'redundant'),
        'redundant',
        np.where(both_scored, 'orthogonal', 'INSUFFICIENT OVERLAP'))
    n_red = int((merged['final_verdict'] == 'redundant').sum())
    print(f"\nFINAL (either pass >= {args.threshold} -> redundant): "
          f"{n_red} redundant, "
          f"{int((merged['final_verdict'] == 'orthogonal').sum())} orthogonal, "
          f"{int((merged['final_verdict'] == 'INSUFFICIENT OVERLAP').sum())} insufficient.")

    if args.output_csv:
        merged.to_csv(args.output_csv, index=False)
        print(f"\nwritten to {args.output_csv}")
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                             cwd=project_root, text=True).strip()
        except Exception:
            commit = 'unknown'
        meta = {
            'run_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'commit': commit,
            'mode': 'combined-frame',
            'catalog': getattr(args, 'catalog', 'sequence'),
            'window': [str(args.start), str(args.end)],
            'stride_m15_bars': args.stride,
            'rows_screened': int(len(strided)),
            'threshold': args.threshold,
            'min_overlap': args.min_overlap,
            'event_min_overlap': args.event_min_overlap,
            'alignment': 'advanced_train.calculate_features: 4h index +4h, daily +24h, ffill onto M15',
            'n_active_features': len(active_names),
            'active_features': active_names,
            'candidates': cand_names,
            'candidates_missing_from_frame': missing,
        }
        meta_path = os.path.splitext(args.output_csv)[0] + '.meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f"meta written to {meta_path}")
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--candidates-csv', default=None,
                   help='CSV with a date column and one column per candidate feature')
    p.add_argument('--combined-frame', action='store_true',
                   help='Screen a catalog family on the training-aligned M15 '
                        'combined frame instead of a daily candidates CSV')
    p.add_argument('--catalog', default='sequence',
                   choices=['sequence', 'calendar', 'session', 'moments', 'volume',
                            'crosspair', 'timing'],
                   help="combined-frame mode: 'sequence' (A5), 'calendar' (A6), "
                        "'session' (A7), 'moments' (A8), 'volume' (A9), "
                        "'crosspair' (A10) or 'timing' (B6)")
    p.add_argument('--stride', type=int, default=8,
                   help='combined-frame mode: keep every Nth M15 bar (default 8)')
    p.add_argument('--event-min-overlap', type=int, default=300,
                   help='combined-frame mode: min event bars for the event-conditional pass')
    p.add_argument('--date-column', default='date')
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--threshold', type=float, default=REDUNDANCY_THRESHOLD)
    p.add_argument('--min-overlap', type=int, default=500)
    p.add_argument('--output-csv', default=None)
    args = p.parse_args()

    if args.combined_frame:
        run_combined_frame_screen(args)
        return
    if not args.candidates_csv:
        p.error('either --candidates-csv or --combined-frame is required')

    cand = pd.read_csv(args.candidates_csv)
    cand[args.date_column] = pd.to_datetime(cand[args.date_column], format='ISO8601')
    cand = cand.set_index(args.date_column).sort_index()
    if args.start:
        cand = cand[cand.index >= pd.to_datetime(args.start)]
    if args.end:
        cand = cand[cand.index <= pd.to_datetime(args.end)]
    print(f"candidates: {len(cand.columns)} columns, {len(cand):,} rows "
          f"({cand.index.min().date()} .. {cand.index.max().date()})")

    active = active_daily_features(start=args.start, end=args.end)
    table = screen(cand, active, threshold=args.threshold, min_overlap=args.min_overlap)
    print_report(table, args.threshold)
    if args.output_csv:
        table.to_csv(args.output_csv, index=False)
        print(f"\nwritten to {args.output_csv}")


if __name__ == '__main__':
    main()
