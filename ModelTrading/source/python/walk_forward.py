"""
Simple walk-forward runner around advanced_train.py.

Unlike iterative_training.py (config grids, scenarios, backtest sweeps, parallel
execution) this script does exactly one thing: derive N train/backtest windows
walking BACKWARDS from a fixed anchor date and run advanced_train.py once per fold,
forwarding every other CLI argument 1:1.

Window derivation (all windows in calendar months):
    fold 1:  backtest_end   = --backtest-window-end
             backtest_start = backtest_end - backtest_window
             train_end      = backtest_start - 1 day
             train_start    = backtest_start - train_window      (--train-window)
                              or the fixed --train-start date    (anchored: every
                              fold trains from the same start, so the window
                              EXPANDS as folds get newer)
             fast_train_start = backtest_start - fast_train_window   (only with
             --fast-train-window; forwarded as --fast-train-start so the M15 fast
             scope gets its own window length per fold) — or the fixed
             --fast-train-start date, shared by every fold.
    fold k+1: backtest_end  = backtest_end(fold k) - walk_forward_step
             (all other dates re-derived from that anchor)

Usage:
    python walk_forward.py --train-window 18m --backtest-window 6m \
        --walk-forward-step 3m --backtest-window-end 2026-04-19 --folds 4 \
        --run-id wf_trendonly --label-mode trend_only --pip-target 150 --stop-pips 35 ...

    # anchored (expanding) instead of rolling: fixed slow + fast starts
    python walk_forward.py --train-start 2015-01-01 --fast-train-start 2017-01-01 \
        --backtest-window 6m --backtest-window-end 2026-04-19 --folds 4 ...

Everything after the walk-forward flags above (label mode, hyperparameters, --seed,
--run-backtest, ...) is passed through to advanced_train.py unchanged. The window
flags --train-end/--backtest-start/--backtest-end are computed here and therefore
rejected as inputs; --train-start and --fast-train-start are walk_forward flags
(fixed-anchor mode) and never forwarded verbatim.

By default each fold's advanced_train.py output is captured to
generated/{run_id}/foldNN/train.log (--verbose streams it to the console instead).

Each fold trains into generated/{run_id}/foldNN/ and a walk_forward_summary.json
with the fold plan, the per-fold gate metrics (cv_final_metrics) and a cross-fold
aggregate is written to generated/{run_id}/. The aggregate pools the trades of all
folds (when --run-backtest is forwarded) into a mean P&L per trade with a
month-clustered 95% interval, and reports each model's AUC/F1/MCC as mean +/- std
across the folds.

--op-select adds per-fold operating-point selection (see operating_point.py):
after each fold trains, a grid of cheap backtests (slow threshold x fast-gate
option, plus the trained-threshold status quo) runs on the fold's own training
window, the best arm is frozen and evaluated ONCE on the test window — that
frozen run is the fold's official result the pooled aggregate reads. The full
grid additionally runs on the test window as a recorded diagnostic (per-arm
OOS curves, look-ahead oracle + regret). Measurement settings for all these
backtests go through --op-backtest-args; --run-backtest must not be forwarded
alongside.
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config
from ModelTrading.source.python import operating_point
from ModelTrading.source.python.utils.stats import (
    clustered_se, MIN_CLUSTERS_FOR_SIGNIFICANCE,
)

# Window flags this script computes itself — passing them through would silently
# fight the fold derivation, so they are rejected up front. --train-start and
# --fast-train-start are NOT listed: they are walk_forward's own fixed-anchor
# flags now, consumed by parse_args and applied per fold.
FORBIDDEN_PASSTHROUGH = (
    '--train-end', '--backtest-start', '--backtest-end',
)


@dataclass
class FoldWindow:
    """One walk-forward fold. All dates are YYYY-MM-DD strings."""
    fold: int
    train_start: str
    train_end: str
    backtest_start: str
    backtest_end: str
    # Only set with --fast-train-window: per-fold start of the M15 fast scope.
    fast_train_start: str = None


def parse_months(value):
    """
    Parse a window length given in months: '6', '6m', '18M' -> int months.

    Raises argparse.ArgumentTypeError on anything else (so argparse reports it
    as a clean CLI error instead of a traceback).
    """
    match = re.fullmatch(r'(\d+)\s*[mM]?', str(value).strip())
    if not match or int(match.group(1)) <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid month window '{value}' (expected e.g. '6', '6m', '18m')")
    return int(match.group(1))


def parse_date(value):
    """Parse a fixed anchor date 'YYYY-MM-DD' -> normalized string, or argparse error."""
    text = str(value).strip()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        raise argparse.ArgumentTypeError(
            f"invalid date '{value}' (expected YYYY-MM-DD)")
    try:
        pd.Timestamp(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid date '{value}' (expected YYYY-MM-DD)")
    return text


def compute_folds(backtest_window_end, backtest_window_months, train_window_months,
                  step_months, n_folds, fast_train_window_months=None,
                  train_start_fixed=None, fast_train_start_fixed=None):
    """
    Derive the walk-forward fold windows, newest fold first.

    Fold 1 is anchored at backtest_window_end; every further fold moves the anchor
    back by step_months. train_end is always one day before backtest_start, so
    there is never an overlap between training features and the backtest window
    (the label-horizon embargo inside the training window stays advanced_train's
    --cv-gap concern, unchanged).

    Args:
        backtest_window_end (str): Anchor date (YYYY-MM-DD) = backtest end of fold 1.
        backtest_window_months (int): Length of each backtest window in months.
        train_window_months (int or None): Length of each training window in months
            (rolling window). None requires train_start_fixed instead.
        step_months (int): Shift between consecutive folds in months.
        n_folds (int): Number of folds to generate.
        fast_train_window_months (int, optional): Length of the M15 fast-scope
            training window in months; anchored at backtest_start exactly like
            train_window. None leaves fast_train_start unset (fast scope inherits
            the slow window inside advanced_train).
        train_start_fixed (str, optional): Fixed train start (YYYY-MM-DD) shared by
            every fold — the training window expands as folds get newer. Must be
            strictly before every fold's train_end.
        fast_train_start_fixed (str, optional): Fixed M15 fast-scope start
            (YYYY-MM-DD) shared by every fold; mutually exclusive with
            fast_train_window_months.

    Returns:
        list[FoldWindow]

    Raises:
        ValueError: fixed start dates that are not strictly before a fold's
            train_end (the fold would have an empty/negative training window).
    """
    if (train_window_months is None) == (train_start_fixed is None):
        raise ValueError('exactly one of train_window_months / train_start_fixed '
                         'is required')
    if fast_train_window_months is not None and fast_train_start_fixed is not None:
        raise ValueError('fast_train_window_months and fast_train_start_fixed are '
                         'mutually exclusive')
    anchor = pd.Timestamp(backtest_window_end)
    folds = []
    for i in range(1, n_folds + 1):
        bt_end = anchor - pd.DateOffset(months=(i - 1) * step_months)
        bt_start = bt_end - pd.DateOffset(months=backtest_window_months)
        train_end = bt_start - pd.Timedelta(days=1)
        if train_window_months is not None:
            train_start = bt_start - pd.DateOffset(months=train_window_months)
        else:
            train_start = pd.Timestamp(train_start_fixed)
            if train_start >= train_end:
                raise ValueError(
                    f"fixed --train-start {train_start_fixed} is not before fold "
                    f"{i}'s train_end {train_end.strftime('%Y-%m-%d')} — move the "
                    "start earlier or reduce --folds/--walk-forward-step")
        fast_train_start = None
        if fast_train_window_months is not None:
            fast_train_start = (bt_start - pd.DateOffset(
                months=fast_train_window_months)).strftime('%Y-%m-%d')
        elif fast_train_start_fixed is not None:
            if pd.Timestamp(fast_train_start_fixed) >= train_end:
                raise ValueError(
                    f"fixed --fast-train-start {fast_train_start_fixed} is not "
                    f"before fold {i}'s train_end "
                    f"{train_end.strftime('%Y-%m-%d')} — move the start earlier "
                    "or reduce --folds/--walk-forward-step")
            fast_train_start = fast_train_start_fixed
        folds.append(FoldWindow(
            fold=i,
            train_start=train_start.strftime('%Y-%m-%d'),
            train_end=train_end.strftime('%Y-%m-%d'),
            backtest_start=bt_start.strftime('%Y-%m-%d'),
            backtest_end=bt_end.strftime('%Y-%m-%d'),
            fast_train_start=fast_train_start,
        ))
    return folds


def validate_passthrough(passthrough):
    """Reject window flags that this script computes itself."""
    offending = [flag for flag in FORBIDDEN_PASSTHROUGH
                 if flag in passthrough
                 or any(arg.startswith(flag + '=') for arg in passthrough)]
    if offending:
        raise SystemExit(
            f"walk_forward.py computes {', '.join(offending)} from the window flags — "
            "remove them and use --train-window/--train-start / "
            "--fast-train-window/--fast-train-start / --backtest-window / "
            "--backtest-window-end / --walk-forward-step / --folds instead.")


def fold_run_id(base_run_id, fold):
    """Run id of one fold: nested under the base run directory."""
    return f"{base_run_id}/fold{fold:02d}"


def build_fold_command(fold_window, base_run_id, passthrough,
                       train_script=None, python=None):
    """
    Build the advanced_train.py argv for one fold.

    Pure function (no subprocess, no filesystem) so the forwarding is unit-testable.
    """
    if train_script is None:
        train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'advanced_train.py')
    args = [
        python or sys.executable, train_script,
        '--run-id', fold_run_id(base_run_id, fold_window.fold),
        '--train-start', fold_window.train_start,
        '--train-end', fold_window.train_end,
        '--backtest-start', fold_window.backtest_start,
        '--backtest-end', fold_window.backtest_end,
    ]
    if fold_window.fast_train_start:
        args.extend(['--fast-train-start', fold_window.fast_train_start])
    args.extend(passthrough)
    return args


def extract_fold_metrics(run_id):
    """
    Read the gate metrics of a finished fold from its run directory.

    Returns a dict with cv_final_metrics (per model: auc/f1/mcc/threshold/rounds)
    and, if a backtest ran, the headline backtest numbers — or None when the
    training summary is missing (fold failed before writing it).
    """
    run_dir = os.path.join(dir_config.GENERATED_DIR, run_id)
    summary_path = os.path.join(run_dir, 'training_summary.json')
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    metrics = {'cv_final_metrics': summary.get('cv_final_metrics')}

    backtest_path = os.path.join(run_dir, 'report', 'backtest_summary.json')
    if os.path.exists(backtest_path):
        with open(backtest_path) as f:
            bt = json.load(f)
        metrics['backtest'] = {
            'total_trades': bt.get('total_trades'),
            'win_rate_pct': bt.get('win_rate_pct'),
            'total_pnl_eur': bt.get('total_pnl_eur'),
            'total_pnl_pips': bt.get('total_pnl_pips'),
            'max_drawdown_pct': bt.get('max_drawdown_pct'),
        }
    return metrics


def load_fold_trades(run_id):
    """
    Read the trades of one finished fold (report/trade_list.csv) as a list of dicts.

    Only exists when --run-backtest was forwarded; returns [] otherwise. Keeps just
    the columns the pooled interval needs (pnl, pnl_pips, open_time).
    """
    path = os.path.join(dir_config.GENERATED_DIR, run_id, 'report', 'trade_list.csv')
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return []
        cols = [c for c in ('pnl', 'pnl_pips', 'open_time') if c in df.columns]
        return df[cols].to_dict('records')
    except Exception as e:
        print(f"  WARNING: could not read trades for {run_id}: {e}")
        return []


def aggregate_folds(fold_results):
    """
    Pool the trades of every fold and summarize the gate metrics across folds.

    Pooling trades (rather than averaging fold totals) keeps the per-trade spread
    the confidence interval is built from; the interval is clustered on calendar
    months because with a walk-forward step smaller than the backtest window the
    same market days appear in several folds. Gate metrics (AUC/F1/MCC per model,
    from cv_final_metrics) are summarized as mean +/- std across the folds instead
    — they cannot be pooled, every fold measures its own CV.

    Args:
        fold_results (list[dict]): main()'s per-fold records; each may carry
            'trades' (list of dicts with pnl/pnl_pips/open_time) and 'metrics'.

    Returns:
        dict with a 'trades' block (None when no fold produced trades), a
        'cv_final_across_folds' block {model: {metric: mean/std/min/max/n_folds}}
        and 'backtest_period' — the overall start/end covered by the fold
        backtest windows (None when no fold carries a window).
    """
    starts = [r['window']['backtest_start'] for r in fold_results
              if (r.get('window') or {}).get('backtest_start')]
    ends = [r['window']['backtest_end'] for r in fold_results
            if (r.get('window') or {}).get('backtest_end')]
    backtest_period = ({'start': min(starts), 'end': max(ends)}
                       if starts and ends else None)

    pnl, pips, months, fold_pnls = [], [], [], []
    for res in fold_results:
        trades = res.get('trades') or []
        fold_pnls.append(sum(float(t.get('pnl', 0.0)) for t in trades))
        for t in trades:
            pnl.append(float(t.get('pnl', 0.0)))
            pips.append(float(t.get('pnl_pips', 0.0)))
            months.append(str(t.get('open_time', ''))[:7])   # 'YYYY-MM'

    trades_block = None
    n = len(pnl)
    if n > 0:
        mean = sum(pnl) / n
        se, n_clusters = clustered_se(pnl, months if any(months) else None)
        naive_se = (statistics.stdev(pnl) / math.sqrt(n)) if n > 1 else float('nan')
        ci = 1.96 * se if n > 1 and se == se else float('nan')
        trades_block = {
            'n_trades': n,
            'n_month_clusters': n_clusters,
            'total_pnl_eur': sum(pnl),
            'total_pnl_pips': sum(pips),
            'mean_pnl_per_trade_eur': mean,
            'se_clustered': se,
            'se_naive': naive_se,
            'ci95_low': mean - ci if ci == ci else float('nan'),
            'ci95_high': mean + ci if ci == ci else float('nan'),
            't_stat': mean / se if n > 1 and se == se and se > 0 else 0.0,
            # Below about a year of distinct months the clustered interval itself is
            # too unreliable to license any claim, however it comes out.
            'clusters_reliable': n_clusters >= MIN_CLUSTERS_FOR_SIGNIFICANCE,
            'significant': bool(n > 1 and ci == ci and (mean - ci) > 0
                                and n_clusters >= MIN_CLUSTERS_FOR_SIGNIFICANCE),
            'win_rate': sum(1 for p in pnl if p > 0) / n,
            'folds_profitable': sum(1 for p in fold_pnls if p > 0),
            'fold_pnls_eur': fold_pnls,
        }

    # Gate metrics across folds, per model.
    models = []
    for res in fold_results:
        cv = (res.get('metrics') or {}).get('cv_final_metrics') or {}
        for mk in cv:
            if mk not in models:
                models.append(mk)
    metrics_block = {}
    for mk in models:
        metrics_block[mk] = {}
        for metric in ('auc', 'f1', 'mcc', 'precision', 'recall'):
            vals = []
            for res in fold_results:
                cv = (res.get('metrics') or {}).get('cv_final_metrics') or {}
                val = (cv.get(mk) or {}).get(metric)
                if isinstance(val, (int, float)) and val == val:
                    vals.append(float(val))
            metrics_block[mk][metric] = {
                'mean': statistics.mean(vals) if vals else None,
                'std': statistics.stdev(vals) if len(vals) > 1 else None,
                'min': min(vals) if vals else None,
                'max': max(vals) if vals else None,
                'n_folds': len(vals),
            }

    return {
        'n_folds': len(fold_results),
        'n_folds_with_metrics': sum(1 for r in fold_results if r.get('metrics')),
        'backtest_period': backtest_period,
        'trades': trades_block,
        'cv_final_across_folds': metrics_block,
    }


def print_aggregate(aggregate):
    """Console rendering of aggregate_folds(): pooled trades + metric spread."""
    print("\n" + "=" * 100)
    print("AGGREGATE ACROSS FOLDS")
    print("=" * 100)

    period = aggregate.get('backtest_period')
    if period:
        print(f"  Backtest period:    {period['start']} .. {period['end']} "
              f"(all fold backtest windows combined)")

    tb = aggregate.get('trades')
    if tb:
        print(f"  Pooled trades:      {tb['n_trades']} over "
              f"{tb['n_month_clusters']} calendar months "
              f"({tb['folds_profitable']}/{aggregate['n_folds']} folds profitable)")
        print(f"  Total P&L:          EUR {tb['total_pnl_eur']:,.2f} "
              f"({tb['total_pnl_pips']:.1f} pips)")
        se_txt = (f"SE {tb['se_clustered']:.2f} clustered / {tb['se_naive']:.2f} naive"
                  if tb['se_naive'] == tb['se_naive'] else '')
        print(f"  Mean P&L/trade:     EUR {tb['mean_pnl_per_trade_eur']:+,.2f}   "
              f"95% CI [{tb['ci95_low']:+,.2f}, {tb['ci95_high']:+,.2f}] "
              f"(month-clustered)   t={tb['t_stat']:.2f}   {se_txt}")
        print(f"  Win rate:           {tb['win_rate'] * 100:.1f}%")
        if not tb['clusters_reliable']:
            print(f"  WARNING: only {tb['n_month_clusters']} month clusters "
                  f"(< {MIN_CLUSTERS_FOR_SIGNIFICANCE}) — the interval is too "
                  "unreliable to license any claim.")
        elif tb['significant']:
            print("  The clustered 95% interval excludes zero — positive per-trade "
                  "expectancy over this walk-forward.")
        else:
            print("  The clustered 95% interval contains zero — no demonstrated "
                  "per-trade expectancy.")
    else:
        print("  No trades pooled (forward --run-backtest to get per-fold trade lists).")

    metrics = aggregate.get('cv_final_across_folds') or {}
    if metrics:
        print(f"\n  CV gates across folds (mean +/- std over "
              f"{aggregate['n_folds_with_metrics']} folds, at resolved rounds):")
        for mk, block in metrics.items():
            def _fmt(metric):
                stats_ = block.get(metric) or {}
                if stats_.get('mean') is None:
                    return f"{metric} -"
                if stats_.get('std') is None:
                    return f"{metric} {stats_['mean']:.4f}"
                return f"{metric} {stats_['mean']:.4f}+/-{stats_['std']:.4f}"
            print(f"    {mk:<12} {_fmt('auc')}   {_fmt('f1')}   {_fmt('mcc')}")
    print("=" * 100)


def print_fold_plan(folds, base_run_id):
    with_fast = any(fw.fast_train_start for fw in folds)
    print(f"\nWalk-forward plan — run id '{base_run_id}', {len(folds)} folds "
          "(fold 1 = newest, walking backwards):")
    fast_header = f"  {'fast_train_start':>16}" if with_fast else ''
    print(f"  {'fold':>4}  {'train_start':>11}  {'train_end':>11}  "
          f"{'backtest_start':>14}  {'backtest_end':>12}{fast_header}")
    for fw in folds:
        fast_cell = f"  {fw.fast_train_start or '-':>16}" if with_fast else ''
        print(f"  {fw.fold:>4}  {fw.train_start:>11}  {fw.train_end:>11}  "
              f"{fw.backtest_start:>14}  {fw.backtest_end:>12}{fast_cell}")
    print()


def print_summary(fold_results):
    """Per-fold one-liner for the slow pair plus backtest, after all folds ran."""
    print("\n" + "=" * 116)
    print("WALK-FORWARD SUMMARY (cv_final_metrics = gates at the resolved boost rounds)")
    print("=" * 116)
    header = (f"  {'fold':>4}  {'backtest_period':<24}  {'status':<8}  "
              f"{'ls_auc':>7}  {'ls_f1':>7}  {'ls_mcc':>7}  "
              f"{'ss_auc':>7}  {'ss_f1':>7}  {'ss_mcc':>7}  {'trades':>6}  {'pnl_eur':>10}")
    print(header)
    for res in fold_results:
        window = res.get('window') or {}
        period = (f"{window['backtest_start']}..{window['backtest_end']}"
                  if window.get('backtest_start') and window.get('backtest_end')
                  else '-')
        metrics = res.get('metrics')
        if not metrics or not metrics.get('cv_final_metrics'):
            print(f"  {res['fold']:>4}  {period:<24}  {res['status']:<8}  "
                  f"(no training_summary.json)")
            continue

        def _cell(model, key):
            val = (metrics['cv_final_metrics'].get(model) or {}).get(key)
            return f"{val:.4f}" if isinstance(val, (int, float)) else '-'

        bt = metrics.get('backtest') or {}
        trades = bt.get('total_trades')
        pnl = bt.get('total_pnl_eur')
        print(f"  {res['fold']:>4}  {period:<24}  {res['status']:<8}  "
              f"{_cell('long_slow', 'auc'):>7}  {_cell('long_slow', 'f1'):>7}  "
              f"{_cell('long_slow', 'mcc'):>7}  "
              f"{_cell('short_slow', 'auc'):>7}  {_cell('short_slow', 'f1'):>7}  "
              f"{_cell('short_slow', 'mcc'):>7}  "
              f"{trades if trades is not None else '-':>6}  "
              f"{f'{pnl:,.0f}' if pnl is not None else '-':>10}")
    print("=" * 116)
    print("Reminder: single-fold numbers carry huge variance — judge the folds together, "
          "never the best one alone.\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Sequential walk-forward wrapper around advanced_train.py. '
                    'All unrecognised arguments are forwarded to advanced_train.py 1:1.',
        allow_abbrev=False,
    )
    parser.add_argument('--train-window', type=parse_months, default=None,
                        help="Training window length in months (e.g. '18m'): "
                             'train-start = backtest-start - train-window (rolling '
                             'window). Exactly one of --train-window / '
                             '--train-start is required.')
    parser.add_argument('--train-start', type=parse_date, default=None,
                        help='Fixed training start date (YYYY-MM-DD) shared by '
                             'every fold — the training window expands as folds '
                             'get newer (anchored walk-forward). Exactly one of '
                             '--train-window / --train-start is required.')
    parser.add_argument('--backtest-window', type=parse_months, required=True,
                        help="Backtest window length in months (e.g. '6m'): "
                             'backtest-start = backtest-end - backtest-window.')
    parser.add_argument('--fast-train-window', type=parse_months, default=None,
                        help="Training window length of the M15 fast scope in months "
                             "(e.g. '9m'): fast-train-start = backtest-start - "
                             'fast-train-window, forwarded per fold as '
                             '--fast-train-start. Default: unset, the fast scope '
                             'trains on the full slow window. Mutually exclusive '
                             'with --fast-train-start.')
    parser.add_argument('--fast-train-start', type=parse_date, default=None,
                        help='Fixed start date (YYYY-MM-DD) of the M15 fast scope, '
                             'shared by every fold and forwarded per fold as '
                             '--fast-train-start. Mutually exclusive with '
                             '--fast-train-window.')
    parser.add_argument('--backtest-window-end', type=str, required=True,
                        help='Backtest end date of fold 1 (YYYY-MM-DD); later folds '
                             'walk backwards from here.')
    parser.add_argument('--walk-forward-step', type=parse_months, default=None,
                        help='Shift between consecutive folds in months '
                             '(default: --backtest-window, i.e. disjoint backtests).')
    parser.add_argument('--folds', type=int, required=True,
                        help='Number of folds to run.')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Base run id; fold N trains into generated/{run-id}/foldNN. '
                             'Default: wf_<timestamp>.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the fold plan and the advanced_train.py commands '
                             'without running anything.')
    parser.add_argument('--quiet', dest='quiet', action='store_true', default=True,
                        help="Suppress advanced_train.py's console output: each fold "
                             'writes to generated/{run-id}/foldNN/train.log instead, '
                             'only the fold plan and the summaries stay on the console '
                             '(default: on).')
    parser.add_argument('--verbose', dest='quiet', action='store_false',
                        help="Stream advanced_train.py's output live to the console "
                             'instead of the per-fold train.log.')

    # --- operating-point selection (grid of cheap backtests per fold) ---
    parser.add_argument('--op-select', action='store_true',
                        help='After each fold trains, backtest a grid of operating '
                             'points (slow threshold x fast-gate option, plus the '
                             'trained-threshold status quo) on the fold\'s own '
                             'TRAINING window, freeze the best arm and evaluate it '
                             'once on the test window. The full grid also runs on '
                             'the test window as a recorded diagnostic sweep '
                             '(per-arm OOS curves, look-ahead oracle, regret) — '
                             'the official fold result stays the in-sample-selected '
                             'arm. Incompatible with forwarding --run-backtest.')
    parser.add_argument('--op-select-months', type=parse_months, default=None,
                        help='Trailing months of the training window the grid is '
                             'selected on. Default: the FULL training window, so a '
                             'short single-regime stretch cannot starve arms whose '
                             'regime it lacks.')
    parser.add_argument('--op-min-trades', type=int, default=6,
                        help='Eligibility floor: arms with fewer trades on the '
                             'selection window cannot win (fallback: most trades). '
                             'Default 6.')
    parser.add_argument('--op-grid-slow', type=str,
                        default=operating_point.DEFAULT_SLOW_GRID,
                        help='Comma list of slow entry thresholds for the grid '
                             f'(default {operating_point.DEFAULT_SLOW_GRID}).')
    parser.add_argument('--op-grid-fast', type=str,
                        default=operating_point.DEFAULT_FAST_GRID,
                        help="Comma list of fast-gate options: 'off' or a fast "
                             f'threshold (default {operating_point.DEFAULT_FAST_GRID}).')
    parser.add_argument('--op-no-trained', action='store_true',
                        help='Drop the --use-trained-threshold status-quo arm from '
                             'the grid (kept by default so the old rule competes).')
    parser.add_argument('--op-selection-scores', type=str, default='oof',
                        choices=['oof', 'model'],
                        help="Probabilities the SELECTION backtests rank arms on. "
                             "'oof' (default): the run's calibrated out-of-fold "
                             'scores via --proba-file — in-sample model scores '
                             'are memorized and degenerate the selection to the '
                             "lowest threshold. 'model': raw in-sample behaviour "
                             '(comparison only). Test sweep and the official '
                             'frozen run always use the real model scores.')
    parser.add_argument('--op-backtest-args', type=str, default='',
                        help='Extra backtest.py arguments applied to EVERY grid and '
                             'final backtest, e.g. "--cost-model data '
                             '--slippage-pips 0.2 --risk-model fixed_fractional". '
                             'Window/threshold/gate flags are rejected — the '
                             'selection machinery owns them.')

    args, passthrough = parser.parse_known_args(argv)

    if (args.train_window is None) == (args.train_start is None):
        parser.error('exactly one of --train-window / --train-start is required')
    if args.fast_train_window is not None and args.fast_train_start is not None:
        parser.error('--fast-train-window and --fast-train-start are mutually '
                     'exclusive')
    if args.folds <= 0:
        parser.error('--folds must be >= 1')
    if args.walk_forward_step is None:
        args.walk_forward_step = args.backtest_window
    if args.run_id is None:
        args.run_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    validate_passthrough(passthrough)
    if args.op_select and '--run-backtest' in passthrough:
        parser.error('--op-select runs its own backtests (selection grid + frozen '
                     'final) — remove --run-backtest from the forwarded arguments.')
    return args, passthrough


def main(argv=None):
    args, passthrough = parse_args(argv)

    try:
        folds = compute_folds(
            backtest_window_end=args.backtest_window_end,
            backtest_window_months=args.backtest_window,
            train_window_months=args.train_window,
            step_months=args.walk_forward_step,
            n_folds=args.folds,
            fast_train_window_months=args.fast_train_window,
            train_start_fixed=args.train_start,
            fast_train_start_fixed=args.fast_train_start,
        )
    except ValueError as e:
        raise SystemExit(f"walk-forward fold derivation: {e}")
    print_fold_plan(folds, args.run_id)
    if passthrough:
        print(f"Forwarded to advanced_train.py: {' '.join(passthrough)}\n")

    op_grid, op_bt_args = None, None
    if args.op_select:
        try:
            op_grid = operating_point.build_grid(
                operating_point.parse_float_list(args.op_grid_slow),
                operating_point.parse_fast_options(args.op_grid_fast),
                include_trained=not args.op_no_trained)
            op_bt_args = operating_point.parse_backtest_args(args.op_backtest_args)
        except ValueError as e:
            raise SystemExit(f"operating-point grid: {e}")
        scope = (f'last {args.op_select_months}m of the training window'
                 if args.op_select_months else 'full training window')
        print(f"Operating-point selection: {len(op_grid)} arms "
              f"({[op.arm for op in op_grid]}), selected on the {scope} "
              f"using {args.op_selection_scores} scores, "
              f"min {args.op_min_trades} trades; extra backtest args: "
              f"{' '.join(op_bt_args) or '(none)'}\n")

    if args.dry_run:
        for fw in folds:
            print(' '.join(build_fold_command(fw, args.run_id, passthrough)))
            if args.op_select:
                sel_start, sel_end = operating_point.selection_window(
                    fw.train_start, fw.train_end, args.op_select_months)
                example = operating_point.build_backtest_command(
                    fold_run_id(args.run_id, fw.fold), sel_start, sel_end,
                    op_grid[0], op_bt_args)
                print(f"#   + {len(op_grid)} selection backtests "
                      f"{sel_start}..{sel_end} and {len(op_grid)} test-sweep "
                      f"backtests {fw.backtest_start}..{fw.backtest_end}, e.g.: "
                      + ' '.join(example))
        return 0

    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning,ignore::FutureWarning'
    env['PYTHONIOENCODING'] = 'utf-8'

    fold_results = []
    for fw in folds:
        cmd = build_fold_command(fw, args.run_id, passthrough)
        rid = fold_run_id(args.run_id, fw.fold)
        print(f"\n{'=' * 80}\nFOLD {fw.fold}/{len(folds)}  "
              f"train {fw.train_start}..{fw.train_end}  "
              f"backtest {fw.backtest_start}..{fw.backtest_end}\n{'=' * 80}")
        started = time.time()
        log_path = None
        if args.quiet:
            # Child output goes to a per-fold log file; the console keeps only the
            # fold plan and the summaries.
            fold_dir = os.path.join(dir_config.GENERATED_DIR, rid)
            os.makedirs(fold_dir, exist_ok=True)
            log_path = os.path.join(fold_dir, 'train.log')
            with open(log_path, 'w') as log_fh:
                process = subprocess.run(cmd, env=env,
                                         stdout=log_fh, stderr=subprocess.STDOUT)
        else:
            # Output streams straight to the console — advanced_train.py's own
            # diagnostics ARE the progress report.
            process = subprocess.run(cmd, env=env)
        elapsed = time.time() - started
        if args.quiet:
            print(f"  finished in {elapsed / 60:.1f} min "
                  f"(exit code {process.returncode}) — log: {log_path}")

        metrics = extract_fold_metrics(rid)
        # advanced_train can exit non-zero on post-training extras (report, plots)
        # after the summary was already written — the summary decides.
        status = 'ok' if metrics is not None else 'FAILED'
        log_ref = log_path if log_path else 'the log above'
        if process.returncode != 0 and metrics is not None:
            status = 'ok*'
            print(f"WARNING: fold {fw.fold} exited with code {process.returncode} "
                  f"but wrote training_summary.json — metrics kept, check {log_ref}.")
        elif metrics is None:
            print(f"ERROR: fold {fw.fold} produced no training_summary.json "
                  f"(exit code {process.returncode}, see {log_ref}).")

        op_result = None
        if args.op_select and metrics is not None:
            op_result = operating_point.run_fold_selection(
                run_id=rid,
                run_dir=os.path.join(dir_config.GENERATED_DIR, rid),
                train_start=fw.train_start, train_end=fw.train_end,
                backtest_start=fw.backtest_start, backtest_end=fw.backtest_end,
                grid=op_grid, select_months=args.op_select_months,
                min_trades=args.op_min_trades, extra_args=op_bt_args, env=env,
                selection_scores=args.op_selection_scores)
            # metrics['backtest'] was read before the sweep (or not at all
            # without --run-backtest) — refresh it from the frozen final run.
            metrics = extract_fold_metrics(rid)

        trades = load_fold_trades(rid)
        fold_results.append({
            'fold': fw.fold,
            'run_id': rid,
            'window': asdict(fw),
            'status': status,
            'returncode': process.returncode,
            'elapsed_seconds': round(elapsed, 1),
            'log_file': log_path,
            'metrics': metrics,
            'operating_point': op_result,
            'trades': trades,
            'n_trades': len(trades),
        })

    aggregate = aggregate_folds(fold_results)

    summary = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'command': sys.argv,
        'run_id': args.run_id,
        'train_window_months': args.train_window,
        'train_start_fixed': args.train_start,
        'fast_train_window_months': args.fast_train_window,
        'fast_train_start_fixed': args.fast_train_start,
        'backtest_window_months': args.backtest_window,
        'walk_forward_step_months': args.walk_forward_step,
        'backtest_window_end': args.backtest_window_end,
        'n_folds': args.folds,
        'passthrough_args': passthrough,
        'operating_point_selection': ({
            'enabled': True,
            'grid': [op.arm for op in op_grid],
            'select_months': args.op_select_months,
            'min_trades': args.op_min_trades,
            'selection_scores': args.op_selection_scores,
            'backtest_args': op_bt_args,
        } if args.op_select else {'enabled': False}),
        # Per-trade rows stay in each fold's report/trade_list.csv — the summary only
        # carries the counts and the pooled aggregate.
        'folds': [{k: v for k, v in r.items() if k != 'trades'} for r in fold_results],
        'aggregate': aggregate,
        'operating_point_aggregate': (
            operating_point.aggregate_operating_points(fold_results)
            if args.op_select else None),
        'operating_point_arm_curves': (
            operating_point.aggregate_arm_curves(fold_results)
            if args.op_select else None),
    }
    out_dir = os.path.join(dir_config.GENERATED_DIR, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'walk_forward_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print_summary(fold_results)
    if args.op_select:
        operating_point.print_op_summary(fold_results)
    print_aggregate(aggregate)
    print(f"Summary written to {summary_path}")
    return 0 if all(r['status'] != 'FAILED' for r in fold_results) else 1


if __name__ == '__main__':
    sys.exit(main())
