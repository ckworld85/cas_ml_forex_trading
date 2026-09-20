"""
Operating-point grid selection for walk-forward folds.

A trained model is not a strategy until an operating point (entry thresholds and
the optional fast entry gate) is fixed. The fixed rules used so far can misfire
in both directions — measured 2026-09-11 on run `trend_only`:
`--use-trained-threshold` mapped a raw slow threshold of 0.347 through the
direction calibrators to an EFFECTIVE threshold of 0.971 and produced 6 trades
in 7 months (1 with the fast gate), while hand thresholds around 0.5–0.6 sat on
a plateau of ~23 trades. Either way the walk-forward measured the operating
RULE, not the model. See docs/pipeline_controls_2026-09-11.md for the
instrument-calibration context.

This module makes the operating point part of the measured procedure:

1. **Selection (in-sample)**: after a fold trains, every arm of a small grid
   (slow threshold x fast-gate option, plus the trained-threshold status quo)
   is backtested on the LAST `select_months` of the fold's own training window.
   The best arm by total pips (with a minimum-trade eligibility floor) becomes
   the fold's operating point.
2. **Evaluation (out-of-sample)**: ONE backtest with the frozen operating point
   runs on the fold's test window. That is the fold's official result — the
   selection never sees the test window.
3. **Diagnostic sweep**: the full grid is ALSO run on the test window and
   recorded (cheap next to training). The best arm there is reported as the
   look-ahead "oracle" upper bound together with the selection's regret — it is
   diagnostics, never the result: best-of-K selected on the evaluation window
   is exactly the bias that produced the +45k artefact (best_20260808).

Everything that decides is a pure function (grid construction, window
arithmetic, command building, arm selection) so it is unit-testable without
running a single backtest; the subprocess runner is a thin shell around them.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

MODELS = ('long_fast', 'short_fast', 'long_slow', 'short_slow')

# Flags the machinery sets itself — user-supplied backtest args must not fight them.
FORBIDDEN_BACKTEST_ARGS = (
    '--backtest-start', '--backtest-end', '--run-id',
    '--p-open-slow', '--p-open-fast', '--opening-requires-fast-signal',
    '--use-trained-threshold', '--proba-file', '--unseal-holdout',
)

DEFAULT_SLOW_GRID = '0.45,0.50,0.55,0.60,0.65,0.70'
DEFAULT_FAST_GRID = 'off,0.50,0.55'

# Keys copied from each arm's backtest_summary.json into its result row.
_ROW_KEYS = ('total_trades', 'win_rate_pct', 'total_pnl_pips', 'total_pnl_eur',
             'avg_pnl_per_trade_pips', 'profit_factor', 'max_drawdown_pct',
             'thresholds')


@dataclass
class OperatingPoint:
    """One grid arm: entry thresholds + fast-gate option (or the trained rule)."""
    arm: str
    p_open_slow: float = None
    p_open_fast: float = None
    fast_gate: bool = False
    use_trained: bool = False


def parse_float_list(spec):
    """Parse '0.45,0.5,0.6' into a list of floats (order kept, must be non-empty)."""
    values = [v.strip() for v in str(spec).split(',') if v.strip()]
    if not values:
        raise ValueError(f"empty threshold list: '{spec}'")
    return [float(v) for v in values]


def parse_fast_options(spec):
    """Parse 'off,0.50,0.55' into [None, 0.5, 0.55]; None = fast gate off."""
    options = []
    for value in str(spec).split(','):
        value = value.strip()
        if not value:
            continue
        options.append(None if value.lower() == 'off' else float(value))
    if not options:
        raise ValueError(f"empty fast-gate option list: '{spec}'")
    return options


def _arm_label(slow, fast):
    slow_part = f"slow{round(slow * 100):02d}"
    return f"{slow_part}_nogate" if fast is None else f"{slow_part}_fast{round(fast * 100):02d}"


def build_grid(slow_thresholds, fast_options, include_trained=True):
    """
    Cross slow thresholds with fast-gate options; optionally append the
    trained-threshold status quo as its own arm so it competes on equal terms.
    """
    grid = []
    for slow in slow_thresholds:
        for fast in fast_options:
            grid.append(OperatingPoint(
                arm=_arm_label(slow, fast),
                p_open_slow=slow,
                p_open_fast=fast if fast is not None else None,
                fast_gate=fast is not None,
            ))
    if include_trained:
        grid.append(OperatingPoint(arm='trained', use_trained=True))
    return grid


def parse_backtest_args(spec):
    """
    Split the pass-through backtest argument string (measurement settings such
    as '--cost-model data --slippage-pips 0.2') and reject flags the selection
    machinery controls itself.
    """
    args = shlex.split(spec or '')
    offending = [flag for flag in FORBIDDEN_BACKTEST_ARGS
                 if flag in args or any(a.startswith(flag + '=') for a in args)]
    if offending:
        raise ValueError(
            f"--op-backtest-args must not contain {', '.join(offending)} — "
            "the operating-point machinery sets these itself.")
    return args


def selection_window(train_start, train_end, months=None):
    """
    The in-sample window the grid is selected on. Default (months=None): the
    WHOLE training window — a short trailing window can sit in one regime and
    starve arms whose regime never occurs in it, which would reject the
    configuration for the window, not for the config. A trailing window stays
    available via `months` for studies. Returns (start, end) as YYYY-MM-DD
    strings; end == train_end, so the selection never touches the test window.
    """
    start_ts = pd.Timestamp(train_start)
    end_ts = pd.Timestamp(train_end)
    sel_start = start_ts if months is None else max(
        start_ts, end_ts - pd.DateOffset(months=months))
    return sel_start.strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d')


def build_backtest_command(run_id, start, end, op, extra_args=(),
                           python=None, script=None):
    """argv for one arm's backtest. Pure function for testability."""
    if script is None:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'backtest.py')
    cmd = [python or sys.executable, script,
           '--run-id', run_id,
           '--backtest-start', start,
           '--backtest-end', end]
    if op.use_trained:
        cmd.append('--use-trained-threshold')
    else:
        cmd.extend(['--p-open-slow', str(op.p_open_slow)])
        if op.fast_gate:
            cmd.extend(['--opening-requires-fast-signal',
                        '--p-open-fast', str(op.p_open_fast)])
    cmd.extend(extra_args)
    return cmd


def select_operating_point(rows, min_trades):
    """
    Pick the arm to freeze from the selection-window rows.

    Rule: among arms that ran and traded at least `min_trades` times, the
    highest total_pnl_pips wins (first of equals in grid order — deterministic).
    If no arm reaches the floor, fall back to the arm with the most trades so
    the fold still gets a defined out-of-sample measurement; if nothing traded
    at all there is no operating point.

    Returns (row_or_None, reason).
    """
    ok_rows = [r for r in rows if r.get('ok')]
    eligible = [r for r in ok_rows if (r.get('total_trades') or 0) >= min_trades]
    if eligible:
        best = eligible[0]
        for row in eligible[1:]:
            if row['total_pnl_pips'] > best['total_pnl_pips']:
                best = row
        return best, 'best_pips'
    traded = [r for r in ok_rows if (r.get('total_trades') or 0) > 0]
    if traded:
        best = traded[0]
        for row in traded[1:]:
            if row['total_trades'] > best['total_trades']:
                best = row
        return best, 'fallback_max_trades'
    return None, 'no_arm_traded'


def oracle_row(rows):
    """Best arm by total pips on the rows it is given (diagnostic upper bound)."""
    ok_rows = [r for r in rows if r.get('ok') and (r.get('total_trades') or 0) > 0]
    if not ok_rows:
        return None
    best = ok_rows[0]
    for row in ok_rows[1:]:
        if row['total_pnl_pips'] > best['total_pnl_pips']:
            best = row
    return best


# ------------------------------------------------------------ OOF proba file

def _apply_calibrator(raw, calibrator):
    """Same mapping backtest.py applies to live scores (Platt or isotonic)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(np.asarray(raw))
    return raw


def build_oof_proba_file(run_dir, out_path=None, stage='final'):
    """
    Turn the run's pooled CV validation scores (oof_predictions.parquet) into a
    backtest.py --proba-file so the SELECTION backtests rank arms on
    validation-honest probabilities instead of memorized in-sample scores.

    Why this exists (measured on wf_trendonly, 2026-09-11): an in-sample
    backtest of the trained model memorizes its training bars, so "more trades
    = more pips" and the selection degenerates to the lowest threshold
    (slow45 won 3 of 4 folds in-sample while ranking WORST out-of-sample).
    OOF scores carry no memorization, so the arm ranking becomes meaningful.

    Each model's column is passed through the run's own calibrator
    (calibrator_target_<model>.pkl) — the final backtest applies thresholds to
    CALIBRATED probabilities, so the selection must rank thresholds in the
    same space (the 0.347-raw -> 0.971-calibrated mapping is exactly what made
    --use-trained-threshold a starvation rule).

    Bars without an OOF score for a model stay NaN and become 0.0 in
    backtest.py — no trade. Slow models therefore only signal on their 4h
    cadence bars here, which is identical across arms and fine for ranking.

    Returns {'path', 'rows', 'stage', 'coverage': {model: n_scores}} or None
    when the run has no oof_predictions.parquet (caller falls back to model
    scores and says so).
    """
    import joblib

    oof_path = os.path.join(run_dir, 'oof_predictions.parquet')
    if not os.path.exists(oof_path):
        return None
    df = pd.read_parquet(oof_path)
    df = df[df['stage'] == stage]
    if len(df) == 0:
        return None

    wide = df.pivot_table(index='timestamp', columns='model', values='y_score',
                          aggfunc='last').astype('float64')
    coverage = {}
    for model in MODELS:
        if model not in wide.columns:
            wide[model] = np.nan
        valid = wide[model].notna()
        coverage[model] = int(valid.sum())
        cal_path = os.path.join(run_dir, f'calibrator_target_{model}.pkl')
        if valid.any() and os.path.exists(cal_path):
            calibrator = joblib.load(cal_path)
            wide.loc[valid, model] = _apply_calibrator(
                wide.loc[valid, model].to_numpy(), calibrator)
    wide = wide[list(MODELS)].sort_index()

    if out_path is None:
        out_dir = os.path.join(run_dir, 'op_selection')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'oof_proba.parquet')
    wide.to_parquet(out_path)
    return {'path': out_path, 'rows': int(len(wide)), 'stage': stage,
            'coverage': coverage}


# --------------------------------------------------------------------- runner

def _report_paths(run_dir):
    report = os.path.join(run_dir, 'report')
    return (os.path.join(report, 'backtest_summary.json'),
            os.path.join(report, 'trade_list.csv'))


def _clear_report(run_dir):
    """Remove the previous arm's artefacts so a failed run cannot be misread
    as the current arm's result."""
    for path in _report_paths(run_dir):
        if os.path.exists(path):
            os.remove(path)


def run_arm(op, run_id, run_dir, start, end, extra_args, env, phase,
            python=None, script=None, proba_file=None):
    """
    Run one arm's backtest and return its result row. The arm's summary and log
    are archived under <run_dir>/op_selection/<phase>_<arm>.{json,log}; the
    trade list is NOT archived (only the officially selected arm's final run
    leaves its trade_list.csv in report/). `proba_file` replaces the model
    scores (selection on OOF probabilities); the row records it.
    """
    op_dir = os.path.join(run_dir, 'op_selection')
    os.makedirs(op_dir, exist_ok=True)
    log_path = os.path.join(op_dir, f"{phase}_{op.arm}.log")
    summary_path, _ = _report_paths(run_dir)

    _clear_report(run_dir)
    cmd = build_backtest_command(run_id, start, end, op, extra_args,
                                 python=python, script=script)
    if proba_file:
        cmd.extend(['--proba-file', proba_file])
    with open(log_path, 'w') as log_fh:
        proc = subprocess.run(cmd, env=env, stdout=log_fh,
                              stderr=subprocess.STDOUT)

    row = {'arm': op.arm, 'operating_point': asdict(op), 'phase': phase,
           'returncode': proc.returncode, 'log': log_path}
    if not os.path.exists(summary_path):
        row.update(ok=False, error=f"no backtest_summary.json (exit {proc.returncode})")
        return row
    with open(summary_path) as fh:
        summary = json.load(fh)
    shutil.copy(summary_path, os.path.join(op_dir, f"{phase}_{op.arm}.json"))
    row.update({key: summary.get(key) for key in _ROW_KEYS})
    row['ok'] = True
    return row


def run_fold_selection(run_id, run_dir, train_start, train_end,
                       backtest_start, backtest_end, grid, select_months,
                       min_trades, extra_args, env, python=None, script=None,
                       verbose=True, selection_scores='oof'):
    """
    The full per-fold procedure: in-sample grid -> freeze one arm -> test-window
    sweep with the frozen arm LAST (so report/ keeps the official artefacts) ->
    oracle/regret diagnostics. Writes operating_point_selection.json into the
    run directory and returns the same dict.

    selection_scores: 'oof' (default) ranks the selection arms on the run's
    calibrated out-of-fold scores via --proba-file — in-sample MODEL scores are
    memorized and degenerate the selection to the lowest threshold. 'model'
    keeps the raw in-sample behaviour for comparison. Falls back to 'model'
    (recorded + printed) when the run has no oof_predictions.parquet. The
    test-window sweep and the official frozen run always use the real model
    scores — that is the deployment path being measured.
    """
    sel_start, sel_end = selection_window(train_start, train_end, select_months)

    oof_meta, proba_file = None, None
    scores_used = selection_scores
    if selection_scores == 'oof':
        oof_meta = build_oof_proba_file(run_dir)
        if oof_meta is None:
            scores_used = 'model'
            print(f"  [op] WARNING: no oof_predictions.parquet in {run_dir} — "
                  "selection falls back to in-sample MODEL scores (memorized; "
                  "biased toward low thresholds).")
        else:
            proba_file = oof_meta['path']
    if verbose:
        scope = ('full training window' if select_months is None
                 else f'last {select_months}m of the training window')
        print(f"  [op] selection grid: {len(grid)} arms on {sel_start}..{sel_end} "
              f"({scope}), scored on {scores_used} scores")

    select_rows = [run_arm(op, run_id, run_dir, sel_start, sel_end, extra_args,
                           env, 'select', python=python, script=script,
                           proba_file=proba_file)
                   for op in grid]
    selected_row, reason = select_operating_point(select_rows, min_trades)
    selected_arm = selected_row['arm'] if selected_row else None

    # Test-window sweep: every arm once, the selected arm last so its
    # backtest_summary.json / trade_list.csv remain in report/ as the fold's
    # official out-of-sample result.
    sweep = [op for op in grid if op.arm != selected_arm]
    if selected_arm is not None:
        sweep.append(next(op for op in grid if op.arm == selected_arm))
    if verbose:
        print(f"  [op] selected: {selected_arm or '-'} ({reason}) — test sweep: "
              f"{len(sweep)} arms on {backtest_start}..{backtest_end}")
    test_rows = [run_arm(op, run_id, run_dir, backtest_start, backtest_end,
                         extra_args, env, 'test', python=python, script=script)
                 for op in sweep]
    if selected_arm is None:
        # No official arm: leave no artefacts a pooled aggregate could misread.
        _clear_report(run_dir)

    oracle = oracle_row(test_rows)
    selected_test = next((r for r in test_rows if r['arm'] == selected_arm), None)
    regret = None
    if oracle is not None and selected_test is not None and selected_test.get('ok'):
        regret = float(oracle['total_pnl_pips']) - float(selected_test['total_pnl_pips'])

    result = {
        'selection_window': {'start': sel_start, 'end': sel_end,
                             'months': select_months},
        'selection_scores': scores_used,
        'oof_proba': oof_meta,
        'min_trades': min_trades,
        'grid_size': len(grid),
        'selected_arm': selected_arm,
        'selection_reason': reason,
        'selected_selection_row': selected_row,
        'selected_test_row': selected_test,
        # Look-ahead diagnostics — never a result:
        'oracle_test_row': oracle,
        'regret_pips': regret,
        'select_rows': select_rows,
        'test_rows': test_rows,
    }
    with open(os.path.join(run_dir, 'operating_point_selection.json'), 'w') as fh:
        json.dump(result, fh, indent=2)
    if verbose and selected_test is not None and selected_test.get('ok'):
        oracle_txt = (f"oracle {oracle['arm']} {oracle['total_pnl_pips']:+.1f} pips, "
                      f"regret {regret:+.1f}" if oracle else 'oracle -')
        print(f"  [op] OOS ({selected_arm}): {selected_test['total_trades']} trades, "
              f"{selected_test['total_pnl_pips']:+.1f} pips — {oracle_txt}")
    return result


# ----------------------------------------------------------------- aggregation

def aggregate_arm_curves(fold_results):
    """
    Per-ARM pooled out-of-sample results across every fold's FULL test window.

    Each arm is a fixed a-priori configuration applied identically in every
    fold, so its pooled number is honest out-of-sample — this is the right way
    to ask "is slow 0.55 + gate a good operating point overall" without
    selecting per fold. Comparing arms and quoting the best one is still a
    best-of-K statement (charge the grid size), but the SHAPE of the curve —
    a plateau vs. a spike, sign stability across folds — is diagnostics the
    per-fold selection cannot show. Returns {arm: {...}} or None.
    """
    per_arm = {}
    order = []
    for res in fold_results:
        op = res.get('operating_point')
        if not op:
            continue
        for row in op.get('test_rows') or []:
            if not row.get('ok'):
                continue
            arm = row['arm']
            if arm not in per_arm:
                per_arm[arm] = {'n_folds': 0, 'total_trades': 0,
                                'total_pnl_pips': 0.0, 'total_pnl_eur': 0.0,
                                'folds_positive': 0, 'fold_pips': []}
                order.append(arm)
            entry = per_arm[arm]
            pips = float(row.get('total_pnl_pips') or 0.0)
            entry['n_folds'] += 1
            entry['total_trades'] += int(row.get('total_trades') or 0)
            entry['total_pnl_pips'] += pips
            entry['total_pnl_eur'] += float(row.get('total_pnl_eur') or 0.0)
            entry['folds_positive'] += 1 if pips > 0 else 0
            entry['fold_pips'].append(pips)
    return {arm: per_arm[arm] for arm in order} if per_arm else None


def aggregate_operating_points(fold_results):
    """
    Cross-fold view of the selection: which arms were chosen, how the selected
    out-of-sample pips compare to the look-ahead oracle, and how often the
    selection had to fall back. Reads the 'operating_point' block main()
    attaches to each fold record; returns None when no fold carries one.
    """
    ops = [r['operating_point'] for r in fold_results if r.get('operating_point')]
    if not ops:
        return None
    arm_counts = {}
    for op in ops:
        arm = op.get('selected_arm') or '-'
        arm_counts[arm] = arm_counts.get(arm, 0) + 1
    regrets = [op['regret_pips'] for op in ops if op.get('regret_pips') is not None]
    selected_pips = [op['selected_test_row']['total_pnl_pips'] for op in ops
                     if (op.get('selected_test_row') or {}).get('ok')]
    oracle_pips = [op['oracle_test_row']['total_pnl_pips'] for op in ops
                   if op.get('oracle_test_row')]
    return {
        'n_folds_with_selection': len(ops),
        'selected_arm_counts': arm_counts,
        'selection_reasons': {reason: sum(1 for op in ops
                                          if op.get('selection_reason') == reason)
                              for reason in {op.get('selection_reason') for op in ops}},
        'pooled_selected_oos_pips': sum(selected_pips) if selected_pips else None,
        # Sum of per-fold best-on-test arms — a LOOK-AHEAD upper bound, kept
        # only to show how much the honest selection leaves on the table.
        'pooled_oracle_oos_pips_lookahead': sum(oracle_pips) if oracle_pips else None,
        'mean_regret_pips': (sum(regrets) / len(regrets)) if regrets else None,
        'regrets_pips': regrets,
    }


def print_op_summary(fold_results):
    """Per-fold operating-point table + the cross-fold aggregate."""
    ops = [(r['fold'], r.get('operating_point')) for r in fold_results
           if r.get('operating_point')]
    if not ops:
        return
    print("\n" + "=" * 100)
    print("OPERATING-POINT SELECTION (selected in-sample, evaluated once "
          "out-of-sample; oracle = look-ahead diagnostic)")
    print("=" * 100)
    print(f"  {'fold':>4}  {'selected':<16} {'reason':<20} {'sel pips':>9}  "
          f"{'OOS trades':>10}  {'OOS pips':>9}  {'oracle':<16} {'regret':>8}")
    for fold, op in ops:
        sel = op.get('selected_selection_row') or {}
        test = op.get('selected_test_row') or {}
        oracle = op.get('oracle_test_row') or {}
        regret = op.get('regret_pips')
        print(f"  {fold:>4}  {op.get('selected_arm') or '-':<16} "
              f"{op.get('selection_reason') or '-':<20} "
              f"{sel.get('total_pnl_pips', float('nan')):>9.1f}  "
              f"{test.get('total_trades', 0) if test.get('ok') else '-':>10}  "
              f"{test.get('total_pnl_pips', float('nan')) if test.get('ok') else float('nan'):>9.1f}  "
              f"{oracle.get('arm', '-'):<16} "
              f"{regret if regret is not None else float('nan'):>8.1f}")
    records = [{'fold': f, 'operating_point': op} for f, op in ops]
    agg = aggregate_operating_points(records)
    print(f"\n  Selected pooled OOS pips: {agg['pooled_selected_oos_pips']}"
          f"   vs. look-ahead oracle: {agg['pooled_oracle_oos_pips_lookahead']}"
          f"   (mean regret {agg['mean_regret_pips'] if agg['mean_regret_pips'] is not None else float('nan'):+.1f} pips)")
    print(f"  Arm stability across folds: {agg['selected_arm_counts']}")

    curves = aggregate_arm_curves(records)
    if curves:
        print("\n  Per-arm pooled OOS across all fold test windows (fixed arm, "
              "honest OOS; comparing arms = best-of-K, mind the grid size):")
        print(f"    {'arm':<16} {'folds':>5} {'trades':>7} {'pips':>10} "
              f"{'EUR':>12} {'folds>0':>8}")
        for arm, entry in curves.items():
            print(f"    {arm:<16} {entry['n_folds']:>5} {entry['total_trades']:>7} "
                  f"{entry['total_pnl_pips']:>10.1f} {entry['total_pnl_eur']:>12.0f} "
                  f"{entry['folds_positive']:>8}")
    print("=" * 100)
