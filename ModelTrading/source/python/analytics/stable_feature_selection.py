"""
Stable feature selection — an automated wrapper search that writes a features YAML.

WHY THIS SHAPE
--------------
The selection criterion is **performance contribution**: every candidate move retrains
the model with and without the features in question and reads the paired multi-seed AUC
delta in the operative regime. That is multivariate by construction, so a feature that
only pays off in combination is protected — the measured example is `daily_adx`, MI
0.0000 under its own noise floor yet 10.5% of the within-trend SHAP spread (see
training/feature_selection.py). Neither MI nor gain can provide this criterion.

Textbook backward elimination (sklearn's SequentialFeatureSelector) would automate the
very instability it is meant to remove: each greedy step takes an argmax over ~30
candidates whose true differences sit far below the measured ±0.017 AUC seed noise, so
the search path is chosen by the random draw and changes with every window. Three
adaptations make the search decidable:

1. **Moves at cluster level.** Redundant features (label-free |corr| clustering —
   feature_pruning step 1) are collapsed to representatives first; afterwards each
   elimination candidate is a single surviving representative. The contribution of one
   member of a correlated cluster is not identifiable — only the cluster's is.
2. **Non-inferiority instead of argmin.** A candidate may only be removed when the
   paired AUC delta's CI lower bound stays above -margin (default 0.017, the measured
   reseed movement) — the feature_ab / feature_pruning acceptance rule. When no
   candidate passes, the search stops. It is a stopping rule, not a ranking.
3. **Agreement across time windows.** Every candidate is evaluated on N rolling
   train/test windows carved out of ONE run's artefacts (X_{model}.parquet spans the
   full history, so no pipeline retraining) and must pass in >= k of them (default:
   all). The LAST --confirm-windows windows never see the search; the final set must
   be non-inferior to the FULL set there too, otherwise the model keeps its full set
   in the emitted config. Selecting on the same data that judges the selection was the
   best_20260808 failure mode.

WHAT IT WRITES
--------------
--emit-config produces a features YAML via feature_pruning.emit_config: `models:` tags
are narrowed to the models that kept each feature, features kept by no model are set
`enabled: false`, rows the search never saw are copied through untouched. Emit to a
*-proposal.yaml, never onto a canonical features-<mode>.yaml; an existing file is only
overwritten with --force.

Usage
-----
    python -m ModelTrading.source.python.analytics.stable_feature_selection \
        --run-id <run> --model all --seeds 8 \
        --train-months 18 --test-months 6 --step-months 6 --confirm-windows 1 \
        --emit-config ModelTrading/config/features-<mode>-stable-proposal.yaml \
        --out docs/results/stable_selection_<run>.json

Cost: one elimination step is (1 baseline + m candidates) x windows x seeds single
xgb.train fits at the model's cadence — minutes per step for the slow models (4h
cadence), substantially more for the fast models (M15 cadence). The baseline is fitted
once per (window, seed) and shared across all candidates of the step.

The decision statistic is AUC non-inferiority; AUC has repeatedly failed to track P&L
in this project, so confirm an accepted proposal with a walk-forward before treating it
as an earnings change.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config  # noqa: E402
from ModelTrading.source.python.analytics import feature_ab  # noqa: E402
from ModelTrading.source.python.analytics import feature_pruning  # noqa: E402

MODELS = feature_pruning.MODELS
NON_INFERIORITY_MARGIN = feature_pruning.NON_INFERIORITY_MARGIN

# Below this many training positives a window trains an effectively constant model whose
# paired deltas are ~0 — which would wave every candidate through. Refuse instead.
MIN_TRAIN_POSITIVES = 10


# --- Windows ----------------------------------------------------------------------------

def build_windows(start, end, train_months, test_months, step_months, embargo_days=5):
    """Rolling train/test windows between `start` and `end`.

    The test stretch starts `embargo_days` after train_end (purge for the barrier
    horizon, same role as the walk-forward embargo) and a window is only emitted when
    its full test stretch fits before `end`.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out, t0 = [], start
    while True:
        train_end = t0 + pd.DateOffset(months=train_months)
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        if test_start < test_end:
            out.append({'train_start': t0, 'train_end': train_end,
                        'test_start': test_start, 'test_end': test_end,
                        'label': f"{t0:%Y-%m}..{train_end:%Y-%m}->{test_end:%Y-%m}"})
        t0 = t0 + pd.DateOffset(months=step_months)
    return out


def prepare_contexts(X, y, regimes, windows, model, label_mode, spw_factor):
    """Masks, class balance and the operative-regime scope per window.

    Aborts loudly on a window that cannot carry a verdict — no/one-class training
    labels, or a single-class evaluation slice. A degenerate window would not fail the
    comparison, it would trivially PASS every candidate, which is worse.
    """
    ctxs = []
    for w in windows:
        train_mask = (X.index >= w['train_start']) & (X.index < w['train_end'])
        test_mask = (X.index >= w['test_start']) & (X.index < w['test_end'])
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise SystemExit(
                f"[{model}] window {w['label']}: empty split "
                f"({train_mask.sum()} train / {test_mask.sum()} test bars). "
                f"Change the window layout (--train-months/--test-months/--step-months).")

        ytr = y[train_mask]
        n_pos = int((ytr == 1).sum())
        if ytr.nunique() < 2 or n_pos < MIN_TRAIN_POSITIVES:
            raise SystemExit(
                f"[{model}] window {w['label']}: degenerate training labels "
                f"({n_pos} positives). A constant model passes every candidate "
                f"trivially — shift the windows instead.")

        y_test = y[test_mask].values
        op_name = op_mask = None
        if regimes is not None:
            for name, mask, operative in feature_ab._regime_groups(
                    regimes.loc[test_mask], label_mode):
                if operative:
                    op_name, op_mask = name, mask
                    break

        scope_y = y_test[op_mask] if op_mask is not None else y_test
        if len(scope_y) < 20 or np.unique(scope_y).size < 2:
            raise SystemExit(
                f"[{model}] window {w['label']}: the evaluation scope "
                f"({op_name or 'GLOBAL'}) is single-class or too small "
                f"({len(scope_y)} bars) — no AUC is defined there. "
                f"Change the window layout.")

        spw = ((ytr == 0).sum() / max(n_pos, 1)) * spw_factor
        ctxs.append({**w, 'train_mask': train_mask, 'test_mask': test_mask,
                     'y_test': y_test, 'spw': float(spw),
                     'op_name': op_name, 'op_mask': op_mask})
    return ctxs


# --- Paired evaluation ------------------------------------------------------------------

def paired_stats(deltas):
    """Mean and 95% CI of per-seed paired deltas (NaN seeds dropped)."""
    clean = [float(d) for d in deltas if d == d]
    k = len(clean)
    if k == 0:
        return {'mean': float('nan'), 'ci95': (float('nan'), float('nan')), 'n_seeds': 0}
    m = float(np.mean(clean))
    if k == 1:
        return {'mean': m, 'ci95': (float('nan'), float('nan')), 'n_seeds': 1}
    half = 1.96 * float(np.std(clean, ddof=1)) / (k ** 0.5)
    return {'mean': m, 'ci95': (m - half, m + half), 'n_seeds': k}


def window_passes(stats, margin):
    """Non-inferiority in one window: the CI lower bound stays above -margin.

    A missing interval (single seed, all-NaN deltas) is NOT a pass — absence of
    evidence must not remove a feature.
    """
    lo = stats['ci95'][0]
    return bool(lo == lo and lo > -margin)


def _scope_auc(ctx, pred):
    if ctx['op_mask'] is not None:
        return feature_ab._auc_or_nan(ctx['y_test'][ctx['op_mask']],
                                      pred[ctx['op_mask']])
    return feature_ab._auc_or_nan(ctx['y_test'], pred)


def evaluate_candidates(X, y, ctxs, params, rounds, base_cols, candidates, seeds,
                        margin, agree_k):
    """Paired evaluation of every candidate column set against `base_cols`.

    The baseline is fitted ONCE per (window, seed) and shared across all candidates —
    the whole point of batching an elimination step. A candidate passes when it is
    non-inferior (window_passes) in >= agree_k of the windows.
    """
    if not candidates:
        return {}
    deltas = {name: {c['label']: [] for c in ctxs} for name in candidates}
    for ctx in ctxs:
        for seed in range(1, seeds + 1):
            _, _, base_pred = feature_ab._fit_score(
                X, y, base_cols, ctx['train_mask'], ctx['test_mask'],
                params, rounds, ctx['spw'], seed)
            b_auc = _scope_auc(ctx, base_pred)
            for name, cols in candidates.items():
                _, _, pred = feature_ab._fit_score(
                    X, y, cols, ctx['train_mask'], ctx['test_mask'],
                    params, rounds, ctx['spw'], seed)
                deltas[name][ctx['label']].append(_scope_auc(ctx, pred) - b_auc)

    results = {}
    for name in candidates:
        wstats = {lbl: paired_stats(d) for lbl, d in deltas[name].items()}
        n_pass = sum(window_passes(s, margin) for s in wstats.values())
        means = [s['mean'] for s in wstats.values() if s['mean'] == s['mean']]
        results[name] = {
            'windows': {lbl: {**s, 'pass': window_passes(s, margin)}
                        for lbl, s in wstats.items()},
            'n_pass': int(n_pass),
            'n_windows': len(ctxs),
            'pass': bool(n_pass >= agree_k),
            'mean_delta': float(np.mean(means)) if means else float('nan'),
        }
    return results


# --- The search -------------------------------------------------------------------------

def select_model(run_id, model, *, generated_dir=None, threshold=0.80, seeds=8,
                 margin=NON_INFERIORITY_MARGIN, agree=0, train_months=18,
                 test_months=6, step_months=6, embargo_days=5, confirm_windows=1,
                 max_steps=40, search_start=None, search_end=None, verbose=True):
    """Collapse + backward elimination for one model. Returns a result dict.

    `agree=0` means every search window must agree; a positive value is the minimum
    number of agreeing windows. `max_steps=0` runs the redundancy collapse only.
    """
    generated_dir = generated_dir or dir_config.GENERATED_DIR
    run_dir = os.path.join(generated_dir, run_id)
    X, y, args = feature_ab.load_run(run_id, model, generated_dir=generated_dir)
    regimes = feature_ab.load_regimes(run_id, X.index, generated_dir=generated_dir)
    label_mode = args.get('label_mode', 'static')
    params, rounds, spw_factor = feature_ab.resolve_params(args, model)

    start = max(pd.Timestamp(search_start or args['train_start']), X.index.min())
    end = min(pd.Timestamp(search_end or args.get('backtest_end') or X.index.max()),
              X.index.max())
    windows = build_windows(start, end, train_months, test_months, step_months,
                            embargo_days)
    if len(windows) < confirm_windows + 2:
        raise SystemExit(
            f"[{model}] only {len(windows)} windows fit between {start:%Y-%m-%d} and "
            f"{end:%Y-%m-%d}; need at least {confirm_windows + 2} "
            f"(>= 2 search + {confirm_windows} confirmation). Shorten --train-months/"
            f"--step-months or lower --confirm-windows.")

    n_search = len(windows) - confirm_windows
    search_ctxs = prepare_contexts(X, y, regimes, windows[:n_search], model,
                                   label_mode, spw_factor)
    confirm_ctxs = prepare_contexts(X, y, regimes, windows[n_search:], model,
                                    label_mode, spw_factor) if confirm_windows else []
    agree_k = len(search_ctxs) if not agree else min(int(agree), len(search_ctxs))

    scope_name = search_ctxs[0]['op_name'] or 'GLOBAL'
    if verbose:
        print(f"\n[{model}] {len(search_ctxs)} search + {len(confirm_ctxs)} "
              f"confirmation windows, scope {scope_name}, agree >= {agree_k}, "
              f"margin {margin:.3f}, {seeds} paired seeds")

    full_cols = list(X.columns)
    current = list(full_cols)

    # Phase 1: label-free redundancy collapse, clustered on the SEARCH training rows
    # only so the confirmation windows stay untouched by every part of the selection.
    train_union = np.zeros(len(X), dtype=bool)
    for c in search_ctxs:
        train_union |= np.asarray(c['train_mask'])
    clusters, reps = feature_pruning.cluster_features(X.loc[train_union],
                                                      threshold=threshold)
    phase1 = {'clusters': clusters, 'representatives': reps,
              'result': None, 'applied': False}
    if len(reps) < len(current):
        if verbose:
            print(f"[{model}] phase 1: {len(current)} features -> {len(reps)} "
                  f"representatives, verifying the collapse "
                  f"({len(search_ctxs) * seeds * 2} fits)")
        res = evaluate_candidates(X, y, search_ctxs, params, rounds, current,
                                  {'collapse': reps}, seeds, margin, agree_k)['collapse']
        phase1.update(result=res, applied=res['pass'])
        if res['pass']:
            current = list(reps)
    elif verbose:
        print(f"[{model}] phase 1: no cluster reaches |corr| >= {threshold:.2f} "
              f"— nothing to collapse")

    # Phase 2: backward elimination over the survivors.
    steps, stop_reason = [], None
    while len(current) > 1:
        if len(steps) >= max_steps:
            stop_reason = f'step cap {max_steps} reached'
            break
        candidates = {f: [c for c in current if c != f] for f in current}
        if verbose:
            print(f"[{model}] step {len(steps) + 1}: {len(candidates)} candidates on "
                  f"{len(search_ctxs)} windows x {seeds} seeds "
                  f"({len(search_ctxs) * seeds * (1 + len(candidates))} fits)")
        res = evaluate_candidates(X, y, search_ctxs, params, rounds, current,
                                  candidates, seeds, margin, agree_k)
        passing = [f for f, r in res.items() if r['pass']]
        if not passing:
            stop_reason = 'no remaining candidate is non-inferior in the required windows'
            break
        drop = sorted(passing, key=lambda f: (-res[f]['mean_delta'], f))[0]
        current.remove(drop)
        steps.append({'step': len(steps) + 1, 'dropped': drop,
                      'result': res[drop], 'n_candidates': len(candidates),
                      'n_passing': len(passing),
                      'candidates': {f: {'mean_delta': r['mean_delta'],
                                         'n_pass': r['n_pass']}
                                     for f, r in res.items()}})
        if verbose:
            print(f"[{model}]   dropped {drop} (mean delta "
                  f"{res[drop]['mean_delta']:+.4f}, {res[drop]['n_pass']}/"
                  f"{res[drop]['n_windows']} windows, {len(passing)} passing)")
    if stop_reason is None:
        stop_reason = 'only one feature left'

    # Confirmation on the untouched windows: final set vs FULL set.
    changed = current != full_cols
    confirmation = None
    if changed and confirm_ctxs:
        confirmation = evaluate_candidates(
            X, y, confirm_ctxs, params, rounds, full_cols, {'final': current},
            seeds, margin, len(confirm_ctxs))['final']
        accept = confirmation['pass']
        reason = (f"final set non-inferior to the full set in all "
                  f"{len(confirm_ctxs)} untouched confirmation windows"
                  if accept else
                  f"confirmation failed: non-inferior in only "
                  f"{confirmation['n_pass']} of {len(confirm_ctxs)} untouched "
                  f"windows — the full set is kept")
    elif changed:
        accept = True
        reason = ('accepted on the search windows alone — NO untouched '
                  'confirmation windows were reserved (--confirm-windows 0)')
    else:
        accept, reason = True, 'no change: nothing could be removed under the rule'

    return {
        'run_id': run_id, 'model': model, 'label_mode': label_mode,
        'scope': scope_name, 'scope_is_operative': search_ctxs[0]['op_name'] is not None,
        'windows': {'search': [c['label'] for c in search_ctxs],
                    'confirm': [c['label'] for c in confirm_ctxs],
                    'train_months': train_months, 'test_months': test_months,
                    'step_months': step_months, 'embargo_days': embargo_days},
        'agree_k': agree_k, 'margin': margin, 'threshold': threshold,
        'seeds': seeds, 'rounds': int(rounds),
        'phase1': phase1, 'steps': steps, 'stop_reason': stop_reason,
        'keep': current, 'drop': [c for c in full_cols if c not in set(current)],
        'confirmation': confirmation,
        'verdict': {'accept': bool(accept), 'reason': reason},
        'budget_before': feature_pruning.capacity_budget(X, y, run_dir, model),
        'budget_after': feature_pruning.capacity_budget(X[current], y, run_dir, model),
    }


# --- Reporting and emission -------------------------------------------------------------

def _fmt_windows(res):
    parts = []
    for lbl, s in res['windows'].items():
        lo, hi = s['ci95']
        parts.append(f"{lbl}: {s['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                     f"{' PASS' if s['pass'] else ' fail'}")
    return parts


def print_report(out):
    b, a = out['budget_before'], out['budget_after']
    w = out['windows']
    print()
    print('=' * 88)
    print(f"STABLE FEATURE SELECTION - {out['run_id']} / {out['model']} "
          f"(label mode {out['label_mode']})")
    print('=' * 88)
    print(f"\n  windows: {len(w['search'])} search + {len(w['confirm'])} confirmation "
          f"(train {w['train_months']}m, test {w['test_months']}m, "
          f"step {w['step_months']}m, embargo {w['embargo_days']}d)")
    for lbl in w['search']:
        print(f"    search  {lbl}")
    for lbl in w['confirm']:
        print(f"    confirm {lbl}")
    print(f"  scope: {out['scope']}"
          + ('' if out['scope_is_operative'] else
             '  (WARNING: global AUC — no operative-regime breakdown available)'))
    print(f"  rule: drop only if the AUC-delta CI lower bound stays above "
          f"-{out['margin']:.3f} in >= {out['agree_k']} search windows, "
          f"{out['seeds']} paired seeds")

    p1 = out['phase1']
    print(f"\n1. REDUNDANCY COLLAPSE (label-free, |corr| >= {out['threshold']:.2f})")
    n_red = sum(len(c) - 1 for c in p1['clusters'] if len(c) > 1)
    if p1['result'] is None:
        print(f"   nothing to collapse ({n_red} redundant members)")
    else:
        state = 'APPLIED' if p1['applied'] else 'REJECTED (full set kept for phase 2)'
        print(f"   {n_red} redundant members -> {len(p1['representatives'])} "
              f"representatives: {state}")
        for line in _fmt_windows(p1['result']):
            print(f"     {line}")

    print("\n2. BACKWARD ELIMINATION")
    if not out['steps']:
        print("   no elimination step was possible")
    for s in out['steps']:
        r = s['result']
        print(f"   step {s['step']}: dropped {s['dropped']}  (mean "
              f"{r['mean_delta']:+.4f}, {r['n_pass']}/{r['n_windows']} windows, "
              f"{s['n_passing']}/{s['n_candidates']} candidates passing)")
    print(f"   stopped: {out['stop_reason']}")

    print("\n3. CONFIRMATION (untouched windows, final vs FULL set)")
    if out['confirmation'] is None:
        print("   not run (no change, or --confirm-windows 0)")
    else:
        for line in _fmt_windows(out['confirmation']):
            print(f"     {line}")
    v = out['verdict']
    print(f"   -> {'ACCEPT' if v['accept'] else 'REJECT'}: {v['reason']}")

    print(f"\n  capacity: {b['n_features']} -> {a['n_features']} features, "
          f"{b['eff_positives_per_feature']:.1f} -> "
          f"{a['eff_positives_per_feature']:.1f} eff. positives/feature"
          + ('' if b['n_eff_measured'] else '  (n_eff = row count, upper bound)'))
    print(f"\nFINAL SET ({len(out['keep'])} features):")
    for col in out['keep']:
        print(f'  {col}')
    print('=' * 88)


def check_emit_target(path, force=False):
    """Refuse to overwrite an existing config without --force.

    Canonical features-<mode>.yaml files can carry hand-curated selections; a
    mechanical emission must not land on one by accident. New *-proposal.yaml names
    are always safe.
    """
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"{path} exists — refusing to overwrite. Emit to a new *-proposal.yaml, "
            f"or pass --force if overwriting is intended.")


def emit_proposal(results, source_path, out_path, label_mode=None):
    """Write the features YAML via feature_pruning.emit_config.

    Inherits its contract: a model whose verdict rejected the selection (failed
    confirmation) keeps its FULL set; features kept by no model are disabled; `models:`
    tags are narrowed to the models that kept each feature.
    """
    return feature_pruning.emit_config(
        results, source_path, out_path, label_mode=label_mode,
        generated_by='analytics/stable_feature_selection.py --emit-config')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id', required=True, help='Training run whose artefacts to read.')
    ap.add_argument('--model', default='all', choices=list(MODELS) + ['all'])
    ap.add_argument('--threshold', type=float, default=0.80,
                    help='|corr| at which two features count as interchangeable (0.80).')
    ap.add_argument('--seeds', type=int, default=8,
                    help='Paired seeds per evaluation (default 8, minimum 2).')
    ap.add_argument('--margin', type=float, default=NON_INFERIORITY_MARGIN,
                    help='Non-inferiority margin in AUC (default 0.017, the measured '
                         'movement from reseeding alone).')
    ap.add_argument('--agree', type=int, default=0,
                    help='Minimum number of agreeing search windows per move '
                         '(0 = all of them, the default).')
    ap.add_argument('--train-months', type=int, default=18)
    ap.add_argument('--test-months', type=int, default=6)
    ap.add_argument('--step-months', type=int, default=6)
    ap.add_argument('--embargo-days', type=int, default=5)
    ap.add_argument('--confirm-windows', type=int, default=1,
                    help='How many trailing windows are reserved for confirmation and '
                         'never seen by the search (default 1; 0 disables and is '
                         'recorded in the verdict).')
    ap.add_argument('--max-steps', type=int, default=40,
                    help='Cap on elimination steps; 0 runs the redundancy collapse only.')
    ap.add_argument('--search-start', default=None,
                    help="First usable bar (default: the run's train_start).")
    ap.add_argument('--search-end', default=None,
                    help="Last usable bar (default: the run's backtest_end).")
    ap.add_argument('--emit-config', default=None,
                    help='Write the resulting features YAML here. Requires --model all, '
                         'since the config carries a per-model tag for every feature. '
                         'Use a *-proposal.yaml name.')
    ap.add_argument('--source-config', default=None,
                    help='Base YAML to narrow (default config/features.yaml).')
    ap.add_argument('--label-mode', default=None,
                    help='Recorded in the emitted config header.')
    ap.add_argument('--force', action='store_true',
                    help='Allow --emit-config to overwrite an existing file.')
    ap.add_argument('--out', default=None, help='Write the full result to this JSON.')
    args = ap.parse_args()

    if args.seeds < 2:
        raise SystemExit('--seeds must be >= 2: one seed has no confidence interval, '
                         'and a missing interval can never justify a removal.')
    if args.emit_config:
        if args.model != 'all':
            raise SystemExit('--emit-config requires --model all: the config assigns '
                             'each feature to the models that kept it, so all four '
                             'must be known.')
        check_emit_target(args.emit_config, force=args.force)

    models = list(MODELS) if args.model == 'all' else [args.model]
    results = {}
    for model in models:
        out = select_model(
            args.run_id, model, threshold=args.threshold, seeds=args.seeds,
            margin=args.margin, agree=args.agree, train_months=args.train_months,
            test_months=args.test_months, step_months=args.step_months,
            embargo_days=args.embargo_days, confirm_windows=args.confirm_windows,
            max_steps=args.max_steps, search_start=args.search_start,
            search_end=args.search_end)
        print_report(out)
        results[model] = out

    if args.emit_config:
        src = args.source_config or os.path.join(dir_config.CONFIG_DIR, 'features.yaml')
        label_mode = args.label_mode or next(
            (r.get('label_mode') for r in results.values() if r.get('label_mode')), None)
        disabled, narrowed, restored = emit_proposal(results, src, args.emit_config,
                                                     label_mode=label_mode)
        print(f'\nWrote {args.emit_config}')
        print(f"  disabled ({len(disabled)}): {', '.join(disabled) or '-'}")
        for name, kept in narrowed:
            print(f"  narrowed {name} -> [{', '.join(kept)}]")
        if restored:
            print(f"  KEPT FULL SET for {', '.join(restored)} — the confirmation "
                  f"windows rejected the selection for them")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'\nResult written to {args.out}')


if __name__ == '__main__':
    main()
