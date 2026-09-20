"""
Feature A/B — decide whether a feature-set change is worth making.

**Why this exists.** A single backtest cannot answer the question. Measured on run
feature_eval: 34 trades, ±303 pips standard error on the total, so a +525 pip result
has a 95% interval of −68…+1119 — it cannot even establish that the strategy has an
edge, let alone that a feature change helped. Two runs differing only by the random
draw share ~70% of their trades. And the walk-forward is no help either at this size:
its minimum detectable difference is ~21% of total P&L, far more than a feature tweak
moves.

**What this does instead.** It compares the two feature sets on **test AUC**, which is
formed over every bar of the test window (~860 at 4h cadence) rather than over 34
trades, giving roughly 25x the resolution. Both variants are trained on the SAME seeds
and the difference is analysed per seed (paired), so the seed component cancels out of
the comparison instead of sitting on both sides of it.

**And it decides on the regime that trades, not on the aggregate.** A single global AUC
is the wrong criterion here. Under trend_only every range bar is a forced 0, so the
range population is trivially separable and inflates the overall number — a change that
sharpens range separation while blunting the trend regime can look like an improvement
and be worth nothing, because the strategy only opens positions in trend bars and the
live regime gate discards the rest. The verdict below is therefore taken from the
**operative regime** (trend, for trend_only and regime_conditional labels); the other
regimes are reported alongside so a change that trades one against the other is visible
instead of averaged away. Same reasoning as the `TREND (all)` row in
advanced_train._print_per_regime_metrics.

**What it cannot do.** It reads the feature matrix a training run already wrote, so it
can only remove (or subset) features — adding a feature that is not in the parquet
needs a full retrain with an updated features.yaml. It also reports AUC, not P&L: in
this project AUC has repeatedly failed to predict P&L (see label_mode_comparison.md),
so treat a small AUC gain as "no reason to avoid the change", not as "this earns money".
The honest use is to rule changes IN or OUT cheaply, not to rank them.

Usage:
    python -m ModelTrading.source.python.analytics.feature_ab --run-id feature_eval \
        --model long_slow --drop daily_regime_trend,4hours_is_asia_session --seeds 8
"""

import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add project root to path so we can import sibling modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config

MODELS = ('long_fast', 'short_fast', 'long_slow', 'short_slow')

# Below this the difference is not worth a decision even when it is measurable: it is
# far inside the spread of everything else that moves the result (seed, window choice).
PRACTICAL_AUC_THRESHOLD = 0.005


def _find_label_file(run_dir, model):
    """Locate the label parquet. Naming is inconsistent across cadences."""
    direction = model.split('_')[0]
    for name in (f'y_target_{model}.parquet',
                 f'y_target_{direction}_{model}.parquet'):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    hits = glob.glob(os.path.join(run_dir, f'y_target_*{model}.parquet'))
    if hits:
        return hits[0]
    raise SystemExit(f"No label parquet for '{model}' in {run_dir}")


def load_run(run_id, model, generated_dir=None):
    """Feature matrix, labels and the run's own hyperparameters, at the model's cadence."""
    run_dir = os.path.join(generated_dir or dir_config.GENERATED_DIR, run_id)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")

    X = pd.read_parquet(os.path.join(run_dir, f'X_{model}.parquet'))
    y = pd.read_parquet(_find_label_file(run_dir, model)).iloc[:, 0]

    with open(os.path.join(run_dir, 'training_summary.json'), encoding='utf-8') as f:
        args = json.load(f)['args']

    # Slow models train on 4h bars — the M15 forward-fill would inflate the sample ~16x.
    if model.endswith('_slow'):
        X, y = X.resample('4h').first(), y.resample('4h').first()

    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok].astype(int), args


def resolve_params(args, model):
    """The run's own hyperparameters for this model's cadence."""
    fast = model.endswith('_fast')
    pick = lambda specific, shared, default: (
        args.get(specific) if args.get(specific) is not None
        else args.get(shared, default))

    prefix = 'fast' if fast else 'slow'
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': pick(f'{prefix}_max_depth', 'max_depth', 3),
        'eta': args.get('eta', 0.05),
        'subsample': args.get('subsample', 1.0),
        'colsample_bytree': args.get('colsample_bytree', 1.0),
        'min_child_weight': pick(f'{prefix}_min_child_weight', 'min_child_weight', 3),
        'lambda': args.get(f'{prefix}_lambda', 1.0),
        'verbosity': 0,
    }
    rounds = pick(f'{prefix}_num_boost_round', 'num_boost_round', 200)
    spw_factor = args.get(f'{prefix}_spw_factor', 1.0)
    return params, int(rounds), float(spw_factor)


def _fit_score(X, y, cols, train_mask, test_mask, params, rounds, spw, seed):
    """Train on the training window; return (train AUC, test AUC, test predictions)."""
    scaler = StandardScaler().fit(X.loc[train_mask, cols])
    dtrain = xgb.DMatrix(scaler.transform(X.loc[train_mask, cols]),
                         label=y[train_mask].values)
    dtest = xgb.DMatrix(scaler.transform(X.loc[test_mask, cols]))
    booster = xgb.train({**params, 'scale_pos_weight': spw, 'seed': seed},
                        dtrain, num_boost_round=rounds)
    pred_test = booster.predict(dtest)
    return (roc_auc_score(y[train_mask], booster.predict(dtrain)),
            roc_auc_score(y[test_mask], pred_test),
            pred_test)


def load_regimes(run_id, index, generated_dir=None):
    """
    Regime label per test bar, aligned to `index`. None when the run has no labels.

    Returned as the combined label (e.g. TREND_HIGH_VOL) plus the coarse trend flag,
    both reindexed onto the model's own cadence.
    """
    path = os.path.join(generated_dir or dir_config.GENERATED_DIR, run_id,
                        'regime_analysis', 'regime_labels.parquet')
    if not os.path.exists(path):
        return None
    reg = pd.read_parquet(path)
    if index.freq is None and len(index) > 1:
        # Slow models were resampled to 4h; take the regime of each period's first bar.
        reg = reg.resample('4h').first() if (index[1] - index[0]) >= pd.Timedelta('4h') else reg
    return reg.reindex(index)


def _regime_groups(regimes, label_mode):
    """
    (name, mask, is_operative) per regime population, coarse first.

    The operative regime is the one the strategy can actually open positions in. Under
    trend_only / regime_conditional the labels put every positive in a trend bar, so a
    change is only worth anything if it holds up there.
    """
    if regimes is None or 'regime_trend' not in regimes.columns:
        return []
    trend = (regimes['regime_trend'] > 0.5).values
    operative_is_trend = label_mode in ('trend_only', 'regime_conditional')

    groups = [('TREND (all)', trend, operative_is_trend),
              ('RANGE (all)', ~trend, False)]
    if 'regime_combined' in regimes.columns:
        combined = regimes['regime_combined'].astype(str).values
        for name in sorted(set(combined) - {'nan'}):
            groups.append((name, combined == name, False))
    return groups


def _auc_or_nan(y_true, pred):
    """AUC, or NaN when the slice carries a single class (undefined, not zero)."""
    if len(y_true) < 20 or len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, pred)


def compare(X, y, args, model, drop=(), keep_only=None, seeds=8,
            train_end=None, test_end=None, regimes=None):
    """
    Train both feature sets on the same seeds and analyse the paired difference.

    ``regimes`` (optional, from load_regimes) splits the test window into its regime
    populations so the verdict can be taken from the one the strategy actually trades
    rather than from an aggregate that averages it with bars no position is ever opened
    in. Without it the comparison falls back to the global AUC.

    Returns a dict with per-variant means, the paired delta statistics, and per-regime
    breakdowns keyed by regime name.
    """
    baseline_cols = list(X.columns)
    if keep_only is not None:
        missing = [c for c in keep_only if c not in baseline_cols]
        if missing:
            raise SystemExit(f"Not in the feature matrix: {', '.join(missing)}")
        variant_cols = [c for c in baseline_cols if c in set(keep_only)]
    else:
        missing = [c for c in drop if c not in baseline_cols]
        if missing:
            raise SystemExit(f"Not in the feature matrix: {', '.join(missing)}")
        variant_cols = [c for c in baseline_cols if c not in set(drop)]

    if not variant_cols:
        raise SystemExit("The variant would have no features left.")
    if variant_cols == baseline_cols:
        raise SystemExit("Variant and baseline are identical — nothing to compare.")

    train_end = pd.to_datetime(train_end or args['train_end'])
    train_start = pd.to_datetime(args['train_start'])
    test_end = pd.to_datetime(test_end or args.get('backtest_end') or X.index.max())
    train_mask = (X.index >= train_start) & (X.index <= train_end)
    test_mask = (X.index > train_end) & (X.index <= test_end)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise SystemExit(f"Empty split: {train_mask.sum()} train / {test_mask.sum()} test bars")

    params, rounds, spw_factor = resolve_params(args, model)
    ytr = y[train_mask]
    spw = ((ytr == 0).sum() / max((ytr == 1).sum(), 1)) * spw_factor

    y_test = y[test_mask].values
    groups = _regime_groups(
        regimes.loc[test_mask] if regimes is not None else None,
        args.get('label_mode', 'static'))
    per_regime = {name: {'baseline': [], 'variant': [], 'delta': [],
                         'n_bars': int(mask.sum()),
                         'pos_rate': float(y_test[mask].mean()) if mask.any() else float('nan'),
                         'operative': operative}
                  for name, mask, operative in groups}

    rows = []
    for seed in range(1, seeds + 1):
        b_tr, b_te, b_pred = _fit_score(X, y, baseline_cols, train_mask, test_mask,
                                        params, rounds, spw, seed)
        v_tr, v_te, v_pred = _fit_score(X, y, variant_cols, train_mask, test_mask,
                                        params, rounds, spw, seed)
        rows.append((seed, b_tr, b_te, v_tr, v_te))
        for name, mask, _ in groups:
            b_auc = _auc_or_nan(y_test[mask], b_pred[mask])
            v_auc = _auc_or_nan(y_test[mask], v_pred[mask])
            per_regime[name]['baseline'].append(b_auc)
            per_regime[name]['variant'].append(v_auc)
            per_regime[name]['delta'].append(v_auc - b_auc)

    b_test = [r[2] for r in rows]
    v_test = [r[4] for r in rows]
    b_gap = [r[1] - r[2] for r in rows]
    v_gap = [r[3] - r[4] for r in rows]
    deltas = [v - b for b, v in zip(b_test, v_test)]

    def _paired(values):
        """Mean, 95% CI and verdict flags for a list of per-seed deltas."""
        clean = [v for v in values if v == v]          # drop NaN (single-class slices)
        k = len(clean)
        if k == 0:
            return {'mean': float('nan'), 'ci95': (float('nan'),) * 2,
                    'measurable': False, 'practically_relevant': False, 'n_seeds': 0}
        m = st.mean(clean)
        sd = st.stdev(clean) if k > 1 else 0.0
        half = 1.96 * sd / (k ** 0.5) if k > 1 else float('nan')
        return {
            'mean': m,
            'ci95': (m - half, m + half) if half == half else (float('nan'),) * 2,
            'measurable': bool(half == half and abs(m) > half),
            'practically_relevant': abs(m) >= PRACTICAL_AUC_THRESHOLD,
            'n_seeds': k,
        }

    n = len(deltas)
    mean_d = st.mean(deltas)
    sd_d = st.stdev(deltas) if n > 1 else 0.0
    se_d = sd_d / (n ** 0.5) if n > 1 else float('nan')
    ci = 1.96 * se_d if se_d == se_d else float('nan')

    regime_stats = {}
    for name, d in per_regime.items():
        stats = _paired(d['delta'])
        b_clean = [v for v in d['baseline'] if v == v]
        v_clean = [v for v in d['variant'] if v == v]
        regime_stats[name] = {
            **stats,
            'n_bars': d['n_bars'],
            'pos_rate': d['pos_rate'],
            'operative': d['operative'],
            'baseline_auc': st.mean(b_clean) if b_clean else float('nan'),
            'variant_auc': st.mean(v_clean) if v_clean else float('nan'),
            'evaluable': bool(b_clean),
        }

    operative = next((k for k, v in regime_stats.items()
                      if v['operative'] and v['evaluable']), None)

    return {
        'operative_regime': operative,
        'label_mode': args.get('label_mode', 'static'),
        'regimes': regime_stats,
        'model': model,
        'n_seeds': n,
        'n_train_bars': int(train_mask.sum()),
        'n_test_bars': int(test_mask.sum()),
        'train_window': (str(train_start.date()), str(train_end.date())),
        'test_window': (str((train_end + pd.Timedelta(days=1)).date()), str(test_end.date())),
        'baseline_n_features': len(baseline_cols),
        'variant_n_features': len(variant_cols),
        'removed': [c for c in baseline_cols if c not in set(variant_cols)],
        'baseline_test_auc': st.mean(b_test),
        'baseline_test_std': st.stdev(b_test) if n > 1 else 0.0,
        'baseline_gap': st.mean(b_gap),
        'variant_test_auc': st.mean(v_test),
        'variant_test_std': st.stdev(v_test) if n > 1 else 0.0,
        'variant_gap': st.mean(v_gap),
        'delta_mean': mean_d,
        'delta_std': sd_d,
        'delta_ci95': (mean_d - ci, mean_d + ci) if ci == ci else (float('nan'),) * 2,
        'measurable': bool(ci == ci and abs(mean_d) > ci),
        'practically_relevant': abs(mean_d) >= PRACTICAL_AUC_THRESHOLD,
        'per_seed': rows,
    }


def print_report(res):
    tw, ew = res['train_window'], res['test_window']
    print(f"\n{'=' * 76}")
    print(f"FEATURE A/B — {res['model']}")
    print(f"{'=' * 76}")
    print(f"  train {tw[0]}..{tw[1]} ({res['n_train_bars']} bars)   "
          f"test {ew[0]}..{ew[1]} ({res['n_test_bars']} bars)")
    print(f"  {res['n_seeds']} paired seeds, "
          f"{res['baseline_n_features']} -> {res['variant_n_features']} features")
    print(f"  removed: {', '.join(res['removed'])}")

    print(f"\n  {'':10s} {'test AUC':>10s} {'std':>8s} {'overfit gap':>13s}")
    print(f"  {'baseline':10s} {res['baseline_test_auc']:10.4f} "
          f"{res['baseline_test_std']:8.4f} {res['baseline_gap']:13.4f}")
    print(f"  {'variant':10s} {res['variant_test_auc']:10.4f} "
          f"{res['variant_test_std']:8.4f} {res['variant_gap']:13.4f}")

    lo, hi = res['delta_ci95']
    print(f"\n  delta (variant - baseline), paired per seed:")
    print(f"    overall {res['delta_mean']:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

    regimes = res.get('regimes') or {}
    if regimes:
        print(f"\n  Per regime on the test window "
              f"(operative = the population positions are opened in):")
        print(f"    {'regime':18s} {'bars':>6s} {'pos%':>6s} {'baseline':>9s} "
              f"{'variant':>8s} {'delta':>8s} {'95% CI':>19s}")
        for name, r in regimes.items():
            mark = ' <<' if name == res.get('operative_regime') else ''
            if not r['evaluable']:
                # Under trend_only every range bar is a forced 0 — a single-class slice
                # has no AUC at all. Saying "n/a" is the honest report; a 0 would be read
                # as "bad" when it means "not defined".
                print(f"    {name:18s} {r['n_bars']:6d} {r['pos_rate']:6.1%} "
                      f"{'n/a':>9s} {'n/a':>8s} {'n/a':>8s} {'single-class':>19s}{mark}")
                continue
            clo, chi = r['ci95']
            print(f"    {name:18s} {r['n_bars']:6d} {r['pos_rate']:6.1%} "
                  f"{r['baseline_auc']:9.4f} {r['variant_auc']:8.4f} "
                  f"{r['mean']:+8.4f} [{clo:+.4f}, {chi:+.4f}]{mark}")

    print()
    op = res.get('operative_regime')
    verdict_on = regimes.get(op) if op else None
    if verdict_on is None:
        verdict_on = {'mean': res['delta_mean'], 'measurable': res['measurable'],
                      'practically_relevant': res['practically_relevant']}
        scope = "overall AUC (no regime labels available)"
    else:
        scope = f"{op} — label_mode={res['label_mode']}"

    print(f"  VERDICT scope: {scope}")
    if not verdict_on['measurable']:
        print(f"  -> no measurable difference — the interval spans zero.")
        print(f"     The change neither helps nor hurts where it matters.")
        print(f"     Decide on other grounds: fewer features is the safer default")
        print(f"     while the overfitting gap sits near {res['baseline_gap']:.2f}.")
    elif not verdict_on['practically_relevant']:
        print(f"  -> measurable but below {PRACTICAL_AUC_THRESHOLD} AUC — too small to act on.")
        print(f"     Seed choice and window choice move the result by more than this.")
    else:
        direction = 'improves' if verdict_on['mean'] > 0 else 'degrades'
        print(f"  -> the change {direction} AUC by {abs(verdict_on['mean']):.4f} in the "
              f"operative regime, outside noise.")
        print(f"     AUC has not tracked P&L in this project — confirm with a")
        print(f"     walk-forward before treating it as an earnings improvement.")

    # A change that buys its overall gain outside the traded population is worth nothing.
    if op and regimes.get(op) and regimes[op]['evaluable']:
        if res['delta_mean'] > 0 and regimes[op]['mean'] < 0:
            print(f"\n  WARNING: overall AUC improves ({res['delta_mean']:+.4f}) while the "
                  f"operative regime degrades ({regimes[op]['mean']:+.4f}).")
            print(f"           The gain sits in bars no position is ever opened in. Reject.")
    print(f"{'=' * 76}\n")


def main():
    p = argparse.ArgumentParser(
        description='Compare two feature sets on paired multi-seed test AUC.')
    p.add_argument('--run-id', required=True,
                   help='Run whose feature matrix and hyperparameters to reuse')
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--drop', default='',
                   help='Comma-separated features to remove from the baseline')
    p.add_argument('--keep-only', default=None,
                   help='Comma-separated features to keep (alternative to --drop)')
    p.add_argument('--seeds', type=int, default=8,
                   help='Number of paired seeds (default: 8)')
    p.add_argument('--train-end', default=None,
                   help="Override the run's train_end (test is everything after it)")
    p.add_argument('--test-end', default=None, help="Override the run's backtest_end")
    p.add_argument('--no-regimes', action='store_true',
                   help='Judge on overall AUC only, ignoring the regime split. Rarely '
                        'what you want: under trend_only the range bars are forced '
                        'negatives and inflate the aggregate.')
    p.add_argument('--json', default=None, help='Also write the result as JSON here')
    args = p.parse_args()

    split = lambda s: [x.strip() for x in s.replace(',', ' ').split() if x.strip()]
    drop = split(args.drop)
    keep_only = split(args.keep_only) if args.keep_only else None
    if not drop and keep_only is None:
        raise SystemExit('Give --drop or --keep-only.')

    X, y, run_args = load_run(args.run_id, args.model)
    regimes = None if args.no_regimes else load_regimes(args.run_id, X.index)
    if regimes is None and not args.no_regimes:
        print("Note: no regime_labels.parquet in this run — falling back to overall AUC.")
    res = compare(X, y, run_args, args.model, drop=drop, keep_only=keep_only,
                  seeds=args.seeds, train_end=args.train_end, test_end=args.test_end,
                  regimes=regimes)
    print_report(res)

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in res.items() if k != 'per_seed'}, f, indent=2)
        print(f"Written to {args.json}")


if __name__ == '__main__':
    main()
