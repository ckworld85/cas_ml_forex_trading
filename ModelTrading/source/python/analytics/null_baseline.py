"""
Null-Baseline / Leakage Check (standalone diagnostics)
======================================================

Answers one question per model: **is the CV AUC real, or an artifact?**

It reruns the *exact* cross-validation used in training (same
``run_time_series_cv`` — same TimeSeriesSplit folds, same
precision-at-target-recall threshold rule, same pooled metrics) twice:

  1. once on the real labels                       -> real AUC / F1
  2. many times on *permuted* labels (shuffled y)  -> null distribution

Interpretation:

  * shuffled AUC ~ 0.50           -> the harness is clean (no leakage)
      - real >> shuffled (p<0.05) -> REAL SIGNAL
      - real ~ shuffled           -> NO SIGNAL (model is at chance)
  * shuffled AUC clearly > 0.50   -> LEAKAGE SUSPECTED: the model can
      "predict" even randomised labels, which only happens when
      information about the label leaks into train/val (feature without
      shift(1), label horizon bleeding into the fold, forward-fill
      contamination, no purge gap, ...).

This script is READ-ONLY. It loads the artefacts training already
persisted (``X_{model}.parquet``, ``label_targets.parquet``,
``regime_analysis/regime_labels.parquet``) and never touches the training
code path — so it cannot regress training/backtest/live behaviour. It is a
dev-only diagnostic and is NOT staged to TEST/PRODUCTION.

Examples
--------
    # All 4 models, full population:
    python -m ModelTrading.source.python.analytics.null_baseline

    # Just the range regime (where fast showed AUC~0.95 — leakage suspect):
    python -m ModelTrading.source.python.analytics.null_baseline \
        --models long_fast --regime RANGE --n-shuffles 50

    # List available regimes and their sizes:
    python -m ModelTrading.source.python.analytics.null_baseline --list-regimes
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config
import ModelTrading.source.python.features.config as feature_config
from ModelTrading.source.python.advanced_train import run_time_series_cv


# Fallback per-model target files (used only if label_targets.parquet is absent).
# Note the quirky slow filenames are exactly what advanced_train.save writes.
_TARGET_FALLBACK = {
    'long_fast':  ('y_target_long_fast.parquet',       'target_long_fast'),
    'short_fast': ('y_target_short_fast.parquet',      'target_short_fast'),
    'long_slow':  ('y_target_long_long_slow.parquet',  'target_long_slow'),
    'short_slow': ('y_target_short_short_slow.parquet', 'target_short_slow'),
}


# --------------------------------------------------------------------------
# Data loading (read-only)
# --------------------------------------------------------------------------

def load_model_data(generated_dir, model_key):
    """Load the persisted (X, y) for one model, aligned on their index.

    Returns (X: DataFrame, y: Series[int]) with rows dropped where the label
    is NaN. Feature NaNs are kept — XGBoost handles them natively and no
    scaling is needed (trees are scale-invariant).
    """
    x_path = os.path.join(generated_dir, f"X_{model_key}.parquet")
    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"Missing {x_path}. Run advanced_train first so X_{model_key}.parquet exists.")
    X = pd.read_parquet(x_path)
    X.index = pd.to_datetime(X.index)

    y = _load_target(generated_dir, model_key)
    y.index = pd.to_datetime(y.index)

    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    return X, y


def _load_target(generated_dir, model_key):
    """Prefer the aligned label_targets.parquet; fall back to per-model file."""
    combined = os.path.join(generated_dir, "label_targets.parquet")
    col = f"target_{model_key}"
    if os.path.exists(combined):
        df = pd.read_parquet(combined)
        if col in df.columns:
            return df[col].copy()
    fname, fcol = _TARGET_FALLBACK[model_key]
    fpath = os.path.join(generated_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"No target found for '{model_key}' (looked for column '{col}' in "
            f"label_targets.parquet and file {fname}).")
    return pd.read_parquet(fpath)[fcol].copy()


def load_regime_series(generated_dir):
    """Load regime_combined (e.g. RANGE_HIGH_VOL) as a Series, or None."""
    path = os.path.join(generated_dir, "regime_analysis", "regime_labels.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if 'regime_combined' not in df.columns:
        return None
    s = df['regime_combined']
    s.index = pd.to_datetime(s.index)
    return s


def filter_by_regime(X, y, regime_series, regime_prefix):
    """Keep only rows whose regime_combined starts with `regime_prefix`.

    A prefix ('RANGE') selects the whole block; a full name ('RANGE_HIGH_VOL')
    selects exactly that regime. Temporal order is preserved, so downstream
    TimeSeriesSplit stays valid on the subset.
    """
    if regime_series is None:
        raise FileNotFoundError(
            "Regime filtering requested but regime_labels.parquet was not found "
            "(regime_analysis/regime_labels.parquet).")
    reg = regime_series.reindex(X.index)
    mask = reg.astype('object').fillna('').str.startswith(regime_prefix)
    return X.loc[mask.values], y.loc[mask.values], reg.loc[mask.values]


# --------------------------------------------------------------------------
# Core leakage / signal test
# --------------------------------------------------------------------------

def build_xgb_params(max_depth, eta, subsample, colsample_bytree):
    """Binary classifier params matching the project's classifier defaults."""
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'verbosity': 0,
    }


def _cv_auc_f1(X, y, params, n_splits, num_boost_round, target_recall, spw_factor):
    """Run one CV pass silently; return (global_auc, global_f1, shuffled_mean_pos)."""
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_time_series_cv(
            X, y, params,
            n_splits=n_splits,
            num_boost_round=num_boost_round,
            spw_factor=spw_factor,
            target_recall=target_recall,
        )
    return res['global_val_auc_roc'], res['global_val_f1']


def run_leakage_check(X, y, *, params, n_splits=5, num_boost_round=200,
                      target_recall=0.5, spw_factor=1.0, n_shuffles=20, seed=42):
    """Permutation test: compare real-label CV AUC/F1 against shuffled-label runs.

    Labels are permuted globally (marginal class balance preserved, every
    feature->label association destroyed). The p-value is the standard
    permutation-test estimate p = (1 + #{shuffled >= real}) / (1 + n_shuffles).

    Returns a dict of scalars suitable for printing/serialisation.
    """
    n_pos = int((y == 1).sum())
    real_auc, real_f1 = _cv_auc_f1(
        X, y, params, n_splits, num_boost_round, target_recall, spw_factor)

    rng = np.random.default_rng(seed)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    shuf_aucs, shuf_f1s = [], []
    for _ in range(n_shuffles):
        y_perm = rng.permutation(y_arr)
        a, f = _cv_auc_f1(
            X, y_perm, params, n_splits, num_boost_round, target_recall, spw_factor)
        shuf_aucs.append(a)
        shuf_f1s.append(f)

    shuf_aucs = np.array(shuf_aucs, dtype=float)
    shuf_f1s = np.array(shuf_f1s, dtype=float)
    valid_auc = shuf_aucs[~np.isnan(shuf_aucs)]

    if np.isnan(real_auc) or len(valid_auc) == 0:
        p_auc = float('nan')
        shuf_auc_mean = shuf_auc_std = float('nan')
    else:
        shuf_auc_mean = float(np.mean(valid_auc))
        shuf_auc_std = float(np.std(valid_auc))
        p_auc = (1 + int(np.sum(valid_auc >= real_auc))) / (1 + len(valid_auc))

    return {
        'n_rows': int(len(y)),
        'n_pos': n_pos,
        'pos_rate': (n_pos / len(y)) if len(y) else float('nan'),
        'real_auc': float(real_auc) if real_auc is not None else float('nan'),
        'real_f1': float(real_f1),
        'shuf_auc_mean': shuf_auc_mean,
        'shuf_auc_std': shuf_auc_std,
        'shuf_f1_mean': float(np.nanmean(shuf_f1s)) if len(shuf_f1s) else float('nan'),
        'n_shuffles': int(len(valid_auc)),
        'p_value_auc': p_auc,
        'delta_auc': (float(real_auc) - shuf_auc_mean)
                     if not (np.isnan(real_auc) or np.isnan(shuf_auc_mean)) else float('nan'),
        'verdict': _verdict(real_auc, shuf_auc_mean, p_auc),
    }


def _verdict(real_auc, shuf_auc_mean, p_auc, leak_thr=0.55, sig_delta=0.02, alpha=0.05):
    """Classify the outcome into LEAKAGE / REAL SIGNAL / NO SIGNAL / N/A."""
    if np.isnan(real_auc) or np.isnan(shuf_auc_mean):
        return "N/A (single-class fold)"
    if shuf_auc_mean > leak_thr:
        return f"LEAKAGE SUSPECTED (shuffled AUC {shuf_auc_mean:.3f} >> 0.5)"
    delta = real_auc - shuf_auc_mean
    if p_auc < alpha and delta > sig_delta:
        return "REAL SIGNAL"
    return "NO SIGNAL (indistinguishable from noise)"


# --------------------------------------------------------------------------
# Ablation: localize the edge to individual features
# --------------------------------------------------------------------------

def run_ablation(X, y, *, params, n_splits=5, num_boost_round=200,
                 target_recall=0.5, spw_factor=1.0, do_loo=True, progress=None):
    """Localize predictive power to individual features.

    For each feature computes:
      * solo_auc  — CV AUC training on THAT feature alone. A leak concentrates:
                    a single feature scoring ~0.9 is almost certainly peeking.
      * loo_auc   — CV AUC training on all features EXCEPT that one.
      * loo_delta — baseline_auc - loo_auc (how much AUC drops without it). High
                    positive delta = the feature is load-bearing. Weaker than
                    solo_auc under feature correlation (a twin substitutes).

    Returns (baseline_auc, feature_rows) where feature_rows is a list of dicts
    sorted by solo_auc descending (prime suspect first). ``do_loo=False`` skips
    the leave-one-out pass (halves runtime). ``progress`` is an optional
    callback(feature_name, i, total).
    """
    cols = list(X.columns)
    baseline_auc, _ = _cv_auc_f1(
        X, y, params, n_splits, num_boost_round, target_recall, spw_factor)

    rows = []
    for i, col in enumerate(cols):
        if progress:
            progress(col, i, len(cols))
        solo_auc, _ = _cv_auc_f1(
            X[[col]], y, params, n_splits, num_boost_round, target_recall, spw_factor)
        if do_loo:
            loo_auc, _ = _cv_auc_f1(
                X.drop(columns=[col]), y, params, n_splits, num_boost_round,
                target_recall, spw_factor)
            loo_delta = (baseline_auc - loo_auc
                         if not (np.isnan(baseline_auc) or np.isnan(loo_auc)) else float('nan'))
        else:
            loo_auc = loo_delta = float('nan')
        rows.append({
            'feature': col,
            'solo_auc': float(solo_auc) if not np.isnan(solo_auc) else float('nan'),
            'loo_auc': float(loo_auc) if not np.isnan(loo_auc) else float('nan'),
            'loo_delta': float(loo_delta) if not np.isnan(loo_delta) else float('nan'),
        })

    rows.sort(key=lambda r: (-1.0 if np.isnan(r['solo_auc']) else r['solo_auc']), reverse=True)
    return (float(baseline_auc) if not np.isnan(baseline_auc) else float('nan')), rows


def _ablation_flag(solo_auc, solo_leak_thr=0.75):
    """Flag a feature whose solo AUC alone is high enough to indicate a leak."""
    if np.isnan(solo_auc):
        return ""
    if solo_auc >= solo_leak_thr:
        return "  <== LEAK SUSPECT"
    return ""


def print_ablation_report(model_key, scope_desc, baseline_auc, feature_rows, solo_leak_thr=0.75):
    """Pretty-print the per-feature ablation table."""
    print("\n" + "=" * 100)
    print(f"FEATURE ABLATION  (model={model_key}, {scope_desc})")
    print("=" * 100)
    base_s = f"{baseline_auc:.3f}" if not np.isnan(baseline_auc) else "N/A"
    print(f"  Baseline AUC (all {len(feature_rows)} features): {base_s}")
    print(f"  {'Feature':<32} {'soloAUC':>8} {'looAUC':>8} {'dLOO':>8}")
    print("  " + "-" * 62)
    for r in feature_rows:
        solo = f"{r['solo_auc']:.3f}" if not np.isnan(r['solo_auc']) else "  N/A"
        loo = f"{r['loo_auc']:.3f}" if not np.isnan(r['loo_auc']) else "  N/A"
        dloo = f"{r['loo_delta']:+.3f}" if not np.isnan(r['loo_delta']) else "   N/A"
        print(f"  {r['feature']:<32} {solo:>8} {loo:>8} {dloo:>8}"
              f"{_ablation_flag(r['solo_auc'], solo_leak_thr)}")
    print("  " + "-" * 62)
    print(f"  soloAUC >= {solo_leak_thr:.2f} on ONE feature => likely lookahead leak; "
          f"inspect its shift(1)/window in indicators.py.")
    print("  A genuine edge is diffuse (no single feature dominates soloAUC).\n")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_report(rows, scope_desc):
    """Pretty-print the per-model leakage-check table."""
    print("\n" + "=" * 100)
    print(f"NULL-BASELINE / LEAKAGE CHECK  ({scope_desc})")
    print("=" * 100)
    print(f"  {'Model':<12} {'N':>8} {'Pos%':>6} {'realAUC':>8} {'shufAUC':>16} "
          f"{'dAUC':>7} {'p':>6}  Verdict")
    print("  " + "-" * 96)
    for mk, r in rows:
        real = f"{r['real_auc']:.3f}" if not np.isnan(r['real_auc']) else " N/A "
        if not np.isnan(r['shuf_auc_mean']):
            shuf = f"{r['shuf_auc_mean']:.3f}+/-{r['shuf_auc_std']:.3f}"
        else:
            shuf = "N/A"
        d = f"{r['delta_auc']:+.3f}" if not np.isnan(r['delta_auc']) else "  N/A"
        p = f"{r['p_value_auc']:.3f}" if not np.isnan(r['p_value_auc']) else " N/A"
        print(f"  {mk:<12} {r['n_rows']:>8} {r['pos_rate']:>6.1%} {real:>8} {shuf:>16} "
              f"{d:>7} {p:>6}  {r['verdict']}")
    print("  " + "-" * 96)
    print("  shufAUC ~0.5 => clean harness; >>0.5 => leakage. p<0.05 & dAUC>0 => real signal.\n")


def list_regimes(regime_series):
    if regime_series is None:
        print("No regime_labels.parquet found (regime_analysis/regime_labels.parquet).")
        return
    counts = regime_series.value_counts()
    total = int(counts.sum())
    print("\nAvailable regimes (regime_combined):")
    print(f"  {'Regime':<22} {'N':>8} {'Share':>7}")
    print("  " + "-" * 40)
    for name, n in counts.items():
        print(f"  {str(name):<22} {int(n):>8} {n / total:>7.1%}")
    print(f"  {'TOTAL':<22} {total:>8} {1.0:>7.1%}\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Shuffled-label leakage / null-baseline check for the CV AUC.")
    p.add_argument('--run-id', default=None,
                   help='Training run id (subfolder of generated/). Default: generated/ root.')
    p.add_argument('--generated-dir', default=None,
                   help='Explicit path to the generated dir (overrides --run-id).')
    p.add_argument('--models', default=None,
                   help=f'Comma-separated model keys. Default: all ({",".join(feature_config.MODELS)}).')
    p.add_argument('--regime', default=None,
                   help='Restrict to a regime_combined prefix, e.g. RANGE or RANGE_HIGH_VOL.')
    p.add_argument('--list-regimes', action='store_true',
                   help='Print available regimes with sizes, then exit.')
    p.add_argument('--ablate', action='store_true',
                   help='Feature-ablation mode: report per-feature solo AUC + leave-one-out '
                        'ΔAUC to localize the edge/leak (instead of the shuffled-label check).')
    p.add_argument('--no-loo', action='store_true',
                   help='In --ablate, skip the leave-one-out pass (solo AUC only; ~half runtime).')
    p.add_argument('--solo-leak-thr', type=float, default=0.75,
                   help='In --ablate, solo AUC at/above this flags a leak suspect (default 0.75).')
    p.add_argument('--n-shuffles', type=int, default=20,
                   help='Number of label permutations for the null distribution (default 20).')
    p.add_argument('--n-splits', type=int, default=5, help='TimeSeriesSplit folds (default 5).')
    p.add_argument('--target-recall', type=float, default=0.5,
                   help='Recall floor for the threshold rule (default 0.5, matches training).')
    p.add_argument('--spw-factor', type=float, default=1.0,
                   help='scale_pos_weight multiplier (default 1.0).')
    p.add_argument('--num-boost-round', type=int, default=200, help='XGBoost rounds (default 200).')
    p.add_argument('--max-depth', type=int, default=6, help='XGBoost max_depth (default 6).')
    p.add_argument('--eta', type=float, default=0.1, help='XGBoost eta (default 0.1).')
    p.add_argument('--subsample', type=float, default=1.0, help='XGBoost subsample (default 1.0).')
    p.add_argument('--colsample-bytree', type=float, default=1.0,
                   help='XGBoost colsample_bytree (default 1.0).')
    p.add_argument('--seed', type=int, default=42, help='RNG seed for permutations (default 42).')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.generated_dir:
        generated_dir = args.generated_dir
    else:
        generated_dir = dir_config.get_run_dirs(args.run_id)['generated_dir']

    regime_series = load_regime_series(generated_dir)

    if args.list_regimes:
        list_regimes(regime_series)
        return 0

    models = ([m.strip() for m in args.models.split(',') if m.strip()]
              if args.models else list(feature_config.MODELS))
    params = build_xgb_params(args.max_depth, args.eta, args.subsample, args.colsample_bytree)
    scope_desc = f"regime={args.regime}" if args.regime else "full population"

    rows = []
    for mk in models:
        X, y = load_model_data(generated_dir, mk)
        if args.regime:
            X, y, _ = filter_by_regime(X, y, regime_series, args.regime)
        if len(y) == 0 or int((y == 1).sum()) == 0:
            print(f"  [skip] {mk}: no positive labels in scope (n={len(y)}).")
            continue

        if args.ablate:
            n_pass = len(X.columns) * (1 if args.no_loo else 2) + 1
            print(f"  Ablating {mk}: {len(y):,} rows, {len(X.columns)} features "
                  f"(~{n_pass} CV passes) ...")
            def _progress(col, i, total):
                print(f"    [{i + 1}/{total}] {col}")
            baseline_auc, feature_rows = run_ablation(
                X, y, params=params, n_splits=args.n_splits,
                num_boost_round=args.num_boost_round, target_recall=args.target_recall,
                spw_factor=args.spw_factor, do_loo=not args.no_loo, progress=_progress)
            print_ablation_report(mk, scope_desc, baseline_auc, feature_rows, args.solo_leak_thr)
            continue

        print(f"  Running {mk}: {len(y):,} rows, {int((y == 1).sum()):,} positives, "
              f"{args.n_shuffles} shuffles ...")
        r = run_leakage_check(
            X, y, params=params, n_splits=args.n_splits,
            num_boost_round=args.num_boost_round, target_recall=args.target_recall,
            spw_factor=args.spw_factor, n_shuffles=args.n_shuffles, seed=args.seed)
        rows.append((mk, r))

    if rows:
        print_report(rows, scope_desc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
