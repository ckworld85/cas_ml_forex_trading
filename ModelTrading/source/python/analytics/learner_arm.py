"""Does the model class matter? A like-for-like comparison of learners.

The information audit already answered this **model-free** at the feature level:
``--mv-learners xgb,rf,rf_leaf`` on purged folds against a block-shift null, recorded in
``docs/learner_robustness_results.md`` — no RF cell beat its null, and every delta
against the booster fell inside the ±0.017 that reseeding alone produces. That is a
statement about information.

This module answers the *trading* version of the same question: on the identical design
matrices, labels and folds the production pipeline uses, does another model class score
better, and does it trade better? Two things make the comparison fair:

* **It does not touch the training pipeline.** XGBoost is wired into
  ``advanced_train.py`` at four places and hangs on calibration, the ONNX export and the
  Java path. Instead this consumes the artefacts a run already wrote —
  ``X_{model}.parquet``, ``y_target_*.parquet`` and ``label_targets.parquet`` (for the
  ``t1_*`` barrier-resolution bars) — and trains every learner on the same purged,
  embargoed splits. Identical features, identical labels, identical folds; zero risk to
  the deployment path.
* **P&L goes through the identical execution engine.** Each learner's out-of-fold
  probabilities are written as a parquet and scored with ``backtest.py --proba-file``.
  Comparing a different model *and* a different backtest at once would measure neither.

The learner set is chosen so a null result means something. ``dummy`` (predict the base
rate) and ``logistic`` are the floor: if the booster does not clearly beat them, the
extra model class is not buying anything. ``rf_leaf`` raises ``min_samples_leaf`` toward
n/n_eff because bagging assumes independent draws and up to 24 consecutive rows share one
barrier outcome — measured on ``long_slow``: n = 2,347, n_eff = 467.

Usage::

    python analytics/learner_arm.py --run-dir ../../generated/<run_id> \\
        --docs-root ../../../docs --stage S4
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

_SOURCE_PYTHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SOURCE_PYTHON)))
for _p in (_SOURCE_PYTHON, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analytics import figures as fg  # noqa: E402  (fixes the Agg backend on import)
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ('long_fast', 'short_fast', 'long_slow', 'short_slow')

# The AUC movement reseeding alone produces, measured in the A2 learner-robustness run.
# A difference smaller than this is not a difference.
SEED_NOISE_AUC = 0.017


# --- Learners ------------------------------------------------------------------------

def build_learner(name, n_rows=None, n_eff=None, seed=42):
    """Instantiate one learner. ``n_eff`` shapes ``rf_leaf``'s leaf size.

    Every learner is wrapped so it exposes ``fit`` / ``predict_proba`` — including the
    booster, which is driven through its sklearn API here rather than the low-level one
    so all arms share exactly the same code path.
    """
    if name == 'xgb':
        import xgboost as xgb
        return xgb.XGBClassifier(
            max_depth=3, learning_rate=0.05, n_estimators=200, subsample=0.7,
            colsample_bytree=0.6, min_child_weight=3, reg_lambda=5.0,
            objective='binary:logistic', eval_metric='logloss',
            random_state=seed, n_jobs=1, verbosity=0)
    if name == 'rf':
        # Out of the box: unlimited depth, sqrt features. The configuration the
        # "an RF is hard to over-optimise" argument actually describes.
        return RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                      random_state=seed, n_jobs=1)
    if name == 'rf_leaf':
        # Bagging assumes independent bootstrap draws; overlapping barrier labels break
        # that. Sizing the leaves by the overlap factor is the version that would
        # actually be deployed.
        leaf = 1
        if n_rows and n_eff and n_eff > 0:
            leaf = max(1, int(round(n_rows / n_eff)))
        return RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                      min_samples_leaf=leaf, random_state=seed, n_jobs=1)
    if name == 'logistic':
        return LogisticRegression(penalty='l2', C=1.0, max_iter=2000,
                                  class_weight='balanced', random_state=seed)
    if name == 'dummy':
        # Predicts the training base rate for every bar: AUC 0.5 by construction. The
        # floor any claim of skill has to clear.
        return DummyClassifier(strategy='prior', random_state=seed)
    raise ValueError('unknown learner: ' + str(name))


DEFAULT_LEARNERS = ('xgb', 'rf', 'rf_leaf', 'logistic', 'dummy')


# --- Run artefacts -------------------------------------------------------------------

def load_design(run_dir, model):
    """The exported design matrix and target for one model. Index = bar timestamps."""
    x_path = os.path.join(run_dir, 'X_' + model + '.parquet')
    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path + ' not found — the run did not export its '
                                         'design matrices.')
    X = pd.read_parquet(x_path)
    # The slow targets carry a doubled name in the export (y_target_long_long_slow).
    direction = model.split('_')[0]
    candidates = ['y_target_' + direction + '_' + model + '.parquet',
                  'y_target_' + model + '.parquet']
    for cand in candidates:
        y_path = os.path.join(run_dir, cand)
        if os.path.exists(y_path):
            y = pd.read_parquet(y_path)
            return X, y[y.columns[0]]
    raise FileNotFoundError('no target parquet for ' + model + ' in ' + run_dir
                            + ' (tried ' + ', '.join(candidates) + ')')


def load_uniqueness(run_dir, model, index):
    """Rows and effective sample size for one model, from the label metadata.

    ``n_eff`` is what ``rf_leaf`` is sized by, and it is also the honest denominator for
    any standard error over these rows.
    """
    path = os.path.join(run_dir, 'label_targets.parquet')
    if not os.path.exists(path):
        return len(index), None
    try:
        from training import sample_weights as sw
        meta = pd.read_parquet(path)
        t1 = sw.t1_from_metadata(meta, model, index=index, fallback_horizon=None)
        if t1 is None:
            return len(index), None
        return len(t1), float(sw.average_uniqueness(t1).sum())
    except Exception:
        return len(index), None


def cadence_of(model):
    return 'fast' if model.endswith('_fast') else 'slow'


def to_training_cadence(X, y, model, label_agg='first'):
    """Put an exported M15 frame on the cadence the model is actually trained at.

    ``X_{model}.parquet`` is exported at M15 because the backtest evaluates on M15 bars,
    but the slow models train on 4h rows — ``X_train[cols].resample('4h').first()`` in
    ``advanced_train``. Training a comparison arm on the M15 export would give the slow
    learners 16x the rows, each a near-duplicate of its neighbours, and would not be the
    same problem the production model solves. This mirrors the pipeline exactly.

    ``label_agg`` is 'max' under window_cascade, where setup labels are sparse M15 bars
    and ``.first()`` would drop any 4h bar whose opening M15 is unlabelled.
    """
    if cadence_of(model) == 'fast':
        return X, y
    Xr = X.resample('4h').first().dropna(how='all')
    yr = getattr(y.resample('4h'), label_agg)().reindex(Xr.index, fill_value=0)
    return Xr, yr


def drop_unusable_rows(X, y):
    """Complete-case filter.

    XGBoost consumes NaN natively; logistic regression and the forests do not. Imputing
    would introduce a difference between the arms that has nothing to do with the model
    class, so every arm — the booster included — is given the same complete rows.
    """
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask].astype(int)


# --- Evaluation ----------------------------------------------------------------------

def evaluate_learner(X, y, learner_name, n_splits=5, gap=0, n_rows=None, n_eff=None,
                     seed=42):
    """Out-of-fold probabilities and per-fold metrics for one learner on one model.

    The splits are ``TimeSeriesSplit(gap=...)`` — the same purged, embargoed geometry the
    production CV uses, so the arms differ only in the model class.
    """
    X_arr = X.to_numpy(dtype=float)
    y_arr = np.asarray(y, dtype=int)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    oof = np.full(len(y_arr), np.nan)
    folds = np.zeros(len(y_arr), dtype=int)
    rows = []
    for fold, (tr, va) in enumerate(tscv.split(X_arr), 1):
        if len(np.unique(y_arr[tr])) < 2:
            rows.append({'fold': fold, 'n_val': len(va), 'auc': np.nan,
                         'avg_precision': np.nan, 'brier': np.nan,
                         'degenerate': True})
            continue
        # Scaling matters for logistic regression and is harmless for the tree models.
        scaler = StandardScaler().fit(X_arr[tr])
        model = build_learner(learner_name, n_rows=n_rows, n_eff=n_eff, seed=seed)
        model.fit(scaler.transform(X_arr[tr]), y_arr[tr])
        p = model.predict_proba(scaler.transform(X_arr[va]))[:, 1]
        oof[va] = p
        folds[va] = fold

        both = len(np.unique(y_arr[va])) > 1
        rows.append({
            'fold': fold, 'n_val': len(va),
            'auc': float(roc_auc_score(y_arr[va], p)) if both else np.nan,
            'avg_precision': (float(average_precision_score(y_arr[va], p))
                              if both else np.nan),
            'brier': float(brier_score_loss(y_arr[va], p)),
            'degenerate': bool(np.ptp(p) < 1e-12),
        })

    mask = ~np.isnan(oof)
    pooled_auc = (float(roc_auc_score(y_arr[mask], oof[mask]))
                  if mask.any() and len(np.unique(y_arr[mask])) > 1 else np.nan)
    return {
        'oof': pd.Series(oof, index=X.index),
        'folds': pd.Series(folds, index=X.index),
        'per_fold': pd.DataFrame(rows),
        'pooled_auc': pooled_auc,
        'pooled_brier': (float(brier_score_loss(y_arr[mask], oof[mask]))
                         if mask.any() else np.nan),
        'pooled_avg_precision': (float(average_precision_score(y_arr[mask], oof[mask]))
                                 if mask.any() and len(np.unique(y_arr[mask])) > 1
                                 else np.nan),
        'n_scored': int(mask.sum()),
    }


def run_arm(run_dir, learners=DEFAULT_LEARNERS, models=MODELS, n_splits=5, gap=0,
            seed=42, label_agg='first', verbose=True):
    """Every learner on every model. Returns (metrics frame, per-fold frame, oof frame)."""
    metric_rows, fold_rows = [], []
    oof_by_learner = {ln: {} for ln in learners}

    for model in models:
        X, y = load_design(run_dir, model)
        y = y.reindex(X.index)
        X, y = to_training_cadence(X, y, model, label_agg=label_agg)
        n_raw = len(X)
        X, y = drop_unusable_rows(X, y)
        n_rows, n_eff = load_uniqueness(run_dir, model, X.index)
        if verbose:
            eff = 'n/a' if n_eff is None else format(n_eff, '.0f')
            dropped = ('' if n_raw == len(X)
                       else ' (' + str(n_raw - len(X)) + ' incomplete rows dropped)')
            print('  ' + model + ' [' + cadence_of(model) + ']: ' + str(len(X))
                  + ' rows, ' + str(X.shape[1]) + ' features, n_eff ' + eff + dropped)
        if len(X) < n_splits * 2 or y.nunique() < 2:
            if verbose:
                print('    skipped — too few usable rows or a single class')
            continue

        for learner in learners:
            res = evaluate_learner(X, y, learner, n_splits=n_splits, gap=gap,
                                   n_rows=n_rows, n_eff=n_eff, seed=seed)
            oof_by_learner[learner][model] = res['oof']
            metric_rows.append({
                'model': model, 'learner': learner,
                'n': len(X), 'n_eff': n_eff, 'n_features': X.shape[1],
                'base_rate': float(y.mean()),
                'auc': res['pooled_auc'],
                'avg_precision': res['pooled_avg_precision'],
                'brier': res['pooled_brier'],
                'n_scored': res['n_scored'],
                'n_degenerate_folds': int(res['per_fold']['degenerate'].sum()),
            })
            pf = res['per_fold'].copy()
            pf['model'], pf['learner'] = model, learner
            fold_rows.append(pf)
            if verbose:
                auc = res['pooled_auc']
                print('    ' + learner.ljust(9) + ' AUC '
                      + ('  n/a' if np.isnan(auc) else format(auc, '.4f')))

    metrics = pd.DataFrame(metric_rows)
    per_fold = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    oof = {ln: pd.DataFrame(cols) for ln, cols in oof_by_learner.items()}
    return metrics, per_fold, oof


def paired_deltas(per_fold, reference='xgb'):
    """Per-fold AUC difference against the reference learner, on the same folds.

    Paired, because the folds differ enormously in difficulty: an unpaired comparison of
    means is dominated by which folds a learner happened to be scored on.
    """
    if per_fold.empty:
        return pd.DataFrame()
    wide = per_fold.pivot_table(index=['model', 'fold'], columns='learner', values='auc')
    if reference not in wide.columns:
        return pd.DataFrame()
    rows = []
    for learner in wide.columns:
        if learner == reference:
            continue
        d = (wide[learner] - wide[reference]).dropna()
        if d.empty:
            continue
        rows.append({
            'learner': learner, 'n_folds': int(len(d)),
            'mean_delta_auc': float(d.mean()),
            'std_delta_auc': float(d.std(ddof=1)) if len(d) > 1 else np.nan,
            'min_delta_auc': float(d.min()), 'max_delta_auc': float(d.max()),
            'folds_better': int((d > 0).sum()),
            # The A2 decision rule: beating the reference by less than the movement
            # reseeding alone produces is not beating it.
            'exceeds_seed_noise': bool(d.mean() > SEED_NOISE_AUC),
        })
    return pd.DataFrame(rows).sort_values('mean_delta_auc', ascending=False)


def write_proba_files(oof, out_dir, suffix='_oof'):
    """One parquet per learner.

    ``suffix='_oof'`` marks these as **out-of-fold** scores, which live inside the
    TRAINING window. They are the right input for AUC, precision/recall and calibration
    — and the wrong input for a backtest, whose window lies after training. Use
    :func:`deployment_probabilities` for P&L.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for learner, frame in oof.items():
        if frame.empty:
            continue
        # Uncovered bars stay NaN here; backtest.py floors them to 0.0, which is below
        # every entry threshold, so an unscored bar simply never trades.
        path = os.path.join(out_dir, 'proba_' + learner + suffix + '.parquet')
        frame.to_parquet(path)
        written[learner] = path
    return written


def training_window(run_dir):
    """(train_start, train_end) of the run, from its own summary. (None, None) if absent."""
    path = os.path.join(run_dir, 'training_summary.json')
    if not os.path.exists(path):
        return None, None
    with open(path, encoding='utf-8') as fh:
        summary = json.load(fh)
    return (pd.to_datetime(summary.get('train_start'), errors='coerce'),
            pd.to_datetime(summary.get('train_end'), errors='coerce'))


def deployment_probabilities(run_dir, learners=DEFAULT_LEARNERS, models=MODELS,
                             train_start=None, train_end=None, seed=42,
                             label_agg='first', verbose=True):
    """Fit on the training window, predict over the whole exported index.

    This is what a backtest needs, and it is a different artefact from the out-of-fold
    scores. Out-of-fold predictions exist only inside the training window — measured on
    run oof_smoke, they covered 4.7 % of the backtest frame's bars — so scoring a
    backtest with them would evaluate a handful of stray bars and call it a strategy.

    The production asymmetry is reproduced exactly: a slow model is **fitted on 4h rows**
    and **predicts on M15 bars**, because that is what ``backtest.py`` does when it runs
    ``bst.predict`` over the M15 design matrix.
    """
    if train_start is None or train_end is None:
        ts, te = training_window(run_dir)
        train_start = train_start if train_start is not None else ts
        train_end = train_end if train_end is not None else te

    out = {ln: {} for ln in learners}
    for model in models:
        X_all, y_all = load_design(run_dir, model)
        y_all = y_all.reindex(X_all.index)
        X_fit, y_fit = to_training_cadence(X_all, y_all, model, label_agg=label_agg)
        if train_start is not None and pd.notna(train_start):
            X_fit = X_fit.loc[X_fit.index >= train_start]
        if train_end is not None and pd.notna(train_end):
            X_fit = X_fit.loc[X_fit.index <= train_end]
        y_fit = y_fit.reindex(X_fit.index)
        X_fit, y_fit = drop_unusable_rows(X_fit, y_fit)
        if len(X_fit) < 20 or y_fit.nunique() < 2:
            if verbose:
                print('  ' + model + ': not enough training rows to fit — skipped')
            continue

        n_rows, n_eff = load_uniqueness(run_dir, model, X_fit.index)
        scaler = StandardScaler().fit(X_fit.to_numpy(dtype=float))
        # Predict on every exported bar. Rows with a missing feature cannot be scored by
        # the sklearn learners, so they stay NaN and simply never trade.
        predictable = X_all.notna().all(axis=1)
        X_pred = scaler.transform(X_all.loc[predictable].to_numpy(dtype=float))

        for learner in learners:
            m = build_learner(learner, n_rows=n_rows, n_eff=n_eff, seed=seed)
            m.fit(scaler.transform(X_fit.to_numpy(dtype=float)), y_fit.to_numpy())
            p = pd.Series(np.nan, index=X_all.index, dtype=float)
            p.loc[predictable] = m.predict_proba(X_pred)[:, 1]
            out[learner][model] = p
        if verbose:
            print('  ' + model + ': fitted on ' + str(len(X_fit)) + ' '
                  + cadence_of(model) + ' rows, scored '
                  + str(int(predictable.sum())) + '/' + str(len(X_all)) + ' bars')

    return {ln: pd.DataFrame(cols) for ln, cols in out.items() if cols}


# --- Figures --------------------------------------------------------------------------

def figure_auc_by_learner(metrics, writer, name='learner_auc'):
    """Pooled out-of-fold AUC, grouped by model, one bar per learner."""
    if metrics.empty:
        return None
    models = [m for m in MODELS if m in set(metrics['model'])]
    learners = list(dict.fromkeys(metrics['learner']))
    xs = np.arange(len(models))
    width = 0.8 / max(len(learners), 1)

    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_FULL, 0.38))
    for i, learner in enumerate(learners):
        vals = [float(metrics[(metrics['model'] == m) &
                              (metrics['learner'] == learner)]['auc'].mean())
                for m in models]
        ax.bar(xs + (i - (len(learners) - 1) / 2) * width, vals, width * 0.9,
               color=fg.CATEGORICAL[i % len(fg.CATEGORICAL)], label=learner, zorder=2)
    ax.axhline(0.5, color=fg.INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xticks(xs, models)
    ax.set_ylabel('Pooled out-of-fold AUC')
    ax.set_ylim(0.35, max(0.75, float(np.nanmax(metrics['auc'])) + 0.05))
    ax.set_title('Model class against the same features, labels and folds '
                 '(0.5 = chance)')
    fg.tidy(ax)
    ax.legend(loc='upper right', ncol=min(len(learners), 5))
    return writer.save_figure(fig, name,
                              caption='Pooled out-of-fold AUC per learner and model.')


def figure_paired_deltas(deltas, writer, name='learner_deltas', reference='xgb'):
    """Mean paired AUC delta against the reference, with the seed-noise band drawn.

    The band is what makes the figure decidable: a bar inside it is not a difference,
    however consistently positive it looks.
    """
    if deltas.empty:
        return None
    d = deltas.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_FULL, 0.3))
    ax.axvspan(-SEED_NOISE_AUC, SEED_NOISE_AUC, color=fg.INK_MUTED, alpha=0.16,
               zorder=1, label='±' + format(SEED_NOISE_AUC, '.3f') + ' seed noise')
    for i, row in d.iterrows():
        colour = (fg.CATEGORICAL[2] if row['exceeds_seed_noise'] else fg.INK_MUTED)
        ax.barh(i, row['mean_delta_auc'], 0.55, color=colour, zorder=2)
        ax.text(row['mean_delta_auc'], i, ' ' + format(row['mean_delta_auc'], '+.4f'),
                va='center', fontsize=6.5, color=fg.INK_SECONDARY)
    ax.axvline(0, color=fg.BASELINE, linewidth=0.9, zorder=1)
    ax.set_yticks(range(len(d)), [str(r) + '  (' + str(int(n)) + ' folds)'
                                  for r, n in zip(d['learner'], d['n_folds'])])
    ax.set_xlabel('Mean paired AUC difference vs. ' + reference)
    ax.set_title('Does another model class beat the booster on the same folds?')
    fg.tidy(ax, grid_axis='x')
    ax.legend(loc='lower right')
    return writer.save_figure(
        fig, name,
        caption='Paired per-fold AUC difference against ' + reference +
                '; the shaded band is the movement reseeding alone produces.')


# --- Entry point ----------------------------------------------------------------------

def run(run_dir, docs_root, stage_id='S4', learners=DEFAULT_LEARNERS, n_splits=5,
        gap=0, seed=42, proba_dir=None, label_agg='first'):
    writer = fg.ArtefactWriter(stage_id, docs_root)
    print('Learner arm on ' + os.path.abspath(run_dir))
    metrics, per_fold, oof = run_arm(run_dir, learners=learners, n_splits=n_splits,
                                     gap=gap, seed=seed, label_agg=label_agg)
    deltas = paired_deltas(per_fold)

    writer.save_table(metrics, 'learner_metrics',
                      caption='Pooled out-of-fold quality per learner and model, on '
                              'identical features, labels and folds.')
    if not deltas.empty:
        writer.save_table(deltas, 'learner_deltas',
                          caption='Paired per-fold AUC difference against the booster.')
    if not per_fold.empty:
        writer.save_table(per_fold, 'learner_per_fold',
                          caption='Per-fold detail behind the pooled figures.')

    figure_auc_by_learner(metrics, writer)
    figure_paired_deltas(deltas, writer)

    proba_dir = proba_dir or os.path.join(run_dir, 'learner_arm')
    written = write_proba_files(oof, proba_dir, suffix='_oof')
    # The out-of-fold scores above are for AUC / precision-recall / calibration. A
    # backtest needs models fitted on the training window and applied to the bars that
    # follow it — a different artefact, produced here.
    print('\nFitting deployment models (training window -> full index):')
    deploy = deployment_probabilities(run_dir, learners=learners, seed=seed,
                                      label_agg=label_agg)
    written_deploy = write_proba_files(deploy, proba_dir, suffix='_deploy')

    writer.save_json({'run_dir': os.path.abspath(run_dir),
                      'learners': list(learners),
                      'n_splits': n_splits, 'cv_gap': gap, 'seed': seed,
                      'label_agg': label_agg,
                      'seed_noise_auc': SEED_NOISE_AUC,
                      'proba_files_oof': written,
                      'proba_files_deploy': written_deploy,
                      'backtest_hint': ('python backtest.py --run-id <id> --proba-file '
                                        '<proba_<learner>_deploy.parquet>. Use the '
                                        '_deploy files, never the _oof ones: '
                                        'out-of-fold scores live inside the TRAINING '
                                        'window and cover almost none of a backtest '
                                        '(measured on run oof_smoke: 4.7 % of bars).'),
                      'artifacts': writer.artifacts}, 'learner_arm')
    return writer, metrics, deltas, written, written_deploy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='a run directory that exported X_*.parquet and y_target_*')
    ap.add_argument('--docs-root', default=None)
    ap.add_argument('--stage', default='S4')
    ap.add_argument('--learners', default=','.join(DEFAULT_LEARNERS),
                    help='comma-separated: ' + ', '.join(DEFAULT_LEARNERS))
    ap.add_argument('--cv-splits', type=int, default=5)
    ap.add_argument('--cv-gap', type=int, default=0,
                    help='embargo rows between each training fold and its validation '
                         'fold. Use the same value the training run used.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label-agg', default='first', choices=['first', 'max'],
                    help="how M15 labels collapse to 4h rows for the slow models. "
                         "Use 'max' for --label-mode window_cascade, whose setup "
                         "labels are sparse M15 bars; 'first' otherwise, matching "
                         "advanced_train.")
    ap.add_argument('--proba-dir', default=None,
                    help='where to write proba_<learner>.parquet '
                         '(default: <run-dir>/learner_arm)')
    args = ap.parse_args(argv)

    docs_root = args.docs_root or os.path.join(_REPO_ROOT, 'docs')
    learners = [x.strip() for x in args.learners.split(',') if x.strip()]
    writer, metrics, deltas, written, written_deploy = run(
        args.run_dir, docs_root, stage_id=args.stage, learners=learners,
        n_splits=args.cv_splits, gap=args.cv_gap, seed=args.seed,
        proba_dir=args.proba_dir, label_agg=args.label_agg)

    print('\n' + '=' * 78)
    print('LEARNER ARM')
    print('=' * 78)
    print(metrics.to_string(index=False))
    if not deltas.empty:
        print('\nPaired AUC delta vs. xgb (band: +/-'
              + format(SEED_NOISE_AUC, '.3f') + '):')
        print(deltas.to_string(index=False))
    print('\nProbability files:')
    for learner, path in written.items():
        print('  ' + learner.ljust(9) + ' [oof   -> AUC/PR/calibration]     ' + path)
    for learner, path in written_deploy.items():
        print('  ' + learner.ljust(9) + ' [deploy -> backtest --proba-file] ' + path)
    print('\nWrote ' + str(len(writer.artifacts)) + ' artefacts to ' + writer.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
