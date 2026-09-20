"""Model-quality diagnostics from persisted out-of-fold scores.

``docs/preregistration.md`` §5 makes ROC, precision/recall, the confusion matrix and
the calibration error mandatory for every reported model. Until the out-of-fold scores
were persisted none of them could be drawn: ``training_summary.json`` carries only
aggregates, and an aggregate cannot be turned back into a curve.

``advanced_train.py`` now writes ``oof_predictions.parquet`` (one row per model, stage
and validation bar: ``timestamp, model, stage, fold, y_true, y_score`` plus the regime
columns). This module turns that file into the figures, as often as wanted, without
retraining anything.

A note on the regime split that matters for reading the output. Under ``trend_only``
and ``regime_conditional`` the label of a RANGE bar is forced to 0, so a within-RANGE
AUC has no positives to rank and is not a measurement. The within-TREND rows keep the
real barrier outcome, so the TREND number *is* the "AUC within the operative regime
against the non-zeroed outcome" the pre-registration asks for — and it is what exposes
a headline AUC that only measures regime detection.

Usage::

    python analytics/model_diagnostics.py --run-dir ../../generated/<run_id> \\
        --docs-root ../../../docs --stage S7
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)

# analytics/__init__.py pulls in model_report, which imports ModelTrading.* — so the
# repo root has to be importable too, not just source/python. Mirrors tests/conftest.py.
_SOURCE_PYTHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SOURCE_PYTHON)))
for _p in (_SOURCE_PYTHON, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analytics import figures as fg  # noqa: E402  (fixes the Agg backend on import)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MODEL_ORDER = ('long_slow', 'short_slow', 'long_fast', 'short_fast')
OOF_FILENAME = 'oof_predictions.parquet'


# --- Loading -----------------------------------------------------------------------

def load_oof(run_dir, stage='final'):
    """Load the out-of-fold scores for one pipeline stage.

    Args:
        run_dir: a run directory under ``ModelTrading/generated/``.
        stage: 'final' (the gate metrics, at the resolved rounds) or 'post' (at the cap).

    Returns:
        pd.DataFrame, sorted by model and timestamp.

    Raises:
        FileNotFoundError: the run predates the OOF export, with a hint to retrain.
    """
    path = os.path.join(run_dir, OOF_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            path + " not found. Runs made before the OOF export was added do not carry "
            "the validation scores, and the aggregates in training_summary.json cannot "
            "be turned back into curves — re-run advanced_train.py for this configuration."
        )
    df = pd.read_parquet(path)
    if stage is not None:
        df = df[df['stage'] == stage]
        if df.empty:
            raise ValueError("no rows for stage '" + str(stage) + "' in " + path)
    return df.sort_values(['model', 'timestamp']).reset_index(drop=True)


def load_summary(run_dir):
    """Load training_summary.json, or an empty dict when the run has none."""
    path = os.path.join(run_dir, 'training_summary.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def model_threshold(summary, model_key, default=0.5):
    """The operating point the training run chose for this model."""
    final = (summary.get('cv_final_metrics') or {}).get(model_key) or {}
    thr = final.get('threshold')
    return float(thr) if thr is not None else float(default)


def present_models(oof):
    """Models in the file, in the canonical order (extras appended)."""
    have = list(dict.fromkeys(oof['model']))
    ordered = [m for m in MODEL_ORDER if m in have]
    return ordered + [m for m in have if m not in ordered]


def _slice(oof, model_key):
    d = oof[oof['model'] == model_key]
    return d['y_true'].to_numpy(), d['y_score'].to_numpy()


def _both_classes(y):
    return len(y) > 0 and len(np.unique(y)) > 1


# --- Figures -----------------------------------------------------------------------

def figure_roc(oof, writer, name='roc_curves'):
    """Pooled ROC per model, one panel. The diagonal is the no-skill reference."""
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.5))
    ax.plot([0, 1], [0, 1], color=fg.INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)),
            zorder=1, label='no skill')
    for i, mk in enumerate(present_models(oof)):
        y, p = _slice(oof, mk)
        if not _both_classes(y):
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=fg.model_color(mk, i), zorder=2,
                label=mk + '  AUC ' + format(auc(fpr, tpr), '.3f'))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('ROC, pooled out-of-fold validation predictions')
    fg.tidy(ax, grid_axis='both')
    ax.legend(loc='lower right')
    return writer.save_figure(fig, name,
                             caption='Pooled out-of-fold ROC per model.')


def figure_precision_recall(oof, summary, writer, name='precision_recall'):
    """PR curve per model with the chosen operating point marked.

    PR is the informative view here: the positive class is a small minority, and a ROC
    can look respectable while precision at the operating threshold is near the base
    rate. The dotted line is that base rate — the precision of guessing.
    """
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.5))
    for i, mk in enumerate(present_models(oof)):
        y, p = _slice(oof, mk)
        if not _both_classes(y):
            continue
        colour = fg.model_color(mk, i)
        prec, rec, thr = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=colour, zorder=2,
                label=mk + '  AP ' + format(average_precision_score(y, p), '.3f'))
        t = model_threshold(summary, mk)
        # thr has one entry fewer than prec/rec; index the point nearest the threshold.
        if len(thr):
            j = int(np.argmin(np.abs(thr - t)))
            ax.plot(rec[j], prec[j], marker='o', markersize=5, color=colour,
                    markeredgecolor=fg.SURFACE, markeredgewidth=1.2, zorder=3)
        ax.axhline(float(y.mean()), color=colour, linewidth=0.7,
                   linestyle=(0, (1, 3)), zorder=1)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_title('Precision/recall, out-of-fold (dot = operating point, dotted = base rate)')
    fg.tidy(ax, grid_axis='both')
    ax.legend(loc='upper right')
    return writer.save_figure(fig, name,
                             caption='Out-of-fold precision/recall with the operating '
                                     'point chosen during training and the base rate.')


def figure_confusion(oof, summary, writer, name='confusion_matrices'):
    """Confusion matrix per model at the operating threshold, as a small-multiple grid."""
    models = present_models(oof)
    ncol = min(len(models), 4) or 1
    fig, axes = plt.subplots(1, ncol, figsize=(fg.WIDTH_FULL, 2.1), squeeze=False)
    cmap = LinearSegmentedColormap.from_list('seq_blue', list(fg.SEQUENTIAL))
    for col, (ax, mk) in enumerate(zip(axes[0], models)):
        y, p = _slice(oof, mk)
        thr = model_threshold(summary, mk)
        cm = confusion_matrix(y, (p > thr).astype(int), labels=[0, 1])
        # Row-normalised: the classes are wildly imbalanced, so raw counts would paint
        # one cell dark and leave the minority row unreadable.
        rows = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, np.maximum(rows, 1))
        # Square cells: a stretched matrix reads as if the two classes differed in a
        # way the counts do not support.
        ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        for r in range(2):
            for c in range(2):
                ax.text(c, r, format(cm[r, c], ','),
                        ha='center', va='center', fontsize=7,
                        color=fg.SURFACE if shown[r, c] > 0.55 else fg.INK_PRIMARY)
        ax.set_xticks([0, 1], ['pred 0', 'pred 1'])
        # Only the leftmost panel carries row labels — repeating them prints text on
        # top of the neighbouring panel's counts.
        if col == 0:
            ax.set_yticks([0, 1], ['true 0', 'true 1'])
        else:
            ax.set_yticks([])
        ax.set_title(mk + '\nthr ' + format(thr, '.3f'), fontsize=8)
        for side in ('top', 'right', 'left', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    for ax in axes[0][len(models):]:
        ax.axis('off')
    fig.suptitle('Confusion matrices at the operating threshold (shading = row share)',
                 fontsize=9, y=1.02)
    fig.subplots_adjust(wspace=0.25)
    return writer.save_figure(fig, name,
                             caption='Out-of-fold confusion matrices, counts printed, '
                                     'shaded by within-row share.')


def figure_calibration(oof, writer, name='calibration', n_bins=10):
    """Reliability diagram: predicted probability against observed frequency.

    Bins are equal-width so the diagram is comparable across models. Bins holding fewer
    than ``min_count`` rows are dropped rather than plotted as noise.
    """
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.5))
    ax.plot([0, 1], [0, 1], color=fg.INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)),
            zorder=1, label='perfectly calibrated')
    edges = np.linspace(0, 1, n_bins + 1)
    for i, mk in enumerate(present_models(oof)):
        y, p = _slice(oof, mk)
        if len(y) == 0:
            continue
        xs, ys = [], []
        for b in range(n_bins):
            hi_inclusive = b == n_bins - 1
            m = (p >= edges[b]) & ((p <= edges[b + 1]) if hi_inclusive else (p < edges[b + 1]))
            if m.sum() >= max(20, 0.001 * len(p)):
                xs.append(float(p[m].mean()))
                ys.append(float(y[m].mean()))
        if xs:
            ax.plot(xs, ys, color=fg.model_color(mk, i), marker='o', zorder=2,
                    label=mk + '  Brier ' + format(brier_score_loss(y, p), '.4f'))
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Calibration, out-of-fold')
    fg.tidy(ax, grid_axis='both')
    ax.legend(loc='upper left')
    return writer.save_figure(fig, name,
                             caption='Reliability diagram over pooled out-of-fold '
                                     'predictions; sparse bins omitted.')


def figure_score_distribution(oof, summary, writer, name='score_distribution'):
    """Score histogram split by true class, one panel per model.

    Separation between the two histograms is what a model is for; a single overlapping
    lump is the picture of a model that has not learned the distinction.
    """
    models = present_models(oof)
    ncol = min(len(models), 2) or 1
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(fg.WIDTH_FULL, 1.8 * nrow),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, mk in zip(flat, models):
        y, p = _slice(oof, mk)
        bins = np.linspace(0, max(float(p.max()) if len(p) else 1.0, 1e-3), 40)
        ax.hist(p[y == 0], bins=bins, color=fg.CATEGORICAL[0], alpha=0.75,
                label='label 0', density=True)
        ax.hist(p[y == 1], bins=bins, color=fg.CATEGORICAL[1], alpha=0.75,
                label='label 1', density=True)
        ax.axvline(model_threshold(summary, mk), color=fg.INK_PRIMARY, linewidth=1.0,
                   linestyle=(0, (4, 3)))
        ax.set_title(mk, fontsize=8)
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        fg.tidy(ax)
        ax.legend(loc='upper right')
    for ax in flat[len(models):]:
        ax.axis('off')
    fig.tight_layout()
    return writer.save_figure(fig, name,
                             caption='Out-of-fold score distribution by true class; '
                                     'dashed line marks the operating threshold.')


# --- Regime breakdown --------------------------------------------------------------

def regime_breakdown(oof, summary):
    """AUC and base rate within TREND bars and within RANGE bars.

    Returns a tidy frame with one row per (model, bucket). ``interpretable`` flags the
    rows where the bucket actually carries both classes — under the regime-gated label
    modes the RANGE bucket is all-zero by construction, and an AUC there is not a
    measurement of anything.
    """
    if 'regime_trend' not in oof.columns:
        return pd.DataFrame(columns=['model', 'bucket', 'n', 'positives', 'base_rate',
                                     'auc', 'interpretable'])
    rows = []
    for mk in present_models(oof):
        d = oof[oof['model'] == mk]
        trend = d['regime_trend']
        buckets = {'TREND': trend.fillna(0) != 0, 'RANGE': trend.fillna(0) == 0}
        for label, mask in buckets.items():
            y = d.loc[mask, 'y_true'].to_numpy()
            p = d.loc[mask, 'y_score'].to_numpy()
            ok = _both_classes(y)
            rows.append({
                'model': mk,
                'bucket': label,
                'n': int(len(y)),
                'positives': int(y.sum()),
                'base_rate': float(y.mean()) if len(y) else np.nan,
                'auc': float(roc_auc_score(y, p)) if ok else np.nan,
                'interpretable': bool(ok),
            })
    out = pd.DataFrame(rows)
    overall = []
    for mk in present_models(oof):
        y, p = _slice(oof, mk)
        overall.append({'model': mk, 'bucket': 'ALL', 'n': int(len(y)),
                        'positives': int(y.sum()),
                        'base_rate': float(y.mean()) if len(y) else np.nan,
                        'auc': float(roc_auc_score(y, p)) if _both_classes(y) else np.nan,
                        'interpretable': _both_classes(y)})
    return pd.concat([pd.DataFrame(overall), out], ignore_index=True)


def figure_regime_auc(breakdown, writer, name='regime_auc'):
    """Overall AUC beside the within-TREND AUC — the gap is the regime-recovery artefact."""
    d = breakdown[breakdown['interpretable']]
    models = [m for m in MODEL_ORDER if m in set(d['model'])]
    if not models:
        return None
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.5))
    width = 0.38
    xs = np.arange(len(models))
    for k, (bucket, colour) in enumerate((('ALL', fg.CATEGORICAL[0]),
                                          ('TREND', fg.CATEGORICAL[1]))):
        vals = [float(d[(d['model'] == m) & (d['bucket'] == bucket)]['auc'].mean())
                for m in models]
        pos = xs + (k - 0.5) * width
        ax.bar(pos, vals, width * 0.92, color=colour, label=bucket, zorder=2)
        for x, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.012, format(v, '.3f'), ha='center', fontsize=6.5,
                        color=fg.INK_SECONDARY)
    ax.axhline(0.5, color=fg.INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xticks(xs, models)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('Overall AUC vs. AUC within trend bars (0.5 = chance)')
    fg.tidy(ax)
    ax.legend(loc='upper right')
    return writer.save_figure(
        fig, name,
        caption='A headline AUC far above the within-trend AUC measures regime '
                'detection, not trade selection.')


# --- Summary table -----------------------------------------------------------------

def metrics_table(oof, summary):
    """One row per model: the numbers the acceptance gates are read on."""
    rows = []
    for mk in present_models(oof):
        y, p = _slice(oof, mk)
        thr = model_threshold(summary, mk)
        pred = (p > thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        final = (summary.get('cv_final_metrics') or {}).get(mk) or {}
        coverage = (summary.get('label_coverage') or {}).get(mk) or {}
        rows.append({
            'model': mk,
            'n': int(len(y)),
            'n_eff': coverage.get('n_eff'),
            'base_rate': float(y.mean()) if len(y) else np.nan,
            'threshold': thr,
            'auc': float(roc_auc_score(y, p)) if _both_classes(y) else np.nan,
            'avg_precision': float(average_precision_score(y, p)) if _both_classes(y) else np.nan,
            'precision': float(tp / (tp + fp)) if (tp + fp) else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) else 0.0,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
            'pos_pred_rate': float(pred.mean()) if len(pred) else np.nan,
            'brier': float(brier_score_loss(y, p)) if len(y) else np.nan,
            'ece': final.get('ece'),
            'rounds': final.get('rounds'),
            'degenerate_folds': len(coverage.get('degenerate_fold_numbers') or []),
        })
    return pd.DataFrame(rows)


# --- Entry point -------------------------------------------------------------------

def run(run_dir, docs_root, stage_id='S7', oof_stage='final'):
    """Produce the whole S7 artefact set for one run. Returns the writer."""
    oof = load_oof(run_dir, stage=oof_stage)
    summary = load_summary(run_dir)
    writer = fg.ArtefactWriter(stage_id, docs_root)

    figure_roc(oof, writer)
    figure_precision_recall(oof, summary, writer)
    figure_confusion(oof, summary, writer)
    figure_calibration(oof, writer)
    figure_score_distribution(oof, summary, writer)

    breakdown = regime_breakdown(oof, summary)
    if not breakdown.empty:
        writer.save_table(breakdown, 'regime_auc',
                          caption='AUC by regime bucket. RANGE rows are not '
                                  'interpretable under regime-gated label modes.')
        figure_regime_auc(breakdown, writer)

    table = metrics_table(oof, summary)
    writer.save_table(table, 'model_quality',
                      caption='Out-of-fold model quality at the operating threshold.')
    writer.save_json({'run_dir': os.path.abspath(run_dir),
                      'oof_stage': oof_stage,
                      'run_id': summary.get('run_id'),
                      'command_line': summary.get('command_line'),
                      'artifacts': writer.artifacts}, 'model_diagnostics')
    return writer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='run directory under ModelTrading/generated/')
    ap.add_argument('--docs-root', default=None,
                    help='docs/ directory (default: <repo>/docs)')
    ap.add_argument('--stage', default='S7', help='stage id for the artefact paths')
    ap.add_argument('--oof-stage', default='final', choices=['final', 'post'],
                    help="'final' = at the resolved rounds (the gate metrics); "
                         "'post' = at the round cap")
    args = ap.parse_args(argv)

    docs_root = args.docs_root
    if docs_root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        docs_root = os.path.join(here, '..', '..', '..', '..', 'docs')

    writer = run(args.run_dir, docs_root, stage_id=args.stage, oof_stage=args.oof_stage)
    print('\nWrote ' + str(len(writer.artifacts)) + ' artefacts to ' + writer.root)
    for a in writer.artifacts:
        for path in a['paths'].values():
            print('  ' + a['kind'].ljust(7) + ' ' + writer.relative(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
