"""S3 — feature selection: what survives, what agrees, and what is stable.

Reads the per-fold diagnostics that ``iterative_training --keep-diagnostics`` rescues
before each run directory is deleted, and answers three questions the existing artefacts
could state but never showed:

1. **What does the noise-floor gate actually remove?** Mutual information is scored
   against a permutation null, and features below it are dropped. The gate is univariate
   and therefore blind to anything that only pays off in combination — measured A/B over
   12 walk-forward folds, gating cost 123k EUR, which is why ``--mi-permutations 0`` is
   mandatory. The figure shows which features the gate would take.

2. **Do the importance measures agree?** Gain says which features the model *used*, PFI
   which it *needed*, SHAP how each one moves a prediction. They answer different
   questions and routinely disagree; a feature story resting on one of them alone is
   resting on the choice of measure.

3. **Is the selection stable when the window moves?** This is the one that decides
   whether a feature set is a finding or a fit, and it is only answerable because the
   per-fold ``selected_features.txt`` now survives the campaign.

Usage::

    python analytics/feature_figures.py \\
        --diagnostics ../../generated/paper_diagnostics/S2c \\
        --reference ../../generated/paper_diagnostics/S3 \\
        --docs-root ../../../docs --stage S3
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_SOURCE_PYTHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SOURCE_PYTHON)))
for _p in (_SOURCE_PYTHON, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analytics import figures as fg  # noqa: E402  (fixes the Agg backend on import)
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ('long_slow', 'short_slow', 'long_fast', 'short_fast')


# --- Reading the rescued diagnostics ---------------------------------------------------

def fold_dirs(root):
    """Every kept fold-run directory under a diagnostics root, with its context."""
    out = []
    for ctx_path in sorted(glob.glob(os.path.join(root, '*', '*', 'run_context.json'))):
        try:
            with open(ctx_path, encoding='utf-8') as fh:
                ctx = json.load(fh)
        except Exception:
            ctx = {}
        out.append({'dir': os.path.dirname(ctx_path), 'context': ctx})
    return out


def read_selected(fold_dir, model):
    path = os.path.join(fold_dir, model + '_selected_features.txt')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [line.strip() for line in fh if line.strip()]


def read_mi(fold_dir, model):
    path = os.path.join(fold_dir, model + '_mi_scores.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df.index.name = 'feature'
    return df


def read_pfi(fold_dir, model):
    path = os.path.join(fold_dir, model + '_pfi_scores.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def read_shap(fold_dir, model):
    for name in ('shap_importance_target_' + model + '.csv',
                 'shap_importance_' + model + '.csv'):
        path = os.path.join(fold_dir, name)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


# --- 1. The noise-floor gate ------------------------------------------------------------

def mi_gate_frame(mi):
    """MI score beside its permutation null, per feature.

    The regime-aware columns are used when present: under a regime-gated label mode the
    global MI would rank trend-vs-range *detectors* rather than features that separate
    winners inside a trend.
    """
    if mi is None or mi.empty:
        return pd.DataFrame()
    if 'mi_score_trend' in mi.columns and 'mi_null_trend' in mi.columns:
        d = pd.DataFrame({'score': mi['mi_score_trend'], 'null': mi['mi_null_trend']})
        bucket = 'trend'
    elif 'mi_null' in mi.columns:
        d = pd.DataFrame({'score': mi['mi_score'], 'null': mi['mi_null']})
        bucket = 'global'
    else:
        d = pd.DataFrame({'score': mi['mi_score'], 'null': np.nan})
        bucket = 'global (no null measured)'
    d['above_floor'] = d['score'] > d['null']
    d.attrs['bucket'] = bucket
    return d.sort_values('score', ascending=False)


def figure_mi_gate(gate, writer, model, name=None, top_n=28):
    """MI against its permutation floor. Bars below the floor are what the gate removes."""
    if gate.empty:
        return None
    d = gate.head(top_n).iloc[::-1]
    ys = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(fg.WIDTH_FULL, max(2.2, 0.20 * len(d) + 1.0)))
    ax.barh(ys, d['score'], 0.62, zorder=2,
            color=[fg.CATEGORICAL[2] if ok else fg.INK_MUTED
                   for ok in d['above_floor']])
    if d['null'].notna().any():
        ax.plot(d['null'], ys, marker='|', markersize=9, linestyle='none',
                color=fg.STATUS['critical'], zorder=3, label='permutation floor')
        ax.legend(loc='lower right')
    ax.set_yticks(ys, list(d.index))
    ax.set_xlabel('Mutual information')
    kept = int(gate['above_floor'].sum())
    ax.set_title('MI against its noise floor — ' + model + '  ·  ' + str(kept) + ' of '
                 + str(len(gate)) + ' features clear it')
    fg.tidy(ax, grid_axis='x')
    return writer.save_figure(
        fig, name or ('mi_vs_noise_floor_' + model),
        caption='Grey bars fall below the permutation floor and would be dropped by the '
                'gate. The gate is univariate and cannot see a feature that only pays '
                'off in combination.')


# --- 2. Do the measures agree? ----------------------------------------------------------

def importance_agreement(fold_dir, model, top_n=20):
    """Rank of the same features under gain, PFI and SHAP."""
    frames = {}
    mi = read_mi(fold_dir, model)
    if mi is not None and 'mi_score' in mi.columns:
        frames['mi'] = mi['mi_score']
    pfi = read_pfi(fold_dir, model)
    if pfi is not None and {'feature', 'importance'} <= set(pfi.columns):
        frames['pfi'] = pfi.set_index('feature')['importance']
    shap = read_shap(fold_dir, model)
    if shap is not None and len(shap.columns) >= 2:
        col = 'mean_abs_shap' if 'mean_abs_shap' in shap.columns else shap.columns[1]
        key = 'feature' if 'feature' in shap.columns else shap.columns[0]
        frames['shap'] = shap.set_index(key)[col]
    if len(frames) < 2:
        return pd.DataFrame()

    wide = pd.DataFrame(frames)
    ranks = wide.rank(ascending=False, na_option='keep')
    ranks = ranks.loc[ranks.min(axis=1).sort_values().index[:top_n]]
    return ranks


def figure_importance_agreement(ranks, writer, model, name=None):
    """Slope chart: where a feature sits under each measure.

    Lines that cross are the message — a feature the model uses heavily need not be one
    it needs, and neither need be the one that moves predictions most.
    """
    if ranks.empty or ranks.shape[1] < 2:
        return None
    cols = list(ranks.columns)
    xs = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.75))
    for i, (feature, row) in enumerate(ranks.iterrows()):
        vals = row.reindex(cols).to_numpy(dtype=float)
        ax.plot(xs, vals, marker='o', markersize=4, linewidth=1.1,
                color=fg.CATEGORICAL[i % len(fg.CATEGORICAL)], zorder=2)
        if np.isfinite(vals[0]):
            ax.text(-0.06, vals[0], str(feature) + ' ', ha='right', va='center',
                    fontsize=6, color=fg.INK_SECONDARY)
    ax.set_xticks(xs, [c.upper() for c in cols])
    ax.set_xlim(-0.9, len(cols) - 0.55)
    ax.invert_yaxis()
    ax.set_ylabel('Rank (1 = most important)')
    ax.set_title('Do the importance measures agree? — ' + model)
    fg.tidy(ax, grid_axis='y')
    return writer.save_figure(
        fig, name or ('importance_agreement_' + model),
        caption='Rank of the same features under each measure; crossing lines mean the '
                'measures disagree about what matters.')


# --- 3. Selection stability -------------------------------------------------------------

def selection_stability(folds, model):
    """How often each feature is selected across the kept fold-runs."""
    picks = []
    for f in folds:
        sel = read_selected(f['dir'], model)
        if sel:
            picks.append(set(sel))
    if not picks:
        return pd.DataFrame(), 0
    universe = sorted(set().union(*picks))
    rows = [{'feature': feat,
             'n_selected': sum(1 for p in picks if feat in p),
             'share': sum(1 for p in picks if feat in p) / len(picks)}
            for feat in universe]
    df = pd.DataFrame(rows).sort_values('share', ascending=False).reset_index(drop=True)
    return df, len(picks)


def selection_is_inactive(stability):
    """True when every feature is chosen in every window.

    Not a bug and not a strong feature set: it is what happens when the gate is switched
    off. The economically-validated configuration runs ``--mi-permutations 0`` with a
    threshold of 0 — the A/B that established this cost 123k EUR when the gate was left
    on — so nothing is ever dropped and the feature set is fixed by ``features.yaml``
    rather than by the data. Worth stating, because a chart of uniformly full bars
    otherwise reads as evidence of stability.
    """
    return (not stability.empty) and bool((stability['share'] >= 1.0).all())


def figure_selection_stability(stability, n_folds, writer, model, name=None, top_n=32):
    """Selection frequency per feature.

    A feature chosen in every window is a finding; one chosen in a third of them is a
    property of those windows. Nothing else in the pipeline distinguishes the two.
    """
    if stability.empty:
        return None
    inactive = selection_is_inactive(stability)
    d = stability.head(top_n).iloc[::-1]
    ys = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(fg.WIDTH_FULL, max(2.2, 0.20 * len(d) + 1.2)))
    ax.barh(ys, d['share'] * 100, 0.62, zorder=2,
            color=[fg.CATEGORICAL[0] if s >= 0.8 else
                   (fg.CATEGORICAL[3] if s >= 0.4 else fg.INK_MUTED)
                   for s in d['share']])
    ax.axvline(100, color=fg.BASELINE, linewidth=0.8)
    ax.set_yticks(ys, list(d['feature']))
    ax.set_xlabel('Selected in % of the ' + str(n_folds) + ' fold-runs')
    ax.set_xlim(0, 105)
    title = 'Feature-selection stability — ' + model
    if inactive:
        title += '\nall ' + str(len(stability)) + ' features chosen every time: the '
        title += 'gate is off (--mi-permutations 0), so nothing is selected away'
    ax.set_title(title, fontsize=8.5 if inactive else 9)
    fg.tidy(ax, grid_axis='x')
    caption = ('How often each feature survives selection across windows. A feature '
               'chosen in a minority of windows is a property of those windows.')
    if inactive:
        caption = ('Every feature survives every window because the noise-floor gate is '
                   'switched off in the economically-validated configuration. The '
                   'feature set is therefore fixed by features.yaml, not chosen by the '
                   'data - which is a statement about the configuration, not evidence '
                   'of a stable selection.')
    return writer.save_figure(fig, name or ('selection_stability_' + model),
                              caption=caption)


# --- Entry point --------------------------------------------------------------------------

def run(docs_root, stage_id='S3', reference=None, diagnostics=None,
        models=('long_slow', 'long_fast')):
    writer = fg.ArtefactWriter(stage_id, docs_root)
    produced = {'mi_gate': [], 'agreement': [], 'stability': []}

    ref_folds = fold_dirs(reference) if reference else []
    for model in models:
        if not ref_folds:
            break
        fold_dir = ref_folds[0]['dir']
        gate = mi_gate_frame(read_mi(fold_dir, model))
        if not gate.empty:
            writer.save_table(gate.reset_index(), 'mi_gate_' + model,
                              caption='MI score against its permutation floor for '
                                      + model + '.')
            if figure_mi_gate(gate, writer, model):
                produced['mi_gate'].append(model)
        ranks = importance_agreement(fold_dir, model)
        if not ranks.empty and figure_importance_agreement(ranks, writer, model):
            writer.save_table(ranks.reset_index(), 'importance_ranks_' + model,
                              caption='Rank under each importance measure for ' + model + '.')
            produced['agreement'].append(model)

    stab_folds = fold_dirs(diagnostics) if diagnostics else []
    for model in models:
        if not stab_folds:
            break
        stability, n = selection_stability(stab_folds, model)
        if stability.empty:
            continue
        if selection_is_inactive(stability):
            produced.setdefault('selection_inactive', []).append(model)
        writer.save_table(stability, 'selection_stability_' + model,
                          caption='Selection frequency across ' + str(n)
                                  + ' fold-runs for ' + model + '.')
        if figure_selection_stability(stability, n, writer, model):
            produced['stability'].append(model)

    writer.save_json({'reference': os.path.abspath(reference) if reference else None,
                      'diagnostics': (os.path.abspath(diagnostics)
                                      if diagnostics else None),
                      'n_reference_folds': len(ref_folds),
                      'n_stability_folds': len(stab_folds),
                      'produced': produced,
                      'artifacts': writer.artifacts}, 'feature_figures')
    return writer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default=None,
                    help='diagnostics root of the full-diagnostics reference run '
                         '(MI, PFI and SHAP come from here)')
    ap.add_argument('--diagnostics', default=None,
                    help='diagnostics root of a walk-forward campaign (selection '
                         'stability is measured across its fold-runs)')
    ap.add_argument('--models', default='long_slow,long_fast')
    ap.add_argument('--docs-root', default=None)
    ap.add_argument('--stage', default='S3')
    args = ap.parse_args(argv)

    if not args.reference and not args.diagnostics:
        ap.error('give --reference and/or --diagnostics')
    docs_root = args.docs_root or os.path.join(_REPO_ROOT, 'docs')
    models = tuple(m.strip() for m in args.models.split(',') if m.strip())
    writer = run(docs_root, stage_id=args.stage, reference=args.reference,
                 diagnostics=args.diagnostics, models=models)
    print('\nWrote ' + str(len(writer.artifacts)) + ' artefacts to ' + writer.root)
    for a in writer.artifacts:
        for path in a['paths'].values():
            print('  ' + a['kind'].ljust(7) + ' ' + writer.relative(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
