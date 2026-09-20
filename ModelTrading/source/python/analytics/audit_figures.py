"""Figures for the screening stages: the information audit (S1) and the label audit (S2).

Both audits already write their results as CSV/JSON and print a report. Neither draws
anything, and both carry a headline that is much easier to see than to read:

* **The information audit is a multiplicity story.** 510 tests produced 5 nominally
  significant results where chance alone gives ~25. A table of p-values does not make
  that visible; a QQ plot against the uniform distribution does — under a true null the
  points lie on the diagonal, and here they sit *below* it, meaning the data produced
  fewer small p-values than noise would.
* **The label audit is an agreement story.** Raw agreement between two label modes runs
  around 79 % and looks like near-duplication; Cohen's kappa on the same pair is 0.58,
  because agreeing on the overwhelmingly common "no trade" label is free. Putting the two
  side by side is the whole finding.

Usage::

    python analytics/audit_figures.py --kind information \\
        --pooled ../../../docs/results/information_audit_v2_2026-08-29.csv \\
        --multivariate ../../../docs/results/information_audit_v2_2026-08-29_multivariate.csv \\
        --stage S1
    python analytics/audit_figures.py --kind label \\
        --label-audit ../../../docs/results/label_mode_audit_2026-08-29.json --stage S2
"""

import argparse
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ALPHA = 0.05


# --- Information audit (S1) ----------------------------------------------------------

def figure_pvalue_qq(pooled, writer, name='pvalue_qq', by='family'):
    """Observed p-values against the uniform distribution they would follow under H0.

    The diagonal is the null. Points **below** it mean more small p-values than chance
    produces — evidence. Points **on or above** it mean the opposite, and that is what a
    family with no information looks like. This is the figure that carries H1.

    Drawn as small multiples, one panel per family: every pair of series would otherwise
    be on screen simultaneously in a scatter, and past three simultaneous categorical
    hues the pairs no longer separate for colour-vision-deficient readers. Faceting also
    makes each family's departure from the diagonal legible on its own.
    """
    d = pooled.dropna(subset=['p_raw'])
    if d.empty:
        return None
    families = list(dict.fromkeys(d[by])) if by in d.columns else ['all']

    ncol = min(4, max(1, len(families)))
    nrow = int(np.ceil(len(families) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(fg.WIDTH_FULL, 1.65 * nrow + 0.35),
                             squeeze=False, sharex=True, sharey=True)
    flat = [a for row in axes for a in row]
    for ax, fam in zip(flat, families):
        sub = d if by not in d.columns else d[d[by] == fam]
        p = np.sort(sub['p_raw'].to_numpy(float))
        if len(p) == 0:
            ax.axis('off')
            continue
        expected = (np.arange(1, len(p) + 1) - 0.5) / len(p)
        ax.plot([0, 1], [0, 1], color=fg.INK_MUTED, linewidth=0.9,
                linestyle=(0, (4, 3)), zorder=1)
        # One hue throughout: identity is carried by the panel title, not by colour.
        ax.plot(expected, p, marker='o', markersize=2.2, linestyle='none',
                color=fg.CATEGORICAL[0], zorder=2)
        hits = int((p < ALPHA).sum())
        # Compact two-line title: side-by-side panels are ~1.5 in wide, so anything
        # longer runs into the neighbouring panel.
        ax.set_title(str(fam) + '\nn=' + str(len(p)) + ' · hits ' + str(hits) + '/'
                     + format(ALPHA * len(p), '.1f'), fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fg.tidy(ax, grid_axis='both')
    for ax in flat[len(families):]:
        ax.axis('off')
    fig.supxlabel('Expected p-value under the null', fontsize=8)
    fig.supylabel('Observed p-value', fontsize=8)
    fig.suptitle('Observed p-values against chance (dashed = the null)', fontsize=9)
    fig.tight_layout()
    return writer.save_figure(
        fig, name,
        caption='Points on the diagonal are what pure noise produces; above it means '
                'fewer small p-values than chance.')


def hits_vs_chance(pooled, alpha=ALPHA, by='family'):
    """Nominally significant results per family against the number chance predicts."""
    if pooled.empty or 'p_raw' not in pooled.columns:
        return pd.DataFrame()
    key = by if by in pooled.columns else None
    groups = pooled.groupby(key) if key else [('all', pooled)]
    rows = []
    for fam, sub in groups:
        n = int(len(sub))
        hits = int((sub['p_raw'] < alpha).sum())
        rows.append({'family': fam, 'n_tests': n, 'hits_observed': hits,
                     'hits_expected': alpha * n,
                     'ratio': hits / (alpha * n) if n else np.nan,
                     'passes_after_correction': int(sub.get(
                         'passes_h1', pd.Series(dtype=bool)).sum())})
    return pd.DataFrame(rows).sort_values('n_tests', ascending=False)


def figure_hits_vs_chance(table, writer, name='hits_vs_chance'):
    """Observed against expected-by-chance hits, per family.

    A family whose bar falls short of its chance marker has produced *less* than noise.
    """
    if table.empty:
        return None
    xs = np.arange(len(table))
    width = 0.38
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_FULL, 0.36))
    ax.bar(xs - width / 2, table['hits_observed'], width * 0.92,
           color=fg.CATEGORICAL[0], label='observed p < ' + str(ALPHA), zorder=2)
    ax.bar(xs + width / 2, table['hits_expected'], width * 0.92,
           color=fg.CATEGORICAL[3], label='expected by chance', zorder=2)
    for x, (obs, exp) in enumerate(zip(table['hits_observed'], table['hits_expected'])):
        if exp > 0:
            ax.text(x, max(obs, exp) + 0.15, format(obs / exp, '.2f') + 'x',
                    ha='center', fontsize=6.5, color=fg.INK_SECONDARY)
    ax.set_xticks(xs, [str(f) + '\n' + str(int(n)) + ' tests'
                       for f, n in zip(table['family'], table['n_tests'])])
    ax.set_ylabel('Nominally significant results')
    ax.set_title('Findings against what noise alone produces')
    fg.tidy(ax)
    ax.legend(loc='upper right')
    return writer.save_figure(fig, name,
                              caption='Observed and chance-expected hit counts per '
                                      'information family.')


def figure_ic_heatmap(pooled, writer, name='ic_heatmap', family=None, top_n=18):
    """Information coefficient per feature and horizon, diverging around zero.

    Diverging because the sign is the whole point — a warm cell and a cool cell mean
    opposite trades, and a sequential ramp would hide that.
    """
    d = pooled if family is None else pooled[pooled['family'] == family]
    d = d.dropna(subset=['ic'])
    if d.empty or 'horizon' not in d.columns:
        return None
    pivot = d.pivot_table(index='feature', columns='horizon', values='ic',
                          aggfunc='mean')
    if pivot.empty:
        return None
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index[:top_n]
    pivot = pivot.loc[order]

    cmap = LinearSegmentedColormap.from_list(
        'div', [fg.DIVERGING_LOW, fg.DIVERGING_MID, fg.DIVERGING_HIGH])
    lim = float(np.nanmax(np.abs(pivot.to_numpy()))) or 0.01
    fig, ax = plt.subplots(figsize=(fg.WIDTH_FULL, max(2.0, 0.22 * len(pivot) + 1.0)))
    im = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), list(pivot.index))
    ax.set_xlabel('Forward horizon (trading days)')
    ax.set_title('Information coefficient'
                 + ('' if family is None else ' — ' + str(family)))
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label('Spearman IC', fontsize=7)
    return writer.save_figure(fig, name,
                              caption='Spearman information coefficient per feature and '
                                      'horizon; zero is the neutral midpoint.')


def figure_multivariate(mv, writer, name='multivariate_vs_null'):
    """Booster AUC against the block-shift null, per cell — with the null band drawn.

    The null shifts the entire feature block by one common offset, preserving each
    feature's autocorrelation *and* the cross-feature correlation structure. A bar inside
    the band has found nothing the shuffled data does not also contain.
    """
    if mv.empty or 'auc' not in mv.columns:
        return None
    d = mv.copy()
    label_cols = [c for c in ('learner', 'outcome', 'horizon') if c in d.columns]
    d['label'] = d[label_cols].astype(str).agg('\n'.join, axis=1) if label_cols else \
        d.index.astype(str)
    d = d.sort_values('auc', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(fg.WIDTH_FULL, max(1.8, 0.30 * len(d) + 1.0)))
    for i, row in d.iterrows():
        nm, ns = row.get('null_mean'), row.get('null_sd')
        if pd.notna(nm) and pd.notna(ns):
            ax.plot([nm - 2 * ns, nm + 2 * ns], [i, i], color=fg.INK_MUTED,
                    linewidth=4, alpha=0.35, zorder=1,
                    solid_capstyle='butt')
            ax.plot(nm, i, marker='|', color=fg.INK_MUTED, markersize=8, zorder=2)
        passed = bool(row.get('passes', False))
        ax.plot(row['auc'], i, marker='o', markersize=5, zorder=3,
                color=fg.CATEGORICAL[2] if passed else fg.CATEGORICAL[0],
                markeredgecolor=fg.SURFACE, markeredgewidth=1.0)
    ax.axvline(0.5, color=fg.BASELINE, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(d)), list(d['label']))
    ax.set_xlabel('Out-of-sample AUC (grey band = null mean ± 2 sd)')
    ax.set_title('Whole-feature-block test against the block-shift null')
    fg.tidy(ax, grid_axis='x')
    return writer.save_figure(
        fig, name,
        caption='Each cell against its own block-shift null; a point inside the band '
                'found nothing the shuffled data does not also contain.')


def run_information_audit(pooled_path, docs_root, stage_id='S1', multivariate_path=None,
                          conditional_path=None, family_for_heatmap=None):
    writer = fg.ArtefactWriter(stage_id, docs_root)
    pooled = pd.read_csv(pooled_path)

    figure_pvalue_qq(pooled, writer)
    table = hits_vs_chance(pooled)
    if not table.empty:
        writer.save_table(table, 'hits_vs_chance',
                          caption='Nominally significant results per family against the '
                                  'number chance alone produces.')
        figure_hits_vs_chance(table, writer)
    figure_ic_heatmap(pooled, writer, family=family_for_heatmap)

    layers = [{'layer': 'pooled', 'n_tests': int(len(pooled)),
               'hits': int((pooled['p_raw'] < ALPHA).sum()),
               'expected': ALPHA * len(pooled)}]
    if conditional_path and os.path.exists(conditional_path):
        cond = pd.read_csv(conditional_path)
        layers.append({'layer': 'conditional', 'n_tests': int(len(cond)),
                       'hits': int((cond['p_raw'] < ALPHA).sum()),
                       'expected': ALPHA * len(cond)})
        figure_pvalue_qq(cond, writer, name='pvalue_qq_conditional', by='conditioner')
    if multivariate_path and os.path.exists(multivariate_path):
        mv = pd.read_csv(multivariate_path)
        layers.append({'layer': 'multivariate', 'n_tests': int(len(mv)),
                       'hits': int((mv['p'] < ALPHA).sum()) if 'p' in mv else 0,
                       'expected': ALPHA * len(mv)})
        figure_multivariate(mv, writer)

    layer_table = pd.DataFrame(layers)
    layer_table['ratio'] = layer_table['hits'] / layer_table['expected'].replace(0, np.nan)
    writer.save_table(layer_table, 'layers',
                      caption='Tests, findings and chance expectation per audit layer.')

    writer.save_json({'pooled': os.path.abspath(pooled_path),
                      'conditional': (os.path.abspath(conditional_path)
                                      if conditional_path else None),
                      'multivariate': (os.path.abspath(multivariate_path)
                                       if multivariate_path else None),
                      'alpha': ALPHA, 'artifacts': writer.artifacts},
                     'information_audit_figures')
    return writer


# --- Label audit (S2) -----------------------------------------------------------------

def coverage_table(audit):
    """Positive rate, n, n_eff and empty months per label mode and model."""
    rows = []
    for mode, models in (audit.get('coverage') or {}).items():
        for model, stats in models.items():
            rows.append({'mode': mode, 'model': model,
                         'n': stats.get('n'),
                         'n_positive': stats.get('n_positive'),
                         'positive_rate': stats.get('positive_rate'),
                         'n_eff': stats.get('n_eff'),
                         'mean_uniqueness': stats.get('mean_uniqueness'),
                         'n_months': stats.get('n_months'),
                         'n_months_without_positive':
                             stats.get('n_months_without_positive')})
    return pd.DataFrame(rows)


def figure_kappa_matrix(audit, writer, name='label_agreement'):
    """Cohen's kappa between every pair of label modes, as a heatmap.

    Kappa, not raw agreement: two modes that both say "no trade" on 90 % of bars agree
    90 % of the time without sharing any information about when to trade.
    """
    pairs = audit.get('pairs') or []
    if not pairs:
        return None
    modes = sorted({p['mode_a'] for p in pairs} | {p['mode_b'] for p in pairs})
    m = pd.DataFrame(np.nan, index=modes, columns=modes)
    for p in pairs:
        m.loc[p['mode_a'], p['mode_b']] = p.get('kappa')
        m.loc[p['mode_b'], p['mode_a']] = p.get('kappa')
    # pandas 3 (copy-on-write) returns a read-only array from .values —
    # fill the diagonal on an owned copy and rebuild the frame.
    m_arr = m.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(m_arr, 1.0)
    m = pd.DataFrame(m_arr, index=m.index, columns=m.columns)

    cmap = LinearSegmentedColormap.from_list('seq_blue', list(fg.SEQUENTIAL))
    fig, ax = plt.subplots(figsize=(fg.WIDTH_FULL, fg.WIDTH_FULL * 0.72))
    im = ax.imshow(m.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1)
    for i in range(len(modes)):
        for j in range(len(modes)):
            v = m.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, format(v, '.2f'), ha='center', va='center', fontsize=6.5,
                        color=fg.SURFACE if v > 0.55 else fg.INK_PRIMARY)
    ax.set_xticks(range(len(modes)), modes, rotation=45, ha='right')
    ax.set_yticks(range(len(modes)), modes)
    ax.set_title("Label-mode agreement (Cohen's kappa)")
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    return writer.save_figure(fig, name,
                              caption='Chance-corrected agreement between label modes; '
                                      '1.0 would mean the modes are duplicates.')


def figure_kappa_vs_agreement(audit, writer, name='kappa_vs_agreement'):
    """Raw agreement against kappa for every pair — why the naive number misleads."""
    pairs = audit.get('pairs') or []
    if not pairs:
        return None
    d = pd.DataFrame(pairs)
    if 'agreement' not in d.columns or 'kappa' not in d.columns:
        return None
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.62))
    ax.plot([0, 1], [0, 1], color=fg.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
            zorder=1, label='the two would agree')
    ax.scatter(d['agreement'], d['kappa'], s=26, alpha=0.8,
               color=fg.CATEGORICAL_ALL_PAIRS[0], edgecolors=fg.SURFACE,
               linewidths=0.6, zorder=2)
    ax.set_xlabel('Raw agreement')
    ax.set_ylabel("Cohen's kappa")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Raw agreement overstates how alike two label modes are')
    fg.tidy(ax, grid_axis='both')
    ax.legend(loc='upper left')
    return writer.save_figure(
        fig, name,
        caption='Every pair sits below the diagonal: agreeing on the common '
                '"no trade" label is free.')


def figure_coverage(coverage, writer, name='label_coverage'):
    """Rows against effective rows per mode, plus the months with no positive at all.

    n_eff is the sample size every standard error should have used; the gap between the
    two bars is how much of the row count is duplication. A mode with empty months
    cannot be cross-validated on that window at all.
    """
    if coverage.empty:
        return None
    d = coverage[coverage['model'] == 'long_slow']
    if d.empty:
        d = coverage.groupby('mode', as_index=False).first()
    d = d.sort_values('mode')
    xs = np.arange(len(d))
    width = 0.38

    fig, (ax_n, ax_m) = plt.subplots(2, 1, figsize=(fg.WIDTH_FULL, 3.4), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1],
                                                  'hspace': 0.15})
    ax_n.bar(xs - width / 2, d['n'], width * 0.92, color=fg.CATEGORICAL[0],
             label='rows (n)', zorder=2)
    ax_n.bar(xs + width / 2, d['n_eff'], width * 0.92, color=fg.CATEGORICAL[2],
             label='effective rows (n_eff)', zorder=2)
    ax_n.set_ylabel('long_slow rows')
    ax_n.set_title('Label coverage: how much of the sample is actually independent')
    fg.tidy(ax_n)
    ax_n.legend(loc='upper right')

    ax_m.bar(xs, d['n_months_without_positive'], 0.55, color=fg.STATUS['critical'],
             zorder=2)
    ax_m.set_ylabel('Months with\nno positive')
    ax_m.set_xticks(xs, list(d['mode']), rotation=30, ha='right')
    fg.tidy(ax_m)
    return writer.save_figure(
        fig, name,
        caption='Row count against effective sample size per label mode, with the '
                'months that carry no positive label at all.')


def run_label_audit(audit_path, docs_root, stage_id='S2'):
    writer = fg.ArtefactWriter(stage_id, docs_root)
    with open(audit_path, encoding='utf-8') as fh:
        audit = json.load(fh)

    coverage = coverage_table(audit)
    if not coverage.empty:
        writer.save_table(coverage, 'label_coverage',
                          caption='Positive rate, effective sample size and empty '
                                  'months per label mode and model.')
        figure_coverage(coverage, writer)
    if audit.get('pairs'):
        writer.save_table(pd.DataFrame(audit['pairs']), 'label_agreement',
                          caption='Pairwise agreement between label modes.')
        figure_kappa_matrix(audit, writer)
        figure_kappa_vs_agreement(audit, writer)

    writer.save_json({'source': os.path.abspath(audit_path),
                      'window': audit.get('window'),
                      'params': audit.get('params'),
                      'artifacts': writer.artifacts}, 'label_audit_figures')
    return writer


# --- Entry point ------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', required=True, choices=['information', 'label'])
    ap.add_argument('--pooled', help='information audit pooled CSV')
    ap.add_argument('--conditional', default=None)
    ap.add_argument('--multivariate', default=None)
    ap.add_argument('--family', default=None, help='restrict the IC heatmap to one family')
    ap.add_argument('--label-audit', help='label_mode_audit_*.json')
    ap.add_argument('--docs-root', default=None)
    ap.add_argument('--stage', default=None)
    args = ap.parse_args(argv)

    docs_root = args.docs_root or os.path.join(_REPO_ROOT, 'docs')
    if args.kind == 'information':
        if not args.pooled:
            ap.error('--pooled is required for --kind information')
        writer = run_information_audit(args.pooled, docs_root,
                                       stage_id=args.stage or 'S1',
                                       multivariate_path=args.multivariate,
                                       conditional_path=args.conditional,
                                       family_for_heatmap=args.family)
    else:
        if not args.label_audit:
            ap.error('--label-audit is required for --kind label')
        writer = run_label_audit(args.label_audit, docs_root, stage_id=args.stage or 'S2')

    print('\nWrote ' + str(len(writer.artifacts)) + ' artefacts to ' + writer.root)
    for a in writer.artifacts:
        for path in a['paths'].values():
            print('  ' + a['kind'].ljust(7) + ' ' + writer.relative(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
