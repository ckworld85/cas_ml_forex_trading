"""Walk-forward figures: fold geometry, pooled result, and how much evidence it is.

``iterative_training.py`` prints three tables (EVIDENCE, STABILITY, PER SEED) and writes
``walk_forward_summary_*.json``. Neither produces a single picture, and the two things a
reader most needs to see are geometric: **where** the training and test windows sit
relative to each other, and **how much the test windows overlap**.

The overlap matters because it is the difference between two honest readings of the same
campaign. With ``step_months == test_months`` the test windows are disjoint and the folds
are close to independent — that is the primary evidence. With ``step_months <
test_months`` the windows deliberately overlap, which is the right design for reading
stability across a shifting window but means the same market day appears in several
folds, so the trades are near-copies. ``aggregate_walk_forward`` prices that in by
clustering the confidence interval on calendar months; :func:`figure_standard_errors`
draws the naive and the clustered interval side by side so the size of that correction is
visible rather than asserted.

Usage::

    python analytics/wf_figures.py \\
        --walk-forward ../../generated/walk_forward_summary_20260829_120000.json \\
        --parallel ../../generated/parallel_training_summary_20260829_120000.json \\
        --docs-root ../../../docs --stage S6 --label disjoint
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
import matplotlib.dates as mdates  # noqa: E402

# A cluster-robust interval needs enough clusters to be believed. Mirrors
# iterative_training.MIN_CLUSTERS_FOR_SIGNIFICANCE.
MIN_CLUSTERS = 12


# --- Loading -----------------------------------------------------------------------

def load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def config_table(summary):
    """The EVIDENCE table as a frame, ranked by t-statistic like the console version."""
    rows = []
    for cfg in summary.get('configs', []):
        rows.append({
            'config': cfg.get('base_name'),
            'n_windows': cfg.get('n_windows'),
            'n_seeds': cfg.get('n_seeds'),
            'n_trades': cfg.get('n_trades'),
            'n_month_clusters': cfg.get('n_month_clusters'),
            'total_pnl': cfg.get('total_pnl'),
            'total_pips': cfg.get('total_pips'),
            'mean_pnl_per_trade': cfg.get('mean_pnl_per_trade'),
            'se_clustered': cfg.get('se_pnl_per_trade'),
            'se_naive': cfg.get('se_naive'),
            'ci95_low': cfg.get('ci95_low'),
            'ci95_high': cfg.get('ci95_high'),
            't_stat': cfg.get('t_stat'),
            'significant': cfg.get('significant'),
            'clusters_reliable': cfg.get('clusters_reliable'),
            'win_rate': cfg.get('win_rate'),
            'folds_profitable': cfg.get('folds_profitable'),
            'months_needed': cfg.get('months_needed_for_significance'),
        })
    df = pd.DataFrame(rows)
    if not df.empty and 't_stat' in df.columns:
        df = df.sort_values('t_stat', ascending=False, na_position='last')
    return df.reset_index(drop=True)


def fold_frame(summary, config=None):
    """One row per (config, fold-run) with its windows parsed into timestamps."""
    rows = []
    for cfg in summary.get('configs', []):
        name = cfg.get('base_name')
        if config is not None and name != config:
            continue
        for fd in cfg.get('fold_details', []):
            rows.append({
                'config': name,
                'seed': fd.get('seed'),
                'train_start': pd.to_datetime(fd.get('train_start'), errors='coerce'),
                'test_start': pd.to_datetime(fd.get('test_start'), errors='coerce'),
                'test_end': pd.to_datetime(fd.get('test_end'), errors='coerce'),
                'pnl': fd.get('pnl'),
                'n_trades': fd.get('n_trades'),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(['config', 'test_start', 'seed']).reset_index(drop=True)


def overlap_factor(folds):
    """How many times the average calendar day is counted across the test windows.

    1.0 = disjoint windows. 3.0 = every day appears in three folds, so the naive
    standard error understates the interval by roughly sqrt(3).
    """
    f = folds.drop_duplicates(subset=['test_start', 'test_end']).dropna(
        subset=['test_start', 'test_end'])
    if f.empty:
        return float('nan'), 0, 0
    covered = int(((f['test_end'] - f['test_start']).dt.days + 1).sum())
    span = int((f['test_end'].max() - f['test_start'].min()).days + 1)
    return (covered / span if span else float('nan')), covered, span


# --- Figures -----------------------------------------------------------------------

def figure_fold_gantt(folds, writer, name='fold_geometry', holdout_start=None,
                      title=None):
    """Training and test window of every fold, one row per fold, on a calendar axis.

    The gap between the two bars of a row is the embargo. Without it the last
    ``label_horizon`` bars of the training labels are resolved from price action inside
    the very window the fold is evaluated on, so this picture is the fastest check that
    the embargo was actually applied.
    """
    f = folds.drop_duplicates(subset=['train_start', 'test_start', 'test_end'])
    f = f.dropna(subset=['train_start', 'test_start', 'test_end'])
    if f.empty:
        return None
    f = f.sort_values('test_start').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(fg.WIDTH_FULL, max(1.6, 0.24 * len(f) + 0.9)))
    for i, row in f.iterrows():
        train_end = row['test_start']          # the embargo lives between these two
        ax.barh(i, mdates.date2num(train_end) - mdates.date2num(row['train_start']),
                left=mdates.date2num(row['train_start']), height=0.62,
                color=fg.CATEGORICAL[0], zorder=2)
        ax.barh(i, mdates.date2num(row['test_end']) - mdates.date2num(row['test_start']),
                left=mdates.date2num(row['test_start']), height=0.62,
                color=fg.CATEGORICAL[1], zorder=3)
    if holdout_start is not None:
        ax.axvline(mdates.date2num(pd.to_datetime(holdout_start)),
                   color=fg.STATUS['critical'], linewidth=1.2, linestyle=(0, (4, 3)),
                   zorder=4)
        ax.text(mdates.date2num(pd.to_datetime(holdout_start)), len(f) - 0.2,
                ' sealed hold-out', fontsize=6.5, color=fg.STATUS['critical'],
                va='top')

    ax.set_yticks(range(len(f)), ['fold ' + format(i + 1, '02d') for i in range(len(f))])
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_xlabel('Calendar time')
    ax.set_title(title or 'Walk-forward fold geometry')
    fg.tidy(ax, grid_axis='x')
    # Two series in one panel: a legend is required, and the bars are directly adjacent
    # so the reader can map colour to role without hunting.
    handles = [plt.Rectangle((0, 0), 1, 1, color=fg.CATEGORICAL[0]),
               plt.Rectangle((0, 0), 1, 1, color=fg.CATEGORICAL[1])]
    ax.legend(handles, ['training window', 'test window'], loc='lower left', ncol=2)
    return writer.save_figure(fig, name,
                             caption='Training and test window per fold; the gap '
                                     'between them is the embargo.')


def figure_overlap(folds, writer, name='fold_overlap', title=None):
    """How often each calendar day is reused across the test windows.

    A day counted more than once means the pooled trades are not independent. This is
    the picture behind the month-clustered confidence interval.
    """
    f = folds.drop_duplicates(subset=['test_start', 'test_end']).dropna(
        subset=['test_start', 'test_end'])
    if f.empty:
        return None
    days = pd.date_range(f['test_start'].min(), f['test_end'].max(), freq='D')
    counts = np.zeros(len(days), dtype=int)
    for _, row in f.iterrows():
        counts += ((days >= row['test_start']) & (days <= row['test_end'])).astype(int)

    factor, covered, span = overlap_factor(folds)
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_FULL, 0.32))
    ax.fill_between(days, counts, 0, step='mid', color=fg.CATEGORICAL[0], alpha=0.45,
                    linewidth=0)
    ax.plot(days, counts, drawstyle='steps-mid', color=fg.CATEGORICAL[0], linewidth=1.2)
    ax.axhline(1, color=fg.INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_ylabel('Folds covering the day')
    ax.set_xlabel('Calendar time')
    ax.set_ylim(0, max(counts.max() + 0.6, 1.8))
    ax.set_title(title or ('Test-window overlap — each day counted '
                           + format(factor, '.2f') + 'x on average'))
    fg.tidy(ax)
    return writer.save_figure(
        fig, name,
        caption='Number of folds covering each calendar day (' + str(covered)
                + ' fold-days over ' + str(span) + ' distinct days).')


def figure_fold_pnl(summary, writer, name='fold_pnl', config=None):
    """Per-fold P&L with the pooled mean, so a total carried by one fold is visible."""
    configs = summary.get('configs', [])
    if config is not None:
        configs = [c for c in configs if c.get('base_name') == config]
    configs = [c for c in configs if c.get('fold_pnls')]
    if not configs:
        return None

    nrow = len(configs)
    fig, axes = plt.subplots(nrow, 1, figsize=(fg.WIDTH_FULL, 1.5 * nrow + 0.4),
                             squeeze=False, sharex=False)
    for ax, cfg in zip([a for row in axes for a in row], configs):
        pnls = np.asarray(cfg['fold_pnls'], dtype=float)
        xs = np.arange(1, len(pnls) + 1)
        ax.bar(xs, pnls, 0.7, zorder=2,
               color=[fg.CATEGORICAL[2] if v >= 0 else fg.STATUS['critical']
                      for v in pnls])
        mean = float(np.mean(pnls))
        ax.axhline(mean, color=fg.CATEGORICAL[0], linewidth=1.2)
        ax.axhline(0, color=fg.BASELINE, linewidth=0.8)
        ax.text(len(pnls) + 0.2, mean, ' mean ' + format(mean, ',.0f'),
                fontsize=6.5, color=fg.CATEGORICAL[0], va='center')
        ax.set_title(str(cfg.get('base_name')) + '  ·  '
                     + str(cfg.get('folds_profitable')) + '/' + str(len(pnls))
                     + ' fold-runs profitable', fontsize=8)
        ax.set_ylabel('P&L (EUR)')
        fg.tidy(ax)
    axes[-1][0].set_xlabel('Fold run (window x seed), chronological')
    fig.tight_layout()
    return writer.save_figure(fig, name,
                             caption='Per-fold P&L against the pooled mean.')


def figure_forest(table, writer, name='forest'):
    """Mean P&L per trade with the month-clustered 95 % interval, one row per config.

    The interval, not the point estimate, is the result. A configuration whose interval
    straddles zero has not been shown to make money, however large its total.
    """
    df = table.dropna(subset=['mean_pnl_per_trade']).copy()
    if df.empty:
        return None
    df = df.iloc[::-1].reset_index(drop=True)      # best at the top after invert

    fig, ax = plt.subplots(figsize=(fg.WIDTH_FULL, max(1.5, 0.36 * len(df) + 0.9)))
    for i, row in df.iterrows():
        lo, hi = row.get('ci95_low'), row.get('ci95_high')
        reliable = bool(row.get('clusters_reliable'))
        colour = fg.CATEGORICAL[0] if reliable else fg.INK_MUTED
        if pd.notna(lo) and pd.notna(hi):
            ax.plot([lo, hi], [i, i], color=colour, linewidth=1.4, zorder=2)
            ax.plot([lo, lo, hi, hi], [i - 0.12, i + 0.12, i - 0.12, i + 0.12],
                    linestyle='none', marker='|', color=colour, zorder=2)
        ax.plot(row['mean_pnl_per_trade'], i, marker='o', markersize=5, color=colour,
                markeredgecolor=fg.SURFACE, markeredgewidth=1.0, zorder=3)
    ax.axvline(0, color=fg.STATUS['critical'], linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)

    labels = []
    for _, row in df.iterrows():
        clusters = row.get('n_month_clusters')
        tag = '' if row.get('clusters_reliable') else '  (< ' + str(MIN_CLUSTERS) + ' clusters)'
        labels.append(str(row['config']) + '\n' + str(row.get('n_trades')) + ' trades, '
                      + str(clusters) + ' months' + tag)
    ax.set_yticks(range(len(df)), labels)
    ax.set_xlabel('Mean P&L per trade (EUR), 95 % CI clustered on calendar months')
    ax.set_title('Walk-forward evidence per configuration')
    fg.tidy(ax, grid_axis='x')
    return writer.save_figure(
        fig, name,
        caption='Point estimate and month-clustered interval per configuration; grey '
                'rows have too few clusters for the interval to be believed.')


def figure_standard_errors(table, writer, name='standard_errors'):
    """Naive against month-clustered standard error — the size of the correction.

    Treating overlapping folds as independent shrinks the interval by roughly the square
    root of the overlap factor and manufactures evidence that is not there. This makes
    that inflation a measured quantity instead of an argument.
    """
    df = table.dropna(subset=['se_naive', 'se_clustered']).copy()
    if df.empty:
        return None
    xs = np.arange(len(df))
    width = 0.38
    # With one or two configurations a bar of the default width spans the whole panel,
    # which reads as a design accident rather than a comparison. Keep the pair compact
    # and let the axis carry the empty space instead.
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_FULL, 0.36))
    ax.bar(xs - width / 2, df['se_naive'], width * 0.92, color=fg.CATEGORICAL[3],
           label='naive (trades independent)', zorder=2)
    ax.bar(xs + width / 2, df['se_clustered'], width * 0.92, color=fg.CATEGORICAL[0],
           label='clustered on calendar months', zorder=2)
    top = float(max(df['se_naive'].max(), df['se_clustered'].max()))
    for x, (a, b) in enumerate(zip(df['se_naive'], df['se_clustered'])):
        if a > 0:
            ax.text(x, max(a, b) + top * 0.03, format(b / a, '.2f') + 'x',
                    ha='center', fontsize=6.5, color=fg.INK_SECONDARY)
    ax.set_xticks(xs, list(df['config']))
    ax.set_xlim(-0.9, max(len(df) - 0.1, 1.6))
    ax.set_ylim(0, top * 1.32)          # headroom for the ratio labels and the legend
    ax.set_ylabel('Standard error of the mean (EUR/trade)')
    ax.set_title('What the overlap costs in evidence')
    fg.tidy(ax)
    ax.legend(loc='upper right', ncol=1)
    return writer.save_figure(fig, name,
                             caption='Naive and cluster-robust standard errors; the '
                                     'label is the ratio between them.')


def pooled_equity(parallel_summary, config=None):
    """Trades from every fold of one configuration, pooled and ordered by open time.

    Not an equity curve a trader could have had — the folds overlap in calendar time and
    come from separately trained models. It is a picture of the pooled sample, which is
    what the confidence interval is computed on.
    """
    if isinstance(parallel_summary, (str, os.PathLike)):
        parallel_summary = load_json(parallel_summary)
    rows = []
    for res in parallel_summary.get('results', []):
        cfg = res.get('config') or {}
        name = cfg.get('base_name') or cfg.get('name')
        if config is not None and name != config:
            continue
        for t in res.get('trades') or []:
            rows.append({'config': name, 'open_time': t.get('open_time'),
                         'pnl': t.get('pnl'), 'seed': cfg.get('seed')})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
    df = df.dropna(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
    df['cum_pnl'] = df.groupby('config')['pnl'].cumsum()
    return df


def figure_pooled_equity(pooled, writer, name='pooled_equity'):
    """Cumulative pooled P&L per configuration over calendar time."""
    if pooled.empty:
        return None
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_FULL, 0.42))
    for i, (name_, d) in enumerate(pooled.groupby('config')):
        ax.plot(d['open_time'], d['cum_pnl'], color=fg.CATEGORICAL[i % len(fg.CATEGORICAL)],
                linewidth=1.3, label=str(name_) + '  (' + str(len(d)) + ' trades)')
    ax.axhline(0, color=fg.BASELINE, linewidth=0.8)
    ax.set_ylabel('Cumulative pooled P&L (EUR)')
    ax.set_xlabel('Trade open time')
    ax.set_title('Pooled walk-forward trades (overlapping folds, not a tradeable curve)')
    fg.tidy(ax)
    ax.legend(loc='upper left')
    return writer.save_figure(
        fig, name,
        caption='Cumulative P&L over the pooled fold trades. Folds overlap in calendar '
                'time and come from separately trained models, so this is a view of the '
                'sample, not an achievable equity curve.')


# --- Entry point -------------------------------------------------------------------

def run(walk_forward_path, docs_root, stage_id='S6', parallel_path=None, label=None,
        holdout_start=None, config=None):
    """Produce the walk-forward artefact set for one campaign."""
    summary = load_json(walk_forward_path)
    writer = fg.ArtefactWriter(stage_id, docs_root)
    suffix = '' if not label else '_' + label

    table = config_table(summary)
    writer.save_table(table, 'walk_forward_evidence' + suffix,
                      caption='Pooled walk-forward result per configuration with '
                              'month-clustered intervals.')

    folds = fold_frame(summary, config=config)
    if not folds.empty:
        window = summary.get('window') or {}
        geom = ('step ' + str(window.get('step_months')) + 'm, test '
                + str(window.get('test_months')) + 'm, embargo '
                + str(window.get('embargo_days')) + 'd')
        figure_fold_gantt(folds, writer, 'fold_geometry' + suffix,
                          holdout_start=holdout_start,
                          title='Walk-forward fold geometry — ' + geom)
        figure_overlap(folds, writer, 'fold_overlap' + suffix)

    figure_fold_pnl(summary, writer, 'fold_pnl' + suffix, config=config)
    figure_forest(table, writer, 'forest' + suffix)
    figure_standard_errors(table, writer, 'standard_errors' + suffix)

    if parallel_path:
        figure_pooled_equity(pooled_equity(parallel_path, config=config), writer,
                             'pooled_equity' + suffix)

    factor, covered, span = overlap_factor(folds) if not folds.empty else (np.nan, 0, 0)
    writer.save_json({'walk_forward_summary': os.path.abspath(walk_forward_path),
                      'parallel_summary': (os.path.abspath(parallel_path)
                                           if parallel_path else None),
                      'label': label,
                      'measurement': summary.get('measurement'),
                      'window': summary.get('window'),
                      'overlap_factor': None if np.isnan(factor) else factor,
                      'fold_days': covered, 'calendar_days': span,
                      'artifacts': writer.artifacts},
                     'wf_figures' + suffix)
    return writer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk-forward', required=True,
                    help='walk_forward_summary_*.json')
    ap.add_argument('--parallel', default=None,
                    help='parallel_training_summary_*.json, for the pooled equity curve')
    ap.add_argument('--docs-root', default=None)
    ap.add_argument('--stage', default='S6')
    ap.add_argument('--label', default=None,
                    help="suffix distinguishing the geometry, e.g. 'disjoint' or "
                         "'overlapping' — both are produced for the thesis")
    ap.add_argument('--config', default=None,
                    help='restrict the per-fold figures to one configuration')
    ap.add_argument('--holdout-start', default=None,
                    help='mark the sealed hold-out boundary, e.g. 2026-04-20')
    args = ap.parse_args(argv)

    docs_root = args.docs_root or os.path.join(_REPO_ROOT, 'docs')
    writer = run(args.walk_forward, docs_root, stage_id=args.stage,
                 parallel_path=args.parallel, label=args.label,
                 holdout_start=args.holdout_start, config=args.config)
    print('\nWrote ' + str(len(writer.artifacts)) + ' artefacts to ' + writer.root)
    for a in writer.artifacts:
        for path in a['paths'].values():
            print('  ' + a['kind'].ljust(7) + ' ' + writer.relative(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
