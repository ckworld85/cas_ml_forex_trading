"""Performance aggregation across backtest runs.

``backtest_summary.json`` has been written by every backtest for a long time and read
back by nothing. Every comparison table in the project is built from ``trade_list.csv``
instead, and recomputes its own statistics — so the risk-adjusted numbers that only
exist in the summary (Sharpe, Sortino, Calmar, max drawdown, the cost block, the risk
model, the hold-out flag) are invisible in every comparison ever made, and the two
profit factors in circulation are not even the same quantity (``backtest.py`` computes
it from EUR, ``iterative_training.py`` from pips).

This module reads the summaries, puts them in one table, and draws the figures the
economic chapter needs. In particular the **drawdown** curve: the number is computed at
``backtest.py`` and has never been plotted, while a sizing rule that produced a −38.5 %
drawdown was reported for years through metrics that describe the sizing, not the
strategy (see ``utils/risk.py``).

Usage::

    python analytics/perf_report.py --runs ../../generated/a ../../generated/b \\
        --docs-root ../../../docs --stage S9
"""

import argparse
import glob as globlib
import json
import os
import sys

import numpy as np
import pandas as pd

# analytics/__init__.py pulls in model_report, which imports ModelTrading.* — so the
# repo root has to be importable too, not just source/python. Mirrors tests/conftest.py.
_SOURCE_PYTHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SOURCE_PYTHON)))
for _p in (_SOURCE_PYTHON, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analytics import figures as fg  # noqa: E402  (fixes the Agg backend on import)
import matplotlib.pyplot as plt  # noqa: E402

SUMMARY_NAME = 'backtest_summary.json'
TRADES_NAME = 'trade_list.csv'

# The columns of the comparison table, in reading order: what was traded, what it
# earned, what it cost, and what it risked to earn it.
TABLE_COLUMNS = [
    'run', 'period_start', 'period_end', 'total_trades', 'win_rate_pct',
    'total_pnl_eur', 'total_pnl_pips', 'avg_pnl_per_trade_pips',
    'cost_mode', 'total_cost_pips', 'avg_cost_per_trade_pips', 'cost_share_of_gross_pct',
    'risk_mode', 'return_pct', 'cagr_pct', 'sharpe_ratio', 'sortino_ratio',
    'max_drawdown_pct', 'calmar_ratio', 'profit_factor', 'holdout_unsealed',
]


# --- Loading -----------------------------------------------------------------------

def resolve_summary_path(path):
    """Accept a run directory, its report directory, or the JSON file itself."""
    if os.path.isfile(path):
        return path
    for candidate in (os.path.join(path, 'report', SUMMARY_NAME),
                      os.path.join(path, SUMMARY_NAME)):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError('no ' + SUMMARY_NAME + ' under ' + path)


def resolve_trades_path(path):
    """The trade list beside a summary. Returns None when the run has none."""
    if os.path.isfile(path):
        path = os.path.dirname(path)
    for candidate in (os.path.join(path, 'report', TRADES_NAME),
                      os.path.join(path, TRADES_NAME)):
        if os.path.isfile(candidate):
            return candidate
    return None


def load_summary(path):
    """Read one backtest summary and flatten the nested cost/risk/threshold blocks."""
    summary_path = resolve_summary_path(path)
    with open(summary_path, encoding='utf-8') as fh:
        raw = json.load(fh)

    run_dir = os.path.dirname(os.path.dirname(summary_path))
    flat = {'run': os.path.basename(run_dir) or os.path.basename(summary_path),
            'summary_path': summary_path}
    for key, value in raw.items():
        if key == 'costs' and isinstance(value, dict):
            flat['cost_mode'] = value.get('mode')
            for k in ('spread_pips', 'slippage_pips', 'total_pnl_pips_gross',
                      'total_cost_pips', 'total_commission_eur', 'total_overnight_eur',
                      'avg_cost_per_trade_pips', 'cost_share_of_gross_pct'):
                flat[k] = value.get(k)
        elif key == 'risk_model' and isinstance(value, dict):
            flat['risk_mode'] = value.get('mode')
            flat['risk_pct'] = value.get('risk_pct')
            flat['max_leverage'] = value.get('max_leverage')
        elif key == 'thresholds' and isinstance(value, dict):
            for k, v in value.items():
                flat['threshold_' + k] = v
        elif isinstance(value, dict):
            continue          # any other nested block stays out of the flat table
        else:
            flat[key] = value
    return flat


def collect(paths):
    """One row per run. Missing columns become NaN rather than dropping the run."""
    rows = [load_summary(p) for p in paths]
    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    return pd.DataFrame(rows)


def comparison_table(df, columns=None):
    """Project the collected frame onto the reading-order columns."""
    columns = TABLE_COLUMNS if columns is None else columns
    out = df.reindex(columns=[c for c in columns if c in df.columns] +
                     [c for c in columns if c not in df.columns])
    return out


def load_trades(path):
    """Read a run's trade list with the timestamps parsed. None when absent."""
    trades_path = resolve_trades_path(path)
    if trades_path is None:
        return None
    df = pd.read_csv(trades_path)
    if df.empty:
        return df
    for col in ('open_time', 'close_time'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


# --- Equity and drawdown -----------------------------------------------------------

def equity_curve(trades, start_capital=None):
    """Equity after each closed trade, as a time series.

    Uses ``equity_after`` when the backtest wrote it, and reconstructs it from the
    cumulative P&L otherwise, so older trade lists still plot.
    """
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    t = trades.dropna(subset=['close_time']).sort_values('close_time')
    if 'equity_after' in t.columns and t['equity_after'].notna().any():
        return pd.Series(t['equity_after'].to_numpy(float),
                         index=pd.DatetimeIndex(t['close_time']))
    base = 0.0 if start_capital is None else float(start_capital)
    return pd.Series(base + t['pnl'].cumsum().to_numpy(float),
                     index=pd.DatetimeIndex(t['close_time']))


def daily_equity_curve(trades, start_capital, period_start=None, period_end=None):
    """Equity resampled to calendar days over the whole backtest window.

    This reproduces ``backtest.py``'s definition exactly, and it has to: that function
    spans the FULL window and lets every day before the first trade inherit the start
    capital, so the running peak begins at the start capital rather than at the first
    trade's result. Computing the drawdown on the per-trade curve instead gives a
    shallower number — measured on run oof_smoke, -2.0 % against the summary's -3.0 %.
    Two different figures under one name is precisely what this pipeline exists to stop,
    so the figure and ``max_drawdown_pct`` are derived from the same series.
    """
    events = equity_curve(trades, start_capital)
    if events.empty:
        return events
    start = pd.to_datetime(period_start).normalize() if period_start is not None \
        else events.index.min().normalize()
    end = pd.to_datetime(period_end).normalize() if period_end is not None \
        else events.index.max().normalize()
    if end < start:
        start, end = end, start
    daily_index = pd.date_range(start=start, end=end, freq='D')
    daily = events.resample('1D').last().reindex(daily_index).ffill()
    if start_capital is not None:
        daily = daily.fillna(float(start_capital))
    return daily.dropna()


def drawdown_curve(equity):
    """Drawdown in percent of the running peak. Zero at every new high."""
    if equity.empty:
        return equity
    peak = equity.cummax()
    return (equity - peak) / peak.replace(0, np.nan) * 100.0


def figure_equity_drawdown(trades, writer, name='equity_drawdown', label=None,
                           start_capital=None, period_start=None, period_end=None):
    """Equity over the top, the underwater curve beneath it, sharing one time axis.

    Two stacked panels rather than two y-scales on one panel: a second axis would let
    the reader compare two quantities that have no common scale.
    """
    equity = daily_equity_curve(trades, start_capital, period_start, period_end)
    if equity.empty:
        return None
    dd = drawdown_curve(equity)
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(fg.WIDTH_FULL, 3.4), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.12})

    ax_eq.plot(equity.index, equity.to_numpy(), color=fg.CATEGORICAL[0], linewidth=1.4)
    if start_capital is not None:
        ax_eq.axhline(float(start_capital), color=fg.INK_MUTED, linewidth=0.9,
                      linestyle=(0, (4, 3)))
    ax_eq.set_ylabel('Equity (EUR)')
    ax_eq.set_title('Equity and drawdown' + ('' if label is None else '  ·  ' + label))
    fg.tidy(ax_eq)

    ax_dd.fill_between(dd.index, dd.to_numpy(), 0,
                       color=fg.STATUS['critical'], alpha=0.35, linewidth=0)
    ax_dd.plot(dd.index, dd.to_numpy(), color=fg.STATUS['critical'], linewidth=1.0)
    worst = float(dd.min())
    ax_dd.annotate('max ' + format(worst, '.1f') + ' %',
                   xy=(dd.idxmin(), worst), xytext=(4, 6), textcoords='offset points',
                   fontsize=7, color=fg.INK_SECONDARY)
    ax_dd.set_ylabel('Drawdown (%)')
    ax_dd.set_xlabel('Trade close time')
    fg.tidy(ax_dd)
    return writer.save_figure(fig, name,
                             caption='Equity after each closed trade with the '
                                     'underwater curve beneath it.')


# --- Costs -------------------------------------------------------------------------

def figure_cost_bridge(summary, writer, name='cost_bridge'):
    """Gross to net, one bar per deduction — what the execution actually took.

    Reported in pips, because that is the unit in which the cost model is defined and
    the only one that does not also move with the position size.
    """
    gross = summary.get('total_pnl_pips_gross')
    net = summary.get('total_pnl_pips')
    cost = summary.get('total_cost_pips')
    if gross is None or net is None or cost is None:
        return None

    labels = ['Gross', 'Costs', 'Net']
    values = [float(gross), -float(cost), float(net)]
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.5))
    # Running base so the middle bar floats between gross and net.
    bases = [0.0, float(gross) + values[1], 0.0]
    heights = [values[0], -values[1], values[2]]
    colours = [fg.CATEGORICAL[0], fg.STATUS['critical'], fg.CATEGORICAL[2]]
    for x, (base, height, colour) in enumerate(zip(bases, heights, colours)):
        ax.bar(x, height, 0.6, bottom=base, color=colour, zorder=2)
    for x, v in enumerate(values):
        ax.text(x, max(bases[x] + heights[x], bases[x]) + 0.5,
                format(v, '+.1f'), ha='center', fontsize=7, color=fg.INK_SECONDARY)
    ax.axhline(0, color=fg.BASELINE, linewidth=0.8)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel('P&L (pips)')
    share = summary.get('cost_share_of_gross_pct')
    subtitle = '' if share is None else '  ·  costs are ' + format(float(share), '.1f') + ' % of gross'
    ax.set_title('Gross to net' + subtitle)
    fg.tidy(ax)
    return writer.save_figure(fig, name,
                             caption='Transaction costs deducted from the gross result.')


def figure_exit_reasons(trades, writer, name='exit_reasons'):
    """How positions actually left the book, and what each exit earned.

    A strategy whose result is carried by one exit rule is a different object from one
    whose rules all contribute; the bar chart is the fastest way to see which it is.
    """
    if trades is None or trades.empty or 'exit_reason' not in trades.columns:
        return None
    grouped = (trades.groupby('exit_reason')
               .agg(n=('pnl_pips', 'size'), total_pips=('pnl_pips', 'sum'))
               .sort_values('n', ascending=True))
    fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=(fg.WIDTH_FULL, 2.4), sharey=True)
    ypos = np.arange(len(grouped))

    ax_n.barh(ypos, grouped['n'].to_numpy(), 0.7, color=fg.CATEGORICAL[0], zorder=2)
    ax_n.set_yticks(ypos, list(grouped.index))
    ax_n.set_xlabel('Trades')
    ax_n.set_title('Count by exit reason', fontsize=8)
    fg.tidy(ax_n, grid_axis='x')

    pips = grouped['total_pips'].to_numpy(float)
    ax_p.barh(ypos, pips, 0.7, zorder=2,
              color=[fg.CATEGORICAL[2] if v >= 0 else fg.STATUS['critical'] for v in pips])
    ax_p.axvline(0, color=fg.BASELINE, linewidth=0.8)
    ax_p.set_xlabel('Total P&L (pips)')
    ax_p.set_title('P&L by exit reason', fontsize=8)
    fg.tidy(ax_p, grid_axis='x')
    return writer.save_figure(fig, name,
                             caption='Trade count and total P&L per exit reason.')


def figure_mfe_mae(trades, writer, name='mfe_mae'):
    """How far each trade ran in favour before it ran against — by outcome.

    Capped at three colours: this is a scatter, so every pair of series is on screen at
    once and the palette's all-pairs cap applies.
    """
    if trades is None or trades.empty:
        return None
    if not {'highest_profit_pips', 'lowest_pnl_pips', 'pnl_pips'} <= set(trades.columns):
        return None
    fig, ax = plt.subplots(figsize=fg.size(fg.WIDTH_HALF * 2, 0.55))
    win = trades['pnl_pips'] > 0
    for mask, colour, label in ((win, fg.CATEGORICAL_ALL_PAIRS[0], 'winner'),
                                (~win, fg.CATEGORICAL_ALL_PAIRS[1], 'loser')):
        d = trades[mask]
        ax.scatter(d['lowest_pnl_pips'], d['highest_profit_pips'], s=18, alpha=0.75,
                   color=colour, edgecolors=fg.SURFACE, linewidths=0.5, label=label,
                   zorder=2)
    ax.axhline(0, color=fg.BASELINE, linewidth=0.8)
    ax.axvline(0, color=fg.BASELINE, linewidth=0.8)
    ax.set_xlabel('Maximum adverse excursion (pips)')
    ax.set_ylabel('Maximum favourable excursion (pips)')
    ax.set_title('How far each trade ran, by outcome')
    fg.tidy(ax, grid_axis='both')
    ax.legend(loc='upper left')
    return writer.save_figure(fig, name,
                             caption='MFE against MAE per trade, split by outcome.')


# --- Entry point -------------------------------------------------------------------

def run(paths, docs_root, stage_id='S9'):
    """Build the comparison table and, for a single run, the performance figures."""
    writer = fg.ArtefactWriter(stage_id, docs_root)
    df = collect(paths)
    table = comparison_table(df)
    writer.save_table(table, 'backtest_comparison',
                      caption='Backtest results, including the risk-adjusted and cost '
                              'figures that only backtest_summary.json carries.')

    for path in paths:
        summary = load_summary(path)
        trades = load_trades(path)
        suffix = '' if len(paths) == 1 else '_' + str(summary['run'])
        figure_equity_drawdown(trades, writer, 'equity_drawdown' + suffix,
                               label=summary.get('run'),
                               start_capital=summary.get('start_capital_eur'),
                               period_start=summary.get('period_start'),
                               period_end=summary.get('period_end'))
        figure_cost_bridge(summary, writer, 'cost_bridge' + suffix)
        figure_exit_reasons(trades, writer, 'exit_reasons' + suffix)
        figure_mfe_mae(trades, writer, 'mfe_mae' + suffix)

    writer.save_json({'runs': list(df['run']) if len(df) else [],
                      'summary_paths': list(df['summary_path']) if len(df) else [],
                      'artifacts': writer.artifacts}, 'perf_report')
    return writer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', default=[],
                    help='run directories (or backtest_summary.json paths)')
    ap.add_argument('--glob', default=None,
                    help='glob for backtest_summary.json files, e.g. '
                         '"../../generated/*/report/backtest_summary.json"')
    ap.add_argument('--docs-root', default=None, help='docs/ directory')
    ap.add_argument('--stage', default='S9')
    args = ap.parse_args(argv)

    paths = list(args.runs)
    if args.glob:
        paths += sorted(globlib.glob(args.glob))
    if not paths:
        ap.error('give --runs and/or --glob')

    docs_root = args.docs_root
    if docs_root is None:
        docs_root = os.path.join(_REPO_ROOT, 'docs')

    writer = run(paths, docs_root, stage_id=args.stage)
    print('\nWrote ' + str(len(writer.artifacts)) + ' artefacts to ' + writer.root)
    for a in writer.artifacts:
        for path in a['paths'].values():
            print('  ' + a['kind'].ljust(7) + ' ' + writer.relative(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
