"""Deflated Sharpe Ratio and Probability of Backtest Overfitting.

``docs/preregistration.md`` §7 names this module and requires DSR and PBO "for any
P&L-based claim". It did not exist. This closes that.

Both statistics answer the same question from different directions: **this project has
run roughly 1,800 backtests and 324 walk-forward trainings over one price history**, and
the best of many results is high because many were tried, not because it is good.

* **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) discounts an observed Sharpe
  by how many configurations were tried, how they were spread, and how non-normal the
  returns are. It answers: given that I searched, is this Sharpe still distinguishable
  from zero? A DSR below 0.95 means the result is within what the search itself would
  produce. Note the direction of the correction — reporting the *number of trials
  honestly* is what makes the statistic meaningful, so ``n_trials`` must be the size of
  the whole search, not the size of the table finally shown.

* **Probability of Backtest Overfitting** (Bailey et al., 2015; CSCV) needs no trial
  count. It splits the period into ``S`` blocks, takes every way of halving them into
  in-sample and out-of-sample, picks the configuration that won in-sample, and asks how
  it ranked out-of-sample. PBO is the share of splits in which the in-sample winner
  landed **below the out-of-sample median** — i.e. how often selecting on backtest
  performance is worse than picking at random. Above ~0.5 the selection procedure has no
  skill at all.

Neither is a verdict on a strategy. They are verdicts on a *selection procedure*, which
is exactly what a grid search is.

Usage::

    python analytics/backtest_statistics.py \\
        --walk-forward ../../generated/walk_forward_summary_20260829_120000.json \\
        --docs-root ../../../docs --stage S5
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_SOURCE_PYTHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SOURCE_PYTHON)))
for _p in (_SOURCE_PYTHON, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EULER_MASCHERONI = 0.5772156649015329


# --- Sharpe -------------------------------------------------------------------------

def sharpe_ratio(returns, periods_per_year=None):
    """Sharpe of a return series. Not annualised unless ``periods_per_year`` is given.

    The deflation below operates on the **per-period** Sharpe, so leave the annualisation
    off when feeding ``deflated_sharpe_ratio``: annualising multiplies the estimate but
    not its sampling distribution, and the two must be on the same scale.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float('nan')
    sr = r.mean() / r.std(ddof=1)
    return sr * math.sqrt(periods_per_year) if periods_per_year else sr


def expected_max_sharpe(n_trials, trial_sharpe_std):
    """The Sharpe the *best of N tries* reaches when none of them has any skill.

    This is the benchmark the observed Sharpe has to clear. It grows with the number of
    trials and with how widely the trials scatter — both of which a grid search inflates
    on purpose.
    """
    n_trials = int(n_trials)
    if n_trials < 2 or not np.isfinite(trial_sharpe_std) or trial_sharpe_std <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sharpe_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(returns, n_trials, trial_sharpe_std=None,
                          benchmark_sharpe=None):
    """Probability that the true Sharpe exceeds what the search alone would produce.

    Args:
        returns: the selected configuration's per-period returns.
        n_trials: how many configurations were tried in the search that produced it.
        trial_sharpe_std: spread of the per-period Sharpes across those trials. Required
            unless ``benchmark_sharpe`` is given directly.
        benchmark_sharpe: use this benchmark instead of deriving it from the trials.

    Returns:
        dict with the observed Sharpe, the benchmark, skew, kurtosis, n, and ``dsr``
        (a probability). ``dsr`` is NaN when the series is too short to say anything.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    sr = sharpe_ratio(r)
    out = {'sharpe': sr, 'n_observations': n, 'n_trials': int(n_trials),
           'skew': float('nan'), 'kurtosis': float('nan'),
           'benchmark_sharpe': float('nan'), 'dsr': float('nan')}
    if n < 4 or not np.isfinite(sr):
        return out

    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False, fisher=False))   # non-excess
    sr0 = (float(benchmark_sharpe) if benchmark_sharpe is not None
           else expected_max_sharpe(n_trials, trial_sharpe_std))

    # Variance of the Sharpe estimator under non-normal returns (Bailey & LdP eq. 9).
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    out.update({'skew': skew, 'kurtosis': kurt, 'benchmark_sharpe': sr0})
    if denom <= 0:
        return out
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    out['dsr'] = float(stats.norm.cdf(z))
    return out


# --- PBO ----------------------------------------------------------------------------

def probability_of_backtest_overfitting(matrix, n_splits=16, metric=None):
    """Combinatorially symmetric cross-validation (CSCV).

    Args:
        matrix: DataFrame, rows = periods (in time order), columns = configurations.
            One cell is that configuration's P&L in that period.
        n_splits: number of contiguous blocks the period is cut into. Must be even.
            ``C(S, S/2)`` combinations are evaluated — 12,870 at S=16, 252 at S=10.
        metric: performance of a return vector. Defaults to the per-period Sharpe.

    Returns:
        dict with ``pbo``, the logit series, the out-of-sample relative ranks of the
        in-sample winners, and how often each configuration was the in-sample winner.

    PBO is the share of splits where the in-sample winner came out **below the
    out-of-sample median**. 0.5 means selecting on backtest performance is a coin flip;
    above 0.5 it is worse than not selecting at all.
    """
    metric = sharpe_ratio if metric is None else metric
    m = pd.DataFrame(matrix).dropna(axis=1, how='all')
    if m.shape[1] < 2:
        raise ValueError('PBO needs at least 2 configurations, got ' + str(m.shape[1]))
    if n_splits % 2 or n_splits < 2:
        raise ValueError('n_splits must be a positive even number')
    if len(m) < n_splits:
        raise ValueError('need at least one row per split: ' + str(len(m)) +
                         ' rows for ' + str(n_splits) + ' splits')

    blocks = np.array_split(np.arange(len(m)), n_splits)
    half = n_splits // 2
    values = m.to_numpy(dtype=float)
    columns = list(m.columns)

    logits, ranks, winners = [], [], []
    for chosen in itertools.combinations(range(n_splits), half):
        rest = [b for b in range(n_splits) if b not in chosen]
        is_rows = np.concatenate([blocks[b] for b in chosen])
        oos_rows = np.concatenate([blocks[b] for b in rest])

        is_perf = np.array([metric(values[is_rows, j]) for j in range(len(columns))])
        oos_perf = np.array([metric(values[oos_rows, j]) for j in range(len(columns))])
        if not np.isfinite(is_perf).any() or not np.isfinite(oos_perf).any():
            continue

        best = int(np.nanargmax(is_perf))
        winners.append(columns[best])
        # Relative rank of the in-sample winner among the out-of-sample results.
        finite = np.isfinite(oos_perf)
        if finite.sum() < 2 or not finite[best]:
            continue
        order = stats.rankdata(oos_perf[finite])
        pos = int(np.flatnonzero(np.flatnonzero(finite) == best)[0])
        omega = order[pos] / (finite.sum() + 1.0)      # in (0, 1)
        ranks.append(float(omega))
        logits.append(math.log(omega / (1.0 - omega)))

    if not logits:
        return {'pbo': float('nan'), 'n_combinations': 0, 'logits': [],
                'oos_ranks': [], 'winner_counts': {}}

    logits_arr = np.asarray(logits)
    return {
        'pbo': float((logits_arr < 0).mean()),
        'n_combinations': len(logits),
        'n_splits': n_splits,
        'n_configurations': len(columns),
        'median_oos_rank': float(np.median(ranks)),
        'logits': logits_arr.tolist(),
        'oos_ranks': ranks,
        'winner_counts': {c: int(winners.count(c)) for c in sorted(set(winners))},
    }


# --- Building the matrix from campaign artefacts --------------------------------------

def monthly_pnl_matrix(parallel_summary, min_months=12):
    """Periods x configurations matrix of monthly P&L from a campaign summary.

    Reads the per-trade records that ``iterative_training`` stores in
    ``parallel_training_summary_*.json`` (they are captured before the run directories
    are deleted). Monthly rather than per-fold, because CSCV needs more periods than a
    walk-forward has folds — and the calendar month is the unit the project already
    clusters its standard errors on.
    """
    if isinstance(parallel_summary, (str, os.PathLike)):
        with open(parallel_summary, encoding='utf-8') as fh:
            parallel_summary = json.load(fh)

    frames = {}
    for res in parallel_summary.get('results', []):
        trades = res.get('trades') or []
        if not trades:
            continue
        cfg = res.get('config') or {}
        name = cfg.get('base_name') or cfg.get('name') or cfg.get('run_id')
        df = pd.DataFrame(trades)
        if 'open_time' not in df.columns or 'pnl' not in df.columns:
            continue
        months = pd.to_datetime(df['open_time'], errors='coerce').dt.to_period('M')
        series = df.groupby(months)['pnl'].sum()
        frames[name] = series.add(frames[name], fill_value=0) if name in frames else series

    if not frames:
        return pd.DataFrame()
    matrix = pd.DataFrame(frames).sort_index()
    # A month in which a configuration simply did not trade is a zero return, not a gap.
    matrix = matrix.fillna(0.0)
    if len(matrix) < min_months:
        return matrix.iloc[0:0]
    return matrix


def fold_pnl_matrix(walk_forward_summary):
    """Folds x configurations matrix from a walk_forward_summary.

    Coarser than :func:`monthly_pnl_matrix` — a walk-forward has ~12 folds, which is
    barely enough for CSCV — but it works when the per-trade records are unavailable.
    """
    if isinstance(walk_forward_summary, (str, os.PathLike)):
        with open(walk_forward_summary, encoding='utf-8') as fh:
            walk_forward_summary = json.load(fh)
    cols = {}
    for cfg in walk_forward_summary.get('configs', []):
        pnls = cfg.get('fold_pnls')
        if pnls:
            cols[cfg['base_name']] = list(pnls)
    if not cols:
        return pd.DataFrame()
    width = min(len(v) for v in cols.values())
    return pd.DataFrame({k: v[:width] for k, v in cols.items()})


def campaign_statistics(matrix, n_trials=None, n_splits=None):
    """DSR for the best configuration plus PBO for the selection procedure itself."""
    matrix = pd.DataFrame(matrix)
    if matrix.empty or matrix.shape[1] < 2:
        return {'error': 'need a periods x configurations matrix with >= 2 columns'}

    sharpes = {c: sharpe_ratio(matrix[c]) for c in matrix.columns}
    finite = {c: v for c, v in sharpes.items() if np.isfinite(v)}
    if not finite:
        return {'error': 'no configuration has a finite Sharpe'}

    best = max(finite, key=finite.get)
    trial_std = float(np.std(list(finite.values()), ddof=1)) if len(finite) > 1 else 0.0
    trials = int(n_trials if n_trials is not None else matrix.shape[1])

    # As many splits as the period length allows, capped at 16 (12,870 combinations).
    if n_splits is None:
        n_splits = max(2, min(16, (len(matrix) // 2) * 2))

    dsr = deflated_sharpe_ratio(matrix[best], n_trials=trials, trial_sharpe_std=trial_std)
    out = {'best_configuration': best,
           'per_configuration_sharpe': sharpes,
           'trial_sharpe_std': trial_std,
           'n_periods': int(len(matrix)),
           'deflated_sharpe': dsr}
    try:
        out['pbo'] = probability_of_backtest_overfitting(matrix, n_splits=n_splits)
    except ValueError as e:
        out['pbo'] = {'error': str(e)}
    return out


# --- Reporting ------------------------------------------------------------------------

def print_report(result):
    """Console summary. The interpretation lines are part of the output on purpose."""
    print('\n' + '=' * 78)
    print('BACKTEST STATISTICS — deflated Sharpe and probability of overfitting')
    print('=' * 78)
    if 'error' in result:
        print('  ' + result['error'])
        return

    dsr = result['deflated_sharpe']
    print('  Best configuration      : ' + str(result['best_configuration']))
    print('  Periods                 : ' + str(result['n_periods']))
    print('  Trials counted          : ' + str(dsr['n_trials']))
    print('  Sharpe (per period)     : ' + format(dsr['sharpe'], '.4f'))
    print('  Benchmark (best of N)   : ' + format(dsr['benchmark_sharpe'], '.4f'))
    print('  Skew / kurtosis         : ' + format(dsr['skew'], '.3f') + ' / '
          + format(dsr['kurtosis'], '.3f'))
    print('  Deflated Sharpe (DSR)   : ' + format(dsr['dsr'], '.4f'))
    if np.isfinite(dsr['dsr']):
        print('    -> ' + ('clears 0.95: the Sharpe survives the search'
                           if dsr['dsr'] >= 0.95 else
                           'below 0.95: within what the search alone produces'))

    pbo = result.get('pbo') or {}
    if 'error' in pbo:
        print('  PBO                     : not computed (' + pbo['error'] + ')')
        return
    if pbo.get('n_combinations'):
        print('  PBO                     : ' + format(pbo['pbo'], '.4f')
              + '  (' + str(pbo['n_combinations']) + ' splits of '
              + str(pbo['n_splits']) + ' blocks)')
        print('    -> ' + ('at or above 0.5: selecting on backtest performance has no '
                           'skill' if pbo['pbo'] >= 0.5 else
                           'below 0.5: the in-sample winner tends to stay above the '
                           'out-of-sample median'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--parallel-summary',
                     help='parallel_training_summary_*.json (per-trade records -> '
                          'monthly matrix; preferred)')
    src.add_argument('--walk-forward',
                     help='walk_forward_summary_*.json (fold P&L matrix; coarser)')
    ap.add_argument('--n-trials', type=int, default=None,
                    help='size of the whole search, if larger than the number of '
                         'configurations in this file. Understating it flatters the DSR.')
    ap.add_argument('--n-splits', type=int, default=None,
                    help='CSCV blocks (even). Default: as many as the periods allow, '
                         'capped at 16.')
    ap.add_argument('--json', default=None, help='write the result here')
    args = ap.parse_args(argv)

    if args.parallel_summary:
        matrix = monthly_pnl_matrix(args.parallel_summary)
        source = args.parallel_summary
    else:
        matrix = fold_pnl_matrix(args.walk_forward)
        source = args.walk_forward

    if matrix.empty:
        print('No usable periods x configurations matrix in ' + source)
        return 1

    result = campaign_statistics(matrix, n_trials=args.n_trials, n_splits=args.n_splits)
    result['source'] = source
    print_report(result)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=2, default=str)
        print('\nWrote ' + args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
