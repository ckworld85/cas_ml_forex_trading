"""
Pre-compute the cross-asset currency-strength features once and persist them.

Third instance of a pattern the project already uses twice: `update_timesfm_data.py` and
`update_regime_model_data.py` both compute an expensive, multi-bar-dependent feature block
upfront, write it to `data/*.csv`, and let training consume the CSV rather than recomputing it
every cycle. Here the expense is not compute but *scope*: the features are a statement about
ten instruments, and `advanced_train.py` only ever loads one.

Output
------
`data/cross_asset_daily.csv`      — one row per date, the `daily_ccy_*` columns, unshifted
`data/cross_asset_daily.meta.json` — the universe, the window, the coverage and the residual

The CSV is picked up by `features/external_data.get_external_df`, which applies the `shift(1)`
on this sparse daily index *before* reindexing onto the dense bar index. Values here are
therefore deliberately unshifted; see `features/cross_asset.py`.

COVERAGE IS PART OF THE OUTPUT, NOT A SIDE NOTE
------------------------------------------------
The provider export arrives in waves. At the time of writing the nine additional pairs run
2006-12-31 .. 2023-02 and then jump to 2026-01, while EUR/USD is continuous throughout. A
decomposition needs every pair present on the same day, so the usable index is the
**complete-case intersection** — which is a smaller and differently-shaped window than any
single instrument's. Reporting that intersection is the whole point of `--report`: a feature
frame that silently covers a third of the training window would show up as "the A/B did
nothing" rather than as "the feature was absent".

Usage
-----
    python -m ModelTrading.source.python.data.update_cross_asset_data --report
    python -m ModelTrading.source.python.data.update_cross_asset_data
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
import ModelTrading.config.instruments as instruments  # noqa: E402
import ModelTrading.config.timeframes as tf_config  # noqa: E402
import ModelTrading.source.python.features.cross_asset as ca  # noqa: E402
import ModelTrading.source.python.utils.csv as csv_utils  # noqa: E402

OUTPUT_CSV = 'cross_asset_daily.csv'


def load_panel(symbols=None, data_dir=None, start=None, end=None, verbose=True):
    """Daily close panel (date x symbol), rebuilt from M15 for EVERY pair.

    Why not read the daily CSVs: the delivered daily bars are stamped on a **shifted grid**
    for the non-EURUSD pairs (01:00/02:00 UTC instead of 00:00 — a platform-timezone
    artefact of the export), so their bars cover a different 24-hour window than EURUSD's.
    A decomposition assumes every pair's return is measured over the SAME window; mixing
    boundaries builds the classic non-synchronicity bias straight into the panel.

    The M15 files all share one grid, so the daily close is rebuilt as the last M15 close
    of each 00:00-UTC day — for every pair including EURUSD, one code path, no convention
    detection. Verified against the provider daily for EURUSD: 5,555 common days, 100 %
    within one pip (mean |diff| 2.5e-8). The same check against GBPUSD's *delivered* daily
    file agrees on only 14 % of days (median 5.1 pips), which is the shifted window showing
    itself, and the reason this function refuses to read those files.

    Weekend filtering and ISO-8601 parsing come from `utils.csv.load_csv`, so the panel
    inherits exactly the bar selection the rest of the pipeline uses.
    """
    data_dir = data_dir or dir_config.DATA_DIR
    have = instruments.available_symbols('m15', data_dir, symbols)
    missing = [s for s in (symbols or instruments.SYMBOLS) if s not in have]
    if verbose and missing:
        print(f"  no m15 CSV on disk, skipped: {', '.join(missing)}")

    closes, meta = {}, {}
    for sym in have:
        df = csv_utils.load_csv(instruments.csv_path(sym, 'm15', data_dir))
        daily = df['close'].astype(float).groupby(df.index.normalize()).last()
        if start:
            daily = daily[daily.index >= pd.to_datetime(start)]
        if end:
            daily = daily[daily.index <= pd.to_datetime(end)]
        closes[sym] = daily
        meta[sym] = dict(n=int(len(daily)), start=str(daily.index.min().date()),
                         end=str(daily.index.max().date()))
        if verbose:
            print(f"  {sym}: {len(daily):,} days rebuilt from {len(df):,} M15 bars")
    if not closes:
        raise SystemExit(f"no m15 CSVs found in {data_dir}")
    panel = pd.DataFrame(closes).sort_index()
    return panel, meta


def coverage_report(panel, meta):
    """Per-symbol coverage and the complete-case intersection the decomposition can use."""
    complete = panel.dropna()
    seg = ca.segment_ids(complete.index)
    lines = []
    lines.append(f"{'symbol':<8} {'bars':>7} {'from':>12} {'to':>12}")
    lines.append('-' * 43)
    for sym in panel.columns:
        m = meta[sym]
        lines.append(f"{sym:<8} {m['n']:>7} {m['start']:>12} {m['end']:>12}")
    lines.append('')
    lines.append(f"complete cases (every pair present): {len(complete):,} of "
                 f"{len(panel):,} union dates")
    if len(complete):
        lines.append(f"  span {complete.index.min().date()} .. {complete.index.max().date()}")
        for s in sorted(seg.unique()):
            part = complete.index[seg.values == s]
            lines.append(f"  segment {s}: {len(part):>6} bars  "
                         f"{part.min().date()} .. {part.max().date()}")
    return '\n'.join(lines), complete


def build(symbols=None, symbol=instruments.PRIMARY, lookbacks=ca.DEFAULT_LOOKBACKS,
          exclude_self=True, data_dir=None, start=None, end=None, verbose=True):
    panel, meta = load_panel(symbols, data_dir, start, end, verbose=verbose)
    report, complete = coverage_report(panel, meta)
    if verbose:
        print(report)
    if len(complete) < 300:
        raise SystemExit(f"only {len(complete)} complete-case dates — not enough to build a "
                         f"250-day momentum. Wait for the export to finish.")

    returns = np.log(complete).diff().dropna()
    # A diff straight across the export hole would book one ~3-year "daily return".
    # The rolling features are already segment-guarded; this guards the decomposition.
    gap_days = returns.index.to_series().diff().dt.days.fillna(1.0)
    n_cross = int((gap_days > ca.MAX_GAP_DAYS).sum())
    if n_cross and verbose:
        print(f"  dropped {n_cross} return row(s) spanning a gap > {ca.MAX_GAP_DAYS} days")
    returns = returns[gap_days <= ca.MAX_GAP_DAYS]
    feats = ca.build_features(returns, symbol=symbol, lookbacks=lookbacks,
                              exclude_self=exclude_self)
    info = {
        'symbol': symbol,
        'exclude_self': bool(exclude_self),
        'symbols_used': feats.attrs['symbols_used'],
        'currencies': feats.attrs['currencies'],
        'lookbacks': list(lookbacks),
        'resid_rms_mean': feats.attrs['resid_rms_mean'],
        'n_dates': int(len(feats)),
        'first_date': str(feats.index.min().date()),
        'last_date': str(feats.index.max().date()),
        'per_symbol': meta,
        'features': list(feats.columns),
    }
    return feats, info, report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--symbol', default=instruments.PRIMARY,
                   help='the traded pair the features describe')
    p.add_argument('--symbols', default=None,
                   help='comma-separated universe; default every registered pair on disk')
    p.add_argument('--lookbacks', default=','.join(str(x) for x in ca.DEFAULT_LOOKBACKS))
    p.add_argument('--include-self', action='store_true',
                   help='keep the traded pair in the design matrix. Off by default: with it '
                        'in, the feature contains its own target (the DXY failure mode).')
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None,
                   help=f'defaults to the sealed hold-out boundary '
                        f'({tf_config.HOLDOUT_START.date()})')
    p.add_argument('--report', action='store_true',
                   help='print coverage and exit without writing')
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols else None
    lookbacks = tuple(int(x) for x in args.lookbacks.split(',') if x.strip())
    end = args.end or str((tf_config.HOLDOUT_START - pd.Timedelta(days=1)).date())

    if args.report:
        panel, meta = load_panel(symbols, start=args.start, end=end)
        report, _ = coverage_report(panel, meta)
        print(report)
        return

    feats, info, _ = build(symbols=symbols, symbol=args.symbol, lookbacks=lookbacks,
                           exclude_self=not args.include_self, start=args.start, end=end)
    out_csv = os.path.join(dir_config.DATA_DIR, OUTPUT_CSV)
    frame = feats.copy()
    frame.index.name = 'date'
    frame.to_csv(out_csv)
    with open(out_csv.replace('.csv', '.meta.json'), 'w') as fh:
        json.dump(info, fh, indent=2)

    print(f"\nwrote {out_csv}")
    print(f"  {info['n_dates']:,} dates, {info['first_date']} .. {info['last_date']}")
    print(f"  universe: {', '.join(info['symbols_used'])}")
    print(f"  currencies: {', '.join(info['currencies'])}")
    print(f"  mean decomposition residual: {info['resid_rms_mean']:.3e}")
    print(f"  features: {', '.join(info['features'])}")
    na = feats.isna().mean()
    print("\n  NaN share per feature (warm-up plus segment boundaries):")
    for k, v in na.items():
        print(f"    {k:<34} {v:6.1%}")


if __name__ == '__main__':
    main()
