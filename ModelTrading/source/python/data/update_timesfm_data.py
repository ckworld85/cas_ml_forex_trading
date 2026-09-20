"""
Pre-compute and persist TimesFM forecast features for training.

TimesFM inference is expensive (a 200M neural time-series model). Recomputing it
on every training run is wasteful, so this script computes the rolling forecast
once per timeframe and writes the result to a CSV in ModelTrading/data/:

  timesfm_m15.csv
  timesfm_4hours.csv
  timesfm_daily.csv

Each CSV is indexed by bar timestamp (same OHLC source and weekend filter as
training) with one column per horizon-suffixed stat, e.g.:

  date, tfm_mean_144, tfm_q10_144, tfm_q90_144, tfm_spread_144, tfm_conf_144

The forecast horizon is NOT a parameter — it is encoded in each tfm feature-name
suffix in features.yaml (m15 -> 144, 4hours -> 9, daily -> 2). This script reads
those names to discover which horizon(s) to compute per timeframe.

Consumption
-----------
  * Training (advanced_train / train): indicators.add_features(compute_timesfm=False)
    loads these CSVs as external features.
  * Live inference (feature_server): indicators.add_features(compute_timesfm=True)
    recomputes the forecast on the fly — these CSVs are NOT used at inference and
    are therefore not staged to TEST/PRODUCTION.

Incremental by default
-----------------------
Each run MERGES into the existing CSV instead of overwriting it:

  * Bars (rows) already present in the CSV are kept and NOT recomputed.
  * Only new bars (and any horizon/stat columns missing from the CSV) are
    computed and merged in.
  * ``--overwrite`` forces a full recompute of every requested timeframe,
    replacing the existing CSV.

``--start-date`` means "I need features FROM this date". TimesFM needs
``context_length`` prior bars to forecast, so the script automatically loads
those warmup bars from before the start date and drops them from the output —
the saved features begin exactly at the requested date with valid (non-NaN)
values.

Usage:
  python data/update_timesfm_data.py                       # all timeframes, merge new bars
  python data/update_timesfm_data.py --timeframes daily    # one timeframe only
  python data/update_timesfm_data.py --start-date 2026-01-01   # features from 2026-01-01
  python data/update_timesfm_data.py --overwrite           # recompute everything
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path so "ModelTrading" package is importable.
# This file lives at ModelTrading/source/python/data/update_timesfm_data.py
# parents: [0]=data/, [1]=python/, [2]=source/, [3]=ModelTrading/, [4]=project root
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import ModelTrading.config.directories as dir_config
from ModelTrading.source.python.utils import csv as csv_utils
from ModelTrading.source.python.features.config import get_feature_config
from ModelTrading.source.python.features.timesfm_features import (
    TFM_STATS,
    compute_timesfm_features_multi,
    is_tfm_feature,
    parse_tfm_feature,
    timesfm_csv_path,
)

# Timeframe -> OHLC CSV filename. Mirrors advanced_train.load_and_prepare_data.
_TIMEFRAME_CSV = {
    "m15":    "eurusd_m15.csv",
    "4hours": "eurusd_4hours.csv",
    "daily":  "eurusd_daily.csv",
}


def _horizons_for_timeframe(config, timeframe: str) -> dict:
    """Return ``{horizon: set(TFM_STATS)}`` for all tfm features of a timeframe.

    Every horizon found in features.yaml gets the full set of stats so the CSV
    is complete regardless of which individual stats are currently enabled.
    Disabled features are included too, so the data is ready the moment a
    feature is flipped to enabled.
    """
    prefix = f"{timeframe}_"
    horizons: dict[int, set] = {}
    for name in config.get_features():           # enabled or not
        if not name.startswith(prefix):
            continue
        bare = name.removeprefix(prefix)
        if not is_tfm_feature(bare):
            continue
        _, horizon = parse_tfm_feature(bare)
        horizons[horizon] = set(TFM_STATS)
    return horizons


def _load_existing_csv(out_path: Path):
    """Load a previously written timesfm CSV indexed by tz-naive UTC timestamp.

    Mirrors the date parsing in ``timesfm_features._load_timesfm_features`` so
    the index aligns exactly with the OHLC bar timestamps. Returns ``None`` if
    the file is absent or unreadable.
    """
    if not out_path.exists():
        return None
    raw = pd.read_csv(out_path)
    if "date" not in raw.columns:
        return None
    raw["date"] = pd.to_datetime(raw["date"], format="ISO8601")
    if raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return raw.set_index("date").sort_index()


def compute_timeframe(
    timeframe: str,
    config,
    start_date=None,
    end_date=None,
    overwrite: bool = False,
    verbose: bool = True,
) -> bool:
    """Compute and persist TimesFM features for one timeframe. Returns success.

    By default this MERGES into any existing CSV: bars and columns already
    present are kept and not recomputed; only new bars (and missing
    horizon/stat columns) are computed. ``overwrite=True`` recomputes the full
    history and replaces the CSV.
    """
    horizons = _horizons_for_timeframe(config, timeframe)
    if not horizons:
        if verbose:
            print(f"  [{timeframe}] no tfm features defined in features.yaml — skipped.")
        return True

    csv_name = _TIMEFRAME_CSV[timeframe]
    ohlc_path = Path(dir_config.DATA_DIR) / csv_name
    if not ohlc_path.exists():
        print(f"  ERROR [{timeframe}]: OHLC CSV not found: {ohlc_path}", file=sys.stderr)
        return False

    if verbose:
        hz = ", ".join(str(h) for h in sorted(horizons))
        print(f"  [{timeframe}] loading {csv_name} (horizons: {hz}) …")

    params = config.get_parameters()
    context_length = params.get("tfm_context_length", 512)
    out_path = timesfm_csv_path(timeframe)

    # --start-date asks for *features* from that date. TimesFM needs context_length
    # prior bars (plus one for the log-return diff) before a bar can be forecast,
    # so we load those warmup bars too and forecast over the extended window, then
    # drop the pre-start warmup rows from the output. Bars are counted positionally
    # on the weekend-filtered series so the warmup is correct regardless of gaps.
    warmup_bars = context_length + 1
    start_floor = pd.Timestamp(start_date) if start_date is not None else None

    df = csv_utils.load_csv(
        str(ohlc_path),
        start_date=None,            # filtered below, after reserving warmup bars
        end_date=end_date,
        filter_weekends_flag=True,
    )

    if start_floor is not None:
        first_pos = int(df.index.searchsorted(start_floor, side="left"))
        load_start = max(0, first_pos - warmup_bars)
        if verbose:
            avail = first_pos - load_start  # warmup bars actually available
            short = " (insufficient history — leading rows will be NaN)" if avail < warmup_bars else ""
            print(f"  [{timeframe}] features from {start_floor:%Y-%m-%d}; "
                  f"loaded {avail} warmup bars before it (need {warmup_bars}){short}.")
        df = df.iloc[load_start:]

    existing = None if overwrite else _load_existing_csv(out_path)

    # Decide, per horizon, what (if anything) needs computing.
    #   * no existing CSV / --overwrite -> compute every bar.
    #   * a needed column is missing     -> compute it over every bar.
    #   * only new bars are missing      -> compute just the tail (plus enough
    #                                       context bars to forecast the first
    #                                       new bar), keeping prior rows as-is.
    computed = []
    for horizon in sorted(horizons):
        stats = horizons[horizon]
        cols = [f"tfm_{stat}_{horizon}" for stat in sorted(stats)]

        if existing is None:
            sub_df, reason = df, "full history"
        else:
            missing_cols = [c for c in cols if c not in existing.columns]
            new_bars = df.index.difference(existing.index)
            if missing_cols:
                sub_df, reason = df, f"full (new columns: {', '.join(missing_cols)})"
            elif len(new_bars):
                first_pos = int(df.index.get_indexer([new_bars.min()])[0])
                slice_start = max(0, first_pos - context_length)
                sub_df = df.iloc[slice_start:]
                reason = f"{len(new_bars)} new bars"
            else:
                sub_df, reason = None, "up to date"

        if sub_df is None:
            if verbose:
                print(f"  [{timeframe}] horizon {horizon}: {reason} — skipped.")
            continue

        if verbose:
            print(f"  [{timeframe}] horizon {horizon}: {reason} — TimesFM over {len(sub_df)} bars …")

        res = compute_timesfm_features_multi(
            sub_df,
            {horizon: stats},
            context_length=context_length,
            batch_size=params.get("tfm_batch_size", 256),
            freq=params.get(f"tfm_freq_{timeframe}", 0),
            model_repo=params.get("tfm_model_repo", "google/timesfm-2.5-200m-pytorch"),
        )
        computed.append(res)

    if not computed:
        if verbose:
            rows = len(existing) if existing is not None else 0
            print(f"  [{timeframe}] nothing to compute — CSV already current ({rows} rows).")
        return True

    new_data = pd.concat(computed, axis=1) if len(computed) > 1 else computed[0]

    # The warmup bars before --start-date only provided forecast context; drop
    # them so the output starts at the requested date. Existing rows below the
    # floor are still preserved by the merge below.
    if start_floor is not None:
        new_data = new_data[new_data.index >= start_floor]
        if new_data.empty:
            if verbose:
                print(f"  [{timeframe}] nothing to compute at/after {start_floor:%Y-%m-%d} — skipped.")
            return True

    # Newly computed values win where they overlap existing ones (identical for
    # recomputed context bars); existing rows/columns are preserved otherwise.
    result = new_data if existing is None else new_data.combine_first(existing)

    # Keep the requested feature columns first (in a stable order), then any
    # extra columns the existing CSV may still carry.
    target_cols = [f"tfm_{stat}_{h}" for h in sorted(horizons) for stat in sorted(horizons[h])]
    ordered = [c for c in target_cols if c in result.columns]
    ordered += [c for c in result.columns if c not in ordered]
    result = result[ordered].sort_index()
    for col in result.columns:
        result[col] = result[col].astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.index.name = "date"
    result.to_csv(out_path, index=True, date_format="%Y-%m-%d %H:%M:%S")

    if verbose:
        valid = int(result.iloc[:, 0].notna().sum()) if len(result.columns) else 0
        print(f"  [{timeframe}] saved {len(result)} rows ({valid} valid) → {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute TimesFM forecast features and persist them to data/."
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=sorted(_TIMEFRAME_CSV),
        default=sorted(_TIMEFRAME_CSV),
        help="Which timeframes to compute (default: all).",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Compute features FROM this ISO date. Warmup bars before it are loaded "
             "automatically so the first output row is valid (non-NaN).",
    )
    parser.add_argument("--end-date", default=None, help="Optional ISO end date filter.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the full history and replace the CSV (default: merge new bars only).",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    args = parser.parse_args()

    verbose = not args.quiet
    config = get_feature_config()

    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Computing TimesFM features: {args.timeframes}")

    errors = 0
    for timeframe in args.timeframes:
        try:
            ok = compute_timeframe(
                timeframe, config,
                start_date=args.start_date, end_date=args.end_date,
                overwrite=args.overwrite,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001 — report and continue with next timeframe
            print(f"  ERROR [{timeframe}]: {exc}", file=sys.stderr)
            ok = False
        errors += 0 if ok else 1

    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Done: {len(args.timeframes) - errors} ok, {errors} failed.")

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
