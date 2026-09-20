"""
File-based Label Loading Module

Instead of recomputing labels with one of the heuristic label modes
(static / atr_scaled / trend_only / ...), `--label-mode file` reads a
pre-computed label set from a parquet file. This lets labels be developed
somewhere else (e.g. a notebook experiment such as triple-barrier labelling)
and be trained on without porting the label logic into the pipeline first.

Accepted file layouts (index = bar timestamps, M15 grid):

1. Per-model columns — any subset of
   `long_slow`, `short_slow`, `long_fast`, `short_fast`
   (also accepted with a `target_`, `y_` or `y_target_` prefix), values in {0, 1}.
   The slow columns are mandatory; missing fast columns mirror the slow ones.

2. A single signed direction column (`label`, `signal`, `direction`, `y` or
   `target`, or whatever `--label-column` names), values in {-1, 0, +1}:
       +1 -> long_slow = 1
       -1 -> short_slow = 1
        0 -> both 0

The file only has to cover part of the training index — bars that are absent
become NaN and are dropped by the normal NaN-target filter in the pipeline.
"""

import os

import numpy as np
import pandas as pd


LABEL_KEYS = ('long_slow', 'short_slow', 'long_fast', 'short_fast')
SLOW_KEYS = ('long_slow', 'short_slow')

# Prefixes accepted in front of a label key. `y_target_long_slow` is what the
# training pipeline itself writes, so a previous run's export can be re-fed.
_COLUMN_PREFIXES = ('', 'target_', 'y_', 'y_target_')

# Auto-detected names for a single signed (+1/0/-1) direction column.
_SIGNED_COLUMN_CANDIDATES = ('label', 'signal', 'direction', 'y', 'target')

_TIME_COLUMN_CANDIDATES = ('time', 'timestamp', 'date', 'datetime')


def _load_frame(path: str) -> pd.DataFrame:
    """Read the parquet file and return it with a tz-naive DatetimeIndex."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"--label-file not found: {path}")

    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next(
            (c for c in df.columns if str(c).lower() in _TIME_COLUMN_CANDIDATES),
            None,
        )
        if time_col is None:
            raise ValueError(
                f"Label file '{path}' has no DatetimeIndex and no time column "
                f"(looked for {list(_TIME_COLUMN_CANDIDATES)}). "
                "Save the labels with the bar timestamp as index."
            )
        df = df.set_index(time_col)

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if df.index.has_duplicates:
        n_dupes = int(df.index.duplicated().sum())
        raise ValueError(
            f"Label file '{path}' has {n_dupes} duplicate timestamps — "
            "cannot decide which label applies to a bar."
        )

    return df.sort_index()


def _find_column(df: pd.DataFrame, key: str):
    """Return the column of `df` holding label `key`, or None."""
    lookup = {str(c).lower(): c for c in df.columns}
    for prefix in _COLUMN_PREFIXES:
        col = lookup.get(f"{prefix}{key}")
        if col is not None:
            return col
    return None


def _check_values(series: pd.Series, allowed, name: str, path: str) -> pd.Series:
    """Coerce to float and reject anything outside `allowed` (NaN is allowed)."""
    values = pd.to_numeric(series, errors='coerce').astype(float)

    unexpected = values.dropna()
    unexpected = unexpected[~unexpected.isin(allowed)]
    if len(unexpected) > 0:
        raise ValueError(
            f"Label file '{path}' column '{name}' contains values outside "
            f"{sorted(allowed)} (e.g. {sorted(unexpected.unique())[:5]}). "
            f"{len(unexpected)} of {len(values)} rows affected."
        )
    return values


def _labels_from_signed(df: pd.DataFrame, column, path: str) -> dict:
    """Split a +1/0/-1 direction column into long/short slow labels."""
    signed = _check_values(df[column], {-1.0, 0.0, 1.0}, str(column), path)
    valid = signed.notna()

    long_slow = pd.Series(np.nan, index=df.index, dtype=float)
    short_slow = pd.Series(np.nan, index=df.index, dtype=float)
    long_slow[valid] = (signed[valid] > 0).astype(float)
    short_slow[valid] = (signed[valid] < 0).astype(float)

    return {'long_slow': long_slow, 'short_slow': short_slow}


def _labels_from_columns(df: pd.DataFrame, columns: dict, path: str) -> dict:
    """Read the per-model label columns."""
    return {
        key: _check_values(df[col], {0.0, 1.0}, str(col), path)
        for key, col in columns.items()
    }


def _mirror_fast_from_slow(labels: dict) -> dict:
    """A file without fast labels trains the fast models on the slow labels.

    This is the same policy as --fast-use-slow-label, and it keeps the label
    contract (all four keys) intact regardless of the file layout.
    """
    for slow_key, fast_key in (('long_slow', 'long_fast'), ('short_slow', 'short_fast')):
        if fast_key not in labels:
            labels[fast_key] = labels[slow_key].copy()
    return labels


def _print_summary(labels: dict, index: pd.DatetimeIndex, source: str, path: str) -> None:
    covered = int(labels['long_slow'].notna().sum())
    pct = 100.0 * covered / len(index) if len(index) else 0.0
    print(f"\n  [file] Labels loaded from: {path}")
    print(f"  [file] Source layout: {source}")
    print(f"  [file] Covered bars: {covered} / {len(index)} ({pct:.1f}% of the M15 index)")
    for key in LABEL_KEYS:
        series = labels[key]
        n_valid = int(series.notna().sum())
        n_pos = int((series == 1).sum())
        rate = 100.0 * n_pos / n_valid if n_valid else 0.0
        print(f"  [file]   {key:<11} positives: {n_pos} / {n_valid} ({rate:.2f}%)")
    if covered == 0:
        print("  [file] WARNING: no label timestamp matches the M15 index — "
              "check the timezone/grid of the label file.")


def load_labels_from_file(
    path: str,
    index: pd.DatetimeIndex,
    label_column: str = None,
    verbose: bool = True,
) -> dict:
    """
    Load pre-computed labels from a parquet file and align them to `index`.

    Args:
        path:         Path to the parquet file (see module docstring for layouts).
        index:        M15 DatetimeIndex the labels are aligned to. Timestamps that
                      are missing in the file become NaN (dropped downstream).
        label_column: Explicit name of the signed (+1/0/-1) direction column.
                      Forces layout 2 even if per-model columns exist.
        verbose:      Print a coverage/positive-rate summary.

    Returns:
        dict with keys 'long_slow', 'short_slow', 'long_fast', 'short_fast',
        each a float Series reindexed to `index`.
    """
    df = _load_frame(path)

    if label_column is not None:
        lookup = {str(c).lower(): c for c in df.columns}
        column = lookup.get(str(label_column).lower())
        if column is None:
            raise ValueError(
                f"--label-column '{label_column}' not found in '{path}'. "
                f"Available columns: {list(df.columns)}"
            )
        labels = _labels_from_signed(df, column, path)
        source = f"signed column '{column}'"
    else:
        columns = {key: col for key in LABEL_KEYS
                   if (col := _find_column(df, key)) is not None}
        if all(key in columns for key in SLOW_KEYS):
            labels = _labels_from_columns(df, columns, path)
            source = "per-model columns " + ", ".join(
                f"{k}<-{columns[k]}" for k in LABEL_KEYS if k in columns
            )
        else:
            lookup = {str(c).lower(): c for c in df.columns}
            column = next((lookup[c] for c in _SIGNED_COLUMN_CANDIDATES if c in lookup), None)
            if column is None:
                raise ValueError(
                    f"Label file '{path}' provides neither the slow label columns "
                    f"{list(SLOW_KEYS)} nor a signed direction column "
                    f"{list(_SIGNED_COLUMN_CANDIDATES)}. Available columns: "
                    f"{list(df.columns)}. Use --label-column to name it explicitly."
                )
            labels = _labels_from_signed(df, column, path)
            source = f"signed column '{column}' (auto-detected)"

    labels = _mirror_fast_from_slow(labels)
    labels = {key: labels[key].reindex(index) for key in LABEL_KEYS}

    if verbose:
        _print_summary(labels, index, source, path)

    return labels
