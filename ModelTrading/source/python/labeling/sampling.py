"""
Training-Data Sampling Module

Shared utilities for sub-sampling training data before model fit.
Used by train.py, advanced_train.py, and iterative_training.py.

Goal: keep all minority-class (label=1) bars and thin out majority-class
(label=0) bars. Multiple inclusive strategies can be combined:

  d1) Baseline (always on, non-configurable):
      - All bars where y == 1 are always kept.

  d2) Stride sampling (default on):
      - Every Xth label=0 bar is kept.

  d3) Context sampling (default on):
      - All bars within [t - hours_before, t + hours_after] of any t
        where y == 1 are kept.

Final index = (y==1)  ∪  (d2 picks if enabled)  ∪  (d3 picks if enabled).
If the master flag is off, the original index is returned unchanged.
"""

import argparse
import numpy as np
import pandas as pd


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """
    Add training-data sampling CLI arguments to an argparse parser.

    The master flag and the two strategy flags are default-on; they are
    disabled via `--no-*` variants to keep the existing codebase convention
    (defaults express recommended behaviour).

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        '--no-training-sampling',
        dest='training_sampling',
        action='store_false',
        default=True,
        help='Disable training-data sampling entirely (default: sampling on).',
    )
    parser.add_argument(
        '--no-sampling-stride',
        dest='sampling_stride',
        action='store_false',
        default=True,
        help='Disable stride-based sub-sampling of label=0 bars (default: on).',
    )
    parser.add_argument(
        '--no-sampling-context',
        dest='sampling_context',
        action='store_false',
        default=True,
        help='Disable context-window inclusion around label=1 bars (default: on).',
    )
    parser.add_argument(
        '--sampling-stride-x',
        type=int,
        default=10,
        help='Stride X: keep every Xth label=0 bar when stride sampling is on (default: 10).',
    )
    parser.add_argument(
        '--sampling-hours-before',
        type=float,
        default=4.0,
        help='Hours before each label=1 bar kept as context (default: 4.0).',
    )
    parser.add_argument(
        '--sampling-hours-after',
        type=float,
        default=4.0,
        help='Hours after each label=1 bar kept as context (default: 4.0).',
    )


def validate_sampling_args(args: argparse.Namespace) -> None:
    """
    Validate sampling arguments and raise clear errors.

    Args:
        args: Parsed CLI arguments

    Raises:
        SystemExit: If any numeric parameter is out of range.
    """
    stride_x = getattr(args, 'sampling_stride_x', 10)
    hours_before = getattr(args, 'sampling_hours_before', 4.0)
    hours_after = getattr(args, 'sampling_hours_after', 4.0)

    if stride_x < 1:
        raise SystemExit(
            f"ERROR: --sampling-stride-x must be >= 1, got {stride_x}."
        )
    if hours_before < 0:
        raise SystemExit(
            f"ERROR: --sampling-hours-before must be >= 0, got {hours_before}."
        )
    if hours_after < 0:
        raise SystemExit(
            f"ERROR: --sampling-hours-after must be >= 0, got {hours_after}."
        )

    training_sampling = getattr(args, 'training_sampling', True)
    sampling_stride = getattr(args, 'sampling_stride', True)
    sampling_context = getattr(args, 'sampling_context', True)
    if training_sampling and not sampling_stride and not sampling_context:
        print(
            "WARNING: training sampling is on but both strategies (stride & context) "
            "are disabled. Only label=1 bars will be kept (baseline d1)."
        )


def compute_sampling_index(
    y: pd.Series,
    args: argparse.Namespace,
    label_name: str | None = None,
) -> pd.DatetimeIndex:
    """
    Compute the DatetimeIndex to keep for a single label series.

    Rules (all active strategies are unioned):
      - d1 (baseline): always keep bars where y == 1.
      - d2 (stride): if enabled, keep every Xth bar where y == 0.
      - d3 (context): if enabled, keep all bars within
        [t - hours_before, t + hours_after] of any t where y == 1.

    If the master flag `args.training_sampling` is false, returns y.index
    unchanged (pass-through).

    Args:
        y: Binary label Series (0/1) with a DatetimeIndex.
        args: Parsed CLI arguments (see add_sampling_args).
        label_name: Optional label name used only for the console log.

    Returns:
        Sorted, deduplicated DatetimeIndex of bars to keep.
    """
    if not getattr(args, 'training_sampling', True):
        return y.index

    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError(
            f"compute_sampling_index requires a DatetimeIndex on y, got {type(y.index)}."
        )

    use_stride = getattr(args, 'sampling_stride', True)
    use_context = getattr(args, 'sampling_context', True)
    stride_x = int(getattr(args, 'sampling_stride_x', 10))
    hours_before = float(getattr(args, 'sampling_hours_before', 4.0))
    hours_after = float(getattr(args, 'sampling_hours_after', 4.0))

    y_values = y.to_numpy()
    pos_mask = y_values == 1
    zero_mask = y_values == 0

    pos_idx = y.index[pos_mask]
    zero_idx = y.index[zero_mask]

    keep_mask = pos_mask.copy()

    if use_stride and stride_x > 1 and len(zero_idx) > 0:
        stride_positions = np.where(zero_mask)[0][::stride_x]
        keep_mask[stride_positions] = True
    elif use_stride and stride_x == 1:
        # stride=1 means keep every zero bar as well
        keep_mask |= zero_mask

    if use_context and len(pos_idx) > 0 and (hours_before > 0 or hours_after > 0):
        before = pd.Timedelta(hours=hours_before)
        after = pd.Timedelta(hours=hours_after)
        sorted_index_values = y.index.values
        pos_times = pos_idx.values
        starts = np.searchsorted(sorted_index_values, pos_times - before, side='left')
        ends = np.searchsorted(sorted_index_values, pos_times + after, side='right')
        for s, e in zip(starts, ends):
            if e > s:
                keep_mask[s:e] = True

    keep_idx = y.index[keep_mask]

    n_before = len(y)
    n_kept = int(keep_mask.sum())
    n_pos = int(pos_mask.sum())
    tag = f" [{label_name}]" if label_name else ""
    pct = (n_kept / n_before) if n_before else 0.0
    print(
        f"SAMPLING{tag}: kept {n_kept:,}/{n_before:,} ({pct:.1%}) — "
        f"label=1: {n_pos:,}, stride={'on' if use_stride else 'off'}"
        f"(x={stride_x}), context={'on' if use_context else 'off'}"
        f"(-{hours_before:g}h/+{hours_after:g}h)"
    )

    return keep_idx
