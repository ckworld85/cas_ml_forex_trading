"""
Direction-Horizon Label Generation Module

Generates slow-model labels as a simple fixed-horizon sign-of-return
classification with a dead zone, instead of a TP-before-SL race.

Label logic (per bar t):
  fwd_move_pips = (close[t+horizon] - close[t]) / pip_value
  long_slow[t]  = 1 if fwd_move_pips >  dead_zone_pips else 0
  short_slow[t] = 1 if fwd_move_pips < -dead_zone_pips else 0
  Bars inside the dead zone, or at the end of data (close[t+horizon]
  does not exist), get long_slow = short_slow = 0.
"""

import pandas as pd


def generate_direction_horizon_labels(
    df: pd.DataFrame,
    horizon: int = 192,
    dead_zone_pips: float = 12.0,
    pip_value: float = 0.0001,
    verbose: bool = True,
) -> dict:
    """
    Slow labels from the sign of the fixed-horizon forward return, with a
    symmetric dead zone around zero mapping to "no label" (0/0).

    Returns dict with:
        'long_slow'      : pd.Series[int]
        'short_slow'     : pd.Series[int]
        'fwd_move_pips'  : pd.Series[float]  (diagnostic; NaN at end of data)
    """
    close = df['close']

    fwd_move_pips = (close.shift(-horizon) - close) / pip_value

    long_slow = (fwd_move_pips > dead_zone_pips).astype(int)
    short_slow = (fwd_move_pips < -dead_zone_pips).astype(int)

    # End-of-data bars: fwd_move_pips is NaN -> comparisons already False,
    # but be explicit, matching lookahead.py's convention of never leaving NaN.
    valid_mask = fwd_move_pips.notna()
    long_slow[~valid_mask] = 0
    short_slow[~valid_mask] = 0

    if verbose:
        n_valid = valid_mask.sum()
        lr = long_slow[valid_mask].mean()
        sr = short_slow[valid_mask].mean()
        dead_rate = 1.0 - lr - sr
        print(f"  Direction-horizon labels ({horizon}-bar horizon, dead zone +/-{dead_zone_pips:.1f} pips):")
        print(f"    Valid bars: {n_valid:,}  (end-of-data: {(~valid_mask).sum():,})")
        print(f"    Long  rate: {lr:.1%}")
        print(f"    Short rate: {sr:.1%}")
        print(f"    Dead-zone rate: {dead_rate:.1%}")

    return {
        'long_slow':     long_slow,
        'short_slow':    short_slow,
        'fwd_move_pips': fwd_move_pips,
    }
