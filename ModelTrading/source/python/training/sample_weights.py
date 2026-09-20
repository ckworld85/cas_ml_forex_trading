"""Sample weights from label uniqueness (López de Prado, *Advances in Financial ML*, ch. 4).

The problem
-----------
Labels here are barrier races: the label at bar t is decided by the price path over the
following bars, up to `horizon_max` (default 384 M15 bars = 96 h). Labels are emitted on
**every** bar, so bar t and bar t+1 are decided by almost the same price path. On the 4h
cadence a 96 h horizon means up to **24 consecutive rows share one outcome**.

Two consequences, both of which the pipeline used to absorb silently:

1. **The effective sample size is far smaller than the row count.** 36,913 training rows
   at an average uniqueness of 1/24 carry roughly 1,500 independent observations. Every
   standard error computed as if n were 36,913 is understated by a factor of ~5, which is
   a large part of why this project keeps producing results that look significant and then
   do not replicate.
2. **Training over-weights crowded periods.** A quiet stretch where every barrier takes the
   full horizon contributes 24 near-copies of the same observation; a volatile stretch
   where barriers resolve in two bars contributes two distinct ones. Unweighted, the model
   spends its capacity on the quiet regime in proportion to an overlap artefact.

Average uniqueness
------------------
For each bar t with resolution bar t1[t], concurrency c[i] counts how many labels are
"live" over bar i. A label's uniqueness is the mean of 1/c[i] over its own span:

    u[t] = mean_{i in [t, t1[t]]} 1 / c[i]

A label that never overlaps anything scores 1.0; one of 24 fully-overlapping labels scores
~1/24. `sum(u)` is the effective sample size.

`t1` comes from `labeling/dynamic.generate_dynamic_labels`, which records the bar each
barrier race actually broke on (`t1_{model}` columns on the metadata frame). Where it is
absent, `constant_horizon_t1` falls back to a fixed horizon — that yields uniform weights
for interior bars and therefore does not change training, but still reports a truthful
effective sample size.
"""

import numpy as np
import pandas as pd


def constant_horizon_t1(n, horizon):
    """Fallback t1 when the labeller did not record resolution bars: t1 = t + horizon."""
    t1 = np.arange(n, dtype=np.int64) + int(horizon)
    return np.minimum(t1, n - 1)


def concurrency(t1, n=None):
    """Number of labels live over each bar.

    Args:
        t1: array of resolution bar positions, t1[i] >= i. Entries < 0 mark labels that
            never resolved (too close to the end of the data) and are ignored.
        n: number of bars; defaults to len(t1).

    Returns:
        int array of length n.
    """
    t1 = np.asarray(t1, dtype=np.int64)
    n = int(len(t1) if n is None else n)
    # Difference array: +1 when a label starts, -1 one bar after it ends.
    delta = np.zeros(n + 1, dtype=np.int64)
    for i, end in enumerate(t1):
        if end < i or i >= n:
            continue
        end = min(int(end), n - 1)
        delta[i] += 1
        delta[end + 1] -= 1
    return np.cumsum(delta)[:n]


def average_uniqueness(t1, n=None):
    """Mean 1/concurrency over each label's own span.

    Returns:
        float array in (0, 1]; 0.0 for labels that never resolved.
    """
    t1 = np.asarray(t1, dtype=np.int64)
    n = int(len(t1) if n is None else n)
    c = concurrency(t1, n).astype(np.float64)
    inv = np.divide(1.0, c, out=np.zeros_like(c), where=c > 0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])

    u = np.zeros(n, dtype=np.float64)
    for i, end in enumerate(t1):
        if end < i or i >= n:
            continue
        end = min(int(end), n - 1)
        span = end - i + 1
        u[i] = (cum[end + 1] - cum[i]) / span
    return u


def effective_sample_size(t1, n=None):
    """Sum of average uniqueness — the number of independent observations.

    This is the `n` that standard errors should be computed with, not the row count.
    """
    return float(np.sum(average_uniqueness(t1, n)))


def sample_weights(t1, n=None, normalize=True):
    """Training weights proportional to average uniqueness.

    Args:
        normalize: scale so the weights sum to the row count, which keeps XGBoost's
            effective learning rate and `scale_pos_weight` on the same footing as an
            unweighted run. The RELATIVE weighting is what matters; the scale is not.
    """
    u = average_uniqueness(t1, n)
    if not normalize:
        return u
    total = u.sum()
    if total <= 0:
        return np.ones(len(u), dtype=np.float64)
    return u * (len(u) / total)


def t1_from_metadata(metadata, model, index=None, fallback_horizon=None):
    """Read `t1_{model}` from the label metadata frame as positions in ``index``.

    The column holds the resolution **timestamp** of each label. It is stored that way on
    purpose: a row position does not survive the trimming and reindexing the frame goes
    through before it is persisted, and when it silently does not, the horizon it implies
    is wrong by orders of magnitude rather than obviously broken (measured 2026-08-29:
    positions up to 110,918 on a 102,423-row index, turning a 384-bar horizon into 8,500).

    Args:
        metadata: the frame `generate_dynamic_labels` returns as `labels['metadata']`.
        model: 'long_fast' | 'short_fast' | 'long_slow' | 'short_slow'.
        index: the (possibly subset/resampled) index the weights are needed for. A
            resolution timestamp that is not itself in ``index`` maps to the next row at
            or after it, which is the first bar on which the outcome is fully known.
        fallback_horizon: used when the column is missing (older artefacts, or a label
            mode that does not run the barrier race).

    Returns:
        int array of resolution positions relative to ``index``, -1 where unresolved, or
        None when neither the column nor a fallback is available.
    """
    col = f't1_{model}'
    have = metadata is not None and col in getattr(metadata, 'columns', [])
    if have and not pd.api.types.is_datetime64_any_dtype(metadata[col]):
        # Artefacts written before 2026-08-29 stored t1 as an INTEGER row position.
        # pd.to_datetime would silently read those as nanoseconds since the epoch, land
        # every label in 1970, and return "all unresolved" — a wrong answer dressed as a
        # computed one. Treat the column as absent instead.
        have = False
    if not have:
        if fallback_horizon is None or index is None:
            return None
        return constant_horizon_t1(len(index), fallback_horizon)

    target = pd.DatetimeIndex(metadata.index if index is None else index)
    stamps = pd.to_datetime(pd.Series(metadata[col].values, index=metadata.index),
                            errors='coerce').reindex(target)

    out = np.full(len(target), -1, dtype=np.int64)
    valid = stamps.notna().values
    if not valid.any():
        return out

    mapped = target.searchsorted(pd.DatetimeIndex(stamps[valid].values), side='left')
    out[valid] = np.minimum(mapped, len(target) - 1)

    # A label can never resolve before its own bar.
    own = np.arange(len(target), dtype=np.int64)
    resolved = out >= 0
    out[resolved] = np.maximum(out[resolved], own[resolved])
    return out
