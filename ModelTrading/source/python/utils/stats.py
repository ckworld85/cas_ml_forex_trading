"""
Shared statistical helpers for walk-forward evaluation.

Home of the month-clustered standard error used by both orchestrators
(iterative_training.py and walk_forward.py), so the estimator exists exactly once.
"""

import math
import statistics
from collections import defaultdict

# Cluster-robust standard errors are unreliable with few clusters. A year of distinct
# calendar months is the minimum before the interval is allowed to license a claim.
MIN_CLUSTERS_FOR_SIGNIFICANCE = 12


def clustered_se(values, clusters):
    """
    Standard error of the mean that tolerates correlated observations.

    With overlapping test windows the same market day appears in several folds, so
    its trades are near-copies of one another. Treating them as independent shrinks
    the interval by roughly the square root of the overlap factor and manufactures
    evidence that is not there. Clustering on the calendar month — the standard
    cluster-robust variance for a sample mean — prices that in: correlated trades
    inside a month contribute jointly, and the effective sample size becomes the
    number of distinct months rather than the number of trades.

        Var(mean) = G/(G-1) * (1/n^2) * sum_g( (sum of within-cluster deviations)^2 )

    The G/(G-1) term is the usual finite-cluster correction. The estimator itself
    does not assume the correlation is total — it measures how much the clusters
    actually move together, so partially correlated folds land between the naive and
    the worst-case interval.

    Falls back to the plain standard error when clusters are unavailable or when the
    between-cluster variance degenerates to zero (possible with very few clusters).

    Args:
        values (list[float]): The observations (e.g. per-trade P&L).
        clusters (list | None): Cluster label per observation (e.g. 'YYYY-MM'),
            same length as values, or None for the naive SE.

    Returns:
        tuple[float, int]: (standard error, number of clusters used) — the cluster
        count is n for the naive fallback with missing clusters.
    """
    n = len(values)
    if n < 2:
        return float('nan'), 0
    mean = sum(values) / n
    naive = statistics.stdev(values) / math.sqrt(n)

    if not clusters or len(clusters) != n:
        return naive, n

    sums = defaultdict(float)
    for v, c in zip(values, clusters):
        sums[c] += (v - mean)
    g = len(sums)
    if g < 2:
        return naive, g

    var = (g / (g - 1)) * sum(s * s for s in sums.values()) / (n * n)
    if not (var > 0):
        return naive, g
    return math.sqrt(var), g
