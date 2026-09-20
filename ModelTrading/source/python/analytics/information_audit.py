"""
Information audit — is there a causally exploitable directional edge at all?

WHY THIS RUNS BEFORE ANY MODEL
------------------------------
A model can only extract information that is present. This project has spent ~1,800
backtests and 324 walk-forward trainings searching for a configuration, and the binding
constraint turned out not to be the configuration:

* the trend/SMA/momentum family has already been tested over 20 years and carries **no**
  causal directional edge (pooled AUC 0.481-0.515 under causal de-levelling);
* the shipped models' AUC measures trend/range detection, not trade selection (within-trend
  AUC 0.41-0.63 against the non-zeroed outcome);
* nothing survives correction for the number of configurations tried.

So the question that decides whether the economic goal is reachable is not "which model"
but "which information". This script answers it **model-free**, on daily bars, before a
single tree is grown, for the families the feature config currently has switched OFF:
carry / rate differentials, COT positioning, US yields, DXY, equity risk sentiment, VIX,
and realized-volatility state.

Decision rule is fixed in advance — see docs/preregistration.md, H1:
accept a family if, after Benjamini-Hochberg correction over every (feature x horizon)
test, some member has p < 0.05 AND |IC| >= 0.03 AND a stable sign in >= 60% of half-years.

THE TRAP THIS IS BUILT TO AVOID
-------------------------------
The trend family looked like a *stable mean-reversion edge* (per-half-year AUC 0.44, below
0.5 in 34/43 periods, p=0.0002) and it was an artefact: it only existed when the feature
was de-levelled against its own period, which peeks. Under **causal** de-levelling the edge
vanished. Every feature here is therefore reduced to a trailing z-score computed only from
its own past (W in {125, 250, 500} trading days), and raw levels are excluded outright.

THE NULL
--------
Financial series are heavily autocorrelated, so an i.i.d. permutation null produces
p-values that are far too small. The null here is a **circular shift** of the feature
series by a random offset >= the horizon: it destroys the alignment between feature and
outcome while preserving the feature's autocorrelation structure exactly.

Because Spearman(x, y) = Pearson(rank x, rank y) and a circular shift merely permutes the
ranks the same way, the whole null distribution is computed from one ranking — which is
what makes 500+ draws per test affordable.

PUBLICATION LAGS — two pipeline defects this audit found, now fixed upstream
---------------------------------------------------------------------------
Both were discovered while building this script and are fixed in
`features/external_data.py` as of 2026-08-29:

* **COT carried up to 3 days of lookahead.** The CSV is dated by the CFTC *reference*
  Tuesday (516 of 522 rows are Tuesdays) while the report is published the following
  Friday. `load_cot` now shifts its index by `COT_PUBLICATION_LAG_DAYS`, so every
  consumer gets it right; `--cot-lag-days` therefore defaults to 0 here.
* **Sparse sources were never forward-filled.** `reindex(method='ffill')` fills missing
  index entries, not NaN values that are present, so COT reached 522 of 5,556 daily bars.
  `get_external_df` now calls `.ffill()` on the merged frame.

Usage
-----
    python -m ModelTrading.source.python.analytics.information_audit
    python -m ModelTrading.source.python.analytics.information_audit \
        --horizons 5,10,20 --n-shuffles 1000 --out audit.csv

Read-only: it loads CSVs and writes a report. It never touches training or backtest state.
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
import ModelTrading.source.python.features.external_data as external_data  # noqa: E402
import ModelTrading.source.python.utils.csv as csv_utils  # noqa: E402

# Which raw external column belongs to which information family. The audit reports per
# family because that is the unit the decision rule in the pre-registration is stated on.
FAMILIES = {
    # The interest-rate DIFFERENTIAL — added 2026-08-29 together with the ECB euro-area
    # curve. It did not exist in the data set before: `us_yield_spread` is the US curve
    # slope and `carry` is a policy-rate step function, so the variable FX theory
    # actually points at was never testable. The CHANGE columns matter more than the
    # level: spot responds to repricing of the expected differential, not to its level.
    'ratediff':  ['rate_diff_2y', 'rate_diff_2y_chg5', 'rate_diff_2y_chg20',
                  'rate_diff_10y', 'rate_diff_10y_chg5', 'rate_diff_10y_chg20'],
    'carry':     ['carry_diff', 'carry_diff_chg20'],
    'cot':       ['cot_net_position', 'cot_net_change', 'cot_index'],
    'yields':    ['us_10y_yield', 'us_yield_spread'],
    'dxy':       ['dxy_level', 'dxy_1d_change', 'dxy_zscore_20'],
    'equity':    ['es_1d_change', 'es_zscore_20'],
    'vix':       ['vix_level', 'vix_1d_change', 'vix_percentile'],
    # Cross-asset currency strength (features/cross_asset.py) — the family the first two
    # passes structurally could not contain: it is formed from nine OTHER pairs' prices,
    # with EURUSD excluded from the decomposition, so it is price-derived yet not
    # derivable from the target's own history. Only the redundancy-screen survivors
    # enter (docs/results/ccy_redundancy_screen.csv): the three 60-day spread features
    # correlate 0.86-0.94 with daily_momentum_xlong — the relative motion of the pair's
    # own two legs at the pair's own horizon IS mostly the pair's momentum — and
    # auditing them would spend BH budget re-testing a feature the model already has.
    'ccy':       ['ccy_rank_spread_250', 'ccy_spread_z_250', 'ccy_eur_rank_250',
                  'ccy_usd_rank_250', 'ccy_usd_breadth_250',
                  'ccy_eur_rank_60', 'ccy_usd_breadth_60', 'ccy_dispersion_pct'],
}
COT_COLUMNS = set(FAMILIES['cot'])


def causal_zscore(s, window):
    """Trailing z-score using only the series' own past.

    `shift(1)` before the rolling statistics so the mean and std at date D are computed
    from bars strictly before D. Without it the current observation contributes to its own
    normalisation, which is the exact leak that manufactured the trend family's apparent
    edge.
    """
    past = s.shift(1)
    mu = past.rolling(window, min_periods=window // 2).mean()
    sd = past.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def forward_return(close, horizon):
    """Log return from bar t's close to bar t+horizon's close."""
    return np.log(close.shift(-horizon) / close)


def barrier_outcome(high, low, close, horizon, atr, k=1.0):
    """Symmetric k*ATR barrier race over `horizon` bars: +1 up first, -1 down first, 0 neither.

    A pure forward return answers "did it drift", the barrier race answers "was it
    tradeable" — the two disagree exactly when a move is preceded by an adverse excursion
    that a stop would have caught, which is the case the strategy actually lives in.
    """
    n = len(close)
    h, l, c = high.values, low.values, close.values
    a = atr.values
    out = np.zeros(n)
    for i in range(n):
        if not np.isfinite(a[i]) or a[i] <= 0:
            out[i] = np.nan
            continue
        up, dn = c[i] + k * a[i], c[i] - k * a[i]
        end = min(i + horizon + 1, n)
        res = 0.0
        for j in range(i + 1, end):
            hit_up, hit_dn = h[j] >= up, l[j] <= dn
            if hit_up and hit_dn:
                res = 0.0          # both inside one bar — unresolvable at this resolution
                break
            if hit_up:
                res = 1.0
                break
            if hit_dn:
                res = -1.0
                break
        out[i] = res
        if i + horizon >= n:
            out[i] = np.nan
    return pd.Series(out, index=close.index)


def spearman_with_null(x, y, n_shuffles, horizon, rng):
    """Spearman IC plus a circular-shift null p-value.

    Returns (ic, p_value, n). The p-value is two-sided: the share of shifted draws whose
    |IC| reaches the observed |IC|. A +1 correction in numerator and denominator keeps it
    from ever being exactly 0, which would overstate the evidence.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 100:
        return np.nan, np.nan, n

    # copy=True: pandas 3 (copy-on-write) hands out read-only arrays from
    # .values/.to_numpy(), and the in-place centering below would raise.
    rx = pd.Series(x).rank().to_numpy(copy=True)
    ry = pd.Series(y).rank().to_numpy(copy=True)
    rx -= rx.mean(); ry -= ry.mean()
    dx, dy = np.sqrt((rx ** 2).sum()), np.sqrt((ry ** 2).sum())
    if dx == 0 or dy == 0:
        return np.nan, np.nan, n

    ic = float(rx @ ry / (dx * dy))

    # Offsets at least `horizon` away from both ends, so no draw accidentally reproduces
    # the true alignment.
    lo = max(int(horizon), 1)
    if n <= 2 * lo + 2:
        return ic, np.nan, n
    offsets = rng.integers(lo, n - lo, size=int(n_shuffles))
    null = np.abs(np.array([np.roll(rx, int(o)) @ ry for o in offsets])) / (dx * dy)
    p = float((np.sum(null >= abs(ic)) + 1) / (len(null) + 1))
    return ic, p, n


def half_year_sign_stability(x, y, index):
    """Share of half-year periods whose IC has the same sign as the pooled IC.

    A real edge should not need every period to agree, but it should not flip half the
    time either. Reported because the trend family's artefact was *stable* in sign — sign
    stability alone is not evidence, it is a necessary condition.
    """
    df = pd.DataFrame({'x': x, 'y': y}, index=index).dropna()
    if df.empty:
        return np.nan, 0
    pooled = df['x'].corr(df['y'], method='spearman')
    if not np.isfinite(pooled):
        return np.nan, 0
    periods = df.groupby([df.index.year, (df.index.month > 6).astype(int)])
    signs = []
    for _, chunk in periods:
        if len(chunk) < 30:
            continue
        c = chunk['x'].corr(chunk['y'], method='spearman')
        if np.isfinite(c):
            signs.append(np.sign(c) == np.sign(pooled))
    if not signs:
        return np.nan, 0
    return float(np.mean(signs)), len(signs)


# ---------------------------------------------------------------------------
# Conditional analysis — closes the "pooled over 21 years" blind spot
# ---------------------------------------------------------------------------
# A relationship that flips sign with the macro regime nets to zero when pooled.
# Carry is the textbook case: it should work when rate differentials are wide and do
# nothing at the effective lower bound, so a 2005-2026 pooled IC is the average of two
# opposite states. The half-year sign-stability column can see that something is wrong;
# it cannot say *when* the feature works.
#
# Buckets are cut at the tercile points of a standard normal (+/-0.4307) applied to the
# CAUSAL z-score, never at sample quantiles — a sample quantile is computed from the
# whole history and would put a mild lookahead into the bucket definition itself.

TERCILE_CUT = 0.4307  # +/- this splits a standard normal into three equal parts


def tercile_buckets(z, labels=('low', 'mid', 'high')):
    """Causal three-way split of a z-scored series. NaN stays NaN."""
    out = pd.Series(pd.NA, index=z.index, dtype=object)
    out[z <= -TERCILE_CUT] = labels[0]
    out[(z > -TERCILE_CUT) & (z < TERCILE_CUT)] = labels[1]
    out[z >= TERCILE_CUT] = labels[2]
    return out


def build_conditioners(daily, ext, zwin=250, shock_z=2.0, shock_window=5):
    """Regime/event stratifications to test each feature *within*.

    All causal. Returns {name: Series of bucket labels aligned to daily.index}.

    * ``vol``    — realized-volatility state; the regime most likely to flip a carry or
                   momentum relationship.
    * ``trend``  — price efficiency, i.e. trending vs. choppy.
    * ``rates``  — the level of the policy-rate differential. Splits the ZLB years from
                   the tightening years, which is exactly where carry should differ.
    * ``shock``  — EVENT conditioner, and the third blind spot: everything else in this
                   script is level-shaped, while FX information is often event-shaped.
                   Marks bars within ``shock_window`` days after a >= ``shock_z`` sigma
                   two-day move in VIX, DXY or the yield spread. It is a proxy for
                   "something happened", not a calendar — a real surprise-vs-consensus
                   series would be better and is not in the data set.
    """
    conds = {}

    logret = np.log(daily['close']).diff()
    rv20 = logret.rolling(20, min_periods=10).std()
    conds['vol'] = tercile_buckets(causal_zscore(rv20, zwin))

    move = (daily['close'] - daily['close'].shift(20)).abs()
    path = daily['close'].diff().abs().rolling(20, min_periods=10).sum()
    efficiency = move / path.replace(0.0, np.nan)
    conds['trend'] = tercile_buckets(causal_zscore(efficiency, zwin),
                                     labels=('choppy', 'mid', 'trending'))

    if 'carry_diff' in ext.columns and ext['carry_diff'].notna().sum() > 500:
        conds['rates'] = tercile_buckets(causal_zscore(ext['carry_diff'], zwin),
                                         labels=('eur_favourable', 'mid', 'usd_favourable'))

    shock = pd.Series(False, index=daily.index)
    for col in ('vix_level', 'dxy_level', 'us_yield_spread'):
        if col not in ext.columns:
            continue
        z2 = causal_zscore(ext[col].astype(float).diff(2), zwin)
        shock |= (z2.abs() >= shock_z).fillna(False)
    conds['shock'] = shock.rolling(shock_window, min_periods=1).max().astype(bool).map(
        {True: 'post_shock', False: 'quiet'})

    return conds


def conditional_ic(x, y, bucket, n_shuffles, horizon, rng):
    """IC of x on y within each bucket, with a null that keeps the serial structure.

    The shift is applied to the FULL feature series and the bucket mask is then applied
    to the shifted series — never the other way round. Subsetting first would leave a
    non-contiguous vector whose circular shift no longer preserves the autocorrelation
    the null exists to respect.

    Returns {bucket_label: (ic, p, n)}.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = pd.Series(np.asarray(bucket, dtype=object))
    n_all = len(x)
    lo = max(int(horizon), 1)
    if n_all <= 2 * lo + 2:
        return {}

    offsets = rng.integers(lo, n_all - lo, size=int(n_shuffles))
    finite_y = np.isfinite(y)
    out = {}
    for label in pd.unique(labels.dropna()):
        # .eq().fillna(False) rather than `labels == label`: an object Series carrying
        # pd.NA yields an object mask whose truth value is ambiguous under `&`.
        in_bucket = labels.eq(label).fillna(False).to_numpy(dtype=bool)
        mask = in_bucket & np.isfinite(x) & finite_y
        n = int(mask.sum())
        if n < 100:
            continue
        ic = _rank_corr(x[mask], y[mask])
        if not np.isfinite(ic):
            continue
        null = []
        for o in offsets:
            xs = np.roll(x, int(o))
            m = in_bucket & finite_y & np.isfinite(xs)
            if m.sum() < 100:
                continue
            c = _rank_corr(xs[m], y[m])
            if np.isfinite(c):
                null.append(abs(c))
        p = (float((np.sum(np.array(null) >= abs(ic)) + 1) / (len(null) + 1))
             if null else np.nan)
        out[str(label)] = (float(ic), p, n)
    return out


def _rank_corr(a, b):
    """Spearman via ranks; NaN when either side is constant."""
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean(); rb = rb - rb.mean()
    da, db = np.sqrt((ra ** 2).sum()), np.sqrt((rb ** 2).sum())
    if da == 0 or db == 0:
        return np.nan
    return float(ra @ rb / (da * db))


def benjamini_hochberg(pvals):
    """BH-adjusted p-values, NaNs preserved."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        adj = p[i] * m / (m - rank + 1)
        prev = min(prev, adj)
        out[i] = min(prev, 1.0)
    return out


# ---------------------------------------------------------------------------
# Multivariate test — closes the "univariate" blind spot
# ---------------------------------------------------------------------------
# Spearman IC measures one monotone pairwise relationship. It is blind to exactly the
# thing a tree model exists for: a feature that pays only in combination with another
# ("carry works when volatility is low"). This project already knows that blindness —
# it is why `--mi-permutations 0` is mandatory, MI having stripped 27 of 37 slow
# features and cost 123k EUR in an A/B. The same objection applies to the univariate
# part of this audit, so it has to be answered rather than acknowledged.
#
# The null shifts the ENTIRE feature block by ONE common offset. That destroys the
# feature->outcome alignment while preserving both each feature's autocorrelation and
# the cross-feature correlation structure — so a model that "finds" something in the
# null found it in the covariance, not in the outcome. Shifting each column
# independently would be a weaker null: it would also destroy the correlations between
# features, which is not what we are testing.


def purged_folds(n, n_splits, horizon):
    """Chronological folds with an embargo of `horizon` rows before each validation block.

    Without the embargo the last rows of every training block are labelled from bars
    inside the validation block — the same seam leak the CV embargo closes in training.
    """
    fold_size = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        val_start = k * fold_size
        val_end = n if k == n_splits else (k + 1) * fold_size
        train_end = max(val_start - int(horizon), 0)
        if train_end < 100 or val_end - val_start < 50:
            continue
        yield np.arange(0, train_end), np.arange(val_start, val_end)


# Learner families. The multivariate test originally used one model, which made
# "no combination carries information" a statement about that model as much as about the
# data. The Random Forest arms are the robustness check on that claim — same folds, same
# null, same features.
#
# `rf` uses out-of-the-box settings on purpose: it tests the argument *as stated*, that an
# RF is hard to over-optimise and would therefore find what a mis-tuned booster memorises
# away. `rf_leaf` raises min_samples_leaf because bagging assumes INDEPENDENT bootstrap
# draws, and with up to 24 consecutive rows sharing one barrier outcome the draws are
# strongly correlated — RF's variance reduction is largely illusory at this label overlap.
LEARNERS = ('xgb', 'rf', 'rf_leaf')


def _fit_predict(learner, X_tr, y_tr, X_va, params, num_boost_round, seed):
    """Train `learner` on (X_tr, y_tr) and return its scores for X_va."""
    if learner == 'xgb':
        import xgboost as xgb
        booster = xgb.train(params, xgb.DMatrix(X_tr, label=y_tr),
                            num_boost_round=num_boost_round)
        return booster.predict(xgb.DMatrix(X_va))

    from sklearn.ensemble import RandomForestClassifier
    kwargs = dict(n_estimators=150, max_features='sqrt', n_jobs=4,
                  random_state=int(seed), bootstrap=True)
    if learner == 'rf_leaf':
        # Capacity matched to the effective sample size, not the row count.
        kwargs.update(min_samples_leaf=50)
    elif learner != 'rf':
        raise ValueError(f"unknown learner {learner!r}; expected one of {LEARNERS}")
    rf = RandomForestClassifier(**kwargs).fit(X_tr, y_tr)
    return rf.predict_proba(X_va)[:, 1]


def _multivariate_auc(X, y, n_splits, horizon, params, num_boost_round,
                      learner='xgb', seed=0):
    """Mean out-of-sample AUC of `learner` over purged chronological folds."""
    from sklearn.metrics import roc_auc_score

    aucs = []
    for tr, va in purged_folds(len(y), n_splits, horizon):
        ytr, yva = y[tr], y[va]
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            continue
        scores = _fit_predict(learner, X[tr], ytr, X[va], params, num_boost_round, seed)
        aucs.append(roc_auc_score(yva, scores))
    return float(np.mean(aucs)) if aucs else np.nan, len(aucs)


def multivariate_test(feature_frame, outcome, horizon, n_shuffles, rng,
                      n_splits=4, max_depth=3, num_boost_round=60, min_coverage=0.8,
                      learner='xgb', verbose=True):
    """Can ANY combination of the candidate features beat the block-shift null?

    ``min_coverage`` drops columns available on less than that share of the outcome's
    rows BEFORE the complete-case filter. Without it one short series decides the
    sample: COT starts in 2016, so requiring every column present collapses 5,556 daily
    bars to 2,425 and silently turns this into a COT-era test. Pass 0.0 to include the
    sparse families and accept the shorter sample.

    Returns dict with the real OOS AUC, the null mean/sd, the p-value, the fold count,
    the features actually used and the resulting sample size.
    """
    params = {'max_depth': max_depth, 'eta': 0.1, 'objective': 'binary:logistic',
              'eval_metric': 'auc', 'subsample': 0.8, 'colsample_bytree': 0.8,
              'verbosity': 0, 'nthread': 4}

    outcome_rows = np.isfinite(outcome)
    if min_coverage > 0 and outcome_rows.any():
        cov = feature_frame.loc[outcome_rows].notna().mean()
        keep = cov[cov >= min_coverage].index.tolist()
        dropped = [c for c in feature_frame.columns if c not in keep]
        if dropped and verbose:
            fams = sorted({c.split('::')[0] for c in dropped})
            print(f"      dropped {len(dropped)} low-coverage features "
                  f"(families: {', '.join(fams)}) to keep the sample long")
        feature_frame = feature_frame[keep]

    mask = outcome_rows & feature_frame.notna().all(axis=1).values
    X_all = feature_frame.values[mask]
    y_sign = (outcome[mask] > 0).astype(int)
    n = len(y_sign)
    if n < 500 or len(np.unique(y_sign)) < 2:
        return {'auc': np.nan, 'p': np.nan, 'n': n, 'n_folds': 0,
                'null_mean': np.nan, 'null_sd': np.nan,
                'n_features': feature_frame.shape[1], 'learner': learner}

    real, n_folds = _multivariate_auc(X_all, y_sign, n_splits, horizon, params,
                                      num_boost_round, learner=learner, seed=0)
    if not np.isfinite(real):
        return {'auc': np.nan, 'p': np.nan, 'n': n, 'n_folds': n_folds,
                'null_mean': np.nan, 'null_sd': np.nan,
                'n_features': feature_frame.shape[1], 'learner': learner}

    lo = max(int(horizon), 1)
    null = []
    for i in range(int(n_shuffles)):
        offset = int(rng.integers(lo, n - lo))
        a, _ = _multivariate_auc(np.roll(X_all, offset, axis=0), y_sign, n_splits,
                                 horizon, params, num_boost_round,
                                 learner=learner, seed=i + 1)
        if np.isfinite(a):
            null.append(a)
        if verbose and (i + 1) % 25 == 0:
            print(f"      null draw {i + 1}/{n_shuffles}")
    if not null:
        return {'auc': real, 'p': np.nan, 'n': n, 'n_folds': n_folds,
                'null_mean': np.nan, 'null_sd': np.nan,
                'n_features': feature_frame.shape[1], 'learner': learner}

    null = np.array(null)
    p = float((np.sum(null >= real) + 1) / (len(null) + 1))   # one-sided: only better counts
    return {'auc': real, 'p': p, 'n': n, 'n_folds': n_folds,
            'null_mean': float(null.mean()), 'null_sd': float(null.std()),
            'n_features': feature_frame.shape[1], 'learner': learner}


def build_features(daily, zscore_windows, cot_lag_days, verbose=True, families=None):
    """Causal, de-levelled candidate features from the external sources.

    Returns {feature_name: Series} where the name encodes family, source column and the
    z-score window. ``families`` (a set/list of family names, or None for all)
    restricts which candidate families are built — the BH correction then runs
    across exactly the requested family's tests, which is what "BH as its own
    family" means for a single-family amendment run (A8's rmom leg). Known
    names: the FAMILIES keys plus 'volstate', 'technical' and 'rmom'.
    """
    def _want(fam):
        return families is None or fam in families

    ext = external_data.get_external_df(daily.index)
    if ext is None:
        if any(_want(f) for f in FAMILIES):
            raise SystemExit("No external CSVs found — run data/update_external_data.py first.")
        ext = pd.DataFrame(index=daily.index)

    # Belt and braces: get_external_df forward-fills the merged frame itself since
    # 2026-08-29 (before that, sparse weekly sources reached only their own ~520 report
    # dates out of 5,556 daily bars). Kept so the audit is correct against older
    # revisions of the pipeline too; it is a no-op against the current one — EXCEPT for
    # the ccy_* columns, whose NaNs are deliberate: the staleness guard in
    # get_external_df masks values older than CROSS_ASSET_MAX_STALE_DAYS (the export
    # hole is ~3 years), and re-filling them here would hand the audit three years of
    # confidently stale currency ranks.
    _ccy = [c for c in ext.columns if c.startswith('ccy_')]
    _keep = ext[_ccy].copy()
    ext = ext.ffill()
    ext[_ccy] = _keep

    # COT carries the CFTC reference Tuesday, published the following Friday afternoon.
    # A one-row shift leaves up to three days of lookahead; close it explicitly.
    if cot_lag_days > 0:
        for col in COT_COLUMNS:
            if col in ext.columns:
                shifted = ext[col].copy()
                shifted.index = shifted.index + pd.Timedelta(days=int(cot_lag_days))
                ext[col] = shifted.reindex(ext.index, method='ffill')

    feats = {}
    for family, cols in FAMILIES.items():
        if not _want(family):
            continue
        for col in cols:
            if col not in ext.columns:
                continue
            raw = ext[col].astype(float)
            if raw.notna().sum() < 250:
                if verbose:
                    print(f"  skip {family}/{col}: only {int(raw.notna().sum())} observations")
                continue
            for w in zscore_windows:
                feats[f'{family}::{col}::z{w}'] = causal_zscore(raw, w)

    # Realized-volatility state from the price itself. Included because it is the one
    # non-price-direction signal the price series can legitimately supply, and it is the
    # input the regime models already use.
    if _want('volstate'):
        logret = np.log(daily['close']).diff()
        for w in (20, 60):
            rv = logret.rolling(w, min_periods=w // 2).std()
            for z in zscore_windows:
                feats[f'volstate::rv{w}::z{z}'] = causal_zscore(rv, z)

    # --- realized moments & path asymmetry (A8, daily leg) -----------------
    # The 1d/5d/20d-window members computed from M15 log-returns and sampled
    # at each calendar day's LAST M15 bar — the value is complete at that
    # day's close, exactly like every external day-D column here. The raw
    # series then go through the same causal z de-levelling as every other
    # family; the roster's own z/pct members are deliberately absent (a z of
    # a z would double-count). Roster: analytics/realized_moments_catalog.py,
    # rule: docs/preregistration.md A8.
    if _want('rmom'):
        from ModelTrading.source.python.analytics import realized_moments_catalog as rm_cat
        rm_daily = rm_cat.daily_information_frame(end=daily.index.max())
        rm_daily = rm_daily.reindex(daily.index.normalize())
        rm_daily.index = daily.index
        for col in rm_cat.INFO_AUDIT_MEMBERS:
            raw = rm_daily[col].astype(float)
            if raw.notna().sum() < 250:
                if verbose:
                    print(f"  skip rmom/{col}: only {int(raw.notna().sum())} observations")
                continue
            for w in zscore_windows:
                feats[f'rmom::{col}::z{w}'] = causal_zscore(raw, w)

    # --- price-derived technical family ------------------------------------
    # Closes a real gap: the first audit covered the external families and the volatility
    # state, but the price-derived technical family had only been tested in a separate
    # 20-year study restricted to the trend/SMA/momentum group. These are the seven
    # features an external paper reported as its top importances, plus bb_percent (the
    # paper's `bb_pband`, which already existed in features.yaml, disabled).
    #
    # Same `ta` library and the same formulas as features/indicators.py, computed here on
    # the daily bars so this script stays the standalone read-only diagnostic it is.
    # shift(1) throughout: no bar contributes to its own feature.
    import ta.trend
    import ta.volatility

    if not _want('technical'):
        return feats

    h, l, c = daily['high'].shift(1), daily['low'].shift(1), daily['close'].shift(1)
    o = daily['open'].shift(1)
    adx_ind = ta.trend.ADXIndicator(high=h, low=l, close=c, window=14, fillna=True)
    atr_s = ta.volatility.AverageTrueRange(
        high=h, low=l, close=c).average_true_range().replace(0.0, np.nan)
    bb = ta.volatility.BollingerBands(close=c)
    hi10, lo10 = h.rolling(10).max(), l.rolling(10).min()

    technical = {
        'adx_pos': adx_ind.adx_pos(),
        'adx_neg': adx_ind.adx_neg(),
        'cci': ta.trend.CCIIndicator(high=h, low=l, close=c, window=20, fillna=True).cci(),
        'channel_pos_10': (c - lo10) / (hi10 - lo10).replace(0.0, np.nan),
        'hl_range_atr': (h - l) / atr_s,
        'oc_range_atr': (c - o) / atr_s,
        'log_return_atr_1': np.log(c / c.shift(1)) * c / atr_s,
        'bb_percent': bb.bollinger_pband(),
    }
    for name, series in technical.items():
        for z in zscore_windows:
            feats[f'technical::{name}::z{z}'] = causal_zscore(series, z)

    return feats


def run_audit(start=None, end=None, horizons=(1, 3, 5, 10, 20),
              zscore_windows=(125, 250, 500), n_shuffles=500, cot_lag_days=0,
              atr_period=14, barrier_k=1.0, seed=7, verbose=True,
              conditional=True, cond_shuffles=200,
              multivariate=True, mv_horizons=(5, 20), mv_shuffles=100,
              mv_min_coverage=0.8, mv_learners=('xgb',), families=None):
    """Run the audit.

    Returns (pooled_df, conditional_df, multivariate_df). The three answer three
    different questions and must not be pooled into one BH family:
      * pooled       — does the feature carry a monotone edge over the whole history?
      * conditional  — does it carry one inside a particular regime or after a shock?
      * multivariate — does any COMBINATION of the candidates beat the block-shift null?
    """
    rng = np.random.default_rng(seed)

    daily = csv_utils.load_csv(os.path.join(dir_config.DATA_DIR, 'eurusd_daily.csv'))
    if start is not None:
        daily = daily[daily.index >= pd.to_datetime(start)]
    if end is not None:
        daily = daily[daily.index <= pd.to_datetime(end)]
    if verbose:
        print(f"Daily bars: {len(daily)}  {daily.index.min().date()} -> {daily.index.max().date()}")

    prev_close = daily['close'].shift(1)
    tr = pd.concat([
        daily['high'] - daily['low'],
        (daily['high'] - prev_close).abs(),
        (daily['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    feats = build_features(daily, zscore_windows, cot_lag_days, verbose=verbose,
                           families=families)
    if verbose:
        print(f"Candidate features: {len(feats)}")

    outcomes = {}
    for h in horizons:
        outcomes[('fwd_return', h)] = forward_return(daily['close'], h)
        outcomes[('barrier', h)] = barrier_outcome(
            daily['high'], daily['low'], daily['close'], h, atr, k=barrier_k)

    rows = []
    for name, series in feats.items():
        family, col, zwin = name.split('::')
        x = series.values
        for (kind, h), y_series in outcomes.items():
            y = y_series.values
            ic, p, n = spearman_with_null(x, y, n_shuffles, h, rng)
            stab, n_periods = half_year_sign_stability(series, y_series, daily.index)
            rows.append({
                'family': family, 'feature': col, 'zscore_window': zwin,
                'outcome': kind, 'horizon': h,
                'ic': ic, 'p_raw': p, 'n': n,
                'sign_stability': stab, 'n_half_years': n_periods,
            })
        if verbose:
            print(f"  {name}: done")

    df = pd.DataFrame(rows)
    df['p_bh'] = benjamini_hochberg(df['p_raw'].values)
    df['passes_h1'] = (
        (df['p_bh'] < 0.05) & (df['ic'].abs() >= 0.03) & (df['sign_stability'] >= 0.60)
    )
    df = df.sort_values('p_bh', na_position='last').reset_index(drop=True)

    # --- conditional -------------------------------------------------------
    cond_df = pd.DataFrame()
    if conditional:
        ext = external_data.get_external_df(daily.index)
        if ext is None:
            ext = pd.DataFrame(index=daily.index)
        else:
            _ccy = [c for c in ext.columns if c.startswith('ccy_')]
            _keep = ext[_ccy].copy()
            ext = ext.ffill()
            ext[_ccy] = _keep  # same staleness rationale as in build_candidates
        conds = build_conditioners(daily, ext)
        if verbose:
            print(f"\nConditional analysis over {len(conds)} stratifications: "
                  f"{', '.join(conds)}")
        crows = []
        for cname, buckets in conds.items():
            for name, series in feats.items():
                family, col, zwin = name.split('::')
                for (kind, h), y_series in outcomes.items():
                    res = conditional_ic(series.values, y_series.values, buckets,
                                         cond_shuffles, h, rng)
                    for label, (ic, p, n) in res.items():
                        crows.append({
                            'conditioner': cname, 'bucket': label,
                            'family': family, 'feature': col, 'zscore_window': zwin,
                            'outcome': kind, 'horizon': h,
                            'ic': ic, 'p_raw': p, 'n': n,
                        })
            if verbose:
                print(f"  conditioner '{cname}': done")
        cond_df = pd.DataFrame(crows)
        if len(cond_df):
            cond_df['p_bh'] = benjamini_hochberg(cond_df['p_raw'].values)
            cond_df['passes'] = (cond_df['p_bh'] < 0.05) & (cond_df['ic'].abs() >= 0.03)
            cond_df = cond_df.sort_values('p_bh', na_position='last').reset_index(drop=True)

    # --- multivariate ------------------------------------------------------
    mv_df = pd.DataFrame()
    if multivariate:
        frame = pd.DataFrame(feats, index=daily.index)
        if verbose:
            print(f"\nMultivariate test on {frame.shape[1]} features, "
                  f"{mv_shuffles} block-shift draws per cell")
        mrows = []
        for learner in mv_learners:
            for h in mv_horizons:
                for kind in ('fwd_return', 'barrier'):
                    key = (kind, h)
                    if key not in outcomes:
                        continue
                    if verbose:
                        print(f"  [{learner}] {kind}@{h}d ...")
                    r = multivariate_test(frame, outcomes[key].values, h, mv_shuffles, rng,
                                          min_coverage=mv_min_coverage, learner=learner,
                                          verbose=verbose)
                    r.update(outcome=kind, horizon=h)
                    mrows.append(r)
        mv_df = pd.DataFrame(mrows)
        if len(mv_df):
            mv_df['p_bh'] = benjamini_hochberg(mv_df['p'].values)
            mv_df['passes'] = mv_df['p_bh'] < 0.05

    return df, cond_df, mv_df


def print_report(df, top=25):
    print("\n" + "=" * 88)
    print("INFORMATION AUDIT — H1: does any non-price family carry a causal directional edge?")
    print("=" * 88)
    print(f"Tests run: {len(df)}  (Benjamini-Hochberg correction applied across all of them)")

    passing = df[df['passes_h1']]
    print(f"\nPassing the pre-registered H1 rule "
          f"(p_BH < 0.05 AND |IC| >= 0.03 AND sign stability >= 60%): {len(passing)}")
    if len(passing):
        print(passing[['family', 'feature', 'zscore_window', 'outcome', 'horizon',
                       'ic', 'p_raw', 'p_bh', 'sign_stability', 'n']]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("  NONE.")

    print(f"\nStrongest {top} by |IC| regardless of significance:")
    show = df.reindex(df['ic'].abs().sort_values(ascending=False).index).head(top)
    print(show[['family', 'feature', 'zscore_window', 'outcome', 'horizon',
                'ic', 'p_raw', 'p_bh', 'sign_stability', 'n']]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nPer family — best |IC| and whether anything survives correction:")
    for fam, chunk in df.groupby('family'):
        best = chunk.reindex(chunk['ic'].abs().sort_values(ascending=False).index).iloc[0]
        print(f"  {fam:10s} best |IC| {abs(best['ic']):.4f} "
              f"({best['feature']}/{best['zscore_window']}/{best['outcome']}@{best['horizon']}d, "
              f"p_BH {best['p_bh']:.3f})   passing: {int(chunk['passes_h1'].sum())}/{len(chunk)}")

    print("\nVERDICT (pre-registered rule, docs/preregistration.md H1):")
    if len(passing):
        fams = sorted(passing['family'].unique())
        print(f"  H1 ACCEPTED for: {', '.join(fams)}. These families are the only ones the "
              f"direction modelling in stages 2-4 may build on.")
    else:
        print("  H1 REJECTED for every family. No non-price information source shows a")
        print("  causal directional edge at swing horizon that survives correction.")
        print("  Per the pre-registration this means the economic goal is not reachable")
        print("  with this information set; it is recorded as the finding and no further")
        print("  direction modelling is attempted.")
    print("=" * 88)


def print_conditional_report(cond_df, top=20):
    """Report the conditional analysis — the 'pooled over 21 years' blind spot."""
    print("\n" + "=" * 88)
    print("CONDITIONAL — does a family carry an edge INSIDE a regime or after a shock?")
    print("=" * 88)
    if cond_df is None or not len(cond_df):
        print("  not run.")
        return
    print(f"Tests: {len(cond_df)} (BH corrected as their own family, separate from the "
          f"pooled tests)")
    passing = cond_df[cond_df["passes"]]
    print(f"Passing (p_BH < 0.05 AND |IC| >= 0.03): {len(passing)}")
    cols = ["conditioner", "bucket", "family", "feature", "zscore_window", "outcome",
            "horizon", "ic", "p_raw", "p_bh", "n"]
    if len(passing):
        print(passing[cols].head(top).to_string(index=False,
                                                float_format=lambda v: f"{v:.4f}"))
    else:
        print("  NONE. Strongest by |IC| regardless of significance:")
        show = cond_df.reindex(cond_df["ic"].abs().sort_values(ascending=False).index)
        print(show[cols].head(top).to_string(index=False,
                                             float_format=lambda v: f"{v:.4f}"))


def print_multivariate_report(mv_df):
    """Report the multivariate test — the 'univariate' blind spot."""
    print("\n" + "=" * 88)
    print("MULTIVARIATE — can ANY combination beat a block-shift null?")
    print("=" * 88)
    if mv_df is None or not len(mv_df):
        print("  not run.")
        return
    print("The null shifts the WHOLE feature block by one common offset, so it keeps both")
    print("each feature's autocorrelation and the cross-feature correlation structure.")
    print()
    cols = ["learner", "outcome", "horizon", "n_features", "n", "n_folds", "auc",
            "null_mean", "null_sd", "p", "p_bh", "passes"]
    cols = [c for c in cols if c in mv_df.columns]
    print(mv_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Pre-registered learner rule (Amendment A2): an alternative learner counts as better
    # only if it beats the null AND exceeds the booster by more than the 0.017 of AUC that
    # reseeding alone produces.
    if 'learner' in mv_df.columns and mv_df['learner'].nunique() > 1:
        base = mv_df[mv_df['learner'] == 'xgb'].set_index(['outcome', 'horizon'])['auc']
        print()
        print("Learner comparison vs. the booster (A2 rule: beat the null AND +0.017 AUC):")
        for _, r in mv_df[mv_df['learner'] != 'xgb'].iterrows():
            ref = base.get((r['outcome'], r['horizon']), np.nan)
            delta = r['auc'] - ref
            verdict = ("BETTER" if (r.get('passes') and delta > 0.017)
                       else "no (within seed noise)" if np.isfinite(delta) else "no baseline")
            print(f"  {r['learner']:8s} {r['outcome']:11s}@{int(r['horizon']):2d}d  "
                  f"AUC {r['auc']:.4f} vs xgb {ref:.4f}  delta {delta:+.4f}  -> {verdict}")
    if mv_df["passes"].any():
        print("\n  At least one cell beats the null — a COMBINATION carries information that")
        print("  no single feature does. This is what the univariate part cannot see.")
    else:
        print("\n  No cell beats the null. The univariate result is not an artefact of")
        print("  testing features one at a time — the combination carries nothing either.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default=None, help='First daily bar (default: all).')
    ap.add_argument('--end', default=None, help='Last daily bar (default: all).')
    ap.add_argument('--horizons', default='1,3,5,10,20',
                    help='Forward horizons in trading days.')
    ap.add_argument('--zscore-windows', default='125,250,500',
                    help='Causal trailing z-score windows in trading days.')
    ap.add_argument('--n-shuffles', type=int, default=500,
                    help='Circular-shift draws for the null (default 500).')
    ap.add_argument('--cot-lag-days', type=int, default=0,
                    help='EXTRA publication lag for COT, on top of what the pipeline now '
                         'applies itself. Was 4 while external_data.load_cot returned the '
                         'CFTC reference Tuesday; since 2026-08-29 load_cot shifts by the '
                         'publication lag, so the default is 0 and a non-zero value here '
                         'would double-count.')
    ap.add_argument('--barrier-k', type=float, default=1.0,
                    help='Barrier distance in ATR multiples (default 1.0).')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=None, help='Write the full result table to this CSV.')
    ap.add_argument('--skip-conditional', action='store_true',
                    help='Skip the per-regime / post-shock analysis.')
    ap.add_argument('--cond-shuffles', type=int, default=200,
                    help='Null draws per conditional cell (default 200).')
    ap.add_argument('--skip-multivariate', action='store_true',
                    help='Skip the combination test. It is the slow part.')
    ap.add_argument('--mv-horizons', default='5,20',
                    help='Horizons for the multivariate test (default 5,20).')
    ap.add_argument('--mv-shuffles', type=int, default=100,
                    help='Block-shift draws for the multivariate null (default 100).')
    ap.add_argument('--mv-learners', default='xgb',
                    help="Comma-separated learners for the multivariate test: any of "
                         "xgb, rf, rf_leaf. 'rf' is out-of-the-box (unlimited depth) and "
                         "tests the claim that an RF cannot be over-optimised; 'rf_leaf' "
                         "raises min_samples_leaf to match the EFFECTIVE sample size, "
                         "because bagging assumes independent draws and up to 24 "
                         "consecutive rows share one barrier outcome.")
    ap.add_argument('--families', default=None,
                    help='Comma-separated candidate families to build (default: all). '
                         'Known: the external FAMILIES keys plus volstate, technical, '
                         'rmom (A8 realized moments). Restricting the run makes the BH '
                         'correction span exactly that family — the "BH as its own '
                         'family" requirement of a single-family amendment.')
    ap.add_argument('--mv-min-coverage', type=float, default=0.8,
                    help='Drop features present on less than this share of rows before '
                         'the multivariate complete-case filter (default 0.8). COT starts '
                         'in 2016, so without it one short series turns the test into a '
                         'COT-era test on 2,425 of 5,556 bars. 0.0 keeps everything.')
    args = ap.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(','))
    zwins = tuple(int(w) for w in args.zscore_windows.split(','))

    df, cond_df, mv_df = run_audit(
        start=args.start, end=args.end, horizons=horizons,
        zscore_windows=zwins, n_shuffles=args.n_shuffles,
        cot_lag_days=args.cot_lag_days, barrier_k=args.barrier_k, seed=args.seed,
        conditional=not args.skip_conditional, cond_shuffles=args.cond_shuffles,
        multivariate=not args.skip_multivariate,
        mv_horizons=tuple(int(h) for h in args.mv_horizons.split(',')),
        mv_shuffles=args.mv_shuffles, mv_min_coverage=args.mv_min_coverage,
        mv_learners=tuple(x.strip() for x in args.mv_learners.split(',') if x.strip()),
        families=(set(x.strip() for x in args.families.split(',') if x.strip())
                  if args.families else None))
    print_report(df)
    print_conditional_report(cond_df)
    print_multivariate_report(mv_df)

    out = args.out or os.path.join(dir_config.GENERATED_DIR, 'information_audit.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    if len(cond_df):
        cond_df.to_csv(out.replace('.csv', '_conditional.csv'), index=False)
    if len(mv_df):
        mv_df.to_csv(out.replace('.csv', '_multivariate.csv'), index=False)
    print(f"\nFull table: {out}")

    meta = out.replace('.csv', '.meta.json')
    with open(meta, 'w', encoding='utf-8') as f:
        json.dump({
            'horizons': list(horizons), 'zscore_windows': list(zwins),
            'n_shuffles': args.n_shuffles, 'cot_lag_days': args.cot_lag_days,
            'barrier_k': args.barrier_k, 'seed': args.seed,
            'start': args.start, 'end': args.end,
            'families': args.families,
            'n_tests': int(len(df)), 'n_passing': int(df['passes_h1'].sum()),
            'cond_shuffles': args.cond_shuffles, 'mv_shuffles': args.mv_shuffles,
            'n_conditional_tests': int(len(cond_df)),
            'n_conditional_passing': int(cond_df['passes'].sum()) if len(cond_df) else 0,
            'n_multivariate_tests': int(len(mv_df)),
            'n_multivariate_passing': int(mv_df['passes'].sum()) if len(mv_df) else 0,
        }, f, indent=2)
    print(f"Run metadata: {meta}")


if __name__ == '__main__':
    main()
