"""
Data profile — what is actually inside the parquet artefacts a training run wrote.

WHY NOT ydata-profiling (the maintained pandas-profiling)
---------------------------------------------------------
A generic profiler assumes i.i.d. rows. Every artefact here violates that in a way that
makes the generic report actively misleading rather than merely incomplete:

* **Row counts are not sample sizes.** ``X_long_slow.parquet`` is written at M15
  frequency, but the slow models train on a 4h resample and their barrier labels overlap,
  so 330,204 rows carry an effective sample of a few hundred. A profiler reporting
  "n = 330204", and any correlation significance derived from it, is wrong by two orders
  of magnitude.
* **The columns are shifted, warmed-up time series.** Leading NaNs are the indicator
  warm-up, mid-series NaN blocks are a data hole (the cross-asset export gap), and the two
  need opposite reactions. A single missing-value percentage cannot tell them apart; a
  month-by-month coverage map can, so that is what this draws.
* **Train/test is a date, not a random split.** The question worth asking of these files
  is whether a feature's distribution moved between the training window and the backtest
  window. That is a PSI/KS question against ``args.train_end``, which a generic profiler
  has no notion of.
* **Redundancy is already measured here**, by ``analytics/feature_pruning.cluster_features``
  at the |corr| >= 0.80 gate the project uses. Re-deriving it with a different rule would
  give a second, disagreeing number for the same thing.

So this module answers the data-quality half — coverage, degeneracy, outliers, index
integrity, label coverage, drift — with the project's own conventions, and delegates
redundancy to the existing clustering. ``--engine ydata`` is kept as an escape hatch for
when a generic column-by-column dump really is what you want; it is an optional import and
nothing else depends on it.

WHAT IT DOES NOT DO
-------------------
Nothing here is evidence about an edge. It reads artefacts; it trains no model, scores no
strategy and can accept or reject nothing. A clean profile means the inputs are what they
were meant to be, not that they carry information — that question belongs to
``information_audit.py``.

Usage
-----
    python -m ModelTrading.source.python.analytics.data_profile --list
    python -m ModelTrading.source.python.analytics.data_profile --run-id trend_only
    python -m ModelTrading.source.python.analytics.data_profile --run-id trend_only \
        --model long_slow --open
    python -m ModelTrading.source.python.analytics.data_profile --run-id trend_only \
        --engine ydata            # generic report, needs `pip install ydata-profiling`

Read-only with respect to training state: the only thing it writes is the report
directory (``<run_dir>/data_profile/`` unless ``--out`` says otherwise).
"""

import argparse
import base64
import glob
import io
import json
import os
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config                # noqa: E402
from ModelTrading.source.python.analytics import feature_ab          # noqa: E402
from ModelTrading.source.python.analytics import feature_pruning     # noqa: E402
from ModelTrading.source.python.analytics import figures             # noqa: E402
from ModelTrading.source.python.training import sample_weights as sw  # noqa: E402

MODELS = ('long_fast', 'short_fast', 'long_slow', 'short_slow')

# Population Stability Index bands. The 0.10/0.25 split is the credit-scoring convention;
# it is a triage aid, not a test — PSI has no null distribution attached here.
PSI_WARN, PSI_ALERT = 0.10, 0.25

# A column whose most frequent value covers this share carries almost no variation, even
# when it is nominally continuous.
NEAR_CONSTANT_SHARE = 0.99

# Robust outlier gate: distance from the median in units of a MAD-derived sigma.
OUTLIER_SIGMA = 6.0

# Above this many distinct values the modal-share scan is skipped — it is a full hash of
# the column and cannot matter for a column that has no dominant value anyway.
TOP_SHARE_MAX_UNIQUE = 20000

# Both sides are subsampled to this before the KS test: the statistic is stable long
# before 300k rows, and its p-value is meaningless at that size regardless (the rows are
# autocorrelated, so they are not the independent draws the test assumes).
KS_SAMPLE = 20000


# --------------------------------------------------------------------------------------
# Artefact discovery
# --------------------------------------------------------------------------------------

def list_runs(generated_dir=None):
    """Run ids under GENERATED_DIR that hold at least one feature matrix."""
    base = generated_dir or dir_config.GENERATED_DIR
    out = []
    for path in sorted(glob.glob(os.path.join(base, '*'))):
        if os.path.isdir(path) and glob.glob(os.path.join(path, 'X_*.parquet')):
            out.append(os.path.basename(path))
    return out


def load_summary(run_dir):
    """The run's training_summary.json, or {} when it was not written."""
    path = os.path.join(run_dir, 'training_summary.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def available_models(run_dir):
    return [m for m in MODELS if os.path.exists(os.path.join(run_dir, f'X_{m}.parquet'))]


# --------------------------------------------------------------------------------------
# Index integrity
# --------------------------------------------------------------------------------------

def _covers_saturday(a, b):
    """True when the interval spanned by (a, b] contains a Saturday — market shut."""
    days = pd.date_range(pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize(), freq='D')
    return bool((days.dayofweek == 5).any())


def index_report(df, name, max_gaps=8):
    """Monotonicity, duplicates, the modal bar step and the gaps that are not weekends."""
    idx = df.index
    out = {'artefact': name, 'rows': int(len(df)), 'columns': int(df.shape[1]),
           'index_type': type(idx).__name__}
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return out

    diffs = idx.to_series().diff().dropna()
    step = diffs.mode().iloc[0] if len(diffs) else pd.Timedelta(0)
    out.update({
        'start': str(idx[0]), 'end': str(idx[-1]),
        'monotonic_increasing': bool(idx.is_monotonic_increasing),
        'duplicate_timestamps': int(idx.duplicated().sum()),
        'modal_step': str(step),
    })

    gaps = diffs[diffs > step * 1.5] if step > pd.Timedelta(0) else diffs.iloc[:0]
    non_weekend = [(ts, gap) for ts, gap in gaps.items()
                   if not _covers_saturday(ts - gap, ts)]
    out['n_gaps'] = int(len(gaps))
    out['n_gaps_non_weekend'] = int(len(non_weekend))
    out['largest_non_weekend_gaps'] = [
        {'ends_at': str(ts), 'gap': str(gap)}
        for ts, gap in sorted(non_weekend, key=lambda kv: kv[1], reverse=True)[:max_gaps]
    ]
    return out


# --------------------------------------------------------------------------------------
# Per-column statistics
# --------------------------------------------------------------------------------------

def _longest_run(mask):
    """Length of the longest run of True in a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max())


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, pd.Timedelta)):
        return str(value)
    return value


def column_stats(name, s):
    """Quality and shape statistics for one column.

    Infinities are counted and then excluded from every moment and quantile — leaving them
    in turns a single bad row into a NaN mean and hides the 329,999 good ones.
    """
    n = int(len(s))
    isna = s.isna().to_numpy()
    out = {
        'column': name,
        'dtype': str(s.dtype),
        'n': n,
        'missing': int(isna.sum()),
        'missing_pct': float(isna.mean() * 100.0) if n else 0.0,
        'longest_missing_run': _longest_run(isna),
        'first_valid': str(s.first_valid_index()) if n and not isna.all() else None,
        'last_valid': str(s.last_valid_index()) if n and not isna.all() else None,
        'n_unique': int(s.nunique(dropna=True)),
        'n_inf': 0,
    }

    present = s.dropna()
    if not len(present):
        out['flags'] = ['all-missing']
        return out

    if out['n_unique'] <= TOP_SHARE_MAX_UNIQUE:
        counts = present.value_counts()
        out['top_value'] = _jsonable(counts.index[0])
        out['top_share'] = float(counts.iloc[0] / len(present))

    if pd.api.types.is_numeric_dtype(s):
        arr = present.to_numpy(dtype='float64')
        finite = np.isfinite(arr)
        out['n_inf'] = int((~finite).sum())
        arr = arr[finite]
        if len(arr):
            q = np.quantile(arr, [0.01, 0.25, 0.50, 0.75, 0.99])
            out.update({
                'min': float(arr.min()), 'p01': float(q[0]), 'p25': float(q[1]),
                'median': float(q[2]), 'p75': float(q[3]), 'p99': float(q[4]),
                'max': float(arr.max()), 'mean': float(arr.mean()),
                'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                'zero_share': float((arr == 0).mean()),
                'skew': float(pd.Series(arr).skew()) if len(arr) > 2 else np.nan,
                'kurtosis': float(pd.Series(arr).kurt()) if len(arr) > 3 else np.nan,
            })
            mad = float(np.median(np.abs(arr - out['median'])))
            scale = 1.4826 * mad
            out['outlier_share'] = (
                float((np.abs(arr - out['median']) > OUTLIER_SIGMA * scale).mean())
                if scale > 0 else 0.0)

    out['flags'] = _column_flags(out)
    return out


def _column_flags(st):
    flags = []
    if st['missing_pct'] >= 100.0:
        flags.append('all-missing')
    elif st['missing_pct'] > 5.0:
        flags.append('high-missing')
    if st['n_unique'] <= 1:
        flags.append('constant')
    elif st.get('top_share', 0.0) >= NEAR_CONSTANT_SHARE:
        flags.append('near-constant')
    if st.get('n_inf', 0):
        flags.append('has-inf')
    kurt = st.get('kurtosis')
    if kurt is not None and np.isfinite(kurt) and abs(kurt) > 50.0:
        flags.append('heavy-tail')
    if (st.get('outlier_share') or 0.0) > 0.01:
        flags.append('outliers')
    return flags


def frame_profile(df):
    """Per-column statistics for every column of a frame, as a DataFrame."""
    return pd.DataFrame([column_stats(c, df[c]) for c in df.columns])


def duplicate_columns(df):
    """Groups of columns holding identical values (NaN positions included)."""
    seen = {}
    for col in df.columns:
        s = df[col]
        key = (str(s.dtype), int(s.isna().sum()),
               float(np.nansum(s.to_numpy(dtype='float64', na_value=np.nan)))
               if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.len().sum())
        seen.setdefault(key, []).append(col)

    groups = []
    for members in seen.values():
        if len(members) < 2:
            continue
        remaining = list(members)
        while len(remaining) > 1:
            head, rest, group = remaining[0], [], [remaining[0]]
            for other in remaining[1:]:
                if df[head].equals(df[other]):
                    group.append(other)
                else:
                    rest.append(other)
            if len(group) > 1:
                groups.append(group)
            remaining = rest
    return groups


# --------------------------------------------------------------------------------------
# Train / test drift
# --------------------------------------------------------------------------------------

def psi(train, test, bins=10):
    """Population Stability Index of `test` against `train`, binned on train quantiles."""
    tr = np.asarray(train, dtype='float64')
    te = np.asarray(test, dtype='float64')
    tr, te = tr[np.isfinite(tr)], te[np.isfinite(te)]
    if len(tr) < 50 or len(te) < 50:
        return np.nan
    edges = np.unique(np.quantile(tr, np.linspace(0.0, 1.0, bins + 1))).astype('float64')
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(tr, bins=edges)[0].astype('float64')
    q = np.histogram(te, bins=edges)[0].astype('float64')
    p, q = p / p.sum(), q / q.sum()
    eps = 1e-6
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(((q - p) * np.log(q / p)).sum())


def drift_report(df, split, start=None, rng=None):
    """PSI and a KS statistic per column, training window against everything after it.

    `start` bounds the training side at `train_start`. The parquet also holds the feature
    warm-up that precedes it — on run trend_only, 2012-11 onward against a train_start of
    2015-01 — and those bars were never trained on, so including them measures drift
    against a window the model never saw.
    """
    if split is None or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    split = pd.Timestamp(split)
    in_train = df.index <= split
    if start is not None:
        in_train &= df.index >= pd.Timestamp(start)
    train, test = df.loc[in_train], df.loc[df.index > split]
    if not len(train) or not len(test):
        return pd.DataFrame()

    try:
        from scipy.stats import ks_2samp
    except ImportError:                                            # pragma: no cover
        ks_2samp = None
    rng = rng or np.random.default_rng(0)

    rows = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        a = train[col].to_numpy(dtype='float64', na_value=np.nan)
        b = test[col].to_numpy(dtype='float64', na_value=np.nan)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        value = psi(a, b)
        ks = np.nan
        if ks_2samp is not None and len(a) > 50 and len(b) > 50:
            sa = a if len(a) <= KS_SAMPLE else rng.choice(a, KS_SAMPLE, replace=False)
            sb = b if len(b) <= KS_SAMPLE else rng.choice(b, KS_SAMPLE, replace=False)
            ks = float(ks_2samp(sa, sb).statistic)
        rows.append({
            'column': col, 'n_train': int(len(a)), 'n_test': int(len(b)),
            'train_mean': float(a.mean()) if len(a) else np.nan,
            'test_mean': float(b.mean()) if len(b) else np.nan,
            'psi': value, 'ks': ks,
            'verdict': ('n/a' if not np.isfinite(value)
                        else 'alert' if value >= PSI_ALERT
                        else 'warn' if value >= PSI_WARN else 'stable'),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('psi', ascending=False, na_position='last')


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

def label_report(run_dir, model, train_start=None, train_end=None):
    """Label coverage at the model's own training cadence, with the effective sample.

    Four row counts are reported and they are not interchangeable: what is stored (M15),
    what the model trains on (4h for the slow pair — the M15 forward-fill would inflate
    the sample ~16x), what survives the complete-case filter, and n_eff once the barrier
    overlap is accounted for. The last one is what a standard error must use.
    """
    x_path = os.path.join(run_dir, f'X_{model}.parquet')
    if not os.path.exists(x_path):
        return None
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(feature_ab._find_label_file(run_dir, model)).iloc[:, 0]

    out = {'model': model, 'rows_stored': int(len(X)), 'n_features': int(X.shape[1])}
    out['cadence'] = 'fast (M15)' if model.endswith('_fast') else 'slow (4h resample)'
    if model.endswith('_slow'):
        X, y = X.resample('4h').first(), y.resample('4h').first()
    out['rows_at_cadence'] = int(len(X))

    ok = X.notna().all(axis=1) & y.notna()
    Xc, yc = X[ok], y[ok].astype(int)
    out['rows_complete_case'] = int(len(Xc))
    out['dropped_incomplete'] = int(len(X) - len(Xc))
    out['positives'] = int(yc.sum())
    out['positive_rate'] = float(yc.mean()) if len(yc) else 0.0

    out['n_eff'] = float(len(Xc))
    out['n_eff_measured'] = False
    meta_path = os.path.join(run_dir, 'label_targets.parquet')
    if os.path.exists(meta_path) and len(Xc):
        meta = pd.read_parquet(meta_path)
        t1 = sw.t1_from_metadata(meta, model, index=Xc.index)
        if t1 is not None:
            out['n_eff'] = float(sw.effective_sample_size(t1))
            out['mean_uniqueness'] = float(np.mean(sw.average_uniqueness(t1)))
            out['n_eff_measured'] = True
    out['n_eff_positives'] = out['n_eff'] * out['positive_rate']
    out['eff_positives_per_feature'] = out['n_eff_positives'] / max(out['n_features'], 1)

    if len(yc):
        monthly = yc.groupby(yc.index.to_period('M')).agg(['size', 'sum'])
        out['monthly'] = {str(p): {'rows': int(r['size']), 'positives': int(r['sum'])}
                          for p, r in monthly.iterrows()}
        empty = [str(p) for p, r in monthly.iterrows() if r['sum'] == 0]
        out['months'] = int(len(monthly))
        out['empty_months'] = empty
        out['n_empty_months'] = int(len(empty))

        # The artefact spans the warm-up and the backtest window as well, and an empty
        # month outside the training window costs nothing. Only the ones INSIDE it can
        # leave a TimeSeriesSplit fold with no positive to train on — which is the
        # failure that turned run best_20260808's headline AUC into an artefact — so
        # that is the count worth acting on.
        if train_start is not None or train_end is not None:
            lo = pd.Period(pd.Timestamp(train_start), freq='M') if train_start else None
            hi = pd.Period(pd.Timestamp(train_end), freq='M') if train_end else None
            in_train = [p for p in empty
                        if (lo is None or pd.Period(p, freq='M') >= lo)
                        and (hi is None or pd.Period(p, freq='M') <= hi)]
            out['empty_months_in_train'] = in_train
            out['n_empty_months_in_train'] = int(len(in_train))
    return out


# --------------------------------------------------------------------------------------
# OHLC and out-of-fold scores
# --------------------------------------------------------------------------------------

def ohlc_report(run_dir):
    """Bar-level sanity of the price frame every label and every backtest is built on."""
    path = os.path.join(run_dir, 'ohlc.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    cols = {c.split('_')[-1]: c for c in df.columns
            if c.split('_')[-1] in ('open', 'high', 'low', 'close')}
    if len(cols) < 4:
        return {'artefact': 'ohlc.parquet', 'rows': int(len(df)),
                'note': f'unexpected columns: {list(df.columns)}'}
    o, h, l, c = (df[cols[k]] for k in ('open', 'high', 'low', 'close'))
    ret = c.pct_change()
    return {
        'artefact': 'ohlc.parquet',
        'rows': int(len(df)),
        'high_below_body': int((h < np.maximum(o, c) - 1e-12).sum()),
        'low_above_body': int((l > np.minimum(o, c) + 1e-12).sum()),
        'high_below_low': int((h < l).sum()),
        'non_positive_price': int((df[list(cols.values())] <= 0).to_numpy().sum()),
        'zero_range_bars': int((h == l).sum()),
        'zero_range_share': float((h == l).mean()) if len(df) else 0.0,
        'extreme_returns_gt_1pct': int((ret.abs() > 0.01).sum()),
        'max_abs_return': float(ret.abs().max()) if len(ret) else np.nan,
        'missing_cells': int(df.isna().to_numpy().sum()),
    }


def oof_report(run_dir):
    """Per (model, stage) shape of the pooled out-of-fold validation scores."""
    path = os.path.join(run_dir, 'oof_predictions.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    rows = []
    for (model, stage), g in df.groupby(['model', 'stage'], sort=True):
        per_fold = g.groupby('fold')['y_score']
        rows.append({
            'model': model, 'stage': stage, 'rows': int(len(g)),
            'folds': int(g['fold'].nunique()),
            'positive_rate': float(g['y_true'].mean()),
            'score_min': float(g['y_score'].min()),
            'score_mean': float(g['y_score'].mean()),
            'score_max': float(g['y_score'].max()),
            # A fold whose scores never move trained a constant model; its AUC of exactly
            # 0.500 is still pooled into the headline number unless it is spotted here.
            'constant_score_folds': int(((per_fold.max() - per_fold.min()) < 1e-12).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def _dataurl(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=figures.SURFACE)
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def export_figures(profile, figures_dir):
    """Write every embedded figure of a built profile as a standalone PNG.

    The report deliberately inlines its figures as base64 data-URLs so the HTML
    stays self-contained — but a document that wants ONE of these figures (a
    train/test distribution grid, the |corr| heatmap) should not have to
    screenshot the report. Files are named ``<model>_<figure>.png`` for
    per-model figures and ``<figure>.png`` for run-level ones; ``None`` figures
    (not renderable for that artefact) are skipped. Returns the written paths.
    """
    prefix = 'data:image/png;base64,'
    os.makedirs(figures_dir, exist_ok=True)
    written = []

    def _write(name, dataurl):
        if not dataurl or not dataurl.startswith(prefix):
            return
        path = os.path.join(figures_dir, f'{name}.png')
        with open(path, 'wb') as fh:
            fh.write(base64.b64decode(dataurl[len(prefix):]))
        written.append(path)

    for model, entry in profile.get('models', {}).items():
        for fig_name, dataurl in (entry.get('figures') or {}).items():
            _write(f'{model}_{fig_name}', dataurl)
    for fig_name, dataurl in (profile.get('figures') or {}).items():
        _write(fig_name, dataurl)
    return written


def fig_coverage(df, split=None):
    """Month x feature map of non-missing share — warm-up, holes and staleness at a glance."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return None
    cov = df.notna().groupby(df.index.to_period('M')).mean()
    fig, ax = plt.subplots(figsize=(figures.WIDTH_FULL,
                                    max(2.2, 0.16 * df.shape[1] + 0.9)))
    im = ax.imshow(cov.to_numpy().T, aspect='auto', vmin=0.0, vmax=1.0,
                   cmap='YlGnBu', interpolation='nearest')
    ax.set_yticks(range(df.shape[1]))
    ax.set_yticklabels(df.columns, fontsize=6)
    ticks = np.unique(np.linspace(0, len(cov) - 1, min(10, len(cov))).astype(int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(cov.index[t]) for t in ticks], fontsize=6, rotation=45,
                       ha='right')
    if split is not None:
        pos = cov.index.get_indexer([pd.Period(pd.Timestamp(split), freq='M')])[0]
        if pos >= 0:
            ax.axvline(pos + 0.5, color=figures.STATUS['critical'], lw=1.0, ls='--')
    ax.set_title('Non-missing share per month (dashed = train_end)', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    return _dataurl(fig)


def fig_distributions(df, split, max_features=40):
    """Train vs test histogram per feature — the drift table made visual."""
    cols = list(df.columns)[:max_features]
    if not cols:
        return None
    split = pd.Timestamp(split) if split is not None else None
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(figures.WIDTH_FULL, 1.35 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        s = df[col].to_numpy(dtype='float64', na_value=np.nan)
        finite_mask = np.isfinite(s)
        if split is not None and isinstance(df.index, pd.DatetimeIndex):
            in_train = np.asarray(df.index <= split)
            groups = [('train', s[finite_mask & in_train]),
                      ('test', s[finite_mask & ~in_train])]
        else:
            groups = [('all', s[finite_mask])]
        finite = s[finite_mask]
        rng = (float(np.quantile(finite, 0.005)),
               float(np.quantile(finite, 0.995))) if len(finite) else None
        if rng and rng[0] == rng[1]:
            rng = None
        for i, (name, part) in enumerate(groups):
            if len(part):
                ax.hist(part, bins=30, range=rng, density=True, histtype='step',
                        color=figures.CATEGORICAL[i], lw=0.9, label=name)
        ax.set_title(col, fontsize=5.5)
        ax.tick_params(labelsize=4.5)
        ax.set_yticks([])
    for ax in axes[len(cols):]:
        ax.axis('off')
    if split is not None:
        axes[0].legend(fontsize=4.5, frameon=False)
    fig.tight_layout()
    return _dataurl(fig)


def fig_monthly_mean(df, split, max_features=40):
    """Monthly mean per feature — a level shift here is drift you can date."""
    cols = list(df.columns)[:max_features]
    if not cols or not isinstance(df.index, pd.DatetimeIndex):
        return None
    monthly = df[cols].groupby(df.index.to_period('M')).mean()
    x = monthly.index.to_timestamp()
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(figures.WIDTH_FULL, 1.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        ax.plot(x, monthly[col].to_numpy(), color=figures.CATEGORICAL[0], lw=0.7)
        if split is not None:
            ax.axvline(pd.Timestamp(split), color=figures.STATUS['critical'], lw=0.7,
                       ls='--')
        ax.set_title(col, fontsize=5.5)
        ax.tick_params(labelsize=4.5)
        ax.set_xticks([])
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.tight_layout()
    return _dataurl(fig)


def fig_correlation(df, max_features=60):
    """|corr| heatmap, ordered by the pruning clusters so redundancy sits on the diagonal."""
    cols = list(df.columns)[:max_features]
    if len(cols) < 2:
        return None
    sub = df[cols].dropna()
    if len(sub) < 10:
        return None
    try:
        clusters, _ = feature_pruning.cluster_features(sub)
        order = [c for group in clusters for c in group]
    except Exception:                                              # pragma: no cover
        order = cols
    corr = sub[order].corr().abs()
    fig, ax = plt.subplots(figsize=(figures.WIDTH_FULL, figures.WIDTH_FULL))
    im = ax.imshow(corr.to_numpy(), vmin=0.0, vmax=1.0, cmap='YlGnBu')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=5, rotation=90)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5)
    ax.set_title('|corr|, ordered by redundancy cluster', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    return _dataurl(fig)


def fig_label_calendar(reports):
    """Positives per month per model. Empty months are what kill a CV fold."""
    reports = [r for r in reports if r and r.get('monthly')]
    if not reports:
        return None
    fig, axes = plt.subplots(len(reports), 1,
                             figsize=(figures.WIDTH_FULL, 1.25 * len(reports)))
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reports):
        periods = list(rep['monthly'].keys())
        pos = [rep['monthly'][p]['positives'] for p in periods]
        colors = [figures.STATUS['critical'] if v == 0 else figures.CATEGORICAL[0]
                  for v in pos]
        ax.bar(range(len(pos)), pos, color=colors, width=0.9)
        ax.set_ylabel(rep['model'], fontsize=6)
        ticks = np.unique(np.linspace(0, len(pos) - 1, min(12, len(pos))).astype(int))
        ax.set_xticks(ticks)
        ax.set_xticklabels([periods[t] for t in ticks], fontsize=5, rotation=45, ha='right')
        ax.tick_params(labelsize=5)
    axes[0].set_title('Positive labels per month (red = empty month)', fontsize=8)
    fig.tight_layout()
    return _dataurl(fig)


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------

def profile_run(run_id, models=None, generated_dir=None, max_features=40,
                corr_threshold=0.80, with_figures=True):
    """Everything the report needs, as one plain-dict structure (also written as JSON)."""
    base = generated_dir or dir_config.GENERATED_DIR
    run_dir = os.path.join(base, run_id)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")

    summary = load_summary(run_dir)
    args = summary.get('args', {})
    split = args.get('train_end') or summary.get('train_end')
    models = list(models or available_models(run_dir))
    if not models:
        raise SystemExit(f"No X_*.parquet in {run_dir}")

    profile = {
        'run_id': run_id,
        'run_dir': run_dir,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'label_mode': args.get('label_mode'),
        'features_config': summary.get('features_config'),
        'train_start': args.get('train_start'), 'train_end': split,
        'backtest_start': args.get('backtest_start'),
        'backtest_end': args.get('backtest_end'),
        'pip_target': args.get('pip_target'), 'stop_pips': args.get('stop_pips'),
        'command_line': summary.get('command_line'),
        'models': {},
        'artefacts': [],
        'figures': {},
    }

    for name in sorted(os.listdir(run_dir)):
        if name.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(run_dir, name))
            profile['artefacts'].append(index_report(df, name))

    profile['ohlc'] = ohlc_report(run_dir)
    oof = oof_report(run_dir)
    profile['oof'] = oof.to_dict('records') if oof is not None else None

    label_reports = []
    for model in models:
        X = pd.read_parquet(os.path.join(run_dir, f'X_{model}.parquet'))
        stats = frame_profile(X)
        drift = drift_report(X, split, args.get('train_start'))
        if not drift.empty:
            verdicts = drift.set_index('column')['verdict']
            stats['flags'] = [
                flags + (['drift'] if verdicts.get(col) == 'alert' else [])
                for col, flags in zip(stats['column'], stats['flags'])]

        complete = X.dropna()
        if len(complete) > 10 and X.shape[1] > 1:
            clusters, keep = feature_pruning.cluster_features(complete, corr_threshold)
        else:
            clusters, keep = [[c] for c in X.columns], list(X.columns)

        rep = label_report(run_dir, model, args.get('train_start'), split)
        label_reports.append(rep)

        entry = {
            'columns': stats.to_dict('records'),
            'drift': drift.to_dict('records') if not drift.empty else [],
            'duplicate_columns': duplicate_columns(X),
            'clusters': [c for c in clusters if len(c) > 1],
            'n_clusters': len(clusters),
            'representatives': keep,
            'labels': rep,
        }
        if with_figures:
            entry['figures'] = {
                'coverage': fig_coverage(X, split),
                'distributions': fig_distributions(X, split, max_features),
                'monthly_mean': fig_monthly_mean(X, split, max_features),
                'correlation': fig_correlation(X, max_features),
            }
        profile['models'][model] = entry

    if with_figures:
        profile['figures']['labels'] = fig_label_calendar(label_reports)
    return profile


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------

_CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--surface:#fcfcfb;--line:#e1e0d9;
      --warn:#fab219;--bad:#d03b3b;--accent:#2a78d6;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
     font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;}
header{padding:18px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;
       background:var(--surface);z-index:5}
h1{font-size:18px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)}
h3{font-size:13px;margin:18px 0 6px;color:var(--ink2)}
main{padding:0 24px 60px;max-width:1400px}
nav a{margin-right:12px;color:var(--accent);text-decoration:none;font-size:12px}
.meta{color:var(--ink2);font-size:12px}
.kv{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0}
.kv div{font-size:12px}.kv b{color:var(--ink2);font-weight:500}
table{border-collapse:collapse;font-size:11px;width:100%;margin:6px 0 14px}
th,td{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap}
th{background:#f4f3ef;position:sticky;top:0;font-weight:600}
td:first-child,th:first-child{text-align:left}
.scroll{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:4px}
.chip{display:inline-block;padding:0 5px;border-radius:3px;font-size:10px;margin-right:3px;
      color:#fff}
.chip.bad{background:var(--bad)}.chip.warn{background:var(--warn);color:#3a2b00}
.chip.info{background:var(--muted)}
img{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff;margin:6px 0}
.note{background:#f4f3ef;border-left:3px solid var(--accent);padding:8px 12px;margin:10px 0;
      font-size:12px;color:var(--ink2)}
code{font-family:ui-monospace,Consolas,monospace;font-size:11px}
ul.meta li{font-family:ui-monospace,Consolas,monospace;font-size:11px}
"""

_BAD_FLAGS = {'all-missing', 'constant', 'has-inf'}
_WARN_FLAGS = {'high-missing', 'near-constant', 'drift', 'outliers', 'heavy-tail'}

_COLUMN_ORDER = ['column', 'flags', 'dtype', 'missing_pct', 'longest_missing_run',
                 'n_unique', 'top_share', 'zero_share', 'outlier_share', 'min', 'p01',
                 'median', 'p99', 'max', 'mean', 'std', 'skew', 'kurtosis',
                 'first_valid', 'last_valid']

_DRIFT_ORDER = ['column', 'verdict', 'psi', 'ks', 'train_mean', 'test_mean',
                'n_train', 'n_test']


def _escape(text):
    return (str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _fmt(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<span class="meta">-</span>'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:,.4g}'
    if isinstance(value, (list, tuple)):
        return ''.join(
            '<span class="chip {}">{}</span>'.format(
                'bad' if f in _BAD_FLAGS else 'warn' if f in _WARN_FLAGS else 'info',
                _escape(f))
            for f in value) or '<span class="meta">ok</span>'
    return _escape(value)


def _table(records, order=None):
    if not records:
        return '<p class="meta">nothing to show</p>'
    cols = order or list(records[0].keys())
    cols = [c for c in cols if any(c in r for r in records)]
    head = ''.join(f'<th>{_escape(c)}</th>' for c in cols)
    body = ''.join(
        '<tr>' + ''.join(f'<td>{_fmt(r.get(c))}</td>' for c in cols) + '</tr>'
        for r in records)
    return ('<div class="scroll"><table><thead><tr>' + head +
            '</tr></thead><tbody>' + body + '</tbody></table></div>')


def _img(src, caption):
    if not src:
        return ''
    return f'<p class="meta">{_escape(caption)}</p><img src="{src}" alt="{_escape(caption)}">'


def render_html(profile):
    p = profile
    nav = ' '.join(f'<a href="#{m}">{m}</a>' for m in p['models'])
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Data profile - {_escape(p['run_id'])}</title><style>{_CSS}</style></head><body>
<header><h1>Data profile &mdash; {_escape(p['run_id'])}</h1>
<div class="meta">{_escape(p['generated_at'])} &middot; label mode
 <code>{_escape(p.get('label_mode'))}</code> &middot; train
 {_escape(p.get('train_start'))} &rarr; {_escape(p.get('train_end'))} &middot; backtest
 {_escape(p.get('backtest_start'))} &rarr; {_escape(p.get('backtest_end'))}</div>
<nav><a href="#artefacts">artefacts</a><a href="#prices">prices</a>{nav}<a href="#oof">oof</a></nav>
</header><main>
<div class="note">Coverage, degeneracy, drift and label sanity of the artefacts this run
wrote. Nothing here is evidence about an edge &mdash; a clean profile means the inputs are
what they were meant to be, not that they carry information.</div>"""]

    parts.append('<h2 id="artefacts">Artefacts and index integrity</h2>')
    parts.append(_table([{k: v for k, v in a.items() if k != 'largest_non_weekend_gaps'}
                         for a in p['artefacts']]))
    gaps = [{'artefact': a['artefact'], **g} for a in p['artefacts']
            for g in a.get('largest_non_weekend_gaps', [])]
    if gaps:
        parts.append('<h3>Largest non-weekend gaps</h3>')
        parts.append(_table(gaps))

    if p.get('ohlc'):
        parts.append('<h2 id="prices">Price frame</h2>')
        parts.append(_table([p['ohlc']]))

    if p['figures'].get('labels'):
        parts.append('<h2>Label coverage</h2>')
        parts.append(_img(p['figures']['labels'], 'Positive labels per month'))

    for model, entry in p['models'].items():
        parts.append(f'<h2 id="{model}">{_escape(model)}</h2>')
        lab = entry.get('labels') or {}
        if lab:
            parts.append('<div class="kv">' + ''.join(
                f'<div><b>{k}</b> {_fmt(lab.get(k))}</div>' for k in
                ('cadence', 'rows_stored', 'rows_at_cadence', 'rows_complete_case',
                 'dropped_incomplete', 'positives', 'positive_rate', 'n_eff',
                 'n_eff_measured', 'eff_positives_per_feature', 'n_empty_months',
                 'n_empty_months_in_train')
            ) + '</div>')
            if lab.get('empty_months_in_train'):
                parts.append('<div class="note"><b>Inside the training window</b>, months '
                             'with no positive label: '
                             + _escape(', '.join(lab['empty_months_in_train']))
                             + '. A CV fold whose training part falls inside these trains '
                               'a constant model, and its AUC of exactly 0.500 is still '
                               'pooled into the headline number.</div>')
            if lab.get('empty_months'):
                parts.append('<div class="note">All months with no positive label '
                             '(training window, warm-up and backtest window together): '
                             + _escape(', '.join(lab['empty_months'])) + '.</div>')
        parts.append('<h3>Columns</h3>')
        parts.append(_table(entry['columns'], _COLUMN_ORDER))

        if entry['duplicate_columns']:
            parts.append('<div class="note">Identical columns: '
                         + _escape('; '.join(' = '.join(g)
                                             for g in entry['duplicate_columns']))
                         + '</div>')
        if entry['clusters']:
            parts.append(f"<h3>Redundancy &mdash; {entry['n_clusters']} clusters from "
                         f"{len(entry['columns'])} features</h3>")
            parts.append('<ul class="meta">' + ''.join(
                f'<li>{_escape(", ".join(c))}</li>' for c in entry['clusters']) + '</ul>')
        if entry['drift']:
            parts.append('<h3>Train &rarr; test drift</h3>')
            parts.append(_table(entry['drift'], _DRIFT_ORDER))

        figs = entry.get('figures') or {}
        parts.append(_img(figs.get('coverage'), 'Non-missing share per month'))
        parts.append(_img(figs.get('distributions'), 'Distribution, train vs test'))
        parts.append(_img(figs.get('monthly_mean'), 'Monthly mean per feature'))
        parts.append(_img(figs.get('correlation'), '|corr| ordered by cluster'))

    if p.get('oof'):
        parts.append('<h2 id="oof">Out-of-fold scores</h2>')
        parts.append(_table(p['oof']))

    if p.get('command_line'):
        parts.append('<h2>Command</h2><p><code>'
                     + _escape(p['command_line']) + '</code></p>')
    parts.append('</main></body></html>')
    return '\n'.join(parts)


def _json_default(obj):
    value = _jsonable(obj)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is obj and not isinstance(obj, (str, int, float, bool, list, dict, type(None))):
        return str(obj)
    return value


def write_report(profile, out_dir):
    """HTML (self-contained, figures inlined) plus the machine-readable JSON beside it."""
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, 'data_profile.html')
    json_path = os.path.join(out_dir, 'data_profile.json')
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(render_html(profile))
    slim = {k: v for k, v in profile.items() if k != 'figures'}
    slim['models'] = {m: {k: v for k, v in e.items() if k != 'figures'}
                      for m, e in profile['models'].items()}
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(slim, fh, indent=2, default=_json_default)
    return html_path, json_path


# --------------------------------------------------------------------------------------
# Optional generic engine
# --------------------------------------------------------------------------------------

def run_ydata(run_dir, models, out_dir, minimal=True):
    """Generic column-by-column report, for when that really is what you want.

    An optional dependency on purpose: the pinned stack (pandas 2.2.2 / numpy 1.26.4) is
    what training and the ONNX export are validated against, and ydata-profiling pulls a
    large transitive tree. Install it into the venv if you want this path.
    """
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        raise SystemExit(
            "ydata-profiling is not installed.\n"
            "  pip install ydata-profiling\n"
            "It is deliberately not in requirements.txt - see the module docstring for "
            "why the native engine is the default.")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for model in models:
        df = pd.read_parquet(os.path.join(run_dir, f'X_{model}.parquet'))
        report = ProfileReport(df, title=f'{model}', tsmode=True, minimal=minimal)
        path = os.path.join(out_dir, f'ydata_{model}.html')
        report.to_file(path)
        written.append(path)
    return written


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def print_summary(profile):
    p = profile
    print('=' * 78)
    print(f"DATA PROFILE - {p['run_id']}   label mode: {p.get('label_mode')}")
    print('=' * 78)
    for a in p['artefacts']:
        gaps = a.get('n_gaps_non_weekend')
        extra = '' if gaps is None else f"   non-weekend gaps {gaps}"
        print(f"  {a['artefact']:<32} {a['rows']:>9,} x {a['columns']:<3}{extra}")

    if p.get('ohlc'):
        o = p['ohlc']
        bad = (o.get('high_below_body', 0) + o.get('low_above_body', 0)
               + o.get('high_below_low', 0))
        print(f"\n  ohlc: {bad} bar-consistency violations, "
              f"{o.get('zero_range_bars', 0):,} zero-range bars, "
              f"{o.get('missing_cells', 0):,} missing cells")

    for model, entry in p['models'].items():
        lab = entry.get('labels') or {}
        flagged = [c for c in entry['columns'] if c.get('flags')]
        alerts = [d for d in entry['drift'] if d.get('verdict') == 'alert']
        print(f"\n  {model}")
        print(f"    rows stored {lab.get('rows_stored', 0):,} -> at cadence "
              f"{lab.get('rows_at_cadence', 0):,} -> complete case "
              f"{lab.get('rows_complete_case', 0):,} -> n_eff {lab.get('n_eff', 0):,.0f}"
              f"{'' if lab.get('n_eff_measured') else '  (UPPER BOUND: t1 unavailable)'}")
        print(f"    positives {lab.get('positives', 0):,} "
              f"({lab.get('positive_rate', 0) * 100:.2f}%), empty months "
              f"{lab.get('n_empty_months_in_train', 0)} in train / "
              f"{lab.get('n_empty_months', 0)} overall, eff. positives/feature "
              f"{lab.get('eff_positives_per_feature', 0):.1f}")
        print(f"    features {len(entry['columns'])} -> {entry['n_clusters']} clusters, "
              f"{len(flagged)} flagged columns, {len(alerts)} PSI alerts")
        for col in flagged[:8]:
            print(f"      - {col['column']}: {', '.join(col['flags'])}")
        if len(flagged) > 8:
            print(f"      ... {len(flagged) - 8} more")


def main():
    ap = argparse.ArgumentParser(
        description='Data-quality profile of the parquet artefacts a training run wrote.')
    ap.add_argument('--run-id', help='run directory under ModelTrading/generated/')
    ap.add_argument('--list', action='store_true', help='list runs that have artefacts')
    ap.add_argument('--model', default='all',
                    help="'all' (default) or a comma-separated subset of "
                         + ','.join(MODELS))
    ap.add_argument('--generated-dir', default=None)
    ap.add_argument('--out', default=None,
                    help='report directory (default <run_dir>/data_profile)')
    ap.add_argument('--engine', choices=('native', 'ydata', 'both'), default='native')
    ap.add_argument('--max-features', type=int, default=40,
                    help='cap on features drawn in the small-multiple figures')
    ap.add_argument('--corr-threshold', type=float, default=0.80,
                    help='|corr| gate for the redundancy clusters (project default 0.80)')
    ap.add_argument('--no-figures', action='store_true',
                    help='tables only, much faster')
    ap.add_argument('--figures-dir', default=None,
                    help='additionally export every embedded figure as a '
                         'standalone PNG into this directory '
                         '(<model>_<figure>.png; requires figures)')
    ap.add_argument('--open', action='store_true', dest='open_browser',
                    help='open the report when it is written')
    args = ap.parse_args()

    if args.list:
        runs = list_runs(args.generated_dir)
        print('\n'.join(runs) if runs else 'no runs with X_*.parquet found')
        return
    if not args.run_id:
        ap.error('--run-id is required (or --list)')
    if args.figures_dir and args.no_figures:
        ap.error('--figures-dir needs the figures that --no-figures skips')

    base = args.generated_dir or dir_config.GENERATED_DIR
    run_dir = os.path.join(base, args.run_id)
    models = (available_models(run_dir) if args.model == 'all'
              else [m.strip() for m in args.model.split(',') if m.strip()])
    out_dir = args.out or os.path.join(run_dir, 'data_profile')

    if args.engine in ('native', 'both'):
        profile = profile_run(args.run_id, models, generated_dir=args.generated_dir,
                              max_features=args.max_features,
                              corr_threshold=args.corr_threshold,
                              with_figures=not args.no_figures)
        print_summary(profile)
        html_path, json_path = write_report(profile, out_dir)
        print(f"\nWritten:\n  {html_path}\n  {json_path}")
        if args.figures_dir:
            for path in export_figures(profile, args.figures_dir):
                print(f"  {path}")
        if args.open_browser:
            webbrowser.open('file://' + os.path.abspath(html_path))

    if args.engine in ('ydata', 'both'):
        for path in run_ydata(run_dir, models, out_dir):
            print(f"  {path}")


if __name__ == '__main__':
    main()
