"""
Feature Stationarity Analysis

Loads OHLC data, computes features, and runs statistical stationarity tests
(ADF, KPSS, Ljung-Box, rolling variance CV) on each selected feature.
Saves per-feature diagnostic plots and a summary CSV.

For Non-Stationary and Inconclusive verdicts, also writes machine-readable
JSON diagnostics that quantify the problem and suggest concrete remedies.

Usage:
    python analytics/analyze_stationarity.py --years 3 --prefix m15
    python analytics/analyze_stationarity.py --years 3 --model long_slow
    python analytics/analyze_stationarity.py --feature m15_rsi daily_adx --no-plot
    python analytics/analyze_stationarity.py --prefix 4hours --helpers --significance 0.01
"""

import json
import os
import sys
import warnings
import argparse
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    from arch.unitroot import VarianceRatio
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False

try:
    from scipy.stats import gaussian_kde, skew as sp_skew, kurtosis as sp_kurtosis
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config
import ModelTrading.source.python.utils.csv as csv_utils
import ModelTrading.source.python.features.config as feature_config
from ModelTrading.source.python.features import indicators as features

_TIMEFRAME_FILE_MAP = {
    'm15': 'eurusd_m15.csv',
    '4hours': 'eurusd_4hours.csv',
    'daily': 'eurusd_daily.csv',
}
_PREFIX_MAP = {
    'm15': 'm15_',
    '4hours': '4hours_',
    'daily': 'daily_',
}

VERDICT_STATIONARY = 'Stationary'
VERDICT_NON_STATIONARY = 'Non-Stationary'
VERDICT_INCONCLUSIVE = 'Inconclusive'

_PROBLEM_VERDICTS = {VERDICT_NON_STATIONARY, VERDICT_INCONCLUSIVE}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_output = os.path.join(dir_config.GENERATED_DIR, 'stationarity')
    p = argparse.ArgumentParser(
        description='Feature stationarity analysis: ADF, KPSS, Ljung-Box, rolling CV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('name', nargs='?', default=None,
                   help='Single feature name to analyse (e.g. daily_volatility_percentile). '
                        'Shorthand for --feature NAME.')

    date_grp = p.add_argument_group('Date range')
    date_grp.add_argument('--start-date', type=str, default=None,
                          help='Explicit start date YYYY-MM-DD (overrides --years)')
    date_grp.add_argument('--end-date', type=str, default=None,
                          help='End date YYYY-MM-DD (default: today)')
    date_grp.add_argument('--years', type=int, default=3,
                          help='Number of years to analyse (used when --start-date is omitted)')

    feat_grp = p.add_argument_group('Feature filtering')
    feat_grp.add_argument('--prefix', choices=['m15', '4hours', 'daily'], default=None,
                          help='Restrict analysis to one timeframe')
    feat_grp.add_argument('--model',
                          choices=['long_fast', 'short_fast', 'long_slow', 'short_slow', 'reg'],
                          default=None,
                          help='Restrict to features assigned to this model')
    feat_grp.add_argument('--feature', nargs='+', default=None,
                          help='Explicit feature names, e.g. m15_rsi daily_adx')
    feat_grp.add_argument('--helpers', action='store_true',
                          help='Include role=helper features (default: model-input only)')
    feat_grp.add_argument('--features-config', type=str, default=None,
                          help='Custom features.yaml path')

    out_grp = p.add_argument_group('Output')
    out_grp.add_argument('--output-dir', type=str, default=default_output,
                         help='Directory for plots and summary CSV')
    out_grp.add_argument('--no-plot', action='store_true',
                         help='Skip PNG generation; only run tests and print summary')
    out_grp.add_argument('--significance', type=float, default=0.05,
                         help='Alpha level for ADF and KPSS tests')
    out_grp.add_argument('--rolling-window', type=int, default=500,
                         help='Bars for rolling mean/std in time-series plot')

    return p.parse_args()


def resolve_date_range(args) -> tuple:
    """Return (start_date_str, end_date_str) as YYYY-MM-DD strings."""
    end_str = args.end_date if args.end_date else date.today().isoformat()
    if args.start_date:
        return args.start_date, end_str
    end_ts = pd.Timestamp(end_str)
    start_ts = end_ts - pd.DateOffset(years=args.years)
    return start_ts.strftime('%Y-%m-%d'), end_str


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(args, fc) -> list:
    """Return ordered list of feature column names matching the CLI filters."""
    prefix_str = _PREFIX_MAP.get(args.prefix) if args.prefix else None

    names = list(fc.get_usedInModel_features(prefix=prefix_str))
    if args.helpers:
        helper_names = fc.get_helper_features(prefix=prefix_str)
        for n in helper_names:
            if n not in names:
                names.append(n)

    if args.model is not None:
        model_set = set(fc.get_model_features(args.model, prefix=prefix_str))
        names = [n for n in names if n in model_set]

    if args.feature is not None:
        requested = set(args.feature)
        all_known = set(fc.get_enabled_features())
        if args.helpers:
            all_known |= set(fc.get_helper_features())
        unknown = requested - all_known
        if unknown:
            print(f"Warning: unknown feature name(s) will be skipped: {sorted(unknown)}")
        names = [n for n in names if n in requested] + \
                [n for n in args.feature if n in all_known and n not in names]

    return names


def _infer_timeframes(feature_names: list) -> set:
    tfs = set()
    for name in feature_names:
        for tf_token, prefix in _PREFIX_MAP.items():
            if name.startswith(prefix):
                tfs.add(tf_token)
                break
    return tfs or set(_TIMEFRAME_FILE_MAP.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_single_timeframe(tf_token: str, start_date: str, end_date: str) -> pd.DataFrame:
    path = os.path.join(dir_config.DATA_DIR, _TIMEFRAME_FILE_MAP[tf_token])
    df = csv_utils.load_csv(path, filter_weekends_flag=True,
                            start_date=start_date, end_date=end_date)
    feat_df = features.add_features(df, tf_token, apply_shift=False)
    return feat_df.add_prefix(_PREFIX_MAP[tf_token])


def load_feature_data(timeframes: set, start_date: str, end_date: str) -> dict:
    """Return {tf_token: prefixed_feature_df} for the requested timeframes."""
    result = {}
    for tf in timeframes:
        print(f"  Loading {tf} data ({start_date} → {end_date})…")
        result[tf] = _load_single_timeframe(tf, start_date, end_date)
    return result


# ---------------------------------------------------------------------------
# Statistical test functions  (pure, no I/O)
# ---------------------------------------------------------------------------

def run_adf_test(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test.
    H0: unit root present (non-stationary). Rejecting H0 is evidence of stationarity.
    """
    base = {'adf_stat': None, 'p_value': None, 'n_lags': None,
            'n_obs': len(series), 'rejects_h0': False, 'error': None}
    if len(series) < 20:
        base['error'] = 'insufficient_data'
        return base
    if not _STATSMODELS_AVAILABLE:
        base['error'] = 'statsmodels_not_available'
        return base
    try:
        stat, p, n_lags, n_obs, *_ = adfuller(series.values, autolag='AIC')
        base.update({'adf_stat': float(stat), 'p_value': float(p),
                     'n_lags': int(n_lags), 'n_obs': int(n_obs),
                     'rejects_h0': p < significance})
    except (ValueError, np.linalg.LinAlgError) as exc:
        base['error'] = str(exc)
    return base


def run_kpss_test(series: pd.Series, significance: float = 0.05) -> dict:
    """
    KPSS test.
    H0: series is level-stationary. Rejecting H0 is evidence of non-stationarity.
    Note: p-values are interpolated from a lookup table capped at [0.01, 0.10].
    """
    base = {'kpss_stat': None, 'p_value': None, 'n_lags': None,
            'rejects_h0': False, 'error': None}
    if len(series) < 20:
        base['error'] = 'insufficient_data'
        return base
    if not _STATSMODELS_AVAILABLE:
        base['error'] = 'statsmodels_not_available'
        return base
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stat, p, n_lags, _ = kpss(series.values, regression='c', nlags='auto')
        base.update({'kpss_stat': float(stat), 'p_value': float(p),
                     'n_lags': int(n_lags), 'rejects_h0': p < significance})
    except (ValueError, np.linalg.LinAlgError) as exc:
        base['error'] = str(exc)
    return base


def run_ljungbox_test(series: pd.Series, lag: int = 10,
                      significance: float = 0.05) -> dict:
    """
    Ljung-Box test for autocorrelation up to `lag`.
    H0: no autocorrelation. Rejecting H0 means autocorrelation is present.
    """
    base = {'lb_stat': None, 'p_value': None, 'rejects_h0': False, 'error': None}
    if len(series) < lag + 2:
        base['error'] = 'insufficient_data'
        return base
    if not _STATSMODELS_AVAILABLE:
        base['error'] = 'statsmodels_not_available'
        return base
    try:
        result = acorr_ljungbox(series.values, lags=list(range(1, lag + 1)),
                                return_df=True)
        # Use the joint statistic at the maximum lag (standard interpretation).
        # Taking min() across lags is too aggressive: 10 tests at α=0.05 gives
        # a ~40% false-positive rate for white noise.
        p_at_lag = float(result['lb_pvalue'].iloc[-1])
        last_stat = float(result['lb_stat'].iloc[-1])
        base.update({'lb_stat': last_stat, 'p_value': p_at_lag,
                     'rejects_h0': p_at_lag < significance})
    except (ValueError, Exception) as exc:
        base['error'] = str(exc)
    return base


def run_variance_ratio_test(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Variance Ratio test (random walk test). Requires the arch package (optional).
    H0: series follows a random walk.
    """
    base = {'vr_stat': None, 'p_value': None, 'rejects_h0': None, 'error': None}
    if not _ARCH_AVAILABLE:
        base['error'] = 'arch_not_available'
        return base
    if len(series) < 20:
        base['error'] = 'insufficient_data'
        return base
    try:
        vr = VarianceRatio(series.values)
        base.update({'vr_stat': float(vr.stat), 'p_value': float(vr.pvalue),
                     'rejects_h0': float(vr.pvalue) < significance})
    except Exception as exc:
        base['error'] = str(exc)
    return base


def compute_rolling_variance_stability(series: pd.Series, n_quarters: int = 4) -> dict:
    """
    Split the series into n_quarters equal windows; compute std per window.
    CV (coefficient of variation of window stds) > 0.5 suggests unstable variance.
    """
    windows = np.array_split(series.values, n_quarters)
    stds = [float(np.std(w)) for w in windows if len(w) > 1]
    if not stds or np.mean(stds) == 0:
        return {'window_stds': stds, 'cv': 0.0, 'is_stable': True}
    cv = float(np.std(stds) / np.mean(stds))
    return {'window_stds': stds, 'cv': cv, 'is_stable': cv < 0.5}


def determine_verdict(adf_result: dict, kpss_result: dict, significance: float) -> str:
    """
    Combine ADF and KPSS results into a three-way verdict.

    Stationary:     ADF rejects H0 (p < sig)  AND  KPSS fails to reject (p > sig)
    Non-Stationary: ADF fails to reject        AND  KPSS rejects
    Inconclusive:   disagreement, both agree, or any error
    """
    if adf_result.get('error') or kpss_result.get('error'):
        return VERDICT_INCONCLUSIVE
    adf_rejects = adf_result['rejects_h0']
    kpss_rejects = kpss_result['rejects_h0']
    if adf_rejects and not kpss_rejects:
        return VERDICT_STATIONARY
    if not adf_rejects and kpss_rejects:
        return VERDICT_NON_STATIONARY
    return VERDICT_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Per-feature analysis
# ---------------------------------------------------------------------------

def analyze_feature(name: str, series: pd.Series,
                    significance: float = 0.05,
                    rolling_window: int = 500) -> dict:
    """Run all tests on one feature series. Returns a result dict."""
    n_obs = len(series)
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    n_valid = len(clean)

    result = {
        'feature': name,
        'n_obs': n_obs,
        'n_valid': n_valid,
        'adf': {},
        'kpss': {},
        'ljungbox': {},
        'variance_ratio': {},
        'rolling_cv': {},
        'verdict': VERDICT_INCONCLUSIVE,
    }

    if n_valid < 20:
        result['adf'] = {'error': 'insufficient_data', 'rejects_h0': False, 'p_value': None}
        result['kpss'] = {'error': 'insufficient_data', 'rejects_h0': False, 'p_value': None}
        result['ljungbox'] = {'error': 'insufficient_data', 'rejects_h0': False, 'p_value': None}
        result['variance_ratio'] = {'error': 'insufficient_data', 'rejects_h0': None, 'p_value': None}
        result['rolling_cv'] = {'cv': None, 'is_stable': None}
        return result

    result['adf'] = run_adf_test(clean, significance)
    result['kpss'] = run_kpss_test(clean, significance)
    result['ljungbox'] = run_ljungbox_test(clean, significance=significance)
    result['variance_ratio'] = run_variance_ratio_test(clean, significance)
    result['rolling_cv'] = compute_rolling_variance_stability(clean)
    result['verdict'] = determine_verdict(result['adf'], result['kpss'], significance)
    return result


# ---------------------------------------------------------------------------
# Machine-readable problem diagnostics
# ---------------------------------------------------------------------------

def _round(v, n=6):
    """Round a float for JSON serialisation; return None if not a number."""
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _compute_acf_values(clean: pd.Series, lags=(1, 2, 5, 10, 20)) -> dict:
    """Return ACF at requested lags. Keys are 'lag_N'."""
    if not _STATSMODELS_AVAILABLE or len(clean) < max(lags) + 2:
        return {f'lag_{l}': None for l in lags}
    try:
        max_lag = max(lags)
        acf_vals = sm_acf(clean.values, nlags=max_lag, fft=True)
        return {f'lag_{l}': _round(acf_vals[l]) for l in lags}
    except Exception:
        return {f'lag_{l}': None for l in lags}


def _compute_rolling_drift(series: pd.Series, rolling_window: int) -> dict:
    """Quantify mean drift: how much the rolling mean moves relative to overall std."""
    min_periods = max(1, rolling_window // 4)
    roll_mean = series.rolling(rolling_window, min_periods=min_periods).mean().dropna()
    roll_std = series.rolling(rolling_window, min_periods=min_periods).std().dropna()
    overall_std = float(series.std()) if series.std() != 0 else 1.0
    mean_range = float(roll_mean.max() - roll_mean.min())
    return {
        'rolling_mean_first': _round(roll_mean.iloc[0]) if len(roll_mean) else None,
        'rolling_mean_last': _round(roll_mean.iloc[-1]) if len(roll_mean) else None,
        'rolling_mean_min': _round(roll_mean.min()) if len(roll_mean) else None,
        'rolling_mean_max': _round(roll_mean.max()) if len(roll_mean) else None,
        'rolling_mean_range': _round(mean_range),
        'rolling_mean_range_pct_of_std': _round(mean_range / overall_std * 100),
        'rolling_std_mean': _round(float(roll_std.mean())) if len(roll_std) else None,
        'rolling_std_min': _round(float(roll_std.min())) if len(roll_std) else None,
        'rolling_std_max': _round(float(roll_std.max())) if len(roll_std) else None,
    }


def _determine_primary_issue(result: dict) -> tuple:
    """
    Return (primary_issue, secondary_issues, inconclusive_subtype).

    primary_issue values:
      'unit_root'              – ADF fails, KPSS rejects (classic random walk / trend)
      'trend_stationary_break' – ADF rejects AND KPSS rejects (structural break / over-differenced)
      'low_signal_noise'       – both fail to reject (noisy, near-constant, or too short)
      'error'                  – test produced an error
    """
    adf = result['adf']
    kpss_r = result['kpss']
    lb = result['ljungbox']
    rv = result['rolling_cv']

    if adf.get('error') or kpss_r.get('error'):
        return 'error', [], None

    adf_rejects = bool(adf.get('rejects_h0', False))
    kpss_rejects = bool(kpss_r.get('rejects_h0', False))

    if not adf_rejects and kpss_rejects:
        primary = 'unit_root'
        subtype = None
    elif adf_rejects and kpss_rejects:
        primary = 'trend_stationary_break'
        subtype = 'both_reject'
    elif not adf_rejects and not kpss_rejects:
        primary = 'low_signal_noise'
        subtype = 'both_fail'
    else:
        primary = 'error'
        subtype = None

    secondary = []
    if lb.get('rejects_h0'):
        secondary.append('autocorrelation')
    if rv.get('cv') is not None and rv['cv'] > 0.5:
        secondary.append('variance_instability')
    if rv.get('cv') is not None and rv['cv'] > 1.5:
        secondary.append('severe_variance_instability')

    return primary, secondary, subtype


_REMEDIES = {
    'unit_root': [
        {
            'id': 'first_difference',
            'priority': 'high',
            'description': (
                'Compute bar-over-bar change instead of the raw level '
                '(e.g. delta_sma_200 = sma_200[t] - sma_200[t-1]). '
                'Eliminates a trending mean and is the standard cure for a unit root.'
            ),
        },
        {
            'id': 'percent_change',
            'priority': 'high',
            'description': (
                'Use relative change: (value[t] - value[t-1]) / value[t-1]. '
                'Removes both level drift and scale effects.'
            ),
        },
        {
            'id': 'normalize_by_atr',
            'priority': 'medium',
            'description': (
                'Divide by the current ATR to remove price-scale drift across '
                'volatility regimes. Effective for price-unit features such as '
                'distances to cloud, SMA, or Bollinger bands.'
            ),
        },
        {
            'id': 'rolling_percentile_rank',
            'priority': 'medium',
            'description': (
                'Replace with the rolling percentile rank over a lookback window '
                '(e.g. 252 bars). Bounds the feature to [0, 1] regardless of absolute '
                'level and is robust to slow structural drifts.'
            ),
        },
    ],
    'trend_stationary_break': [
        {
            'id': 'regime_segment',
            'priority': 'high',
            'description': (
                'ADF rejects but KPSS also rejects, suggesting a structural break rather '
                'than a pure unit root. Train separate models per market regime '
                '(e.g. bull/bear, pre/post-break) so each model sees a locally stationary series.'
            ),
        },
        {
            'id': 'structural_break_detection',
            'priority': 'high',
            'description': (
                'Run a Chow test or CUSUM analysis to identify the break date. '
                'Trim training data to start after the break, or add a binary regime indicator.'
            ),
        },
        {
            'id': 'detrend',
            'priority': 'medium',
            'description': (
                'Apply linear or HP-filter detrending before using the feature. '
                'Use the residual from the trend rather than the raw value.'
            ),
        },
    ],
    'low_signal_noise': [
        {
            'id': 'extend_date_range',
            'priority': 'high',
            'description': (
                'Both ADF and KPSS fail to reject, which often indicates insufficient data '
                'or near-constant output. Extend --years or widen the date range to get '
                'a larger sample before drawing conclusions.'
            ),
        },
        {
            'id': 'verify_indicator_output',
            'priority': 'high',
            'description': (
                'Inspect the raw feature values: a near-constant or all-NaN series '
                'produces this pattern. Check the indicator computation and data coverage.'
            ),
        },
        {
            'id': 'consider_removal',
            'priority': 'medium',
            'description': (
                'Features with inconclusive stationarity and low variation add noise '
                'rather than signal. Consider dropping from the feature set and re-running '
                'MI/PFI selection.'
            ),
        },
    ],
    'autocorrelation': [
        {
            'id': 'difference_to_remove_memory',
            'priority': 'medium',
            'description': (
                'High Ljung-Box autocorrelation means the feature has strong memory. '
                'Differencing (first or seasonal) or using the rate-of-change variant '
                'typically removes serial dependence.'
            ),
        },
        {
            'id': 'check_lookback_period',
            'priority': 'low',
            'description': (
                'Long-lookback indicators (SMA-200, Ichimoku) are inherently autocorrelated '
                'by construction. This is expected and may not require action if the model '
                'already discounts it via regularisation.'
            ),
        },
    ],
    'variance_instability': [
        {
            'id': 'normalize_by_atr',
            'priority': 'high',
            'description': (
                'Rolling variance CV > 0.5 means volatility of this feature changes over time. '
                'Normalising by ATR or using a volatility-scaled version stabilises the distribution '
                'across regimes and prevents the model from overweighting low-vol periods.'
            ),
        },
        {
            'id': 'rolling_zscore',
            'priority': 'medium',
            'description': (
                'Apply a rolling Z-score: subtract the rolling mean and divide by the rolling '
                'std (e.g. 252-bar window). This adaptively re-centres and re-scales the feature.'
            ),
        },
    ],
}


def build_problem_diagnostic(name: str, series: pd.Series, result: dict,
                              significance: float, rolling_window: int = 500) -> dict:
    """
    Build a machine-readable diagnostic dict for a Non-Stationary or Inconclusive feature.

    Includes all numerical statistics visible in the four plot panels, plus
    suggested remedies ranked by priority.
    """
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()

    primary_issue, secondary_issues, subtype = _determine_primary_issue(result)

    # Distribution statistics
    dist = {}
    if len(clean) > 0:
        vals = clean.values.astype(float)
        dist = {
            'mean': _round(float(np.mean(vals))),
            'std': _round(float(np.std(vals))),
            'min': _round(float(np.min(vals))),
            'p5': _round(float(np.percentile(vals, 5))),
            'p25': _round(float(np.percentile(vals, 25))),
            'p50': _round(float(np.percentile(vals, 50))),
            'p75': _round(float(np.percentile(vals, 75))),
            'p95': _round(float(np.percentile(vals, 95))),
            'max': _round(float(np.max(vals))),
        }
        if _SCIPY_AVAILABLE:
            dist['skewness'] = _round(float(sp_skew(vals)))
            dist['excess_kurtosis'] = _round(float(sp_kurtosis(vals)))

    # Rolling drift (from time-series panel)
    drift = _compute_rolling_drift(clean, rolling_window)

    # ACF at key lags (from ACF panel)
    acf_values = _compute_acf_values(clean)

    # Rolling std CV (from rolling-std panel)
    rv = result['rolling_cv']
    variance_info = {
        'window_stds': [_round(v) for v in (rv.get('window_stds') or [])],
        'cv': _round(rv.get('cv')),
        'is_stable': rv.get('is_stable'),
    }

    # Collect relevant remedies
    remedy_ids = set()
    all_remedies = []

    def _add_remedies(key):
        for r in _REMEDIES.get(key, []):
            if r['id'] not in remedy_ids:
                remedy_ids.add(r['id'])
                all_remedies.append(r)

    _add_remedies(primary_issue)
    for issue in secondary_issues:
        _add_remedies(issue)

    # Sort: high > medium > low
    _priority_order = {'high': 0, 'medium': 1, 'low': 2}
    all_remedies.sort(key=lambda r: _priority_order.get(r['priority'], 9))

    # Severity flags
    mean_drift_pct = drift.get('rolling_mean_range_pct_of_std') or 0
    acf_lag1 = acf_values.get('lag_1') or 0
    severity_flags = {
        'large_mean_drift': mean_drift_pct > 50,
        'extreme_mean_drift': mean_drift_pct > 200,
        'high_autocorrelation': abs(acf_lag1) > 0.5,
        'very_high_autocorrelation': abs(acf_lag1) > 0.9,
        'unstable_variance': (rv.get('cv') or 0) > 0.5,
        'severe_variance_instability': (rv.get('cv') or 0) > 1.5,
        'extreme_skewness': abs(dist.get('skewness') or 0) > 2.0,
    }

    return {
        'feature': name,
        'verdict': result['verdict'],
        'n_obs': result['n_obs'],
        'n_valid': result['n_valid'],
        'significance_level': significance,
        'diagnosis': {
            'primary_issue': primary_issue,
            'secondary_issues': secondary_issues,
            'inconclusive_subtype': subtype,
            'severity_flags': severity_flags,
        },
        'test_results': {
            'adf': {
                'stat': _round(result['adf'].get('adf_stat')),
                'p_value': _round(result['adf'].get('p_value')),
                'n_lags_used': result['adf'].get('n_lags'),
                'rejects_unit_root_h0': bool(result['adf'].get('rejects_h0', False)),
                'interpretation': (
                    'Evidence of stationarity (rejects unit root)'
                    if result['adf'].get('rejects_h0')
                    else 'Fails to reject unit root — evidence of non-stationarity'
                ),
            },
            'kpss': {
                'stat': _round(result['kpss'].get('kpss_stat')),
                'p_value': _round(result['kpss'].get('p_value')),
                'rejects_stationarity_h0': bool(result['kpss'].get('rejects_h0', False)),
                'interpretation': (
                    'Rejects level-stationarity — confirms non-stationarity'
                    if result['kpss'].get('rejects_h0')
                    else 'Fails to reject stationarity hypothesis'
                ),
            },
            'ljungbox_lag10': {
                'stat': _round(result['ljungbox'].get('lb_stat')),
                'p_value': _round(result['ljungbox'].get('p_value')),
                'rejects_no_autocorrelation_h0': bool(result['ljungbox'].get('rejects_h0', False)),
                'interpretation': (
                    'Significant autocorrelation detected at lag 10'
                    if result['ljungbox'].get('rejects_h0')
                    else 'No significant autocorrelation at lag 10'
                ),
            },
        },
        'distribution': dist,
        'drift': drift,
        'autocorrelation': acf_values,
        'variance_stability': variance_info,
        'suggested_remedies': all_remedies,
    }


def save_problem_diagnostics(problem_results: list, output_dir: str) -> str:
    """
    Save individual JSON per problematic feature and one combined problems file.
    Returns path to the combined file.
    """
    for diag in problem_results:
        fname = diag['feature'].replace('/', '_') + '_diagnostic.json'
        with open(os.path.join(output_dir, fname), 'w') as f:
            json.dump(diag, f, indent=2)

    combined_path = os.path.join(output_dir, 'problems_diagnostic.json')
    payload = {
        'summary': {
            'total_problems': len(problem_results),
            'non_stationary': sum(1 for d in problem_results
                                  if d['verdict'] == VERDICT_NON_STATIONARY),
            'inconclusive': sum(1 for d in problem_results
                                if d['verdict'] == VERDICT_INCONCLUSIVE),
        },
        'features': problem_results,
    }
    with open(combined_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return combined_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {
    VERDICT_STATIONARY: 'green',
    VERDICT_NON_STATIONARY: 'red',
    VERDICT_INCONCLUSIVE: 'orange',
}


def plot_feature(name: str, series: pd.Series, result: dict,
                 output_dir: str, rolling_window: int = 500) -> str:
    """Create and save a 2×2 diagnostic figure for one feature. Returns saved path."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    verdict = result['verdict']
    adf_p = result['adf'].get('p_value')
    kpss_p = result['kpss'].get('p_value')

    adf_str = f'{adf_p:.4f}' if adf_p is not None else 'n/a'
    kpss_str = f'{kpss_p:.4f}' if kpss_p is not None else 'n/a'
    title = f'{name}  —  {verdict}  (ADF p={adf_str}, KPSS p={kpss_str})'

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=12, fontweight='bold',
                 color=_VERDICT_COLORS.get(verdict, 'black'))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # -- Subplot 1: time series with rolling stats --
    ax1 = fig.add_subplot(gs[0, 0])
    min_periods = max(1, rolling_window // 4)
    roll_mean = series.rolling(rolling_window, min_periods=min_periods).mean()
    roll_std = series.rolling(rolling_window, min_periods=min_periods).std()
    ax1.plot(series.index, series.values, linewidth=0.5, color='steelblue', alpha=0.7,
             label='Feature')
    ax1.plot(roll_mean.index, roll_mean.values, linewidth=1.5, color='darkorange',
             label=f'Rolling mean (w={rolling_window})')
    ax1.fill_between(roll_mean.index,
                     (roll_mean - 2 * roll_std).values,
                     (roll_mean + 2 * roll_std).values,
                     alpha=0.2, color='darkorange')
    ax1.set_title('Time Series  (rolling mean ± 2σ)')
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=30, labelsize=7)

    # -- Subplot 2: ACF --
    ax2 = fig.add_subplot(gs[0, 1])
    if _STATSMODELS_AVAILABLE and len(clean) > 42:
        plot_acf(clean.values, ax=ax2, lags=40, alpha=0.05, zero=False)
        ax2.set_title('Autocorrelation (ACF, lags=40)')
    else:
        ax2.text(0.5, 0.5, 'statsmodels not available\nor insufficient data',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('ACF')

    # -- Subplot 3: histogram + KDE --
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(clean.values, bins=50, density=True, alpha=0.6, color='steelblue',
             label='Histogram')
    if _SCIPY_AVAILABLE and len(clean) > 5:
        try:
            kde = gaussian_kde(clean.values)
            x_grid = np.linspace(float(clean.min()), float(clean.max()), 300)
            ax3.plot(x_grid, kde(x_grid), color='red', linewidth=2, label='KDE')
        except Exception:
            pass
    ax3.set_title('Distribution')
    ax3.legend(fontsize=8)

    # -- Subplot 4: rolling std --
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(roll_std.index, roll_std.values, linewidth=0.8, color='darkorange')
    ax4.set_title('Rolling Std  (variance non-stationarity)')
    ax4.tick_params(axis='x', rotation=30, labelsize=7)

    fname = name.replace('/', '_') + '.png'
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

_ANSI = {
    VERDICT_STATIONARY: '\033[32m',
    VERDICT_NON_STATIONARY: '\033[31m',
    VERDICT_INCONCLUSIVE: '\033[33m',
    'reset': '\033[0m',
}


def _fmt_p(val) -> str:
    if val is None:
        return '  n/a  '
    return f'{val:.4f}'


def print_summary_table(results: list, significance: float) -> None:
    use_color = sys.stdout.isatty()
    header = (
        f"{'Feature':<35} {'N':>7} {'ADF p':>8} {'KPSS p':>8} "
        f"{'LB p':>8} {'CV':>6}  Verdict"
    )
    print()
    print(header)
    print('-' * len(header))
    for r in results:
        verdict = r['verdict']
        adf_p = _fmt_p(r['adf'].get('p_value'))
        kpss_p = _fmt_p(r['kpss'].get('p_value'))
        lb_p = _fmt_p(r['ljungbox'].get('p_value'))
        cv = r['rolling_cv'].get('cv')
        cv_str = f'{cv:.3f}' if cv is not None else '  n/a'
        color = _ANSI.get(verdict, '') if use_color else ''
        reset = _ANSI['reset'] if use_color else ''
        print(
            f"{r['feature']:<35} {r['n_valid']:>7} {adf_p:>8} {kpss_p:>8} "
            f"{lb_p:>8} {cv_str:>6}  {color}{verdict}{reset}"
        )
    print()


def save_summary_csv(results: list, output_dir: str) -> str:
    rows = []
    for r in results:
        rows.append({
            'feature': r['feature'],
            'n_obs': r['n_obs'],
            'n_valid': r['n_valid'],
            'adf_stat': r['adf'].get('adf_stat'),
            'adf_p_value': r['adf'].get('p_value'),
            'adf_rejects_h0': r['adf'].get('rejects_h0'),
            'kpss_stat': r['kpss'].get('kpss_stat'),
            'kpss_p_value': r['kpss'].get('p_value'),
            'kpss_rejects_h0': r['kpss'].get('rejects_h0'),
            'lb_p_value': r['ljungbox'].get('p_value'),
            'lb_rejects_h0': r['ljungbox'].get('rejects_h0'),
            'rolling_cv': r['rolling_cv'].get('cv'),
            'variance_stable': r['rolling_cv'].get('is_stable'),
            'vr_p_value': r['variance_ratio'].get('p_value'),
            'verdict': r['verdict'],
        })
    new_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'stationarity_summary.csv')
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        # Drop rows for features that were just re-analysed, then append fresh rows.
        existing = existing[~existing['feature'].isin(new_df['feature'])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.sort_values('feature', inplace=True)
    combined.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not _STATSMODELS_AVAILABLE:
        raise RuntimeError(
            "statsmodels is required. Install with:\n"
            "  pip install 'statsmodels>=0.14.0'"
        )

    if args.name is not None:
        args.feature = [args.name]

    start_date, end_date = resolve_date_range(args)
    print(f"\nStationarity analysis: {start_date} → {end_date}")

    if args.features_config:
        feature_config.set_feature_config(args.features_config)

    fc = feature_config.get_feature_config()
    feature_names = select_features(args, fc)

    if not feature_names:
        filters = []
        if args.prefix:
            filters.append(f'--prefix {args.prefix}')
        if args.model:
            filters.append(f'--model {args.model}')
        if args.feature:
            filters.append(f'--feature {args.feature}')
        print(
            f"ERROR: no features selected after applying filters: {filters}\n"
            f"  Available enabled model features: {fc.get_usedInModel_features()[:10]}…"
        )
        sys.exit(1)

    print(f"Features to analyse: {len(feature_names)}")

    needed_tfs = _infer_timeframes(feature_names)
    tf_data = load_feature_data(needed_tfs, start_date, end_date)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    problem_diagnostics = []
    total = len(feature_names)

    for idx, name in enumerate(feature_names, 1):
        tf_token = next(
            (tf for tf, prefix in _PREFIX_MAP.items() if name.startswith(prefix)),
            None,
        )
        if tf_token is None or tf_token not in tf_data:
            print(f"[{idx:3d}/{total}] {name:<40} … SKIPPED (no data)")
            continue

        series = tf_data[tf_token].get(name)
        if series is None:
            print(f"[{idx:3d}/{total}] {name:<40} … SKIPPED (column not found)")
            continue

        result = analyze_feature(name, series, args.significance, args.rolling_window)
        results.append(result)

        verdict = result['verdict']
        print(f"[{idx:3d}/{total}] {name:<40} … {verdict}")

        if not args.no_plot:
            plot_feature(name, series, result, args.output_dir, args.rolling_window)

        if verdict in _PROBLEM_VERDICTS:
            diag = build_problem_diagnostic(
                name, series, result, args.significance, args.rolling_window
            )
            problem_diagnostics.append(diag)

    if not results:
        print("No results produced.")
        return

    print_summary_table(results, args.significance)

    csv_path = save_summary_csv(results, args.output_dir)
    print(f"Summary CSV        : {csv_path}")

    if problem_diagnostics:
        combined_path = save_problem_diagnostics(problem_diagnostics, args.output_dir)
        print(f"Problem diagnostics: {combined_path}  ({len(problem_diagnostics)} features)")
        print(f"  Individual JSONs : {args.output_dir}/*_diagnostic.json")

    if not args.no_plot:
        print(f"Plot PNGs          : {args.output_dir}/")

    n_stat = sum(1 for r in results if r['verdict'] == VERDICT_STATIONARY)
    n_nonstat = sum(1 for r in results if r['verdict'] == VERDICT_NON_STATIONARY)
    n_inc = sum(1 for r in results if r['verdict'] == VERDICT_INCONCLUSIVE)
    print(f"\nVerdicts: {n_stat} Stationary | {n_nonstat} Non-Stationary | {n_inc} Inconclusive\n")


if __name__ == '__main__':
    main()
