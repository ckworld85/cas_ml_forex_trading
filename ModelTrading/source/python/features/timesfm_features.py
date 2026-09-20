"""
TimesFM probabilistic forecast features for forex trading.

Computes rolling forecasts of cumulative log-returns using Google's TimesFM
neural time-series model. Produces stationary features (mean, q10, q90, spread)
aligned to the training DataFrame index. The caller (indicators.py) applies
shift(1) to prevent lookahead bias.

The forecast horizon is NOT a configuration parameter — it is encoded in each
feature-name suffix (e.g. ``tfm_mean_144``, ``tfm_q90_9``, ``tfm_conf_2``).
This lets a single timeframe expose several horizons side by side.

Two consumption paths share the rolling forecast in this module:

  * Training (``advanced_train`` / ``train``): TimesFM features are computed
    once upfront by ``data/update_timesfm_data.py`` and persisted to
    ``data/timesfm_<timeframe>.csv``. ``get_timesfm_features(..., compute=False)``
    loads those precomputed columns.
  * Live inference (``feature_server``): the precomputed CSV does not cover the
    live bars, so ``get_timesfm_features(..., compute=True)`` recomputes the
    forecast on the fly from the bars Java sends.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_timesfm_model = None      # loaded weights — created once per process (expensive)
_timesfm_compiled = None   # (context_len, max_horizon, batch_size) currently compiled

# Stat name -> column produced by compute_timesfm_features().
TFM_STATS = ("mean", "q10", "q90", "spread", "conf")

# A de-prefixed tfm feature name: tfm_<stat>_<horizon>, e.g. "tfm_mean_144".
_TFM_NAME_RE = re.compile(r"^tfm_(mean|q10|q90|spread|conf)_(\d+)$")


def parse_tfm_feature(name: str):
    """Parse a de-prefixed tfm feature name ``tfm_<stat>_<horizon>``.

    Returns ``(stat, horizon)`` (e.g. ``("mean", 144)``) or ``None`` if the
    name is not a horizon-suffixed TimesFM feature.
    """
    m = _TFM_NAME_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def is_tfm_feature(name: str) -> bool:
    """True if ``name`` (de-prefixed) is a horizon-suffixed TimesFM feature."""
    return _TFM_NAME_RE.match(name) is not None


def _data_dir() -> Path:
    """Return the absolute path to ModelTrading/data/ (same as external_data)."""
    try:
        import ModelTrading.config.directories as _dir
        return Path(_dir.DATA_DIR)
    except ImportError:
        # Fallback: 4 levels up from this file → project root / ModelTrading / data
        return Path(__file__).resolve().parents[4] / "ModelTrading" / "data"


def timesfm_csv_path(timeframe: str) -> Path:
    """Path to the precomputed TimesFM CSV for a timeframe (m15/4hours/daily)."""
    return _data_dir() / f"timesfm_{timeframe}.csv"


def _group_horizons(feature_names) -> dict:
    """Group de-prefixed tfm feature names into ``{horizon: {stat, ...}}``."""
    horizons: dict[int, set] = {}
    for name in feature_names:
        parsed = parse_tfm_feature(name)
        if parsed is None:
            continue
        stat, horizon = parsed
        horizons.setdefault(horizon, set()).add(stat)
    return horizons


def _get_timesfm_model(context_len: int, horizon_len: int, model_repo: str, batch_size: int = 256):
    """
    Return a TimesFM 2.5 model compiled for at least `horizon_len` steps.

    The 2.5 API (timesfm>=2.0) replaced the 1.x ``TimesFm`` /
    ``TimesFmHparams`` / ``TimesFmCheckpoint`` constructor with
    ``TimesFM_2p5_200M_torch.from_pretrained(...)`` followed by
    ``model.compile(ForecastConfig(...))``.

    Weights are loaded once and cached. ``compile()`` is cheap (no weight
    reload), so we recompile only when the context length, batch size, or a
    larger horizon is required. ``max_horizon`` is kept monotonic so a later
    call with a smaller horizon never shrinks the compiled graph and breaks a
    subsequent larger one.
    """
    global _timesfm_model, _timesfm_compiled
    import timesfm

    if _timesfm_model is None:
        _timesfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_repo)

    need_compile = _timesfm_compiled is None
    if not need_compile:
        c_ctx, c_hz, c_bs = _timesfm_compiled
        need_compile = (c_ctx != context_len) or (c_bs != batch_size) or (horizon_len > c_hz)

    if need_compile:
        max_horizon = horizon_len
        if _timesfm_compiled is not None:
            max_horizon = max(max_horizon, _timesfm_compiled[1])  # never shrink
        _timesfm_model.compile(
            timesfm.ForecastConfig(
                max_context=context_len,
                max_horizon=max_horizon,
                per_core_batch_size=batch_size,
                normalize_inputs=True,
                infer_is_positive=False,   # forecasting log-returns, which go negative
                force_flip_invariance=True,
                fix_quantile_crossing=True,
            )
        )
        _timesfm_compiled = (context_len, max_horizon, batch_size)

    return _timesfm_model


def _build_result(
    index: pd.Index,
    tfm_mean: np.ndarray,
    tfm_q10: np.ndarray,
    tfm_q90: np.ndarray,
    include_conf: bool,
) -> pd.DataFrame:
    spread = (tfm_q90 - tfm_q10).astype(np.float32)
    result = pd.DataFrame(
        {
            "tfm_mean":   tfm_mean.astype(np.float32),
            "tfm_q10":    tfm_q10.astype(np.float32),
            "tfm_q90":    tfm_q90.astype(np.float32),
            "tfm_spread": spread,
        },
        index=index,
    )
    if include_conf:
        result["tfm_conf"] = (
            np.abs(tfm_mean) / (spread + 1e-8)
        ).astype(np.float32)
    return result


def compute_timesfm_features(
    df: pd.DataFrame,
    context_length: int = 512,
    forecast_horizon: int = 144,
    batch_size: int = 256,
    freq: int = 0,
    include_conf: bool = False,
    model_repo: str = "google/timesfm-2.5-200m-pytorch",
) -> pd.DataFrame:
    """
    Vectorised rolling TimesFM forecast over the full price series.

    For each bar i >= context_length, uses log_returns[i-context_length : i]
    as the context window and forecasts the next forecast_horizon steps.
    The cumulative sum of forecasted log-returns is returned as tfm_mean/q10/q90.

    Returns a DataFrame aligned to df.index. First context_length rows are NaN
    (warmup). No shift applied here — indicators.py applies shift(1).

    Parameters
    ----------
    df : OHLC DataFrame with a 'close' column
    context_length : rolling lookback window fed to TimesFM
    forecast_horizon : number of steps to forecast (summed for cumulative return)
    batch_size : number of windows per TimesFM inference call
    freq : ignored — kept for backward compatibility. TimesFM 2.5 has no
        frequency input (the old 1.x ``freq`` hint was removed in the 2.0 API).
    include_conf : if True, also compute tfm_conf = |mean| / (spread + ε)
    model_repo : HuggingFace repo id for the TimesFM 2.5 checkpoint
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)

    tfm_mean = np.full(n, np.nan, dtype=np.float32)
    tfm_q10  = np.full(n, np.nan, dtype=np.float32)
    tfm_q90  = np.full(n, np.nan, dtype=np.float32)

    # log_returns[0] = NaN; log_returns[i] = log(close[i] / close[i-1])
    log_returns = np.empty(n, dtype=np.float64)
    log_returns[0] = np.nan
    log_returns[1:] = np.log(close[1:] / close[:-1])

    # Drop the leading NaN so sliding_window_view gets a contiguous valid series.
    # Window k of size context_length spans lr_valid[k : k+context_length], which
    # corresponds to bars [1+k .. context_length+k] of the original series.
    # The forecast therefore belongs to output bar (context_length + k).
    lr_valid = log_returns[1:]  # length = n - 1
    if len(lr_valid) < context_length:
        return _build_result(df.index, tfm_mean, tfm_q10, tfm_q90, include_conf)

    windows = np.lib.stride_tricks.sliding_window_view(lr_valid, context_length)
    # windows[k] -> output bar index (context_length + k)

    model = _get_timesfm_model(context_length, forecast_horizon, model_repo, batch_size=batch_size)
    n_windows = len(windows)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = windows[start:end]  # (B, context_length)

        point_fc, q_fc = model.forecast(
            horizon=forecast_horizon,
            inputs=[np.ascontiguousarray(row) for row in batch],
        )
        # point_fc : (B, forecast_horizon)         – median (q0.5) forecast per step
        # q_fc     : (B, forecast_horizon, 10)     – col 0=mean, 1..9 = quantiles 0.1 … 0.9

        # Cumulative log-return over the forecast horizon
        cum_mean = point_fc[:, :forecast_horizon].sum(axis=1).astype(np.float32)
        cum_q10  = q_fc[:, :forecast_horizon, 1].sum(axis=1).astype(np.float32)
        cum_q90  = q_fc[:, :forecast_horizon, 9].sum(axis=1).astype(np.float32)

        out_s = context_length + start
        out_e = context_length + end
        tfm_mean[out_s:out_e] = cum_mean
        tfm_q10 [out_s:out_e] = cum_q10
        tfm_q90 [out_s:out_e] = cum_q90

    return _build_result(df.index, tfm_mean, tfm_q10, tfm_q90, include_conf)


def compute_timesfm_features_multi(
    df: pd.DataFrame,
    horizons: dict,
    context_length: int = 512,
    batch_size: int = 256,
    freq: int = 0,
    model_repo: str = "google/timesfm-2.5-200m-pytorch",
) -> pd.DataFrame:
    """
    Compute horizon-suffixed TimesFM features for one or more forecast horizons.

    Parameters
    ----------
    df : OHLC DataFrame with a 'close' column.
    horizons : mapping ``{horizon: {stat, ...}}`` where stat ∈ TFM_STATS.
        Produced by ``_group_horizons`` from the requested feature names.
    context_length, batch_size, freq, model_repo : forwarded to
        ``compute_timesfm_features``.

    Returns a DataFrame aligned to ``df.index`` whose columns are the
    horizon-suffixed names (``tfm_<stat>_<horizon>``). Values are UNSHIFTED —
    the caller applies ``shift(1)`` to prevent lookahead bias.
    """
    out = pd.DataFrame(index=df.index)
    for horizon in sorted(horizons):
        stats = horizons[horizon]
        res = compute_timesfm_features(
            df,
            context_length=context_length,
            forecast_horizon=horizon,
            batch_size=batch_size,
            freq=freq,
            include_conf=("conf" in stats),
            model_repo=model_repo,
        )
        for stat in stats:
            out[f"tfm_{stat}_{horizon}"] = res[f"tfm_{stat}"].values
    return out


def _load_timesfm_features(timeframe: str, horizons: dict, index: pd.Index) -> pd.DataFrame | None:
    """
    Load precomputed horizon-suffixed TimesFM features from
    ``data/timesfm_<timeframe>.csv`` and reindex onto ``index``.

    The CSV is written by ``data/update_timesfm_data.py`` using the same OHLC
    source and weekend filter as training, so bar timestamps align exactly and
    the reindex needs no forward-fill. Values are UNSHIFTED — the caller
    applies ``shift(1)``.

    Returns a DataFrame aligned to ``index`` (missing columns are filled with
    NaN), or ``None`` if the CSV is absent.
    """
    csv_path = timesfm_csv_path(timeframe)
    if not csv_path.exists():
        print(
            f"WARNING [timesfm]: precomputed CSV not found: {csv_path}. "
            "Run data/update_timesfm_data.py to generate it. "
            "TimesFM feature columns will be NaN.",
            file=sys.stderr,
        )
        return None

    raw = pd.read_csv(csv_path)
    raw["date"] = pd.to_datetime(raw["date"], format="ISO8601")
    if raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    raw = raw.set_index("date").sort_index()

    out = pd.DataFrame(index=index)
    for horizon in sorted(horizons):
        for stat in horizons[horizon]:
            col = f"tfm_{stat}_{horizon}"
            if col in raw.columns:
                out[col] = raw[col].reindex(index).astype(np.float32)
            else:
                print(
                    f"WARNING [timesfm]: column '{col}' missing from {csv_path.name}; "
                    "feature will be NaN. Re-run data/update_timesfm_data.py.",
                    file=sys.stderr,
                )
                out[col] = np.float32(np.nan)
    return out


def get_timesfm_features(
    df: pd.DataFrame,
    timeframe: str,
    feature_names,
    params: dict,
    compute: bool,
) -> pd.DataFrame | None:
    """
    Resolve the requested horizon-suffixed TimesFM features for ``timeframe``.

    Parameters
    ----------
    df : OHLC DataFrame (the timeframe's bars).
    timeframe : 'm15' / '4hours' / 'daily' (selects the precomputed CSV).
    feature_names : iterable of de-prefixed tfm feature names (e.g.
        ``"tfm_mean_144"``). Non-tfm names are ignored.
    params : feature-config ``parameters`` block (context length, batch size,
        freq, model repo).
    compute : if True, recompute the forecast live (inference path); if False,
        load the precomputed CSV (training path).

    Returns a DataFrame aligned to ``df.index`` with horizon-suffixed columns,
    or ``None`` when no tfm features were requested or the CSV is missing.
    """
    horizons = _group_horizons(feature_names)
    if not horizons:
        return None

    if compute:
        return compute_timesfm_features_multi(
            df,
            horizons,
            context_length=params.get("tfm_context_length", 512),
            batch_size=params.get("tfm_batch_size", 256),
            freq=params.get(f"tfm_freq_{timeframe}", 0),
            model_repo=params.get("tfm_model_repo", "google/timesfm-2.5-200m-pytorch"),
        )

    return _load_timesfm_features(timeframe, horizons, df.index)
