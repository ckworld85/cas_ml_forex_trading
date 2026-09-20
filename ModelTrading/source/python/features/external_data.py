"""
External (non-OHLC) data loader for the slow model.

Provides free, publicly available macro/sentiment data that complements
OHLC-derived technical indicators for the daily / 4H slow-model scope:

  - COT  : CFTC Commitment of Traders — EUR futures speculative positioning (weekly)
  - VIX  : CBOE Volatility Index — risk-sentiment proxy (daily)
  - Yields: US Treasury 10Y / 2Y from FRED — yield-curve indicator (daily)
  - DXY  : US Dollar Index — USD strength indicator (daily)
  - ES   : S&P 500 Futures — risk sentiment proxy (daily)

All data is stored as CSV files in ModelTrading/data/ and updated via
data/update_external_data.py (run manually before training or via a daily cron
job for live trading).

Live-trading usage:
  feature_server.py calls get_external_df() on every request.  A module-level
  TTL cache (4 hours) avoids re-reading the CSV files on every bar while still
  picking up daily refreshes automatically.

Lookahead-bias guarantee:
  get_external_df() applies shift(1) on the *sparse* (daily / weekly) index
  before forward-filling onto the target index.  This ensures that at any
  timestamp T the value visible in the model is the one that was published
  *before* bar T opened — e.g. Friday's VIX close is only available from
  Monday's bars onward.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data directory (same location as OHLC CSVs)
# ---------------------------------------------------------------------------
def _data_dir() -> Path:
    """Return the absolute path to ModelTrading/data/."""
    try:
        import ModelTrading.config.directories as _dir
        return Path(_dir.DATA_DIR)
    except ImportError:
        # Fallback: 4 levels up from this file → project root / ModelTrading / data
        return Path(__file__).resolve().parents[4] / "ModelTrading" / "data"


# ---------------------------------------------------------------------------
# Module-level TTL cache
# ---------------------------------------------------------------------------
_CACHE_TTL = 4 * 60 * 60  # 4 hours in seconds

_cache: dict = {
    "cot":    {"df": None, "loaded_at": 0.0},
    "vix":    {"df": None, "loaded_at": 0.0},
    "yields": {"df": None, "loaded_at": 0.0},
    "eur_yields": {"df": None, "loaded_at": 0.0},
    "dxy":    {"df": None, "loaded_at": 0.0},
    "es":     {"df": None, "loaded_at": 0.0},
    "cross_asset": {"df": None, "loaded_at": 0.0},
}


def _is_stale(key: str) -> bool:
    return _cache[key]["df"] is None or (time.monotonic() - _cache[key]["loaded_at"]) > _CACHE_TTL


def _warn(msg: str) -> None:
    print(f"WARNING [external_data]: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Rolling percentile rank helper
# ---------------------------------------------------------------------------
def _rolling_pct_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Compute the percentile rank of the current value vs. the preceding
    (window-1) values, scaled to [0, 100].
    raw=True passes a numpy array → fast execution.
    """
    def _rank(x: np.ndarray) -> float:
        if len(x) <= 1 or np.isnan(x[-1]):
            return np.nan
        prev = x[:-1]
        valid = prev[~np.isnan(prev)]
        if len(valid) == 0:
            return np.nan
        return float((valid < x[-1]).mean() * 100)

    return series.rolling(window=window, min_periods=min_periods).apply(_rank, raw=True)


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Compute rolling z-score: (current - rolling_mean) / rolling_std.
    Standardizes values relative to recent history.
    """
    return (series - series.rolling(window=window, min_periods=min_periods).mean()) / \
           series.rolling(window=window, min_periods=min_periods).std()


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

# Trailing window for the rate-differential z-scores, in business days (~1 calendar
# year). Long enough to be stable, short enough to track a monetary regime. The
# information audit tests 125/250/500 — let its per-window result move this rather than
# intuition.
RATE_DIFF_Z_WINDOW = 250

# CFTC reference date is Tuesday; the report is published the following Friday.
COT_PUBLICATION_LAG_DAYS = 3

# ccy_* values older than this many calendar days are masked back to NaN instead of
# being forward-filled — see the staleness guard in get_external_df.
CROSS_ASSET_MAX_STALE_DAYS = 7


def load_cot(force: bool = False) -> pd.DataFrame:
    """
    Load CFTC COT data for EUR futures from data/cot_eur_futures.csv.

    Returns a DatetimeIndex DataFrame (weekly Fridays, UTC tz-naive) with:
      cot_net_position : (NonComm_Long - NonComm_Short) / Open_Interest, ~[-1, 1]
      cot_net_change   : week-over-week change of cot_net_position
      cot_index        : 52-week rolling percentile rank of cot_net_position [0, 100]

    Raises FileNotFoundError if the CSV does not exist.
    """
    if not force and not _is_stale("cot"):
        return _cache["cot"]["df"]

    csv_path = _data_dir() / "cot_eur_futures.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"COT CSV not found: {csv_path}. "
            "Run data/update_external_data.py to download it."
        )

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        oi = df["open_interest"].replace(0, np.nan)
        net = (df["non_commercial_long"] - df["non_commercial_short"]) / oi
        df["cot_net_position"] = net.astype(float)
        df["cot_net_change"] = df["cot_net_position"].diff()
        df["cot_index"] = _rolling_pct_rank(df["cot_net_position"], window=52, min_periods=26)

        result = df[["cot_net_position", "cot_net_change", "cot_index"]].copy()

        # PUBLICATION LAG. The CSV is dated by the CFTC *reference* Tuesday (516 of 522
        # rows are Tuesdays), but the report is published the following Friday at 15:30
        # ET. Returning it on the reference date and relying on get_external_df's
        # one-row shift left up to three days of lookahead in every COT feature.
        # Moving the index forward by the publication lag fixes it for every consumer,
        # not just for whoever remembers to compensate.
        result.index = result.index + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS)

        if len(result) < 52:
            _warn(f"COT CSV has only {len(result)} rows; cot_index needs ≥52 for reliable percentiles.")

    except Exception as exc:
        raise RuntimeError(f"Failed to parse COT CSV ({csv_path}): {exc}") from exc

    _cache["cot"]["df"] = result
    _cache["cot"]["loaded_at"] = time.monotonic()
    return result


def load_vix(force: bool = False) -> pd.DataFrame:
    """
    Load VIX data from data/vix_daily.csv.

    Returns a DatetimeIndex DataFrame (business days, UTC tz-naive) with:
      vix_level      : raw VIX closing level (typically [9, 80])
      vix_1d_change  : day-over-day change, zero-centred
      vix_percentile : 252-day rolling percentile rank [0, 100]

    Raises FileNotFoundError if the CSV does not exist.
    """
    if not force and not _is_stale("vix"):
        return _cache["vix"]["df"]

    csv_path = _data_dir() / "vix_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"VIX CSV not found: {csv_path}. "
            "Run data/update_external_data.py to download it."
        )

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        df["vix_level"] = df["close"].astype(float)
        df["vix_1d_change"] = df["vix_level"].diff()
        df["vix_percentile"] = _rolling_pct_rank(df["vix_level"], window=252, min_periods=126)

        result = df[["vix_level", "vix_1d_change", "vix_percentile"]].copy()

        if len(result) < 252:
            _warn(f"VIX CSV has only {len(result)} rows; vix_percentile needs ≥252 for reliable percentiles.")

    except Exception as exc:
        raise RuntimeError(f"Failed to parse VIX CSV ({csv_path}): {exc}") from exc

    _cache["vix"]["df"] = result
    _cache["vix"]["loaded_at"] = time.monotonic()
    return result


def load_yields(force: bool = False) -> pd.DataFrame:
    """
    Load US Treasury yield + policy rate data from data/us_yields_daily.csv.

    Returns a DatetimeIndex DataFrame (business days, UTC tz-naive) with:
      us_10y_yield     : 10-year Treasury yield (raw level, NON-STATIONARY)
      us_yield_spread  : 10Y - 2Y spread (yield-curve indicator, PARTIAL)
      carry_diff       : Fed Funds (dff) - ECB deposit rate (ecbdfr), PARTIAL.
                         US minus EUR: positive = USD-favorable carry
                         (carry literature: bearish EUR/USD).
      carry_diff_chg20 : 20-business-day change of carry_diff (carry momentum,
                         STATIONARY). NaN when dff/ecbdfr columns are absent
                         (old CSV format) or before ECBDFR history (1999).

    Raises FileNotFoundError if the CSV does not exist.
    """
    if not force and not _is_stale("yields"):
        return _cache["yields"]["df"]

    csv_path = _data_dir() / "us_yields_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Yields CSV not found: {csv_path}. "
            "Run data/update_external_data.py to download it."
        )

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        df["us_10y_yield"] = pd.to_numeric(df["dgs10"], errors="coerce")
        # NOTE: this is the US CURVE SLOPE (10Y-2Y), not a cross-country differential.
        # The cross-country differential lives in rate_diff_* (see get_external_df).
        df["us_yield_spread"] = pd.to_numeric(df["dgs10"], errors="coerce") - pd.to_numeric(df["dgs2"], errors="coerce")
        df["us_2y_yield"] = pd.to_numeric(df["dgs2"], errors="coerce")

        # Carry: US minus EUR policy rate (positive = USD-favorable).
        # Guard column access: an old-format CSV (pre carry columns) must
        # NaN-degrade the carry features, not fail the whole yields source.
        dff = pd.to_numeric(df["dff"], errors="coerce") if "dff" in df.columns else np.nan
        ecbdfr = pd.to_numeric(df["ecbdfr"], errors="coerce") if "ecbdfr" in df.columns else np.nan
        df["carry_diff"] = dff - ecbdfr
        # 20-business-day change on the sparse daily index; get_external_df()
        # applies shift(1) before ffill, so no extra shift needed here.
        df["carry_diff_chg20"] = df["carry_diff"] - df["carry_diff"].shift(20)

        result = df[["us_10y_yield", "us_2y_yield", "us_yield_spread",
                     "carry_diff", "carry_diff_chg20"]].copy()

    except Exception as exc:
        raise RuntimeError(f"Failed to parse yields CSV ({csv_path}): {exc}") from exc

    _cache["yields"]["df"] = result
    _cache["yields"]["loaded_at"] = time.monotonic()
    return result


def load_eur_yields(force: bool = False) -> pd.DataFrame:
    """
    Load the euro area AAA government spot curve from data/eur_yields_daily.csv.

    Returns a DatetimeIndex DataFrame (daily, UTC tz-naive) with:
      eur_2y  : 2-year euro area AAA spot rate (raw level, NON-STATIONARY)
      eur_10y : 10-year euro area AAA spot rate (raw level, NON-STATIONARY)

    This is the EUR leg of the interest-rate differential. Until it was added the data
    set contained no euro-area market rate at all: `us_yield_spread` is the US curve
    slope and `carry_diff` is a policy-rate step function, so the variable FX theory
    actually points at — the expected short-rate differential and its change — could
    not be formed. get_external_df() derives rate_diff_* from this plus the US legs.

    Raises FileNotFoundError if the CSV does not exist.
    """
    if not force and not _is_stale("eur_yields"):
        return _cache["eur_yields"]["df"]

    csv_path = _data_dir() / "eur_yields_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"EUR yields CSV not found: {csv_path}. "
            "Run data/update_external_data.py to download it."
        )

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.tz_localize(None)
        df = df.set_index("date").sort_index()
        for col in ("eur_2y", "eur_10y"):
            df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        result = df[["eur_2y", "eur_10y"]].copy()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse EUR yields CSV ({csv_path}): {exc}") from exc

    _cache["eur_yields"]["df"] = result
    _cache["eur_yields"]["loaded_at"] = time.monotonic()
    return result


def load_dxy(force: bool = False) -> pd.DataFrame:
    """
    Load DXY (US Dollar Index) data from data/dxy_daily.csv.

    Returns a DatetimeIndex DataFrame (business days, UTC tz-naive) with:
      dxy_level      : raw DXY closing level
      dxy_1d_change  : day-over-day change, zero-centred
      dxy_zscore_20  : 20-day rolling z-score

    Raises FileNotFoundError if the CSV does not exist.
    """
    if not force and not _is_stale("dxy"):
        return _cache["dxy"]["df"]

    csv_path = _data_dir() / "dxy_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"DXY CSV not found: {csv_path}. "
            "Run data/update_external_data.py to download it."
        )

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        df["dxy_level"] = df["close"].astype(float)
        df["dxy_1d_change"] = df["dxy_level"].diff()
        df["dxy_zscore_20"] = _rolling_zscore(df["dxy_level"], window=20, min_periods=10)

        result = df[["dxy_level", "dxy_1d_change", "dxy_zscore_20"]].copy()

    except Exception as exc:
        raise RuntimeError(f"Failed to parse DXY CSV ({csv_path}): {exc}") from exc

    _cache["dxy"]["df"] = result
    _cache["dxy"]["loaded_at"] = time.monotonic()
    return result


def load_es(force: bool = False) -> pd.DataFrame:
    """
    Load ES (S&P 500 Futures) data from data/es_daily.csv.

    Returns a DatetimeIndex DataFrame (business days, UTC tz-naive) with:
      es_1d_change    : day-over-day change, zero-centred
      es_zscore_20    : 20-day rolling z-score

    Raises FileNotFoundError if the CSV does not exist.
    """
    if not force and not _is_stale("es"):
        return _cache["es"]["df"]

    csv_path = _data_dir() / "es_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ES CSV not found: {csv_path}. "
            "Run data/update_external_data.py to download it."
        )

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        es_level = df["close"].astype(float)
        df["es_1d_change"] = es_level.diff()
        df["es_zscore_20"] = _rolling_zscore(es_level, window=20, min_periods=10)

        result = df[["es_1d_change", "es_zscore_20"]].copy()

    except Exception as exc:
        raise RuntimeError(f"Failed to parse ES CSV ({csv_path}): {exc}") from exc

    _cache["es"]["df"] = result
    _cache["es"]["loaded_at"] = time.monotonic()
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_cross_asset(force: bool = False) -> pd.DataFrame:
    """
    Load the pre-computed cross-asset currency-strength features from
    data/cross_asset_daily.csv (written by data/update_cross_asset_data.py).

    Returns a DatetimeIndex DataFrame (business days, UTC tz-naive) with the
    daily_ccy_* columns exactly as persisted: rank spreads, per-currency ranks,
    breadth and dispersion. All bounded by construction except the z-spreads.

    Unlike the other loaders this one serves DERIVED features, not a raw source:
    the currency decomposition needs ten instruments at once and is therefore
    computed upfront (same pattern as TimesFM and the regime model). Values in
    the CSV are unshifted; get_external_df applies the same shift(1) it applies
    to every other source, so at any bar the visible value is the one formed on
    an earlier day — one day MORE conservative than strictly necessary (the
    features use only closes up to their own date), and uniform with the rest.

    Raises FileNotFoundError if the CSV does not exist.
    """
    if not force and not _is_stale("cross_asset"):
        return _cache["cross_asset"]["df"]

    csv_path = _data_dir() / "cross_asset_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"cross-asset CSV not found: {csv_path}. "
            "Run data/update_cross_asset_data.py to build it."
        )

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.tz_localize(None)
        result = df.set_index("date").sort_index().astype(float)
        bad = [c for c in result.columns if not c.startswith("daily_ccy_")]
        if bad:
            raise RuntimeError(f"unexpected columns {bad} — regenerate the CSV")
        # indicators.py maps external features by BARE name (it prefixes the timeframe
        # itself), so serve them without the daily_ prefix like every other source.
        result.columns = [c[len("daily_"):] for c in result.columns]
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to parse cross-asset CSV ({csv_path}): {exc}") from exc

    _cache["cross_asset"]["df"] = result
    _cache["cross_asset"]["loaded_at"] = time.monotonic()
    return result


def get_external_df(
    target_index: pd.DatetimeIndex,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Merge all available external data sources and forward-fill onto target_index.

    Lookahead-bias guarantee
    ------------------------
    shift(1) is applied on the *sparse* (daily / weekly) index before the
    reindex/ffill onto the dense target index.  At any timestamp T the value
    visible is the one published *before* T — e.g. today's VIX is only
    visible from tomorrow's bars onward.

    Parameters
    ----------
    target_index : pd.DatetimeIndex
        The DatetimeIndex of the DataFrame being enriched (e.g. daily bars
        or 4H bars).
    force : bool
        Bypass the TTL cache and reload from disk.

    Returns
    -------
    pd.DataFrame aligned to target_index, or None if no CSV files are present.

    Notes
    -----
    - Missing CSV files produce a warning and are skipped (graceful degradation).
    - Values before the start of each data source history remain NaN.
    - Columns use bare names (no timeframe prefix): cot_net_position, vix_level, etc.
      The caller (indicators.py) maps them to the prefixed feature names.
    """
    frames: list[pd.DataFrame] = []

    for loader, label in [
        (load_cot, "COT"),
        (load_vix, "VIX"),
        (load_yields, "Yields"),
        (load_eur_yields, "EUR Yields"),
        (load_dxy, "DXY"),
        (load_es, "ES"),
        (load_cross_asset, "Cross-asset"),
    ]:
        try:
            frames.append(loader(force=force))
        except FileNotFoundError as exc:
            _warn(f"{label} data not available — feature will be NaN. ({exc})")
        except RuntimeError as exc:
            _warn(f"{label} data failed to load — feature will be NaN. ({exc})")

    if not frames:
        return None

    # Merge on date index (outer join so we keep all dates from all sources)
    merged = pd.concat(frames, axis=1).sort_index()

    # Carry each source's last known value across the union index BEFORE anything else.
    # `reindex(method="ffill")` further down fills missing INDEX ENTRIES, not NaN values
    # that are present in the frame — so on the union index (which carries every date any
    # source has) a weekly series stayed NaN on all the days it does not publish.
    # Measured 2026-08-29: COT reached 522 of 5,556 daily bars. This is also what makes
    # the differentials below well defined when the US and EUR bond markets have
    # different holidays. Causal: it only ever carries an older value forward.
    pre_ffill_notna = merged.notna()
    merged = merged.ffill()

    # --- Interest-rate differential --------------------------------------
    # The variable FX theory points at, and the one the data set lacked entirely until
    # the ECB curve was added. The LEVEL is the carry; the CHANGE is what moves spot,
    # which is why both are emitted. Sign convention matches carry_diff: positive =
    # USD-favourable.
    #
    # STATIONARITY — measured 2026-08-29 over 2005-2026, NOT assumed. Dickey-Fuller t
    # (5% critical value -2.86) and the per-era standard deviation:
    #
    #   series                     DF t   std 05-11  12-21  22-26   vol ratio
    #   rate_diff_2y              -1.55       1.109  1.019  0.355      3.1
    #   rate_diff_2y_z250         -5.29       1.539  1.487  1.485      1.0
    #   rate_diff_2y_chg20       -17.96       0.176  0.107  0.178      1.7
    #   rate_diff_2y_chg20_z250  -17.68       1.121  1.108  1.170      1.1
    #
    # The LEVELS have a unit root (DF t -1.55 / -2.01, AC(1) 0.999) and their mean shifts
    # with the monetary regime: rate_diff_2y averages 0.01 in 2005-2011, 1.30 in
    # 2012-2021 and 1.90 in 2022-2026. A tree split at "rate_diff_2y > 1.0" therefore
    # means something entirely different in 2008 than in 2024 — the levels cannot be fed
    # to a model directly.
    #
    # The CHANGES are stationary in MEAN (DF t -18 to -38) but not in VARIANCE: their
    # per-era volatility differs by a factor of 1.7-2.1 between the ZLB years and the
    # hiking cycles.
    #
    # The causal trailing z-score fixes both: it turns the levels stationary (DF t -1.55
    # -> -5.29) and flattens the volatility of the changes (ratio 1.7 -> 1.1). The raw
    # columns stay available as helpers; the *_z250 columns are what a model should use.
    for tenor in ("2y", "10y"):
        us_col, eur_col = f"us_{tenor}_yield", f"eur_{tenor}"
        if us_col in merged.columns and eur_col in merged.columns:
            diff = merged[us_col] - merged[eur_col]
            merged[f"rate_diff_{tenor}"] = diff
            merged[f"rate_diff_{tenor}_chg5"] = diff - diff.shift(5)
            merged[f"rate_diff_{tenor}_chg20"] = diff - diff.shift(20)
            for col in (f"rate_diff_{tenor}", f"rate_diff_{tenor}_chg5",
                        f"rate_diff_{tenor}_chg20"):
                merged[f"{col}_z250"] = _rolling_zscore(
                    merged[col], window=RATE_DIFF_Z_WINDOW, min_periods=RATE_DIFF_Z_WINDOW // 2)

    # --- Staleness guard for the cross-asset block -----------------------
    # The global ffill above is right for weekly/holiday gaps, but the cross-asset CSV
    # currently has a ~3-year export hole (2023-02 .. 2025-12). Carrying a February-2023
    # rank across that hole would hand the models three years of confidently stale
    # "currency strength" with nothing anywhere reporting it. The ccy_* columns all come
    # from one CSV and share one observation calendar, so one staleness series covers
    # them all: beyond CROSS_ASSET_MAX_STALE_DAYS without a fresh row they return to NaN
    # (and NaN they stay through the shift/reindex below).
    ccy_cols = [c for c in merged.columns if c.startswith("ccy_")]
    if ccy_cols:
        observed = pre_ffill_notna[ccy_cols[0]]
        idx_ser = merged.index.to_series()
        last_obs = idx_ser.where(observed).ffill()
        stale_days = (idx_ser - last_obs).dt.days
        too_stale = stale_days.isna() | (stale_days > CROSS_ASSET_MAX_STALE_DAYS)
        merged.loc[too_stale, ccy_cols] = np.nan

    # --- Lookahead-bias: shift on the SPARSE index before ffill ---
    # shift(1) moves each row's value to the NEXT row on the sparse index,
    # meaning at date D we see the value that was published on date D-1 (or
    # the previous week for COT, which additionally carries its publication lag
    # already applied in load_cot).
    merged_shifted = merged.shift(1)

    # Forward-fill onto the target index (handles weekends, holidays, intraday gaps)
    result = merged_shifted.reindex(target_index, method="ffill")

    return result


def invalidate_cache() -> None:
    """Force the next call to get_external_df to reload all data from disk."""
    for key in _cache:
        _cache[key]["loaded_at"] = 0.0
        _cache[key]["df"] = None
