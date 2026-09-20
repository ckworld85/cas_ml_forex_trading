"""
Scheduled central-bank event calendar (FXStreet export).

Serves countdown features — "calendar days until the next scheduled event" —
from ModelTrading/data/fxstreet_calendar_event_list.csv:

    until_fed_ir_decision -> Name == "Fed Interest Rate Decision"
    until_ezb_ir_decision -> Name == "ECB Main Refinancing Operations Rate"
                             (FXStreet's name for the ECB rate decision; the
                             "ECB Press Conference" entry is the presser 45
                             minutes later and does not exist for every meeting)

Semantics: 0 = the decision is TODAY (any intraday time), 1 = tomorrow,
counted in calendar days on the bar's own date.

Lookahead: these are TIMESTAMP-derived features like hour_sin / day_of_week.
FOMC and ECB meeting calendars are published more than a year in advance, so
"days until the next decision" at bar t uses only information available at t.
NO shift(1) is applied — shifting a known schedule would simply make the
countdown wrong by one day. This is also why the features do NOT go through
external_data.get_external_df(), whose sparse-index shift(1) exists for
*measured* data.

Degradation guard: bars dated AFTER the last event in the CSV get NaN, never a
fabricated value — an outdated calendar export degrades to NaN instead of
silently counting toward a meeting that already happened. Re-export the
FXStreet list before its max date approaches (currently covers 2007-08 to
2026-12).

Live trading: feature_server.py reaches this module through add_features when
a calendar feature is enabled. The parsed calendar is TTL-cached (4 h) like
external_data, so a refreshed CSV export is picked up without a restart.
Staging note: enabling a calendar feature makes BOTH this module and the
fxstreet CSV runtime artefacts — stage_jforex.ps1 allowlists must then include
them.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CALENDAR_CSV_NAME = "fxstreet_calendar_event_list.csv"

# feature name (bare, without timeframe prefix) -> FXStreet event name
EVENT_NAME_BY_FEATURE = {
    "until_fed_ir_decision": "Fed Interest Rate Decision",
    "until_ezb_ir_decision": "ECB Main Refinancing Operations Rate",
}

_CACHE_TTL = 4 * 60 * 60  # seconds, mirrors external_data

# csv_path -> {"events": {event_name: np.ndarray[datetime64[D]]}, "loaded_at": float}
_cache: dict = {}


def _data_dir() -> Path:
    """Return the absolute path to ModelTrading/data/ (mirrors external_data)."""
    try:
        import ModelTrading.config.directories as _dir
        return Path(_dir.DATA_DIR)
    except ImportError:
        return Path(__file__).resolve().parents[4] / "ModelTrading" / "data"


def _default_csv_path() -> Path:
    return _data_dir() / CALENDAR_CSV_NAME


def _warn(msg: str) -> None:
    print(f"WARNING [calendar_events]: {msg}", file=sys.stderr)


def is_calendar_feature(name: str) -> bool:
    """True if ``name`` (bare, without timeframe prefix) is a calendar countdown feature."""
    return name in EVENT_NAME_BY_FEATURE


def _reset_cache() -> None:
    """Testing hook: drop every parsed calendar."""
    _cache.clear()


def _load_calendar(csv_path: Path) -> dict:
    """Parse the FXStreet CSV into {event_name: sorted unique datetime64[D] array}
    plus {event_name: sorted unique datetime64[ns] array} under the "ts:" prefix
    (the full published decision TIMES, for intraday countdowns)."""
    df = pd.read_csv(csv_path)
    starts = pd.to_datetime(df["Start"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    events: dict = {}
    for event_name in set(EVENT_NAME_BY_FEATURE.values()):
        dates = starts[(df["Name"] == event_name) & starts.notna()]
        days = np.unique(dates.dt.normalize().values.astype("datetime64[D]"))
        if len(days) == 0:
            _warn(f"event '{event_name}' not found in {csv_path.name}")
        events[event_name] = days
        events[f"ts:{event_name}"] = np.unique(dates.values)
    return events


def load_event_dates(event_name: str, csv_path=None, force: bool = False) -> np.ndarray:
    """
    Sorted unique event DATES (datetime64[D]) for one FXStreet event name.

    TTL-cached per csv_path. Returns an empty array (with a warning) when the
    CSV is missing or the event name does not occur — callers degrade to NaN.
    """
    path = Path(csv_path) if csv_path is not None else _default_csv_path()
    key = str(path)
    entry = _cache.get(key)
    stale = entry is None or force or (time.monotonic() - entry["loaded_at"]) > _CACHE_TTL
    if stale:
        try:
            entry = {"events": _load_calendar(path), "loaded_at": time.monotonic()}
        except Exception as exc:  # missing file, malformed CSV
            _warn(f"cannot load {path}: {exc}")
            entry = {"events": {}, "loaded_at": time.monotonic()}
        _cache[key] = entry
    return entry["events"].get(event_name, np.array([], dtype="datetime64[D]"))


def _load_high_impact_counts(csv_path: Path, currencies: tuple) -> dict:
    """Parse the FXStreet CSV into per-day HIGH-impact event counts.

    Returns {"days": datetime64[D] array, "counts": float array,
    "first": date, "last": date} over events whose Currency is in
    ``currencies``. Coverage bounds come from ALL parsed rows so that a day
    with zero matching events inside the export window still counts as 0
    rather than NaN.
    """
    df = pd.read_csv(csv_path)
    starts = pd.to_datetime(df["Start"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    valid = starts.notna()
    if not valid.any():
        raise ValueError(f"no parsable Start timestamps in {csv_path.name}")
    all_days = starts[valid].dt.normalize().values.astype("datetime64[D]")
    sel = valid & (df["Impact"] == "HIGH") & df["Currency"].isin(list(currencies))
    sel_days = starts[sel].dt.normalize().values.astype("datetime64[D]")
    days, counts = np.unique(sel_days, return_counts=True)
    return {
        "days": days,
        "counts": counts.astype(np.float64),
        "first": all_days.min(),
        "last": all_days.max(),
    }


def high_impact_event_count(index: pd.DatetimeIndex, currencies=("EUR", "USD"),
                            csv_path=None) -> pd.Series:
    """
    Number of scheduled HIGH-impact events for ``currencies`` on each bar's
    own calendar date.

    Timestamp-derived like the countdown features: the schedule is published
    in advance, so no shift is applied here (callers that want the PREVIOUS
    bar's event day shift the result themselves). NaN outside the calendar's
    coverage (before its first / after its last parsed event date) and
    everywhere when the CSV cannot be loaded.
    """
    path = Path(csv_path) if csv_path is not None else _default_csv_path()
    key = (str(path), "high_impact", tuple(sorted(currencies)))
    entry = _cache.get(key)
    stale = entry is None or (time.monotonic() - entry["loaded_at"]) > _CACHE_TTL
    if stale:
        try:
            entry = {"data": _load_high_impact_counts(path, tuple(currencies)),
                     "loaded_at": time.monotonic()}
        except Exception as exc:  # missing file, malformed CSV
            _warn(f"cannot load {path}: {exc}")
            entry = {"data": None, "loaded_at": time.monotonic()}
        _cache[key] = entry

    out = np.full(len(index), np.nan, dtype=np.float64)
    data = entry["data"]
    if data is not None and len(data["days"]) > 0:
        bar_days = index.normalize().values.astype("datetime64[D]")
        in_coverage = (bar_days >= data["first"]) & (bar_days <= data["last"])
        pos = np.searchsorted(data["days"], bar_days)
        pos_clipped = np.clip(pos, 0, len(data["days"]) - 1)
        hit = in_coverage & (data["days"][pos_clipped] == bar_days)
        out[in_coverage] = 0.0
        out[hit] = data["counts"][pos_clipped[hit]]
    return pd.Series(out, index=index, dtype=np.float32)


def days_since_last_event(index: pd.DatetimeIndex, event_name: str, csv_path=None) -> pd.Series:
    """
    Calendar days from the most recent scheduled event to each bar's date
    (0 = the event is today). Takes the FXStreet event NAME (the A6 calendar
    direction features resolve their clock through EVENT_NAME_BY_FEATURE).

    NaN outside the calendar's coverage: before the FIRST event (nothing to
    count from) and after the LAST event date (an outdated export cannot rule
    out a newer meeting it does not contain — same degradation guard as the
    countdown, so both clocks share one validity window).
    """
    event_days = load_event_dates(event_name, csv_path=csv_path)
    out = np.full(len(index), np.nan, dtype=np.float64)
    if len(event_days) > 0:
        bar_days = index.normalize().values.astype("datetime64[D]")
        # last event on or before the bar's own date -> same-day event = 0
        pos = np.searchsorted(event_days, bar_days, side="right") - 1
        valid = (pos >= 0) & (bar_days <= event_days[-1])
        out[valid] = (bar_days[valid] - event_days[pos[valid]]).astype("timedelta64[D]").astype(np.float64)
    return pd.Series(out, index=index, dtype=np.float32)


def hours_until_next_event(index: pd.DatetimeIndex, event_name: str, csv_path=None) -> pd.Series:
    """
    Hours from each bar's timestamp to the next scheduled event TIME (the
    published Start, e.g. the FOMC statement release), fractional, >= 0.

    Timestamp-derived like the day countdown — the schedule (date AND time) is
    published in advance, so no shift is applied. NaN before the first and
    after the last published timestamp.
    """
    # load_event_dates serves the "ts:" key from the same TTL cache entry.
    event_ts = load_event_dates(f"ts:{event_name}", csv_path=csv_path)
    out = np.full(len(index), np.nan, dtype=np.float64)
    if len(event_ts) > 0:
        bar_ts = index.values
        pos = np.searchsorted(event_ts, bar_ts, side="left")
        valid = (pos < len(event_ts)) & (bar_ts >= event_ts[0])
        deltas = event_ts[pos[valid]] - bar_ts[valid]
        out[valid] = deltas.astype("timedelta64[s]").astype(np.float64) / 3600.0
    return pd.Series(out, index=index, dtype=np.float32)


def scheduled_event_density(index: pd.DatetimeIndex, window_days: int = 2,
                            currencies=("EUR", "USD"), csv_path=None) -> pd.Series:
    """
    Number of scheduled HIGH-impact events for ``currencies`` over the bar's
    own date and the following ``window_days - 1`` calendar days.

    Forward-looking over the PUBLISHED schedule only — causal for the same
    reason as the countdown (the calendar is known in advance). NaN whenever
    any day of the window falls outside calendar coverage.
    """
    total = None
    for offset in range(int(window_days)):
        counts = high_impact_event_count(index + pd.Timedelta(days=offset),
                                         currencies=currencies, csv_path=csv_path)
        vals = pd.Series(counts.to_numpy(), index=index)
        total = vals if total is None else total + vals
    return total.astype(np.float32)


def days_until_next_event(index: pd.DatetimeIndex, feature_name: str, csv_path=None) -> pd.Series:
    """
    Calendar days from each bar's date to the next scheduled event (0 = today).

    NaN outside the calendar's coverage — before its FIRST event (the true next
    decision may predate the export, so counting to the export's first entry
    would fabricate a too-large value) and after its LAST event (unknown
    schedule) — and everywhere when the calendar cannot be loaded.
    """
    if feature_name not in EVENT_NAME_BY_FEATURE:
        raise ValueError(
            f"Unknown calendar feature '{feature_name}'; "
            f"valid: {sorted(EVENT_NAME_BY_FEATURE)}"
        )
    event_days = load_event_dates(EVENT_NAME_BY_FEATURE[feature_name], csv_path=csv_path)

    out = np.full(len(index), np.nan, dtype=np.float64)
    if len(event_days) > 0:
        bar_days = index.normalize().values.astype("datetime64[D]")
        # first event on or after the bar's own date -> same-day decision = 0
        pos = np.searchsorted(event_days, bar_days, side="left")
        valid = (pos < len(event_days)) & (bar_days >= event_days[0])
        out[valid] = (event_days[pos[valid]] - bar_days[valid]).astype("timedelta64[D]").astype(np.float64)
    return pd.Series(out, index=index, dtype=np.float32)
