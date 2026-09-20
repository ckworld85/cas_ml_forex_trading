"""Fetch ALFRED vintage CSVs for validating the MT5 calendar export (B4/A14).

For every release in the MT5 export of the registered series, downloads the ALFRED
vintage as of that release date (keyless ``alfredgraph.csv`` endpoint) into
``ModelTrading/data/alfred_vintages/{SERIES}_{YYYY-MM-DD}.csv``. Existing files are
skipped, so the script is resumable and incremental.

Transport note: on the Windows dev box, python HTTP stacks (urllib/requests) are
blocked for stlouisfed.org while ``curl.exe`` (schannel TLS) gets through — curl is
therefore the primary transport, with urllib as fallback for other environments.

Usage:
    python -m ModelTrading.source.python.data.update_alfred_vintages
    python -m ModelTrading.source.python.data.update_alfred_vintages --series PAYEMS
"""

from __future__ import annotations

import argparse
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

from ModelTrading.config.directories import BASE_DIR

MT5_CSV_DEFAULT = Path(BASE_DIR) / "data" / "mt5_calendar_export.csv"
VINTAGE_DIR_DEFAULT = Path(BASE_DIR) / "data" / "alfred_vintages"

ALFRED_URL = (
    "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
    "?id={series}&vintage_date={vintage}&cosd={cosd}&coed={coed}"
)

# MT5 event_code -> ALFRED series id (the validation trio frozen in the B4 memo).
# Note: the headline CPI m/m carries the code 'consumer-price-index-mm' in the MT5
# calendar ('cpi-mm' does not exist; the only *cpi* code is the Cleveland median).
SERIES_MAP = {
    "nonfarm-payrolls": "PAYEMS",
    "consumer-price-index-mm": "CPIAUCSL",
    "unemployment-rate": "UNRATE",
}

THROTTLE_SECONDS = 0.25
RETRIES = 3


def load_release_dates(mt5_csv: Path) -> pd.DataFrame:
    """Return one row per (series, release_date) with an actual value present."""
    df = pd.read_csv(mt5_csv, sep=";", encoding="latin-1", low_memory=False)
    df = df[df["currency"] == "USD"]
    df = df[df["event_code"].isin(SERIES_MAP)]
    df = df[df["actual_raw"].notna()]
    df["release_date"] = pd.to_datetime(df["time"], unit="s").dt.date
    df["series"] = df["event_code"].map(SERIES_MAP)
    out = df[["series", "release_date"]].drop_duplicates().sort_values(
        ["series", "release_date"]
    )
    return out.reset_index(drop=True)


def fetch_url(url: str, timeout: int = 60) -> str:
    """curl.exe first (works through the local TLS-fingerprint block), urllib fallback."""
    try:
        proc = subprocess.run(
            ["curl.exe", "-s", "-m", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        if proc.returncode == 0 and proc.stdout.startswith("observation_date"):
            return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def vintage_path(vintage_dir: Path, series: str, release_date) -> Path:
    return vintage_dir / f"{series}_{release_date}.csv"


def fetch_vintage(series: str, release_date, vintage_dir: Path) -> str:
    """Fetch one vintage; returns 'fetched' | 'skipped' | 'failed'."""
    target = vintage_path(vintage_dir, series, release_date)
    if target.exists():
        return "skipped"

    release = pd.Timestamp(release_date)
    cosd = (release - pd.DateOffset(months=4)).replace(day=1).date()
    url = ALFRED_URL.format(
        series=series, vintage=release_date, cosd=cosd, coed=release_date
    )

    for attempt in range(1, RETRIES + 1):
        try:
            text = fetch_url(url)
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            if len(lines) >= 2 and lines[0].startswith("observation_date"):
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return "fetched"
        except Exception:
            pass
        time.sleep(THROTTLE_SECONDS * attempt * 4)

    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mt5-csv", type=Path, default=MT5_CSV_DEFAULT)
    parser.add_argument("--vintage-dir", type=Path, default=VINTAGE_DIR_DEFAULT)
    parser.add_argument("--series", choices=sorted(set(SERIES_MAP.values())),
                        default=None, help="restrict to one ALFRED series")
    args = parser.parse_args()

    releases = load_release_dates(args.mt5_csv)
    if args.series:
        releases = releases[releases["series"] == args.series]

    args.vintage_dir.mkdir(parents=True, exist_ok=True)

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    failures = []

    for i, row in releases.iterrows():
        status = fetch_vintage(row["series"], row["release_date"], args.vintage_dir)
        counts[status] += 1
        if status == "failed":
            failures.append(f"{row['series']}_{row['release_date']}")
        if status == "fetched":
            time.sleep(THROTTLE_SECONDS)
        if (counts["fetched"] + counts["skipped"]) % 50 == 0:
            print(f"... {counts['fetched'] + counts['skipped']}/{len(releases)}",
                  flush=True)

    print(f"done: {counts} of {len(releases)} releases")
    if failures:
        print("FAILED:", ", ".join(failures[:20]),
              "..." if len(failures) > 20 else "")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
