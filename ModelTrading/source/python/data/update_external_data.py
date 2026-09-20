"""
Download and update external (non-OHLC) data for the slow model.

Data sources (all free, no API key required):

  COT    – CFTC Commitment of Traders, EUR futures (CME)
           https://www.cftc.gov/dea/newcot/  (historical + current-year CSVs)

  VIX    – CBOE Volatility Index via Yahoo Finance (yfinance)
           Ticker: ^VIX

  Yields – US Treasury 10Y / 2Y from FRED direct CSV endpoints
           https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10
           https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2
  EURYLD – Euro area AAA government spot curve (2Y/10Y) via the ECB Data Portal
           https://data-api.ecb.europa.eu/service/data/YC/ — the EUR leg of the rate
           differential, which the data set previously lacked entirely.

  DXY    – US Dollar Index via Yahoo Finance (yfinance)
           Ticker: DX-Y.NYB

  ES     – S&P 500 Futures via Yahoo Finance (yfinance)
           Ticker: ES=F

  GOLD   – Gold (XAU/USD) via Yahoo Finance (yfinance)
           Ticker: GC=F

Output files are written to ModelTrading/data/:
  cot_eur_futures.csv
  vix_daily.csv
  us_yields_daily.csv
  dxy_daily.csv
  es_daily.csv
  gold_daily.csv

Usage:
  python data/update_external_data.py          # update all sources
  python data/update_external_data.py --quiet  # suppress progress output

Run this script before training and via a daily scheduled task for live trading.
"""

from datetime import datetime
import sys
import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Resolve data directory
# ---------------------------------------------------------------------------
def _data_dir() -> Path:
    try:
        import ModelTrading.config.directories as _dir
        return Path(_dir.DATA_DIR)
    except ImportError:
        return Path(__file__).resolve().parents[4] / "ModelTrading" / "data"


# ---------------------------------------------------------------------------
# COT  –  CFTC Commitment of Traders
# ---------------------------------------------------------------------------
# CFTC Disaggregated Futures & Options (legacy "Traders in Financial Futures")
# EUR FX futures are in the "Financial Futures" report, CFTC code 099741.
# Historical files (per year): https://www.cftc.gov/files/dea/history/fut_fin_txt_YYYY.zip
_COT_HIST_BASE = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{}.zip"
_COT_CURR_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"

# Column names in CFTC "Traders in Financial Futures" (TFF) CSV
# Leveraged money = hedge funds / speculators — equivalent to legacy NonCommercial
_COT_DATE_COL  = "Report_Date_as_YYYY-MM-DD"
_COT_NAME_COL  = "Market_and_Exchange_Names"
_COT_LONG_COL  = "Lev_Money_Positions_Long_All"
_COT_SHORT_COL = "Lev_Money_Positions_Short_All"
_COT_OI_COL    = "Open_Interest_All"
_COT_FILTER    = "EURO FX"


def download_cot(output_path: Path, verbose: bool = True) -> bool:
    """
    Download CFTC EUR futures COT data (historical + current year) and save
    to output_path as a CSV with columns:
      date, non_commercial_long, non_commercial_short, open_interest

    Returns True on success, False on error.
    """
    if verbose:
        print("  Downloading COT data from CFTC …")

    frames: list[pd.DataFrame] = []

    # --- Historical files (individual year archives from 2000-2025) ---
    # Download the last 10 years to have sufficient history for COT analysis
    import zipfile
    current_year = pd.Timestamp.now().year
    start_year = max(2000, current_year - 10)

    for year in range(start_year, current_year):
        url = _COT_HIST_BASE.format(year)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        try:
                            chunk = pd.read_csv(f, low_memory=False)
                            if _COT_NAME_COL in chunk.columns and _COT_DATE_COL in chunk.columns:
                                chunk = chunk[chunk[_COT_NAME_COL].str.contains(_COT_FILTER, na=False)]
                                if not chunk.empty:
                                    frames.append(chunk)
                        except Exception:
                            pass
            if verbose:
                print(f"    {year}: {sum(len(f) for f in frames[-1:])} EUR rows loaded")
        except Exception as exc:
            if verbose:
                print(f"    WARNING: Could not download {year} COT file: {exc}")

    # --- Current-year incremental file ---
    try:
        resp = requests.get(_COT_CURR_URL, timeout=30)
        resp.raise_for_status()
        curr = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        if _COT_NAME_COL in curr.columns and _COT_DATE_COL in curr.columns:
            curr = curr[curr[_COT_NAME_COL].str.contains(_COT_FILTER, na=False)]
            if not curr.empty:
                frames.append(curr)
                if verbose:
                    print(f"    Current-year file: {len(curr)} EUR rows loaded")
    except Exception as exc:
        if verbose:
            print(f"    WARNING: Could not download current-year COT file: {exc}")

    if not frames:
        if verbose:
            print("    ERROR: No COT data retrieved.")
        return False

    combined = pd.concat(frames, ignore_index=True)

    # Select and rename columns
    try:
        result = combined[[_COT_DATE_COL, _COT_LONG_COL, _COT_SHORT_COL, _COT_OI_COL]].copy()
    except KeyError as exc:
        if verbose:
            print(f"    ERROR: Expected columns missing from COT data: {exc}")
        return False

    result.columns = ["date", "non_commercial_long", "non_commercial_short", "open_interest"]

    # Parse dates and deduplicate
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"])
    result = result.drop_duplicates(subset="date", keep="last")
    result = result.sort_values("date")
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    if verbose:
        print(f"    Saved {len(result)} rows → {output_path.name}")
    return True


# ---------------------------------------------------------------------------
# VIX  –  CBOE Volatility Index via yfinance
# ---------------------------------------------------------------------------

def _download_yfinance_ticker(
    ticker: str,
    output_path: Path,
    label: str,
    verbose: bool = True,
    start: str = "2000-01-01"
) -> bool:
    """
    Generic function to download a ticker from Yahoo Finance via yfinance.
    Saves CSV with columns: date, close

    Returns True on success, False on error.
    """
    if verbose:
        print(f"  Downloading {label} data from Yahoo Finance (ticker: {ticker}) …")

    try:
        import yfinance as yf
    except ImportError:
        if verbose:
            print("    ERROR: yfinance not installed. Run: pip install yfinance")
        return False

    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            if verbose:
                print(f"    ERROR: yfinance returned empty DataFrame for {ticker}.")
            return False

        # yfinance may return MultiIndex columns; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            if verbose:
                print(f"    ERROR: 'close' column not found. Columns: {list(df.columns)}")
            return False

        result = df[["close"]].copy()
        result.index = pd.to_datetime(result.index).tz_localize(None)
        result = result.sort_index().dropna()
        result.index.name = "date"
        result = result.reset_index()
        result["date"] = result["date"].dt.strftime("%Y-%m-%d")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

        if verbose:
            print(f"    Saved {len(result)} rows → {output_path.name}")
        return True

    except Exception as exc:
        if verbose:
            print(f"    ERROR: {label} download failed: {exc}")
        return False


def download_vix(output_path: Path, verbose: bool = True) -> bool:
    """
    Download ^VIX daily closing prices from Yahoo Finance via yfinance.
    Saves CSV with columns: date, close

    Returns True on success, False on error.
    """
    return _download_yfinance_ticker("^VIX", output_path, "VIX", verbose)


# ---------------------------------------------------------------------------
# US Treasury Yields  –  FRED direct CSV (no API key)
# ---------------------------------------------------------------------------
_FRED_DGS10 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
_FRED_DGS2  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"
_FRED_DFF    = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"     # Effective Fed Funds Rate (daily)
_FRED_ECBDFR = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=ECBDFR"  # ECB Deposit Facility Rate (daily)


def _fetch_fred_series(url: str, col_name: str, verbose: bool) -> pd.DataFrame | None:
    """Download one FRED series and return a DataFrame with columns [date, col_name]."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", col_name]
        # FRED uses '.' for missing values
        df[col_name] = pd.to_numeric(df[col_name].replace(".", np.nan), errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", col_name]).sort_values("date")
        return df
    except Exception as exc:
        if verbose:
            print(f"    WARNING: Could not fetch {url}: {exc}")
        return None


def download_yields(output_path: Path, verbose: bool = True) -> bool:
    """
    Download US Treasury 10Y/2Y yields plus the US/EUR policy rates
    (Effective Fed Funds Rate, ECB Deposit Facility Rate) from FRED direct
    CSV endpoints. The policy rates feed the carry features
    (carry_diff = dff - ecbdfr) in features/external_data.py.
    Saves CSV with columns: date, dgs10, dgs2, dff, ecbdfr

    Returns True on success, False on error.
    """
    if verbose:
        print("  Downloading US Treasury yield + policy rate data from FRED …")

    df10   = _fetch_fred_series(_FRED_DGS10,  "dgs10",  verbose)
    df2    = _fetch_fred_series(_FRED_DGS2,   "dgs2",   verbose)
    dffff  = _fetch_fred_series(_FRED_DFF,    "dff",    verbose)
    dfecb  = _fetch_fred_series(_FRED_ECBDFR, "ecbdfr", verbose)

    if df10 is None and df2 is None and (dffff is None or dfecb is None):
        if verbose:
            print("    ERROR: Could not retrieve any yield data.")
        return False

    if df10 is None:
        if verbose:
            print("    WARNING: 10Y yield unavailable; us_10y_yield will be NaN.")
        df10 = pd.DataFrame(columns=["date", "dgs10"])

    if df2 is None:
        if verbose:
            print("    WARNING: 2Y yield unavailable; us_yield_spread will be NaN.")
        df2 = pd.DataFrame(columns=["date", "dgs2"])

    if dffff is None:
        if verbose:
            print("    WARNING: Fed Funds rate unavailable; carry_diff will be NaN.")
        dffff = pd.DataFrame(columns=["date", "dff"])

    if dfecb is None:
        if verbose:
            print("    WARNING: ECB deposit rate unavailable; carry_diff will be NaN.")
        dfecb = pd.DataFrame(columns=["date", "ecbdfr"])

    result = pd.merge(df10, df2, on="date", how="outer")
    result = pd.merge(result, dffff, on="date", how="outer")
    result = pd.merge(result, dfecb, on="date", how="outer").sort_values("date")
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    if verbose:
        print(f"    Saved {len(result)} rows → {output_path.name}")
    return True


# ---------------------------------------------------------------------------
# EUR yields  –  euro area AAA government spot curve via the ECB Data Portal
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------
# Until this was added the data set had NO euro-area interest rate at all beyond the
# ECB deposit facility rate. What was called `us_yield_spread` is DGS10 - DGS2, i.e. the
# US yield CURVE SLOPE, not a cross-country differential; and `carry_diff` is
# Fed Funds - ECB depo, a step function that moves ~8 times a year. The variable the FX
# literature actually points at — the *expected* short-rate differential, proxied by
# US2Y - DE2Y and above all by its CHANGE — was therefore never in the feature set, and
# the 2026-08-29 information audit could not test it.
#
# Source: ECB Data Portal, dataflow YC (euro area AAA-rated central government bond
# spot rates, Svensson fit). Free, no key, daily, from 2004-09-06 — which covers the
# whole EUR/USD history in this project.

_ECB_YC_BASE = "https://data-api.ecb.europa.eu/service/data/YC/"
_ECB_YC_SERIES = {
    "eur_2y":  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
    "eur_10y": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
}


def _fetch_ecb_series(series_key: str, col_name: str, verbose: bool) -> pd.DataFrame | None:
    """Download one ECB Data Portal series as [date, col_name]."""
    try:
        resp = requests.get(
            _ECB_YC_BASE + series_key,
            params={"format": "csvdata", "startPeriod": "2004-01-01"},
            timeout=90,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            if verbose:
                print(f"    WARNING: unexpected ECB payload for {series_key}.")
            return None
        out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        out.columns = ["date", col_name]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out[col_name] = pd.to_numeric(out[col_name], errors="coerce")
        return out.dropna().sort_values("date")
    except Exception as exc:
        if verbose:
            print(f"    WARNING: Could not fetch ECB {series_key}: {exc}")
        return None


def download_eur_yields(output_path: Path, verbose: bool = True) -> bool:
    """
    Download the euro area AAA government spot curve (2Y, 10Y) from the ECB Data Portal.
    Saves CSV with columns: date, eur_2y, eur_10y

    These are the EUR leg of the rate differential; the US leg comes from FRED via
    download_yields(). features/external_data.py derives the differentials.

    Returns True on success, False on error.
    """
    if verbose:
        print("  Downloading euro area AAA government spot curve from the ECB …")

    frames = []
    for col, key in _ECB_YC_SERIES.items():
        df = _fetch_ecb_series(key, col, verbose)
        if df is None:
            if verbose:
                print(f"    WARNING: {col} unavailable; the {col} differential will be NaN.")
            df = pd.DataFrame(columns=["date", col])
        frames.append(df)

    if all(len(f) == 0 for f in frames):
        if verbose:
            print("    ERROR: Could not retrieve any ECB yield data.")
        return False

    result = frames[0]
    for f in frames[1:]:
        result = pd.merge(result, f, on="date", how="outer")
    result = result.sort_values("date")
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    if verbose:
        print(f"    Saved {len(result)} rows → {output_path.name}")
    return True


# ---------------------------------------------------------------------------
# DXY  –  US Dollar Index via yfinance
# ---------------------------------------------------------------------------

def download_dxy(output_path: Path, verbose: bool = True) -> bool:
    """
    Download DX-Y.NYB (US Dollar Index) daily closing prices from Yahoo Finance.
    Saves CSV with columns: date, close

    Returns True on success, False on error.
    """
    return _download_yfinance_ticker("DX-Y.NYB", output_path, "DXY", verbose)


# ---------------------------------------------------------------------------
# ES  –  S&P 500 Futures via yfinance
# ---------------------------------------------------------------------------

def download_es(output_path: Path, verbose: bool = True) -> bool:
    """
    Download ES=F (S&P 500 Futures) daily closing prices from Yahoo Finance.
    Saves CSV with columns: date, close

    Returns True on success, False on error.
    """
    return _download_yfinance_ticker("ES=F", output_path, "ES", verbose)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def update_external_data(verbose: bool = True) -> tuple[int, int]:
    """
    Download / refresh all external data sources.

    Returns (success_count, error_count).
    """
    data_dir = _data_dir()
    if verbose:
        print(f"Updating external data in: {data_dir}\n")

    tasks = [
        (download_cot,    data_dir / "cot_eur_futures.csv",  "COT"),
        (download_vix,    data_dir / "vix_daily.csv",         "VIX"),
        (download_yields, data_dir / "us_yields_daily.csv",   "Yields"),
        (download_eur_yields, data_dir / "eur_yields_daily.csv", "EUR Yields"),
        (download_dxy,    data_dir / "dxy_daily.csv",         "DXY"),
        (download_es,     data_dir / "es_daily.csv",          "ES"),
    ]

    success = 0
    errors = 0
    for fn, path, label in tasks:
        ok = fn(path, verbose=verbose)
        if ok:
            success += 1
        else:
            errors += 1
        if verbose:
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {label}\n")

    return success, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download external (non-OHLC) data for the slow model."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    if (verbose:=not args.quiet):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting external data update…")

    success, errors = update_external_data(verbose=verbose)

    if (verbose:=not args.quiet):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] External data update completed: {success} succeeded, {errors} failed.")

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
