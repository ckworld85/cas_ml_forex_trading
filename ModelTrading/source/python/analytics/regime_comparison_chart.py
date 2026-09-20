"""
Regime comparison chart: EUR/USD daily price with regime lanes below.

Renders one PNG with the daily ASK close on top and one horizontal colour lane
per regime source underneath — the rule-based ADX/price-efficiency regime
(labeling/regime.generate_regime_labels) and each precomputed ML regime model
(data/regime_daily_{hmm,gmm,kmeans}.csv, mapped through
generate_regime_labels_from_scores so every lane shares the same
UPTREND/DOWNTREND/RANGE contract and the comparison is like-for-like).

Regime labels are computed on the FULL daily history and only then sliced to
the requested window — computing on the window alone would corrupt the warm-up
(ADX, the 500-bar volatility-percentile lookback). Bars where a source has no
defined regime yet (indicator warm-up, model coverage start) stay blank instead
of being painted RANGE.

Usage (from ModelTrading/source/python/):
  python analytics/regime_comparison_chart.py --start 2024-01-01 --end 2026-04-19
  python analytics/regime_comparison_chart.py --start 2020-01-01 --end 2023-12-31 --dimension vol
  python analytics/regime_comparison_chart.py --start 2024-01-01 --end 2026-04-19 --algos hmm --trend-threshold 0.3
  python analytics/regime_comparison_chart.py --start 2015-01-01 --end 2026-04-19 --price-style line
  python analytics/regime_comparison_chart.py --start 2024-01-01 --end 2026-04-19 --no-direction
  # German labels + no embedded title (figure for a German document, caption external):
  python analytics/regime_comparison_chart.py --start 2025-10-01 --end 2026-04-19 --lang de --no-title --dpi 200 --out <path.png>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import ModelTrading.config.directories as dir_config
from ModelTrading.source.python.utils import csv as csv_utils
from ModelTrading.source.python.labeling.regime import (
    generate_regime_labels,
    generate_regime_labels_from_scores,
)
from ModelTrading.source.python.features.regime_model import (
    regime_csv_path,
    regime_meta_path,
)

# --- palette (validated reference palette, light mode) -----------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PRICE_COLOR = "#2a78d6"
BAR_UP = "#008300"    # close >= open (green, trading convention)
BAR_DOWN = "#e34948"  # close < open (red)

# Direction is a polarity → diverging pair (blue/red poles) + neutral gray.
TREND_ORDER = ["UPTREND", "RANGE", "DOWNTREND"]
TREND_COLORS = {"UPTREND": "#2a78d6", "RANGE": "#c3c2b7", "DOWNTREND": "#e34948"}
TREND_LEGEND = {"UPTREND": "Uptrend", "RANGE": "Range", "DOWNTREND": "Downtrend"}

# Collapsed variant (--no-direction, direction_aware=False): trend vs range only.
TREND_BINARY_ORDER = ["TREND", "RANGE"]
TREND_BINARY_COLORS = {"TREND": "#2a78d6", "RANGE": "#c3c2b7"}
TREND_BINARY_LEGEND = {"TREND": "Trend", "RANGE": "Range"}

# Volatility is an ordered magnitude → one-hue ordinal ramp (blue 650/400/250).
VOL_ORDER = ["HIGH_VOL", "MED_VOL", "LOW_VOL"]
VOL_COLORS = {"HIGH_VOL": "#104281", "MED_VOL": "#3987e5", "LOW_VOL": "#86b6ef"}
VOL_LEGEND = {"HIGH_VOL": "High vol", "MED_VOL": "Med vol", "LOW_VOL": "Low vol"}

DEFAULT_ALGOS = ["hmm", "gmm", "kmeans"]

# --- chart text per language -------------------------------------------------
# 'en' is the analysis default and reproduces every previously committed chart
# bit-for-bit; 'de' exists for figures embedded in German documents (thesis
# terminology: Aufwärtstrend/Seitwärtsphase/Abwärtstrend, lane «Heuristik»).
# Console diagnostics stay English in both modes.
LANG_TEXT = {
    "en": {
        "trend_legend": TREND_LEGEND,
        "trend_binary_legend": TREND_BINARY_LEGEND,
        "vol_legend": VOL_LEGEND,
        "rule_lane": "Rule (ADX/PE)",
        "algo_display": {},  # fallback: algo.upper()
        "k_prefix": "k",
        "price_ylabel": "EUR/USD",
        "price_ylabel_line": "EUR/USD close",
        "title": "EUR/USD Daily — regime comparison ({dim}), "
                 "{start:%Y-%m-%d} → {end:%Y-%m-%d}",
        "dim_labels": {"trend": "trend", "trendrange": "trend vs range",
                       "vol": "vol"},
        "footnote": "Blank lane segments: regime undefined "
                    "(indicator warm-up / no model coverage).",
    },
    "de": {
        "trend_legend": {"UPTREND": "Aufwärtstrend", "RANGE": "Seitwärtsphase",
                         "DOWNTREND": "Abwärtstrend"},
        "trend_binary_legend": {"TREND": "Trendphase", "RANGE": "Seitwärtsphase"},
        "vol_legend": {"HIGH_VOL": "hohe Volatilität",
                       "MED_VOL": "mittlere Volatilität",
                       "LOW_VOL": "niedrige Volatilität"},
        "rule_lane": "Heuristik",
        "algo_display": {"hmm": "HMM", "gmm": "GMM", "kmeans": "k-Means"},
        "k_prefix": "K",
        "price_ylabel": "EUR/USD",
        "price_ylabel_line": "EUR/USD Schlusskurs",
        "title": "EUR/USD Daily — Regime-Vergleich ({dim}), "
                 "{start:%Y-%m-%d} → {end:%Y-%m-%d}",
        "dim_labels": {"trend": "Trend",
                       "trendrange": "Trendphase vs. Seitwärtsphase",
                       "vol": "Volatilität"},
        "footnote": "Leere Lane-Abschnitte: Regime undefiniert "
                    "(Indikator-Warm-up / keine Modellabdeckung).",
    },
}


def _undefined_mask(regime_df: pd.DataFrame, dimension: str) -> pd.Series:
    """Bars where the source has no defined regime yet (warm-up / no coverage).

    generate_regime_labels* label such bars RANGE/MED_VOL because their trend
    condition evaluates False on NaN — for a chart that would fabricate a
    regime, so these bars are masked and rendered blank instead.
    """
    if dimension == "vol":
        return regime_df["volatility_percentile"].isna()
    if "rgm_trend_score" in regime_df.columns:
        return regime_df["rgm_trend_score"].isna()
    return regime_df["adx"].isna() & regime_df["price_efficiency"].isna()


def regime_codes(regime_df: pd.DataFrame, dimension: str,
                 direction_aware: bool = True) -> pd.Series:
    """Map a regime frame to float category codes; NaN = regime undefined.

    dimension 'trend' → codes over TREND_ORDER (direction_aware=True labels) or
    TREND_BINARY_ORDER (direction_aware=False, trend vs range only); the frame's
    labels must have been generated with the SAME direction_aware setting.
    dimension 'vol' → codes over VOL_ORDER.
    """
    if dimension == "trend":
        labels = regime_df["regime_trend_label"]
        order = TREND_ORDER if direction_aware else TREND_BINARY_ORDER
    elif dimension == "vol":
        labels, order = regime_df["regime_volatility_label"], VOL_ORDER
    else:
        raise ValueError(f"Unknown dimension '{dimension}' (use 'trend' or 'vol').")
    codes = labels.map({c: i for i, c in enumerate(order)}).astype(float)
    codes[_undefined_mask(regime_df, dimension)] = np.nan
    return codes


def _read_meta(algo: str) -> dict | None:
    meta_path = regime_meta_path("daily", algo)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not read {meta_path.name}: {exc}", file=sys.stderr)
        return None


def build_lanes(df_daily: pd.DataFrame, algos, trend_threshold: float = 0.15,
                direction_aware: bool = True, lang: str = "en"):
    """Return [(lane_name, regime_df), ...] — rule-based first, then one per algo.

    Every lane uses the same direction_aware setting so all sources share one
    label contract (UPTREND/DOWNTREND/RANGE, or TREND/RANGE when collapsed). A
    missing regime CSV skips that lane with a warning instead of failing the
    whole chart. ``lang`` selects the lane naming (see LANG_TEXT); the default
    'en' keeps the historical names.
    """
    text = LANG_TEXT[lang]
    daily = df_daily[["high", "low", "close"]].copy()
    lanes = [(text["rule_lane"],
              generate_regime_labels(daily, direction_aware=direction_aware))]

    for algo in algos:
        csv_path = regime_csv_path("daily", algo)
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — skipping lane '{algo}'. "
                  f"Run: python data/update_regime_model_data.py --algo {algo}",
                  file=sys.stderr)
            continue
        raw = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
        raw = raw.reindex(df_daily.index)
        scores = generate_regime_labels_from_scores(
            raw, trend_threshold=trend_threshold, direction_aware=direction_aware
        )
        name = text["algo_display"].get(algo, algo.upper())
        meta = _read_meta(algo)
        if meta:
            name = f"{name} ({text['k_prefix']}={meta.get('n_states', '?')})"
            print(f"  {algo.upper()}: n_states={meta.get('n_states')}, "
                  f"train_end={meta.get('train_end')}, written {meta.get('written_at')}")
        lanes.append((name, scores))
    return lanes


def slice_window(index: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    """Window sub-index [start, end]; raises when the window misses the data."""
    window = index[(index >= start) & (index <= end)]
    if len(window) == 0:
        raise ValueError(
            f"Window {start:%Y-%m-%d}..{end:%Y-%m-%d} contains no bars "
            f"(data covers {index.min():%Y-%m-%d}..{index.max():%Y-%m-%d})."
        )
    return window


def lane_statistics(lane_codes, categories):
    """Per-lane category shares (over defined bars) + agreement vs the first lane.

    lane_codes: [(name, codes Series over the SAME window index), ...].
    Agreement counts only bars where both lanes have a defined regime.
    """
    _, base = lane_codes[0]
    rows = []
    for name, codes in lane_codes:
        valid = codes.dropna()
        n = len(valid)
        row = {"lane": name, "n_defined": n}
        for i, cat in enumerate(categories):
            row[cat] = float((valid == i).mean()) if n else float("nan")
        both = codes.notna() & base.notna()
        row["agreement_vs_rule"] = (
            float((codes[both] == base[both]).mean()) if both.any() else float("nan")
        )
        rows.append(row)
    return rows


def _print_statistics(rows, categories) -> None:
    print("\nRegime shares in window (defined bars only) + agreement vs rule lane:")
    header = f"  {'lane':<18} {'n':>6}" + "".join(f"{c:>12}" for c in categories) + f"{'agree':>9}"
    print(header)
    for row in rows:
        line = f"  {row['lane']:<18} {row['n_defined']:>6}"
        for cat in categories:
            line += f"{row[cat]:>12.1%}"
        line += f"{row['agreement_vs_rule']:>9.1%}"
        print(line)


def _draw_price(ax, ohlc: pd.DataFrame, x: np.ndarray, style: str) -> None:
    """Draw the price panel: 'candle' sticks (default), 'ohlc' bars or a 'line'.

    Bars are centred in their lane cell (x + 0.5); x units are matplotlib
    date numbers, i.e. days.
    """
    if style == "line":
        ax.plot(x, ohlc["close"].to_numpy(), color=PRICE_COLOR, linewidth=1.6)
        return

    o = ohlc["open"].to_numpy(dtype=float)
    h = ohlc["high"].to_numpy(dtype=float)
    l = ohlc["low"].to_numpy(dtype=float)
    c = ohlc["close"].to_numpy(dtype=float)
    xc = x + 0.5
    colors = np.where(c >= o, BAR_UP, BAR_DOWN)

    # high-low range line, shared by both bar styles
    range_segs = np.stack([np.column_stack([xc, l]), np.column_stack([xc, h])], axis=1)
    ax.add_collection(LineCollection(range_segs, colors=colors, linewidths=0.9))

    if style == "ohlc":
        tick = 0.32
        open_segs = np.stack(
            [np.column_stack([xc - tick, o]), np.column_stack([xc, o])], axis=1)
        close_segs = np.stack(
            [np.column_stack([xc, c]), np.column_stack([xc + tick, c])], axis=1)
        ax.add_collection(LineCollection(open_segs, colors=colors, linewidths=0.9))
        ax.add_collection(LineCollection(close_segs, colors=colors, linewidths=0.9))
    else:  # candle
        body_bottom = np.minimum(o, c)
        body_height = np.abs(c - o)
        # a doji body of literal zero height would vanish — give it a hairline
        min_body = 0.03 * float(np.nanmedian(h - l)) if len(h) else 0.0
        body_height = np.maximum(body_height, min_body)
        ax.bar(xc, body_height, bottom=body_bottom, width=0.62,
               color=colors, linewidth=0)

    pad = 0.03 * float(np.nanmax(h) - np.nanmin(l) or 1.0)
    ax.set_ylim(float(np.nanmin(l)) - pad, float(np.nanmax(h)) + pad)


def plot_regime_comparison(df_daily, lanes, dimension, start, end, out_path,
                           dpi=150, price_style="candle", direction_aware=True,
                           lang="en", show_title=True):
    """Render the price panel + one regime lane per source and save the PNG.

    ``lang`` selects legend/title/footnote wording (LANG_TEXT); ``show_title``
    False omits the embedded title for figures whose caption lives in the
    embedding document. Defaults reproduce the historical output.
    """
    text = LANG_TEXT[lang]
    if dimension == "trend" and direction_aware:
        order, colors, legend = TREND_ORDER, TREND_COLORS, text["trend_legend"]
        dim_label = text["dim_labels"]["trend"]
    elif dimension == "trend":
        order, colors, legend = (TREND_BINARY_ORDER, TREND_BINARY_COLORS,
                                 text["trend_binary_legend"])
        dim_label = text["dim_labels"]["trendrange"]
    else:
        order, colors, legend = VOL_ORDER, VOL_COLORS, text["vol_legend"]
        dim_label = text["dim_labels"]["vol"]

    window = slice_window(df_daily.index, start, end)
    ohlc = df_daily.loc[window, ["open", "high", "low", "close"]]

    x = mdates.date2num(window.to_pydatetime())
    edges = np.append(x, x[-1] + 1.0)  # each cell spans to the next bar

    if price_style != "line" and len(window) > 2000:
        print(f"NOTE: {len(window)} bars in the window — OHLC bars will be dense; "
              f"consider --price-style line for long windows.")

    cmap = ListedColormap([colors[c] for c in order])
    cmap.set_bad(SURFACE)

    n_lanes = len(lanes)
    fig, axes = plt.subplots(
        n_lanes + 1, 1, sharex=True,
        figsize=(14, 4.2 + 0.55 * n_lanes),
        gridspec_kw={"height_ratios": [6] + [1] * n_lanes, "hspace": 0.14},
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)

    # --- price panel ---------------------------------------------------------
    ax_price = axes[0]
    ax_price.set_facecolor(SURFACE)
    _draw_price(ax_price, ohlc, x, price_style)
    ax_price.set_ylabel(text["price_ylabel"] if price_style != "line"
                        else text["price_ylabel_line"],
                        fontsize=10, color=INK_SECONDARY)
    ax_price.grid(True, color=GRID, linewidth=0.6)
    ax_price.set_axisbelow(True)
    for side in ("top", "right"):
        ax_price.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_price.spines[side].set_color(AXIS)
    ax_price.tick_params(colors=INK_MUTED, labelsize=9)
    if show_title:
        ax_price.set_title(
            text["title"].format(dim=dim_label, start=start, end=end),
            loc="left", fontsize=12, color=INK_PRIMARY, pad=12,
        )
    handles = [Patch(facecolor=colors[c], label=legend[c]) for c in order]
    ax_price.legend(handles=handles, loc="upper left", ncol=len(order),
                    frameon=True, framealpha=0.85, facecolor=SURFACE,
                    edgecolor=GRID, fontsize=9, labelcolor=INK_SECONDARY)

    # --- regime lanes --------------------------------------------------------
    any_blank = False
    for ax, (name, regime_df) in zip(axes[1:], lanes):
        codes = regime_codes(regime_df, dimension,
                             direction_aware=direction_aware).reindex(window)
        any_blank = any_blank or bool(codes.isna().any())
        data = np.ma.masked_invalid(codes.to_numpy(dtype=float)[None, :])
        ax.pcolormesh(edges, [0.0, 1.0], data, cmap=cmap,
                      vmin=-0.5, vmax=len(order) - 0.5)
        ax.set_facecolor(SURFACE)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(name, rotation=0, ha="right", va="center",
                      fontsize=9, color=INK_SECONDARY, labelpad=8)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlim(edges[0], edges[-1])

    # The footnote explains blank segments — with none in the window it would
    # only describe something the reader cannot see, so it is drawn on demand.
    if any_blank:
        fig.text(0.01, 0.005, text["footnote"], fontsize=8, color=INK_MUTED)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EUR/USD daily chart with rule/HMM/GMM/KMeans regime lanes.")
    parser.add_argument("--start", default=None,
                        help="Window start, ISO date (default: first available bar).")
    parser.add_argument("--end", default=None,
                        help="Window end, ISO date (default: last available bar).")
    parser.add_argument("--dimension", choices=["trend", "vol"], default="trend",
                        help="Which regime dimension the lanes show (default: trend).")
    parser.add_argument("--no-direction", action="store_true",
                        help="Collapse Uptrend/Downtrend into a single Trend "
                             "category (direction_aware=False) — lanes show only "
                             "trend vs range. Only affects --dimension trend.")
    parser.add_argument("--trend-threshold", type=float, default=0.15,
                        help="|rgm_trend_score| above which an ML bar counts as trending "
                             "(default 0.15 = advanced_train default; ~0.3 separates the "
                             "neutral state cleanly on the 2026-09-12 obs).")
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS,
                        choices=DEFAULT_ALGOS,
                        help="ML regime models to include (default: hmm gmm kmeans).")
    parser.add_argument("--price-style", choices=["candle", "ohlc", "line"],
                        default="candle",
                        help="Price panel style: candlesticks (default, green up / "
                             "red down), OHLC bars, or a close line (recommended "
                             "for very long windows).")
    parser.add_argument("--lang", choices=sorted(LANG_TEXT), default="en",
                        help="Chart language: legend, lane names, title, footnote "
                             "(default en; de uses the thesis terminology).")
    parser.add_argument("--no-title", action="store_true",
                        help="Omit the embedded chart title — for figures whose "
                             "caption lives in the embedding document.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: ModelTrading/generated/"
                             "regime_comparison_<dimension>_<start>_<end>.png).")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    ohlc_path = os.path.join(dir_config.DATA_DIR, "eurusd_daily.csv")
    df_daily = csv_utils.load_csv(ohlc_path, filter_weekends_flag=True)

    start = pd.Timestamp(args.start) if args.start else df_daily.index.min()
    end = pd.Timestamp(args.end) if args.end else df_daily.index.max()
    if start > end:
        parser.error(f"--start {start:%Y-%m-%d} is after --end {end:%Y-%m-%d}")

    direction_aware = not args.no_direction

    print("Building regime lanes (full history, then sliced to the window)...")
    lanes = build_lanes(df_daily, args.algos, trend_threshold=args.trend_threshold,
                        direction_aware=direction_aware, lang=args.lang)

    dim_tag = args.dimension if direction_aware or args.dimension != "trend" \
        else "trendrange"
    out_path = args.out or os.path.join(
        dir_config.GENERATED_DIR,
        f"regime_comparison_{dim_tag}_{start:%Y%m%d}_{end:%Y%m%d}.png",
    )
    plot_regime_comparison(df_daily, lanes, args.dimension, start, end,
                           out_path, dpi=args.dpi, price_style=args.price_style,
                           direction_aware=direction_aware, lang=args.lang,
                           show_title=not args.no_title)

    window = slice_window(df_daily.index, start, end)
    if args.dimension == "trend":
        categories = TREND_ORDER if direction_aware else TREND_BINARY_ORDER
    else:
        categories = VOL_ORDER
    lane_codes = [(name,
                   regime_codes(df, args.dimension,
                                direction_aware=direction_aware).reindex(window))
                  for name, df in lanes]
    _print_statistics(lane_statistics(lane_codes, categories), categories)

    print(f"\nChart written to: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
