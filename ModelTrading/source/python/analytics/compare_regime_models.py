"""
Compare data-driven regime models (HMM / GMM / KMeans) for EUR/USD.

For each algorithm this fits the model across a range of regime counts, reports
the model-selection criteria (BIC/AIC for hmm/gmm, silhouette for kmeans, plus
inertia), and — for the selected k — reports two economic sanity checks:

  * State persistence: average dwell time (bars) per state. Real regimes should
    persist; a model whose states flip every bar is not capturing regimes.
  * State semantics: realized forward return and forward volatility per state,
    confirming that discovered states separate trending vs. ranging / calm vs.
    volatile behaviour.

It ends with a recommended (algo, k) so the precompute script can be run with
those settings:

  python data/update_regime_model_data.py --algo <algo> --n-states <k>

Usage:
  python analytics/compare_regime_models.py                    # daily, k=2..6
  python analytics/compare_regime_models.py --k-min 2 --k-max 8
  python analytics/compare_regime_models.py --train-end 2025-09-30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import ModelTrading.config.directories as dir_config
import ModelTrading.config.timeframes as tf_config
from ModelTrading.source.python.utils import csv as csv_utils
from ModelTrading.source.python.features.config import get_feature_config
from ModelTrading.source.python.features import regime_model as rgm

_TIMEFRAME_CSV = {"m15": "eurusd_m15.csv", "4hours": "eurusd_4hours.csv", "daily": "eurusd_daily.csv"}


def _avg_dwell_time(labels: np.ndarray) -> float:
    """Average consecutive-run length of the label sequence (in bars)."""
    labels = labels[~np.isnan(labels)]
    if len(labels) == 0:
        return float("nan")
    changes = np.count_nonzero(np.diff(labels) != 0) + 1
    return len(labels) / changes


def _print_selection_table(algo: str, sel: dict) -> None:
    print(f"\n=== {algo.upper()} - model selection ===")
    keys = ["bic", "aic", "silhouette", "inertia", "loglik"]
    header = "  k  " + "".join(f"{k:>14}" for k in keys)
    print(header)
    for k, s in sorted(sel["scores"].items()):
        if "error" in s:
            print(f"  {k:<3}  (failed: {s['error'][:40]})")
            continue
        row = f"  {k:<3}"
        for key in keys:
            v = s.get(key)
            row += f"{v:>14.3f}" if isinstance(v, (int, float)) else f"{'-':>14}"
        print(row)
    print(f"  -> best k by {sel['criterion']}: {sel['best_k']}")


def _report_economics(obs: pd.DataFrame, df: pd.DataFrame, algo: str, k: int, seed: int) -> None:
    model = rgm.fit_regime_model(obs, algo=algo, n_states=k, seed=seed)
    inferred = model.infer(obs)
    labels = inferred["rgm_label"].to_numpy(dtype=float)

    # Forward 1-bar return and its magnitude, aligned to the state at bar t.
    fwd_ret = np.log(df["close"].shift(-1) / df["close"])
    tbl = pd.DataFrame({
        "label": labels,
        "fwd_ret": fwd_ret.to_numpy(),
        "trend_score": inferred["rgm_trend_score"].to_numpy(dtype=float),
        "vol_score": inferred["rgm_vol_score"].to_numpy(dtype=float),
    }, index=obs.index).dropna()

    print(f"\n=== {algo.upper()} (k={k}) - state economics ===")
    print(f"  avg dwell time: {_avg_dwell_time(labels):.1f} bars")
    print(f"  {'state':>5} {'n':>7} {'trend_sc':>9} {'vol_sc':>8} "
          f"{'fwd_ret_bp':>11} {'fwd_vol_bp':>11}")
    for s in sorted(tbl["label"].unique()):
        g = tbl[tbl["label"] == s]
        print(f"  {int(s):>5} {len(g):>7} {g['trend_score'].mean():>9.3f} "
              f"{g['vol_score'].mean():>8.3f} {g['fwd_ret'].mean()*1e4:>11.2f} "
              f"{g['fwd_ret'].std()*1e4:>11.2f}")


def main():
    parser = argparse.ArgumentParser(description="Compare HMM/GMM/KMeans regime models.")
    parser.add_argument("--timeframe", choices=sorted(_TIMEFRAME_CSV), default="daily")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--train-end", default=None,
                        help="Fit only on bars up to this ISO date (default: TRAIN_END).")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = get_feature_config()
    params = config.get_parameters()
    seed = args.seed if args.seed is not None else int(params.get("rgm_seed", 42))
    train_end = args.train_end if args.train_end is not None else tf_config.TRAIN_END

    ohlc_path = Path(dir_config.DATA_DIR) / _TIMEFRAME_CSV[args.timeframe]
    df = csv_utils.load_csv(str(ohlc_path), start_date=None, end_date=None, filter_weekends_flag=True)
    obs_full = rgm.build_observations(df, params)

    obs = obs_full
    if train_end is not None:
        obs = obs_full[obs_full.index <= pd.Timestamp(train_end)]
    print(f"Regime model comparison - {args.timeframe}, "
          f"fit on {int(obs.dropna().shape[0])} bars (<= {pd.Timestamp(train_end):%Y-%m-%d})")

    k_range = range(args.k_min, args.k_max + 1)
    recommended = {}
    for algo in rgm.RGM_ALGOS:
        sel = rgm.select_num_regimes(obs, algo, k_range, seed=seed)
        _print_selection_table(algo, sel)
        if sel["best_k"] is not None:
            _report_economics(obs, df.loc[obs.index], algo, sel["best_k"], seed)
            recommended[algo] = sel["best_k"]

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    for algo, k in recommended.items():
        print(f"  {algo:>7}: k={k}  ->  "
              f"python data/update_regime_model_data.py --algo {algo} --n-states {k}")
    print("Pick the algo whose states persist and separate fwd return/vol most cleanly.")


if __name__ == "__main__":
    main()
