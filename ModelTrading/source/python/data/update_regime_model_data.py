"""
Fit the ML regime model and pre-compute its features for training.

The regime model (Gaussian HMM / GMM / KMeans over stationary daily
observations) is fitted once on the training window and then run with *causal*
inference over the full history. The result is persisted to a CSV in
ModelTrading/data/ and the fitted model to ModelTrading/generated/:

  data/regime_daily_{algo}.csv   — one row per bar, columns:
      date, rgm_prob_s0 … rgm_prob_s{k-1}, rgm_label,
      rgm_trend_score, rgm_vol_score, rgm_conf
  data/regime_daily.csv          — identical copy under the unsuffixed "active"
      name, for the live path and for configs that do not pin an algorithm
  data/regime_daily{_algo}.meta.json — provenance: algo, n_states, seed,
      train_end, fit bars, per-state trend/vol scores
  generated/regime_model_daily{_algo}.pkl — the fitted RegimeModel (used live)

Both names are written on every run. The algo-suffixed files let several
algorithms coexist so a single walk-forward run can compare them — a shared
unsuffixed CSV cannot, and an experiment that relied on overwriting it between
runs produced two arms with byte-identical labels. Which arm reads which file is
selected by ``parameters.rgm_algo`` in the feature config
(``features-rgm-{algo}.yaml``).

Consumption
-----------
  * Training (advanced_train / train): indicators.add_features(compute_regime=False)
    loads regime_<tf>.csv as features.
  * Live inference (feature_server): indicators.add_features(compute_regime=True)
    loads generated/regime_model_<tf>.pkl and infers on the fly — the CSV is
    training-only and is NOT staged to TEST/PRODUCTION.

Why full recompute (not incremental like TimesFM)
-------------------------------------------------
HMM filtering posteriors depend on the whole preceding sequence, so the cheap
"append new bars only" trick does not apply. Daily inference over ~5k bars is
fast, so each run recomputes the full history and rewrites the CSV. The model
is refitted only on bars up to --train-end so the fit never sees test data.

Usage:
  python data/update_regime_model_data.py                       # daily, defaults
  python data/update_regime_model_data.py --algo gmm --n-states 4
  python data/update_regime_model_data.py --train-end 2025-09-30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project root on sys.path so "ModelTrading" package is importable.
# parents: [0]=data/, [1]=python/, [2]=source/, [3]=ModelTrading/, [4]=project root
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import ModelTrading.config.directories as dir_config
import ModelTrading.config.timeframes as tf_config
from ModelTrading.source.python.utils import csv as csv_utils
from ModelTrading.source.python.features.config import get_feature_config
from ModelTrading.source.python.features import regime_model as rgm

# Timeframe -> OHLC CSV filename. Mirrors advanced_train.load_and_prepare_data.
_TIMEFRAME_CSV = {
    "m15":    "eurusd_m15.csv",
    "4hours": "eurusd_4hours.csv",
    "daily":  "eurusd_daily.csv",
}


def _output_columns(n_states: int) -> list:
    return ([f"rgm_prob_s{i}" for i in range(n_states)]
            + ["rgm_label", "rgm_trend_score", "rgm_vol_score", "rgm_conf"])


def compute_timeframe(timeframe, config, algo, n_states, seed,
                      train_end=None, verbose=True) -> bool:
    """Fit + infer + persist regime features for one timeframe. Returns success."""
    csv_name = _TIMEFRAME_CSV[timeframe]
    ohlc_path = Path(dir_config.DATA_DIR) / csv_name
    if not ohlc_path.exists():
        print(f"  ERROR [{timeframe}]: OHLC CSV not found: {ohlc_path}", file=sys.stderr)
        return False

    if verbose:
        print(f"  [{timeframe}] loading {csv_name} ...")

    df = csv_utils.load_csv(str(ohlc_path), start_date=None, end_date=None,
                            filter_weekends_flag=True)

    params = config.get_parameters()
    obs = rgm.build_observations(df, params)

    # Fit only on bars up to train_end so the model never sees test data.
    fit_obs = obs
    if train_end is not None:
        fit_obs = obs[obs.index <= pd.Timestamp(train_end)]
    n_valid = int(fit_obs.dropna().shape[0])
    if verbose:
        te = f" (fit <= {pd.Timestamp(train_end):%Y-%m-%d})" if train_end is not None else ""
        print(f"  [{timeframe}] fitting {algo} with {n_states} states on {n_valid} bars{te} ...")

    model = rgm.fit_regime_model(fit_obs, algo=algo, n_states=n_states, seed=seed)

    # Causal inference over the FULL history (fit params fixed).
    if verbose:
        print(f"  [{timeframe}] causal inference over {len(obs)} bars ...")
    out = model.infer(obs)

    out = out[_output_columns(n_states)].copy()
    for col in out.columns:
        out[col] = out[col].astype(np.float32)
    out.index.name = "date"

    # Persist under the algo-suffixed name AND the unsuffixed one. The suffixed
    # files let several algorithms coexist, so one walk-forward run can compare
    # them; the unsuffixed copy stays the "active" model for the live path and
    # for any config that does not pin an algorithm.
    meta = {
        "algo": algo,
        "n_states": int(n_states),
        "seed": int(seed),
        "train_end": None if train_end is None else f"{pd.Timestamp(train_end):%Y-%m-%d}",
        "fit_bars": n_valid,
        "total_bars": int(len(out)),
        "trend_scores": [float(x) for x in model.trend_scores],
        "vol_scores": [float(x) for x in model.vol_scores],
        "written_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    }

    written = []
    for algo_suffix in (algo, None):
        model_path = rgm.regime_model_path(timeframe, algo_suffix)
        rgm.save_regime_model(model, model_path)

        out_path = rgm.regime_csv_path(timeframe, algo_suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=True, date_format="%Y-%m-%d %H:%M:%S")

        meta_path = rgm.regime_meta_path(timeframe, algo_suffix)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written.append((out_path, model_path))

    if verbose:
        valid = int(out.iloc[:, 0].notna().sum())
        dist = out["rgm_label"].dropna().value_counts().sort_index()
        dist_str = ", ".join(f"s{int(s)}:{int(c)}" for s, c in dist.items())
        names = " + ".join(p.name for p, _ in written)
        print(f"  [{timeframe}] saved {len(out)} rows ({valid} valid) -> {names}")
        print(f"  [{timeframe}] fitted model -> {written[0][1].name}  | label dist: {dist_str}")
        print(f"  [{timeframe}] trend_scores: {np.round(model.trend_scores, 4).tolist()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fit the ML regime model and persist its features to data/."
    )
    parser.add_argument("--timeframes", nargs="+", choices=sorted(_TIMEFRAME_CSV),
                        default=["daily"],
                        help="Which timeframes to compute (default: daily).")
    parser.add_argument("--algo", choices=list(rgm.RGM_ALGOS), default=None,
                        help="Clustering algorithm (default: features.yaml rgm_algo).")
    parser.add_argument("--n-states", type=int, default=None,
                        help="Number of regimes (default: features.yaml rgm_n_states).")
    parser.add_argument("--train-end", default=None,
                        help="Fit the model only on bars up to this ISO date "
                             "(default: config.timeframes.TRAIN_END).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: features.yaml rgm_seed or 42).")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    args = parser.parse_args()

    verbose = not args.quiet
    config = get_feature_config()
    params = config.get_parameters()

    algo = args.algo or params.get("rgm_algo", "hmm")
    n_states = args.n_states if args.n_states is not None else int(params.get("rgm_n_states", 4))
    seed = args.seed if args.seed is not None else int(params.get("rgm_seed", 42))
    train_end = args.train_end if args.train_end is not None else tf_config.TRAIN_END

    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Regime model: algo={algo}, "
              f"n_states={n_states}, timeframes={args.timeframes}")

    errors = 0
    for timeframe in args.timeframes:
        try:
            ok = compute_timeframe(timeframe, config, algo, n_states, seed,
                                   train_end=train_end, verbose=verbose)
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"  ERROR [{timeframe}]: {exc}", file=sys.stderr)
            ok = False
        errors += 0 if ok else 1

    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Done: "
              f"{len(args.timeframes) - errors} ok, {errors} failed.")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
