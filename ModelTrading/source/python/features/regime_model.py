"""
ML-based market-regime features for forex trading.

A data-driven regime model identifies latent market states of EUR/USD from a
small vector of stationary daily observations (rolling-mean drift, realized
volatility, ADX, price-efficiency) and emits, per bar:

  * ``rgm_prob_s0 … rgm_prob_s{k-1}`` — posterior state probabilities (sum→1)
  * ``rgm_label``       — argmax discrete state (0 … k-1)
  * ``rgm_trend_score`` — probability-weighted signed trend intensity  (∈[-1, 1])
  * ``rgm_vol_score``   — probability-weighted volatility level         (∈[ 0, 1])
  * ``rgm_conf``        — posterior confidence = max prob               (∈[ 0, 1])

Three algorithms are supported and compared during research (see
``analytics/compare_regime_models.py``): a Gaussian ``hmm`` (captures temporal
persistence), a ``gmm`` (Gaussian mixture), and ``kmeans``.

This module mirrors ``features/timesfm_features.py``: an expensive model whose
outputs become features, fitted once and consumed two ways —

  * Training (``advanced_train`` / ``train``): the regime features are computed
    once upfront by ``data/update_regime_model_data.py`` and persisted to
    ``data/regime_daily.csv`` (and the fitted model to ``generated/``).
    ``get_regime_features(..., compute=False)`` loads those columns.
  * Live inference (``feature_server``): the precomputed CSV does not cover live
    bars, so ``get_regime_features(..., compute=True)`` loads the persisted
    fitted model and runs *causal* inference on the bars Java sends.

Correctness
-----------
* No lookahead: HMM inference uses online **filtering** (forward algorithm),
  never smoothing/Viterbi (which peek at future bars). GMM/KMeans are pointwise
  and naturally causal. Observations use only rolling windows ending at bar t.
* Fitting the model on the training window is acceptable (like fitting a
  scaler); the per-bar state assignment is causal.
* Values are written UNSHIFTED — ``indicators.py`` applies ``shift(1)`` exactly
  as it does for TimesFM features.
* State numbering is made deterministic by ordering states by their semantic
  (trend_score, vol_score) so ``rgm_label``/``rgm_prob_s*`` are stable.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features.indicators import (
        calculate_trend_strength,
        calculate_price_efficiency,
        calculate_volatility_percentile,
    )
except ImportError:  # pragma: no cover - import path depends on caller
    from ModelTrading.source.python.features.indicators import (
        calculate_trend_strength,
        calculate_price_efficiency,
        calculate_volatility_percentile,
    )

# Observation columns fed to the regime model (order matters: it fixes the
# meaning of state-mean columns used for the trend/vol semantic mapping).
#
# Changed 2026-09-12, both measured on the daily fit (<= TRAIN_END, 5 seeds):
# * ``drift`` (rolling-mean log-return, window ``rgm_obs_drift_period``)
#   replaces the 1-day ``log_return`` as the direction dimension. Daily
#   returns are too noisy for the states to separate on, so EM organised the
#   states around volatility instead (k=2: vol_scores [1.0, 0.0]) and no
#   neutral state existed even at k=3 (trend_scores [-1.0, +0.19, +0.70],
#   kappa vs. the rule-based trend/range split 0.12).
# * ``atr_percentile`` was dropped: with two volatility dimensions the fit is
#   seed-UNSTABLE (3 distinct solutions over 5 seeds, kappa 0.38 -> -0.07);
#   with one, 4/5 seeds converge to the same solution with a genuine neutral
#   state (trend_scores [-0.82, -0.09, +0.77], kappa 0.37).
OBS_COLUMNS = ("drift", "realized_vol", "adx", "price_efficiency")

# Still computed by build_observations so RegimeModels persisted BEFORE the
# 2026-09-12 observation change (their ``obs_columns`` include these) keep
# inferring; new fits never see them.
LEGACY_OBS_COLUMNS = ("log_return", "atr_percentile")

# Supported clustering algorithms.
RGM_ALGOS = ("hmm", "gmm", "kmeans")

# A de-prefixed regime feature name.
_RGM_NAME_RE = re.compile(r"^rgm_(prob_s\d+|label|trend_score|vol_score|conf)$")
_RGM_PROB_RE = re.compile(r"^rgm_prob_s(\d+)$")


# ---------------------------------------------------------------------------
# Feature-name helpers (mirror timesfm_features.is_tfm_feature / parse_*)
# ---------------------------------------------------------------------------
def is_rgm_feature(name: str) -> bool:
    """True if ``name`` (de-prefixed) is a regime-model feature."""
    return _RGM_NAME_RE.match(name) is not None


def parse_rgm_feature(name: str):
    """Parse a de-prefixed regime feature name.

    Returns the field string (``'prob_s0'``, ``'label'``, ``'trend_score'``,
    ``'vol_score'``, ``'conf'``) or ``None`` if the name is not a regime
    feature.
    """
    m = _RGM_NAME_RE.match(name)
    return m.group(1) if m else None


def prob_state_index(name: str):
    """Return the state index of a ``rgm_prob_s{i}`` name, else ``None``."""
    m = _RGM_PROB_RE.match(name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Paths (mirror timesfm_features._data_dir / timesfm_csv_path)
# ---------------------------------------------------------------------------
def _data_dir() -> Path:
    try:
        import ModelTrading.config.directories as _dir
        return Path(_dir.DATA_DIR)
    except ImportError:
        return Path(__file__).resolve().parents[4] / "ModelTrading" / "data"


def _generated_dir() -> Path:
    try:
        import ModelTrading.config.directories as _dir
        gen = getattr(_dir, "GENERATED_DIR", None)
        if gen:
            return Path(gen)
    except ImportError:
        pass
    return Path(__file__).resolve().parents[4] / "ModelTrading" / "generated"


def regime_csv_path(timeframe: str, algo: str | None = None) -> Path:
    """Path to the precomputed regime CSV for a timeframe.

    ``algo`` selects the per-algorithm file (``regime_daily_hmm.csv``), which is
    what every experiment must use: a single shared ``regime_daily.csv`` cannot
    hold three fitted models at once, so comparing hmm/gmm/kmeans in one run
    requires one file per algorithm. Omitting ``algo`` returns the unsuffixed
    "active" copy, which ``update_regime_model_data.py`` keeps writing for the
    live path.
    """
    suffix = f"_{algo}" if algo else ""
    return _data_dir() / f"regime_{timeframe}{suffix}.csv"


def regime_model_path(timeframe: str, algo: str | None = None) -> Path:
    """Path to the persisted fitted regime model for a timeframe.

    See :func:`regime_csv_path` for the meaning of ``algo``.
    """
    suffix = f"_{algo}" if algo else ""
    return _generated_dir() / f"regime_model_{timeframe}{suffix}.pkl"


def regime_meta_path(timeframe: str, algo: str | None = None) -> Path:
    """Path to the sidecar recording which model produced the regime files.

    Without it, a finished run cannot be attributed to an algorithm after the
    fact — the CSV carries no provenance, so a run that silently consumed
    another algorithm's file is indistinguishable from a correct one.
    """
    suffix = f"_{algo}" if algo else ""
    return _data_dir() / f"regime_{timeframe}{suffix}.meta.json"


# ---------------------------------------------------------------------------
# Observation matrix
# ---------------------------------------------------------------------------
def build_observations(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Build the stationary observation matrix for the regime model.

    Reuses the same indicator calculations as the rule-based regime labeller
    (``labeling/regime.py``) so the ML model observes the same signals. All
    columns are causal (rolling windows ending at bar t) and unshifted; the
    consumer applies ``shift(1)``. Warmup rows contain NaN.

    Returns a DataFrame aligned to ``df.index`` with columns ``OBS_COLUMNS``
    plus ``LEGACY_OBS_COLUMNS``: models persisted before the 2026-09-12
    observation change carry the legacy names in their ``obs_columns`` and
    :meth:`RegimeModel.infer` selects its own columns, so the superset keeps
    old ``.pkl`` files working while new fits use only ``OBS_COLUMNS``.
    """
    params = params or {}
    adx_period = int(params.get("rgm_obs_adx_period", 14))
    eff_period = int(params.get("rgm_obs_efficiency_period", 20))
    vol_period = int(params.get("rgm_obs_vol_period", 14))
    vol_lookback = int(params.get("rgm_obs_vol_lookback", 252))
    rv_period = int(params.get("rgm_realized_vol_period", 20))
    drift_period = int(params.get("rgm_obs_drift_period", 20))

    close = df["close"].astype(float)
    log_return = np.log(close / close.shift(1))
    drift = log_return.rolling(drift_period, min_periods=drift_period).mean()
    realized_vol = log_return.rolling(rv_period, min_periods=rv_period).std()
    adx = calculate_trend_strength(df, period=adx_period)
    efficiency = calculate_price_efficiency(df, period=eff_period)
    atr_pct = calculate_volatility_percentile(df, period=vol_period, lookback=vol_lookback) / 100.0

    obs = pd.DataFrame(
        {
            "drift": drift,
            "log_return": log_return,
            "realized_vol": realized_vol,
            "adx": adx,
            "price_efficiency": efficiency,
            "atr_percentile": atr_pct,
        },
        index=df.index,
    )
    return obs[list(OBS_COLUMNS) + list(LEGACY_OBS_COLUMNS)]


# ---------------------------------------------------------------------------
# Fitted regime model
# ---------------------------------------------------------------------------
class RegimeModel:
    """A fitted regime model + deterministic state semantics.

    Holds the standardiser, the clustering estimator (hmm/gmm/kmeans), and the
    per-state trend/vol scores in a canonical, stable state order. Provides
    causal ``infer`` producing the ``rgm_*`` feature columns.
    """

    def __init__(self, algo, n_states, obs_columns, scaler, estimator,
                 order, trend_scores, vol_scores, kmeans_temp=1.0):
        self.algo = algo
        self.n_states = int(n_states)
        self.obs_columns = list(obs_columns)
        self.scaler = scaler
        self.estimator = estimator
        # ``order``: original estimator state indices in canonical order.
        self.order = np.asarray(order, dtype=int)
        # trend/vol scores already reordered to canonical order.
        self.trend_scores = np.asarray(trend_scores, dtype=float)
        self.vol_scores = np.asarray(vol_scores, dtype=float)
        self.kmeans_temp = float(kmeans_temp)

    # --- raw (estimator-order) posterior probabilities -----------------
    def _raw_proba(self, Xs: np.ndarray) -> np.ndarray:
        if self.algo == "hmm":
            return _hmm_filter(self.estimator, Xs)
        if self.algo == "gmm":
            return self.estimator.predict_proba(Xs)
        if self.algo == "kmeans":
            return _kmeans_proba(self.estimator, Xs, temp=self.kmeans_temp)
        raise ValueError(f"Unknown algo '{self.algo}'")

    def infer(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Causal inference → DataFrame of ``rgm_*`` columns aligned to obs.index.

        Rows with any NaN observation (warmup) yield NaN for every output.
        """
        cols = ([f"rgm_prob_s{i}" for i in range(self.n_states)]
                + ["rgm_label", "rgm_trend_score", "rgm_vol_score", "rgm_conf"])
        out = pd.DataFrame(index=obs.index, columns=cols, dtype=np.float32)

        X = obs[self.obs_columns].to_numpy(dtype=float)
        valid = ~np.isnan(X).any(axis=1)
        if not valid.any():
            return out

        Xs = self.scaler.transform(X[valid])
        raw = self._raw_proba(Xs)                 # (m, k) in estimator order
        probs = raw[:, self.order]                # reorder to canonical states
        # Guard against tiny numerical drift so probabilities sum to 1.
        probs = probs / probs.sum(axis=1, keepdims=True)

        label = probs.argmax(axis=1).astype(np.float32)
        conf = probs.max(axis=1).astype(np.float32)
        trend = probs @ self.trend_scores
        vol = probs @ self.vol_scores

        vals = np.column_stack([probs, label, trend, vol, conf]).astype(np.float32)
        out.loc[valid, cols] = vals
        return out


# ---------------------------------------------------------------------------
# HMM causal filtering (forward algorithm) — no lookahead
# ---------------------------------------------------------------------------
def _gaussian_framelogprob(estimator, Xs: np.ndarray) -> np.ndarray:
    """Per-frame, per-state Gaussian log-likelihood (n, k)."""
    from scipy.stats import multivariate_normal
    means = estimator.means_
    covars = estimator.covars_  # (k, n_feat, n_feat) for GaussianHMM
    k = means.shape[0]
    return np.column_stack([
        multivariate_normal.logpdf(Xs, mean=means[s], cov=covars[s], allow_singular=True)
        for s in range(k)
    ])


def _hmm_filter(estimator, Xs: np.ndarray) -> np.ndarray:
    """Filtered (causal) state posteriors via the forward algorithm.

    Unlike ``predict_proba`` (forward-backward smoothing, which uses future
    observations), this uses only observations up to and including bar t.
    """
    from scipy.special import logsumexp
    n = Xs.shape[0]
    k = estimator.n_components
    log_startprob = np.log(estimator.startprob_ + 1e-300)
    log_transmat = np.log(estimator.transmat_ + 1e-300)
    framelp = _gaussian_framelogprob(estimator, Xs)  # (n, k)

    post = np.empty((n, k))
    log_alpha = log_startprob + framelp[0]
    post[0] = np.exp(log_alpha - logsumexp(log_alpha))
    for t in range(1, n):
        log_alpha = framelp[t] + logsumexp(log_alpha[:, None] + log_transmat, axis=0)
        post[t] = np.exp(log_alpha - logsumexp(log_alpha))
    return post


def _kmeans_proba(estimator, Xs: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Soft assignment from KMeans distances via softmax(-dist^2 / temp)."""
    d2 = estimator.transform(Xs) ** 2      # (n, k) squared distances to centers
    logits = -d2 / max(temp, 1e-8)
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Fitting + model selection
# ---------------------------------------------------------------------------
def _make_estimator(algo: str, n_states: int, seed: int):
    if algo == "hmm":
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM(
            n_components=n_states, covariance_type="full",
            n_iter=100, random_state=seed, tol=1e-3,
        )
    if algo == "gmm":
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(
            n_components=n_states, covariance_type="full",
            n_init=5, random_state=seed, max_iter=200,
        )
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_states, n_init=10, random_state=seed)
    raise ValueError(f"Unknown algo '{algo}' (expected one of {RGM_ALGOS})")


def _fit_estimator(algo, n_states, Xs, seed):
    est = _make_estimator(algo, n_states, seed)
    est.fit(Xs)
    return est


def _state_means_original(algo, estimator, scaler) -> np.ndarray:
    """Per-state observation means in ORIGINAL units (k, n_feat)."""
    if algo == "kmeans":
        centers = estimator.cluster_centers_
    else:
        centers = estimator.means_
    return scaler.inverse_transform(centers)


def _semantic_scores(state_means: np.ndarray):
    """Map per-state observation means → (order, trend_scores, vol_scores).

    * vol_score: min-max normalised ``realized_vol`` mean, ∈[0, 1].
    * trend_score: relative directional drift × trend quality, ∈[-1, 1].
    * order: original state indices sorted by (trend_score, vol_score) so the
      canonical labelling is deterministic across refits.

    The direction uses each state's mean drift **centred on the cross-state
    average**. Taking the raw ``sign(mean drift)`` collapses whenever every
    state shares the sign of the unconditional drift: for EUR/USD 2005-2025 all
    state means are negative, which made trend_score ≤ 0 on 100% of bars and
    stripped the feature of every bit of directional information. Centring asks
    "how bullish is this state *relative to the other regimes*", which is what a
    regime feature should express.

    Trend quality scales by ``eff / max(eff)`` ∈ (0, 1] rather than a min-max,
    which would zero out the weakest state entirely and discard its direction.
    """
    idx = {c: i for i, c in enumerate(OBS_COLUMNS)}
    ret = state_means[:, idx["drift"]]
    rv = state_means[:, idx["realized_vol"]]
    eff = state_means[:, idx["price_efficiency"]]

    def _minmax(a):
        lo, hi = np.min(a), np.max(a)
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    vol_scores_raw = _minmax(rv)

    ret_centred = ret - ret.mean()
    denom = np.max(np.abs(ret_centred))
    direction = ret_centred / denom if denom > 0 else np.zeros_like(ret_centred)
    eff_max = np.max(np.abs(eff))
    quality = eff / eff_max if eff_max > 0 else np.ones_like(eff)
    trend_scores_raw = np.clip(direction * quality, -1.0, 1.0)

    order = np.lexsort((vol_scores_raw, trend_scores_raw))  # sort by trend, then vol
    return order, trend_scores_raw[order], vol_scores_raw[order]


def fit_regime_model(obs: pd.DataFrame, algo: str, n_states: int,
                     seed: int = 42, kmeans_temp: float = 1.0) -> RegimeModel:
    """Fit a regime model on the (warmup-dropped) observation matrix."""
    from sklearn.preprocessing import StandardScaler

    if algo not in RGM_ALGOS:
        raise ValueError(f"Unknown algo '{algo}' (expected one of {RGM_ALGOS})")

    X = obs[list(OBS_COLUMNS)].dropna().to_numpy(dtype=float)
    if len(X) < n_states * 10:
        raise ValueError(f"Not enough valid observations ({len(X)}) to fit {n_states} states.")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    estimator = _fit_estimator(algo, n_states, Xs, seed)

    state_means = _state_means_original(algo, estimator, scaler)
    order, trend_scores, vol_scores = _semantic_scores(state_means)

    return RegimeModel(
        algo=algo, n_states=n_states, obs_columns=OBS_COLUMNS,
        scaler=scaler, estimator=estimator, order=order,
        trend_scores=trend_scores, vol_scores=vol_scores, kmeans_temp=kmeans_temp,
    )


def _score_fit(algo, estimator, Xs) -> dict:
    """Return selection scores for a fitted estimator (lower BIC/AIC = better)."""
    scores = {}
    n = len(Xs)
    if algo == "gmm":
        scores["bic"] = estimator.bic(Xs)
        scores["aic"] = estimator.aic(Xs)
        scores["loglik"] = estimator.score(Xs) * n
    elif algo == "hmm":
        # hmmlearn exposes .bic()/.aic() from 0.3; fall back to manual if absent.
        loglik = estimator.score(Xs)
        scores["loglik"] = loglik
        try:
            scores["bic"] = estimator.bic(Xs)
            scores["aic"] = estimator.aic(Xs)
        except Exception:  # pragma: no cover - older hmmlearn
            k = estimator.n_components
            n_feat = Xs.shape[1]
            n_params = (k - 1) + k * (k - 1) + k * n_feat + k * n_feat * (n_feat + 1) / 2
            scores["bic"] = -2 * loglik + n_params * np.log(n)
            scores["aic"] = -2 * loglik + 2 * n_params
    elif algo == "kmeans":
        scores["inertia"] = float(estimator.inertia_)
    # silhouette (all algos): higher = better separated clusters
    try:
        from sklearn.metrics import silhouette_score
        labels = _hard_labels(algo, estimator, Xs)
        if len(np.unique(labels)) > 1:
            # subsample for speed on large series
            if n > 5000:
                rng = np.random.RandomState(0)
                sel = rng.choice(n, 5000, replace=False)
                scores["silhouette"] = float(silhouette_score(Xs[sel], labels[sel]))
            else:
                scores["silhouette"] = float(silhouette_score(Xs, labels))
    except Exception:  # pragma: no cover
        pass
    return scores


def _hard_labels(algo, estimator, Xs) -> np.ndarray:
    if algo == "hmm":
        return _hmm_filter(estimator, Xs).argmax(axis=1)
    return estimator.predict(Xs)


def select_num_regimes(obs: pd.DataFrame, algo: str, k_range=range(2, 7),
                       seed: int = 42) -> dict:
    """Fit ``algo`` for each k in ``k_range`` and score model selection.

    Returns ``{'algo', 'best_k', 'criterion', 'scores': {k: {...}}}``.
    Selection uses BIC (hmm/gmm) or silhouette (kmeans).
    """
    from sklearn.preprocessing import StandardScaler

    X = obs[list(OBS_COLUMNS)].dropna().to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    scores = {}
    for k in k_range:
        try:
            est = _fit_estimator(algo, k, Xs, seed)
            scores[k] = _score_fit(algo, est, Xs)
        except Exception as exc:  # pragma: no cover - a k may fail to converge
            scores[k] = {"error": str(exc)}

    valid = {k: s for k, s in scores.items() if "error" not in s}
    if algo == "kmeans":
        criterion = "silhouette"
        best_k = max(valid, key=lambda k: valid[k].get("silhouette", -np.inf)) if valid else None
    else:
        criterion = "bic"
        best_k = min(valid, key=lambda k: valid[k].get("bic", np.inf)) if valid else None

    return {"algo": algo, "best_k": best_k, "criterion": criterion, "scores": scores}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_regime_model(model: RegimeModel, path: Path) -> None:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_regime_model(path: Path) -> RegimeModel | None:
    import joblib
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Consumption bridge (mirrors timesfm_features.get_timesfm_features)
# ---------------------------------------------------------------------------
def _requested_fields(feature_names):
    """Split de-prefixed rgm names into (max_state_index, non-prob fields set)."""
    max_state = -1
    fields = set()
    any_rgm = False
    for name in feature_names:
        field = parse_rgm_feature(name)
        if field is None:
            continue
        any_rgm = True
        si = prob_state_index(name)
        if si is not None:
            max_state = max(max_state, si)
        else:
            fields.add(field)
    return any_rgm, max_state, fields


def _select_columns(frame: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Return only the requested rgm columns from ``frame`` (NaN if missing)."""
    out = pd.DataFrame(index=frame.index)
    for name in feature_names:
        if parse_rgm_feature(name) is None:
            continue
        col = name  # already de-prefixed; regime columns share the same name
        if col in frame.columns:
            out[col] = frame[col].astype(np.float32)
        else:
            print(
                f"WARNING [regime]: column '{col}' missing from regime output; "
                "feature will be NaN. Re-run data/update_regime_model_data.py.",
                file=sys.stderr,
            )
            out[col] = np.float32(np.nan)
    return out


def verify_regime_meta(timeframe: str, algo: str | None, max_state: int = -1) -> dict | None:
    """Check the provenance sidecar next to a regime CSV against this run's request.

    ``update_regime_model_data.py`` writes the fitted algorithm, its state count and
    the seed into ``regime_{timeframe}{_algo}.meta.json`` — but nothing used to read
    it, so a mismatch could only be found by comparing value ranges by hand.

    Two mismatches matter:

    * **Wrong algorithm.** The unsuffixed ``regime_daily.csv`` is rewritten on every
      run of the updater, for whichever ``--algo`` ran last. A config that does not
      pin ``rgm_algo`` therefore silently picks up that algorithm's regime — and the
      algorithms disagree substantively, not cosmetically: the fitted hmm has state
      trend-scores [-0.73, -0.16, +0.64] against kmeans' [-1.00, +0.11, +0.26], so
      the same bar can come out trending up under one and down under the other.
    * **Too few states.** ``rgm_n_states`` must cover every ``rgm_prob_s*`` the
      feature config asks for; asking for ``prob_s3`` of a 3-state model yields a
      silently all-NaN column.

    A suffixed file that contradicts its own name is a corrupted checkout and raises.
    The unsuffixed file only warns — reading it is a legitimate (if fragile) choice.

    Returns the parsed sidecar, or None when there is none to check.
    """
    meta_path = regime_meta_path(timeframe, algo)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"WARNING [regime]: could not read {meta_path.name}: {exc}", file=sys.stderr)
        return None

    actual = meta.get("algo")
    if actual and algo and actual != algo:
        raise ValueError(
            f"Regime file mismatch: {regime_csv_path(timeframe, algo).name} claims to "
            f"hold '{algo}' output but its sidecar records '{actual}'. Regenerate with: "
            f"python data/update_regime_model_data.py --algo {algo}"
        )
    if actual and not algo:
        print(
            f"WARNING [regime]: reading the unsuffixed {regime_csv_path(timeframe).name}, "
            f"which currently holds '{actual}' output (the last algorithm the updater ran). "
            f"Pin parameters.rgm_algo in the feature config to read a specific algorithm.",
            file=sys.stderr,
        )

    n_states = meta.get("n_states")
    if n_states is not None and max_state >= n_states:
        raise ValueError(
            f"Feature config requests rgm_prob_s{max_state}, but the fitted model in "
            f"{meta_path.name} has only {n_states} states (s0..s{n_states - 1}). "
            f"Refit with --n-states {max_state + 1} or drop the extra prob features."
        )
    return meta


def _load_regime_csv(timeframe: str, feature_names, index: pd.Index,
                     algo: str | None = None, max_state: int = -1) -> pd.DataFrame | None:
    csv_path = regime_csv_path(timeframe, algo)
    if not csv_path.exists():
        if algo:
            # Falling back to the unsuffixed CSV here would hand this run another
            # algorithm's regime silently — the exact failure that made an earlier
            # hmm/gmm comparison unattributable. Fail instead.
            raise FileNotFoundError(
                f"Regime CSV for algo '{algo}' not found: {csv_path}. "
                f"Run: python data/update_regime_model_data.py --algo {algo}"
            )
        print(
            f"WARNING [regime]: precomputed CSV not found: {csv_path}. "
            "Run data/update_regime_model_data.py to generate it. "
            "Regime feature columns will be NaN.",
            file=sys.stderr,
        )
        return None
    verify_regime_meta(timeframe, algo, max_state)
    raw = pd.read_csv(csv_path)
    raw["date"] = pd.to_datetime(raw["date"], format="ISO8601")
    if raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    raw = raw.set_index("date").sort_index()
    aligned = raw.reindex(index)
    return _select_columns(aligned, feature_names)


def get_regime_features(df: pd.DataFrame, timeframe: str, feature_names,
                        params: dict, compute: bool) -> pd.DataFrame | None:
    """Resolve the requested ``rgm_*`` features for ``timeframe``.

    Parameters
    ----------
    df : OHLC DataFrame (the timeframe's bars).
    timeframe : 'm15' / '4hours' / 'daily' (selects CSV / persisted model).
    feature_names : iterable of de-prefixed rgm names (e.g. ``"rgm_prob_s0"``).
    params : feature-config ``parameters`` block. ``rgm_algo`` selects which
        fitted model's files this run reads, so two feature configs differing
        only in ``rgm_algo`` give two arms their own regime data.
    compute : True → load the persisted fitted model and run causal inference
        live; False → load the precomputed CSV (training path).

    Returns a DataFrame aligned to ``df.index`` with the requested columns, or
    ``None`` when no regime features were requested.
    """
    any_rgm, _max_state, _fields = _requested_fields(feature_names)
    if not any_rgm:
        return None

    algo = (params or {}).get("rgm_algo")

    if not compute:
        return _load_regime_csv(timeframe, feature_names, df.index, algo, _max_state)

    # Live path: load persisted model and infer causally. The .pkl follows the same
    # suffixed/unsuffixed convention as the CSV, so it carries the same mismatch risk.
    verify_regime_meta(timeframe, algo, _max_state)
    model_path = regime_model_path(timeframe, algo)
    model = load_regime_model(model_path)
    if model is None:
        print(
            f"WARNING [regime]: fitted model not found: {model_path}. "
            f"Run data/update_regime_model_data.py --algo {algo} to fit it. "
            "Regime feature columns will be NaN.",
            file=sys.stderr,
        )
        empty = pd.DataFrame(index=df.index)
        return _select_columns(empty, feature_names)

    obs = build_observations(df, params)
    inferred = model.infer(obs)
    return _select_columns(inferred, feature_names)


# ---------------------------------------------------------------------------
# Regime-score sidecar (training → backtest bridge)
#
# X_{model}.parquet only carries each model's SELECTED features, so a score
# tagged ``role: helper`` in the features config (computed, but fed to no
# model) would never reach backtest.py's --regime-risk / --regime-source ml /
# --direction-source trend lookups. advanced_train.py therefore persists every
# daily_rgm_* column of the combined feature frame — helper or model role —
# as a small sidecar parquet next to the model matrices.
# ---------------------------------------------------------------------------
REGIME_SCORES_FILENAME = "regime_scores.parquet"


def save_regime_scores(frame: pd.DataFrame, index: pd.Index, generated_dir) -> list:
    """Persist the ``daily_rgm_*`` columns of ``frame`` to the run directory.

    ``frame`` is the combined feature frame (already shift(1)-correct), ``index``
    the final training index the model matrices were saved on. Returns the list
    of persisted column names; writes nothing and returns [] when the frame has
    no ``daily_rgm_*`` columns.
    """
    cols = [c for c in frame.columns if c.startswith("daily_rgm_")]
    if not cols:
        return []
    out = frame[cols].reindex(index).astype(np.float32)
    out.to_parquet(str(Path(generated_dir) / REGIME_SCORES_FILENAME))
    return cols


def load_regime_scores(generated_dir, index: pd.Index | None = None) -> pd.DataFrame | None:
    """Load the sidecar written by ``save_regime_scores``.

    Returns the DataFrame (reindexed to ``index`` when given — uncovered bars
    become NaN, which every consumer already treats as "score unavailable"),
    or ``None`` when the file is absent or unreadable.
    """
    path = Path(generated_dir) / REGIME_SCORES_FILENAME
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(str(path))
    except Exception as exc:
        print(f"WARNING [regime]: could not read {path}: {exc}", file=sys.stderr)
        return None
    frame.index = pd.to_datetime(frame.index)
    if index is not None:
        frame = frame.reindex(index)
    return frame


def resolve_rgm_series(name: str, X_slow: pd.DataFrame | None,
                       sidecar: pd.DataFrame | None) -> pd.Series | None:
    """Backtest lookup order for one ``daily_rgm_*`` column.

    The slow-model feature matrix wins when it carries the column
    (``role: model`` — the historical path, e.g. run ``rgm_dir``); the
    regime-score sidecar is the fallback (``role: helper``). ``None`` when
    neither source has it.
    """
    if X_slow is not None and name in X_slow.columns:
        return X_slow[name]
    if sidecar is not None and name in sidecar.columns:
        return sidecar[name]
    return None
