import math
import os
import re

import yaml

# Canonical model keys. Each trained model gets its own feature set, scaler and
# ONNX input shape. The first part of the name encodes the *training cadence*:
#   *_fast -> M15 bars,  *_slow -> 4h-resampled bars.
MODELS = ['long_fast', 'short_fast', 'long_slow', 'short_slow']

# --- Warm-up / data-load horizon ------------------------------------------------
# Bars of each timeframe that exist in one calendar week AFTER weekend filtering
# (load_csv is always called with filter_weekends_flag=True): 5 trading days,
# 24h/4h = 6 four-hour bars per day, 24h*4 = 96 M15 bars per day.
BARS_PER_WEEK = {'m15': 5 * 96, '4hours': 5 * 6, 'daily': 5}

# `parameters:` keys whose value is a rolling-window length in bars. Everything
# else in that block (thresholds, seeds, batch sizes, frequency codes, state
# counts) is not a lookback and must not inflate the warm-up.
_WINDOW_KEY_SUFFIXES = ('_period', '_lookback', '_length', '_window')
_WINDOW_KEY_PREFIXES = ('ma_', 'momentum_')

# Headroom on top of the exact bar requirement: exchange holidays, missing bars
# and the fact that a rolling window is only *fully* populated one bar after it
# is first computable. Cheap insurance -- the warm-up bars are dropped before
# training, they only feed the indicators.
WARMUP_SAFETY_FACTOR = 1.25

# Never load less history than the old hard-coded buffer, even for a config
# whose largest window is tiny.
MIN_WARMUP_DAYS = 90


# Map each model to its training cadence (bar frequency + hyperparameter group).
MODEL_CADENCE = {
    'long_fast': 'fast',
    'short_fast': 'fast',
    'long_slow': 'slow',
    'short_slow': 'slow',
}


def model_cadence(model):
    """Return the training cadence ('fast' or 'slow') for a model key."""
    return MODEL_CADENCE[model]


class FeatureConfig:
    """Load and manage feature configuration.

    The config is a flat ``features:`` list. Each entry is a dict:

        - name:     feature name (with timeframe prefix, e.g. ``m15_rsi``)
        - role:     ``model`` (a model input) or ``helper`` (computed intermediate)
        - models:   list of model keys the feature feeds (required when role==model)
        - enabled:  bool, default True. A disabled feature is excluded from every list.
        - external: bool, default False. Sourced from an external CSV instead of OHLC.
    """

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = "features.yaml"
        # If only a filename (no directory separator and not an existing absolute path),
        # resolve it relative to the project's config directory.
        if not os.path.isabs(config_path) and os.path.dirname(config_path) == "":
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            config_path = os.path.join(base_dir, "config", config_path)

        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._features = self._normalize(self.config.get('features', []))

    @staticmethod
    def _normalize(raw_features):
        """Validate and normalize the raw feature list into dicts with defaults."""
        normalized = []
        for entry in raw_features or []:
            if 'name' not in entry:
                raise ValueError(f"Feature entry missing 'name': {entry}")
            name = entry['name']
            role = entry.get('role', 'model')
            if role not in ('model', 'helper'):
                raise ValueError(f"Feature '{name}' has invalid role '{role}' (expected 'model' or 'helper')")
            enabled = entry.get('enabled', True)
            external = entry.get('external', False)
            models = entry.get('models', []) or []
            if role == 'model' and enabled and not models:
                raise ValueError(f"Feature '{name}' is role=model but has no 'models' list")
            unknown = [m for m in models if m not in MODELS]
            if unknown:
                raise ValueError(f"Feature '{name}' references unknown model(s) {unknown}; valid: {MODELS}")
            normalized.append({
                'name': name,
                'role': role,
                'models': list(models),
                'enabled': bool(enabled),
                'external': bool(external),
            })
        return normalized

    @staticmethod
    def _filter_prefix(names, prefix):
        if prefix is not None:
            return [n for n in names if n.startswith(prefix)]
        return names

    def get_features(self, prefix=None):
        """Returns list of all feature names (enabled or not), optionally by prefix."""
        return self._filter_prefix([f['name'] for f in self._features], prefix)

    def get_enabled_features(self, prefix=None):
        """Returns list of enabled feature names in order."""
        return self._filter_prefix([f['name'] for f in self._features if f['enabled']], prefix)

    def get_usedInModel_features(self, prefix=None):
        """Returns enabled model-input feature names (role==model) in order."""
        names = [f['name'] for f in self._features if f['enabled'] and f['role'] == 'model']
        return self._filter_prefix(names, prefix)

    def get_helper_features(self, prefix=None):
        """Returns enabled helper feature names (role==helper) in order.

        Externally-sourced features are excluded: they are loaded from CSV
        (see get_externalSourced_features), not computed as OHLC-derived
        helper intermediates.
        """
        names = [f['name'] for f in self._features
                 if f['enabled'] and f['role'] == 'helper' and not f['external']]
        return self._filter_prefix(names, prefix)

    def get_externalSourced_features(self, prefix=None):
        """Returns enabled externally-sourced feature names in order."""
        names = [f['name'] for f in self._features if f['enabled'] and f['external']]
        return self._filter_prefix(names, prefix)

    def get_model_features(self, model, prefix=None):
        """Returns enabled model-input features assigned to ``model`` (in order).

        This replaces the old prefix-based fast/slow scope split: a model's
        candidate feature set is exactly the features whose ``models`` list
        contains the model key.
        """
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'; valid: {MODELS}")
        names = [f['name'] for f in self._features
                 if f['enabled'] and f['role'] == 'model' and model in f['models']]
        return self._filter_prefix(names, prefix)

    def get_max_lookback_bars(self, prefix=None):
        """Largest rolling-window length, in bars, any enabled feature can require.

        Two sources are combined:

        1. ``parameters:`` entries that name a window (``*_period``, ``*_lookback``,
           ``*_length``, ``*_window``, ``ma_*``, ``momentum_*``). These are global --
           ``volatility_percentile_lookback: 500`` applies to the M15, 4h and daily
           variant of that feature alike.
        2. The trailing ``_<n>`` of an enabled feature name (``m15_volume_percentile_96``,
           ``daily_sma_200``), so a per-feature window that lives in the name rather
           than in ``parameters:`` is still accounted for.

        ``prefix`` restricts source 2 to one timeframe; source 1 always counts.
        """
        max_bars = 0

        for key, value in (self.config.get('parameters') or {}).items():
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            if key.endswith(_WINDOW_KEY_SUFFIXES) or key.startswith(_WINDOW_KEY_PREFIXES):
                max_bars = max(max_bars, value)

        for name in self.get_enabled_features(prefix):
            m = re.search(r'_(\d+)$', name)
            if m:
                max_bars = max(max_bars, int(m.group(1)))

        return max_bars

    def get_warmup_days(self):
        """Calendar days of history needed *before* the first bar that is trained on.

        The binding timeframe is the slowest one: 500 daily bars span ~2 years of
        calendar time, whereas 500 M15 bars span ~5 days. Loading less than this
        does not fail -- it silently changes what the long-lookback features mean
        (a percentile over the 64 bars that happened to be loaded is not the
        percentile over 500 the config asked for), and the distortion differs from
        run to run and from walk-forward fold to fold.
        """
        bars = self.get_max_lookback_bars()
        slowest_bars_per_week = min(BARS_PER_WEEK.values())
        days = math.ceil(bars / slowest_bars_per_week * 7 * WARMUP_SAFETY_FACTOR)
        return max(days, MIN_WARMUP_DAYS)

    def get_models(self):
        """Returns the canonical list of model keys."""
        return list(MODELS)

    def get_parameters(self):
        """Returns feature calculation parameters"""
        return self.config['parameters']

    def is_feature_usedInModel(self, feature_name):
        """Check if a specific feature is an enabled model input."""
        for f in self._features:
            if f['name'] == feature_name and f['enabled'] and f['role'] == 'model':
                return True
        return False

    def get_feature_count(self):
        """Returns total number of model-input features"""
        return len(self.get_usedInModel_features())

# Singleton instance
_config = None

def get_feature_config(config_path=None):
    """Return the singleton FeatureConfig instance.

    On first call, ``config_path`` (filename inside ``ModelTrading/config/`` or
    an absolute path) selects which YAML to load. Subsequent calls ignore the
    argument and return the already-initialized config. Use ``set_feature_config``
    to switch configs explicitly.
    """
    global _config
    if _config is None:
        _config = FeatureConfig(config_path=config_path)
    return _config


def set_feature_config(config_path):
    """Force the singleton to load from ``config_path``. Call before any
    feature calculation that goes through ``get_feature_config()``.
    """
    global _config
    _config = FeatureConfig(config_path=config_path)
    return _config


def get_warmup_days(config_path=None):
    """Warm-up horizon in calendar days for the active feature config.

    Convenience wrapper so callers that only need the number do not have to reach
    into the singleton. Call *after* ``set_feature_config`` when ``--features-config``
    is in play, otherwise the default config answers.
    """
    return get_feature_config(config_path).get_warmup_days()
