import pandas as pd

DATA_AVAILABLE_START = pd.to_datetime("2005-01-01")
DATA_AVAILABLE_END = pd.to_datetime("2026-04-19")

# ============================================================================
# SEALED HOLD-OUT
# ============================================================================
# Everything up to DATA_AVAILABLE_END has been used by roughly 1,800 backtests and
# 324 walk-forward trainings. There is no untouched data left inside it, so no split
# of it can serve as an honest hold-out — a "test set" that has been looked at that
# often is a validation set with a different name.
#
# The one genuinely unseen period is the data that did NOT exist when the search was
# run: the CSVs end 2026-04-19 while the calendar is well past it. That window is
# sealed here and evaluated EXACTLY ONCE, at the end, on the single configuration the
# pre-registered protocol nominates. See docs/preregistration.md.
#
# The seal is enforced, not merely documented: advanced_train.py and backtest.py refuse
# to touch bars at or after HOLDOUT_START unless --unseal-holdout is passed, and an
# unsealed run records that fact in its summary so it can never be mistaken for a
# sealed one.
#
# Until the CSVs are extended (ModelTrading/source/python/data/update_data.py) this
# window contains no bars, so the guard costs nothing and protects automatically the
# moment the data arrives.
HOLDOUT_START = DATA_AVAILABLE_END + pd.Timedelta(days=1)


def holdout_violation(*timestamps):
    """Return the first timestamp that reaches into the sealed hold-out, or None."""
    for ts in timestamps:
        if ts is None:
            continue
        ts = pd.to_datetime(ts)
        if ts >= HOLDOUT_START:
            return ts
    return None

##TRAIN_START = DATA_AVAILABLE_START + pd.DateOffset(months=24)
##TRAIN_START = pd.to_datetime("20205-01-01")
TRAIN_START = pd.to_datetime("2025-04-01") - pd.DateOffset(months=96) 
TRAIN_END   = pd.to_datetime("2025-09-30")

## auto calculated timeframes based on above
TEST_START = TRAIN_END + pd.DateOffset(days=1)
##TEST_START = pd.to_datetime("2025-12-30")
TEST_END = DATA_AVAILABLE_END

BACKTEST_START = TRAIN_END + pd.DateOffset(days=1)
##BACKTEST_START = pd.to_datetime("2026-01-01")
BACKTEST_END = DATA_AVAILABLE_END
BACKTEST_END = pd.to_datetime("2026-04-19")

## Fallback only. train.py / advanced_train.py / denoise_audit.py recompute this as
## min(TRAIN_START, BACKTEST_START) - FeatureConfig.get_warmup_days(), i.e. from the
## longest rolling window in the active features.yaml (a `lookback: 500` on daily bars
## needs ~2 calendar years). This module cannot do that itself: the feature config is
## only chosen later, by --features-config.
DATALOAD_START = TRAIN_START - pd.DateOffset(months=20) ## we need a long history for some indicators with long lookback periods
DATALOAD_END = BACKTEST_END


def apply_feature_warmup(warmup_days):
    """Move DATALOAD_START back far enough to warm up every feature.

    Callers override TRAIN_START / BACKTEST_START first (from --train-start /
    --backtest-start), then pass FeatureConfig.get_warmup_days(). The load window
    starts `warmup_days` before the first bar features are actually needed for, so
    every rolling window is fully populated by then regardless of which training
    window was requested.
    """
    global DATALOAD_START
    DATALOAD_START = min(TRAIN_START, BACKTEST_START) - pd.Timedelta(days=warmup_days)
    return DATALOAD_START

