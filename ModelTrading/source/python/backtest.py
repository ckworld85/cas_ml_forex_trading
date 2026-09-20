import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Add project root to Python path to enable ModelTrading package imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.source.python.utils.forex as forex
import ModelTrading.source.python.utils.datahandling as datahandling
import ModelTrading.source.python.utils.costs as costs
import ModelTrading.source.python.utils.provenance as provenance
import ModelTrading.source.python.utils.risk as risk_utils
import ModelTrading.config.directories as dir_config
import ModelTrading.config.timeframes as timeframes
from ModelTrading.source.python.labeling.regime import generate_regime_labels
import ModelTrading.source.python.features.regime_model as regime_model
import ModelTrading.source.python.features.config as feature_config
import ModelTrading.source.python.utils.experiment_tracking as experiment_tracking

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser(description='Run backtest on trained models')
parser.add_argument('--features-config', type=str, default=None,
                    help='Features YAML to use (filename inside ModelTrading/config/, '
                         'or an absolute path). Defaults to features.yaml.')
parser.add_argument('--run-id', type=str, default=None,
                    help='Unique run ID for parallel execution (loads from generated/{run_id}/)')
parser.add_argument('--backtest-start', type=str, default=None,
                    help='Backtest start date (YYYY-MM-DD) - overrides timeframes.py')
parser.add_argument('--backtest-end', type=str, default=None,
                    help='Backtest end date (YYYY-MM-DD) - overrides timeframes.py')
parser.add_argument('--p-open-fast', type=float, default=None,
                    help='Fast model entry threshold (overrides P_OPEN_THRESHOLD_FAST)')
parser.add_argument('--p-open-slow', type=float, default=None,
                    help='Slow model entry threshold (overrides P_OPEN_THRESHOLD_SLOW)')
parser.add_argument('--use-trained-threshold', action='store_true',
                    help='Take the entry thresholds from training_summary.json '
                         '(slow_final_global_threshold, and fast_final_global_threshold '
                         'when present — relevant under --opening-requires-fast-signal; '
                         'both chosen in-fold by --target-recall) instead of hand-set '
                         '--p-open-slow/--p-open-fast, and resolve each separately '
                         'for long and short through each direction\'s own calibrator. '
                         'A fixed calibrated threshold lands on a DIFFERENT raw operating '
                         'point per direction and per label mode, which confounds any '
                         'comparison between them. NOTE: this deviates from '
                         'JForexMLStrategy.java, which applies one scalar to both '
                         'directions — use it for studies, not for production parity.')
parser.add_argument('--slow-direction-margin', type=float, default=None,
                    help='Minimum |p_long_slow - p_short_slow| required to trade (overrides SLOW_DIRECTION_MARGIN). 0 disables.')
parser.add_argument('--p-close-pip-threshold', type=float, default=None,
                    help='Exit when realized PnL exceeds this many pips (overrides P_CLOSE_PIP_THRESHOLD)')
parser.add_argument('--closing-after-x-pips', action='store_true',
                    help='Enable exit when realized PnL exceeds P_CLOSE_PIP_THRESHOLD')
parser.add_argument('--report-dir', type=str, default=None,
                    help='Override output directory for backtest reports (for parallel sweeps)')
parser.add_argument('--p-close-threshold', type=float, default=None,
                    help='Exit threshold for signal reversal (overrides P_CLOSE_THRESHOLD)')
parser.add_argument('--stop-pips', type=float, default=None,
                    help='Stop loss in pips (overrides STOP_PIPS)')
parser.add_argument('--hold-bars', type=int, default=None,
                    help='Maximum hold time in M15 bars (overrides HOLD_BARS)')
parser.add_argument('--no-closing-before-weekend', dest='no_closing_before_weekend',
                    action='store_true',
                    help='Disable exit before weekend (default: enabled)')
parser.add_argument('--closing-after-time', action='store_true',
                    help='Enable exit after HOLD_BARS bars')
parser.add_argument('--closing-after-signal-reversal-fast', action='store_true',
                    help='Enable exit on fast model signal reversal')
parser.add_argument('--no-closing-after-signal-reversal-slow',
                    dest='no_closing_after_signal_reversal_slow', action='store_true',
                    help='Disable exit on slow model signal reversal (default: enabled)')
parser.add_argument('--closing-with-trailing-stop', action='store_true',
                    help='Enable trailing stop logic')
parser.add_argument('--closing-on-level-retest', action='store_true',
                    help='Enable exit on level retest')
parser.add_argument('--opening-requires-fast-signal', action='store_true',
                    help='Require the fast model signal for entry (p_fast > --p-open-fast). '
                         'Off by default, matching the Java strategy default — without it '
                         'the fast model has no influence on entries at all.')
parser.add_argument('--no-opening-requires-fast-signal',
                    dest='no_opening_requires_fast_signal', action='store_true',
                    help='Disable the fast model requirement for entry (already the default)')
parser.add_argument('--opening-requires-slow-signal',
                    dest='opening_requires_slow_signal', action='store_true',
                    help='Require the slow model signal for entry (p_slow >= --p-open-slow). '
                         'On by default, matching the Java strategy '
                         '(openingRequiresSlowSignal = true).')
parser.add_argument('--no-opening-requires-slow-signal',
                    dest='no_opening_requires_slow_signal', action='store_true',
                    help='Disable the slow model requirement for entry. With the fast '
                         'requirement also off no entry gate remains, and the backtest '
                         'opens NO trades at all.')
parser.add_argument('--regime-breakdown', action='store_true', default=False,
                    help='Print a per-regime trade breakdown after the direction breakdown')
parser.add_argument('--regime-gate', type=str, default='trending',
                    choices=['off', 'trending', 'ranging'],
                    help='Daily regime entry gate: "trending" allows entries only on trending days '
                         '(ADX>25 or PE>0.5, default), "ranging" allows entries only on ranging days, '
                         '"off" disables the gate.')
parser.add_argument('--trend-aware-exits', action='store_true',
                    help='In trend regime bars, disable the fixed-pip take-profit and apply an '
                         'ATR-based chandelier trail (peak_pnl - atr_trail_mult * ATR pips). '
                         'In range bars, the existing exits are unchanged.')
parser.add_argument('--atr-trail-mult', type=float, default=3.0,
                    help='Chandelier trail distance in ATR multiples (default 3.0).')
parser.add_argument('--atr-trail-activation-mult', type=float, default=1.5,
                    help='Peak profit (in ATR multiples) required before the chandelier trail arms '
                         '(default 1.5).')
parser.add_argument('--atr-period', type=int, default=14,
                    help='ATR period for the chandelier trail (default 14).')
parser.add_argument('--close-on-ranging-min-pips', type=float, default=None,
                    help='If set, close any open trade whose unrealized PnL is >= this many pips '
                         'as soon as the current bar is classified as ranging (regime). Requires '
                         'regime labels (auto-loaded when this is set). Off unless specified.')
parser.add_argument('--breakeven-trigger-pips', type=float, default=None,
                    help='If set, once unrealized PnL reaches this many pips, move the stop loss '
                         'to entry + breakeven_offset_pips (long) / entry - offset (short). '
                         'Off unless specified.')
parser.add_argument('--breakeven-offset-pips', type=float, default=-5.0,
                    help='Offset from entry price for the breakeven stop, in pips. Negative = stop '
                         'sits below entry for longs (above for shorts), locking in a small loss. '
                         'Default -5. Only used when --breakeven-trigger-pips is set.')
# --- ML regime model integration -------------------------------------------
parser.add_argument('--regime-source', type=str, default='rule', choices=['rule', 'ml'],
                    help='Source for the daily trend/range regime used by the entry gate and '
                         'trend-aware exits: "rule" (ADX/price-efficiency, default) or "ml" '
                         '(the fitted regime model via daily_rgm_trend_score). "ml" requires the '
                         'daily_rgm_* features to be enabled and the model retrained.')
parser.add_argument('--regime-ml-trend-threshold', type=float, default=0.15,
                    help='With --regime-source ml, a bar is "trending" when '
                         '|daily_rgm_trend_score| exceeds this (default 0.15).')
parser.add_argument('--direction-source', type=str, default='model', choices=['model', 'trend'],
                    help='How the entry direction is chosen: "model" = slow-model probabilities '
                         '(p_long_slow vs p_short_slow, default), "trend" = the sign of the ML '
                         'regime trend score (daily_rgm_trend_score) — momentum-following instead '
                         'of prediction. Rationale: measured slow-model AUC within trend bars is '
                         '~0.50, so the model direction is a coin flip there; the CTA framing '
                         'harvests trend persistence via asymmetric exits instead. Falls back to '
                         '"model" per-bar when the score is unavailable.')
parser.add_argument('--regime-risk', action='store_true', default=False,
                    help='Enable ML regime-based risk management: scale position size and '
                         'stop-loss distance per entry from the regime scores '
                         '(daily_rgm_trend_score / daily_rgm_vol_score). Off by default.')
parser.add_argument('--regime-size-trend-w', type=float, default=0.5,
                    help='Position-size sensitivity to the direction-aligned trend score '
                         '(size *= 1 + w*aligned_trend). Default 0.5.')
parser.add_argument('--regime-size-vol-w', type=float, default=0.5,
                    help='Position-size sensitivity to the volatility score '
                         '(size *= 1 - w*vol_score). Default 0.5.')
parser.add_argument('--regime-size-min', type=float, default=0.25,
                    help='Lower clip for the regime position-size multiplier (default 0.25).')
parser.add_argument('--regime-size-max', type=float, default=2.0,
                    help='Upper clip for the regime position-size multiplier (default 2.0).')
parser.add_argument('--regime-stop-vol-w', type=float, default=0.5,
                    help='Stop-distance sensitivity to the volatility score '
                         '(stop_pips *= 1 + w*(2*vol_score-1)). Default 0.5.')
parser.add_argument('--regime-stop-min-mult', type=float, default=0.5,
                    help='Lower clip for the regime stop-distance multiplier (default 0.5).')
parser.add_argument('--regime-stop-max-mult', type=float, default=2.0,
                    help='Upper clip for the regime stop-distance multiplier (default 2.0).')

# --- Transaction costs (see utils/costs.py) ---------------------------------
parser.add_argument('--cost-model', type=str, default='none',
                    choices=['none', 'fixed', 'data'],
                    help="Execution cost model. 'none' (default, historical behaviour) "
                         "fills both legs at the ASK close, so a long buys and sells on "
                         "the same side of the book and the result is GROSS — not "
                         "tradeable. 'fixed' charges --spread-pips on every bar. 'data' "
                         "charges the per-bar spread 2*(ask_close-mid) from the provider "
                         "CSV (median 0.4 pips, mean 1.8 on EUR/USD M15). Decision prices "
                         "are unchanged in every mode, so triggers fire on the same bars.")
parser.add_argument('--spread-pips', type=float, default=0.4,
                    help='Full spread in pips for --cost-model fixed (default 0.4, the '
                         'measured EUR/USD M15 median).')
parser.add_argument('--min-spread-pips', type=float, default=0.2,
                    help='Floor for the per-bar spread under --cost-model data. 2.1%% of '
                         'bars have mid == ask_close (a stalled mid feed, not a free '
                         'trade). Default 0.2.')
parser.add_argument('--slippage-pips', type=float, default=0.0,
                    help='Slippage charged against the trader on EVERY leg, so a round '
                         'trip pays twice this. Default 0.')
parser.add_argument('--commission-per-million', type=float,
                    default=costs.DEFAULT_COMMISSION_PER_MILLION,
                    help='Trade commission per 1M USD traded per side, quoted in USD '
                         '(default 18.0 = the Dukascopy net-deposit > 50k tier). For '
                         'EUR/USD with a EUR account the currency conversion cancels, '
                         'so it is charged as EUR per 1M EUR notional. A round trip '
                         'pays twice this. Ignored under --cost-model none.')
parser.add_argument('--overnight-long-per-million', type=float,
                    default=costs.DEFAULT_OVERNIGHT_LONG_PER_MILLION,
                    help='Overnight financing per 1M notional per settlement night for '
                         'LONG positions (default 63.65, Dukascopy EUR/USD; same '
                         'currency cancellation as the commission). Nights are Mon-Fri '
                         '21:00 UTC settlements, Wednesday 3x (T+2 weekend). Ignored '
                         'under --cost-model none.')
parser.add_argument('--overnight-short-per-million', type=float,
                    default=costs.DEFAULT_OVERNIGHT_SHORT_PER_MILLION,
                    help='Overnight financing per 1M notional per settlement night for '
                         'SHORT positions (default 28.65, Dukascopy EUR/USD). Ignored '
                         'under --cost-model none.')
parser.add_argument('--data-csv', type=str, default=None,
                    help='Provider M15 CSV used by --cost-model data for the mid column. '
                         'Defaults to ModelTrading/data/eurusd_m15.csv.')

# --- Position sizing / risk (see position_notional) -------------------------
parser.add_argument('--unseal-holdout', action='store_true',
                    help='Allow this run to read bars from the SEALED HOLD-OUT '
                         '(timeframes.HOLDOUT_START onward). The hold-out is the only '
                         'genuinely unseen data this project has — everything before it '
                         'has been through ~1,800 backtests. It is meant to be evaluated '
                         'exactly once, on the single configuration the pre-registered '
                         'protocol nominates (docs/preregistration.md). Every unsealed run '
                         'is recorded as such in backtest_summary.json.')
parser.add_argument('--risk-model', type=str, default='fixed_notional',
                    choices=['fixed_notional', 'fixed_fractional'],
                    help="Position sizing. 'fixed_notional' (default, historical) trades "
                         "NOTIONAL on every entry regardless of the stop distance — that "
                         "is not a risk model and makes every drawdown/Sharpe number a "
                         "statement about the sizing rule. 'fixed_fractional' sizes each "
                         "trade so the stop always costs --risk-pct of current equity, "
                         "which is what keeps risk constant across stop distances.")
parser.add_argument('--risk-pct', type=float, default=1.0,
                    help='Percent of current equity risked per trade under '
                         '--risk-model fixed_fractional (default 1.0).')
parser.add_argument('--max-leverage', type=float, default=30.0,
                    help='Cap on notional/equity under fixed_fractional sizing '
                         '(default 30). A very tight stop would otherwise ask for an '
                         'unbounded position.')

# --- Externally supplied probabilities --------------------------------------
parser.add_argument('--proba-file', type=str, default=None,
                    help='Parquet with columns long_fast/short_fast/long_slow/short_slow '
                         'indexed by bar timestamp, used INSTEAD of running the four '
                         'ONNX/pkl models. This is how an alternative learner is measured '
                         'through the identical execution engine: comparing a different '
                         'model AND a different backtest at the same time measures '
                         'neither. Calibration is skipped — the file supplies final '
                         'probabilities. Mirrors --label-mode file / --label-file.')

# Experiment tracking (opt-in)
experiment_tracking.add_wandb_args(parser)

args = parser.parse_args()

if args.features_config:
    feature_config.set_feature_config(args.features_config)
    print(f"Using features config: {feature_config.get_feature_config().config_path}")

RUN_ID = args.run_id

# Override backtest dates if provided
if args.backtest_start:
    timeframes.BACKTEST_START = pd.to_datetime(args.backtest_start)
if args.backtest_end:
    timeframes.BACKTEST_END = pd.to_datetime(args.backtest_end)

# Sealed hold-out guard. Enforced here rather than left to discipline: the whole point
# of a hold-out is that it cannot be consumed by accident.
_holdout_hit = timeframes.holdout_violation(timeframes.BACKTEST_END)
if _holdout_hit is not None and not args.unseal_holdout:
    raise SystemExit(
        f"--backtest-end {_holdout_hit.date()} reaches into the SEALED HOLD-OUT "
        f"(from {timeframes.HOLDOUT_START.date()}). That window is the only genuinely "
        f"unseen data in this project and is meant to be scored exactly once, on the "
        f"configuration the pre-registered protocol nominates. Shorten the window, or "
        f"pass --unseal-holdout if this IS that single final evaluation."
    )
if args.unseal_holdout:
    print("=" * 78)
    print("SEALED HOLD-OUT UNSEALED — this result is the one-shot final evaluation.")
    print("Any configuration choice made after reading it is no longer out-of-sample.")
    print("=" * 78)

# Get directory paths (uses subdirectory if run_id provided)
run_dirs = dir_config.get_run_dirs(RUN_ID)
GENERATED_DIR = run_dirs['generated_dir']
REPORT_DIR = run_dirs['report_dir']
if args.report_dir is not None:
    REPORT_DIR = args.report_dir

# ============================================================================
# Configuration
# ============================================================================
os.makedirs(REPORT_DIR, exist_ok=True)

# Trading parameters
NOTIONAL = 1000000  # EUR — only used by --risk-model fixed_notional
START_CAPITAL = 50000  # EUR
STOP_PIPS = 35
SYMBOL = "EURUSD"

# Position sizing (see position_notional)
RISK_MODEL = args.risk_model
RISK_PCT = args.risk_pct
MAX_LEVERAGE = args.max_leverage

# Strategy thresholds (matching JForexMLStrategy.java)
P_OPEN_THRESHOLD_FAST = 0.5  # Fast model entry threshold
P_OPEN_THRESHOLD_SLOW = 0.5  # Slow model entry threshold
SLOW_DIRECTION_MARGIN = 0.0  # Minimum |p_long_slow - p_short_slow|; 0 disables
P_CLOSE_THRESHOLD = 0.2      # Exit threshold for signal reversal
P_CLOSE_PIP_THRESHOLD = 50  # Exit threshold for realized PnL (pips)
HOLD_BARS = 144              # Maximum hold time (36 hours = 144 M15 bars)

# ML regime-based risk management (see --regime-* args). All no-ops unless
# --regime-risk / --regime-source ml are set and the daily_rgm_* features exist.
REGIME_SOURCE = args.regime_source
DIRECTION_SOURCE = args.direction_source
REGIME_ML_TREND_THRESHOLD = args.regime_ml_trend_threshold
REGIME_RISK = args.regime_risk
REGIME_SIZE_TREND_W = args.regime_size_trend_w
REGIME_SIZE_VOL_W = args.regime_size_vol_w
REGIME_SIZE_MIN = args.regime_size_min
REGIME_SIZE_MAX = args.regime_size_max
REGIME_STOP_VOL_W = args.regime_stop_vol_w
REGIME_STOP_MIN_MULT = args.regime_stop_min_mult
REGIME_STOP_MAX_MULT = args.regime_stop_max_mult

# Closing strategy flags (matching Java @Configurable parameters)
CLOSING_BEFORE_WEEKEND = False             # Exit before weekend if profitable
CLOSING_AFTER_TIME = False                # Exit after HOLD_BARS (time-based)
CLOSING_AFTER_SIGNAL_REVERSAL_FAST = False  # Exit on fast model signal reversal
CLOSING_AFTER_SIGNAL_REVERSAL_SLOW = True  # Exit on slow model signal reversal
CLOSING_WITH_TRAILING_STOP = False         # Use trailing stop logic
CLOSING_AFTER_X_PIPS = False              # Exit when realized PnL exceeds P_CLOSE_PIP_THRESHOLD
CLOSING_ON_LEVEL_RETEST = False           # Exit on level retest
TREND_AWARE_EXITS = False                 # In trend bars: disable pip TP, use ATR chandelier trail
ATR_TRAIL_MULT = 3.0                      # Chandelier trail distance (ATR multiples)
ATR_TRAIL_ACTIVATION_MULT = 1.5           # Peak profit required to arm trail (ATR multiples)
ATR_PERIOD = 14                           # ATR lookback period

# Opening strategy flags (matching Java @Configurable parameters)
OPENING_REQUIRES_FAST_SIGNAL = False       # Require fast model signal for entry
OPENING_REQUIRES_SLOW_SIGNAL = True       # Require slow model signal for entry

# Trailing stop parameters
TRAILING_ACTIVATION_PIPS = 100   # start trailing only after this profit
TRAILING_DISTANCE_PIPS   = 100   # trail this many pips behind the peak

# Breakeven stop move (off unless --breakeven-trigger-pips is set)
BREAKEVEN_TRIGGER_PIPS = None   # MFE required to arm the breakeven move (None = disabled)
BREAKEVEN_OFFSET_PIPS  = -5.0   # stop offset from entry, signed: -5 = lock in -5 pips

# Close profitable trade when regime turns ranging (off unless --close-on-ranging-min-pips is set)
CLOSE_ON_RANGING_MIN_PIPS = None  # minimum unrealized pips to harvest on a ranging bar

# Level retest closing strategy
# Logic: once price peaks, retraces MIN_RETRACE pips, then returns within PROXIMITY pips of the peak → close
# Only relevant for CLOSING_ON_LEVEL_RETEST = True
LEVEL_RETEST_MIN_RETRACE = 15   # pips below peak before "level is set"
LEVEL_RETEST_PROXIMITY   = 5    # pips from peak to trigger close on retest

# Override thresholds from CLI args
if args.p_open_fast is not None:
    P_OPEN_THRESHOLD_FAST = args.p_open_fast
if args.p_open_slow is not None:
    P_OPEN_THRESHOLD_SLOW = args.p_open_slow

# Per-direction entry thresholds. They default to the single scalar, which is what
# JForexMLStrategy.java applies, so the production path and Java parity are unchanged.
# --use-trained-threshold resolves them separately per direction instead; see below.
P_OPEN_SLOW_LONG = P_OPEN_THRESHOLD_SLOW
P_OPEN_SLOW_SHORT = P_OPEN_THRESHOLD_SLOW
P_OPEN_FAST_LONG = P_OPEN_THRESHOLD_FAST
P_OPEN_FAST_SHORT = P_OPEN_THRESHOLD_FAST
if args.slow_direction_margin is not None:
    SLOW_DIRECTION_MARGIN = args.slow_direction_margin
if args.p_close_pip_threshold is not None:
    P_CLOSE_PIP_THRESHOLD = args.p_close_pip_threshold
if args.closing_after_x_pips:
    CLOSING_AFTER_X_PIPS = True
if args.p_close_threshold is not None:
    P_CLOSE_THRESHOLD = args.p_close_threshold
if args.stop_pips is not None:
    STOP_PIPS = args.stop_pips
if args.hold_bars is not None:
    HOLD_BARS = args.hold_bars
if args.no_closing_before_weekend:
    CLOSING_BEFORE_WEEKEND = False
if args.closing_after_time:
    CLOSING_AFTER_TIME = True
if args.closing_after_signal_reversal_fast:
    CLOSING_AFTER_SIGNAL_REVERSAL_FAST = True
if args.no_closing_after_signal_reversal_slow:
    CLOSING_AFTER_SIGNAL_REVERSAL_SLOW = False
if args.closing_with_trailing_stop:
    CLOSING_WITH_TRAILING_STOP = True
if args.closing_on_level_retest:
    CLOSING_ON_LEVEL_RETEST = True
if args.trend_aware_exits:
    TREND_AWARE_EXITS = True
if args.atr_trail_mult is not None:
    ATR_TRAIL_MULT = args.atr_trail_mult
if args.atr_trail_activation_mult is not None:
    ATR_TRAIL_ACTIVATION_MULT = args.atr_trail_activation_mult
if args.breakeven_trigger_pips is not None:
    BREAKEVEN_TRIGGER_PIPS = args.breakeven_trigger_pips
    BREAKEVEN_OFFSET_PIPS = args.breakeven_offset_pips
if args.close_on_ranging_min_pips is not None:
    CLOSE_ON_RANGING_MIN_PIPS = args.close_on_ranging_min_pips
if args.atr_period is not None:
    ATR_PERIOD = args.atr_period
if args.opening_requires_fast_signal:
    OPENING_REQUIRES_FAST_SIGNAL = True
if args.no_opening_requires_fast_signal:
    OPENING_REQUIRES_FAST_SIGNAL = False
if args.opening_requires_slow_signal:
    OPENING_REQUIRES_SLOW_SIGNAL = True
if args.no_opening_requires_slow_signal:
    OPENING_REQUIRES_SLOW_SIGNAL = False

# With both requirements off, nothing probability-based is left to gate an entry:
# direction_is_long is an argmax that always resolves to one side, so every flat bar
# would open a trade. That is "always in the market", not a strategy -- so the entry
# is disabled entirely and the run reports zero trades.
ENTRY_GATES_ACTIVE = OPENING_REQUIRES_FAST_SIGNAL or OPENING_REQUIRES_SLOW_SIGNAL

# Debug mode
DEBUG_MODE = False  # Set to True to see entry/exit decisions
ENABLE_BACKTEST_LOGGING = False  # Set to True to enable feature/prediction logging (skips trading)
## python backtest.py --backtest-start 2026-04-10 --backtest-end 2026-04-19 > "C:\Users\mail\Daten\06_Projekte\github\forex_trading\ModelTrading\report\backtest\python_backtest.log"

# ============================================================================
# Load Data and Models
# ============================================================================
print("Loading features and OHLC data...")
MODEL_KEYS = ['long_fast', 'short_fast', 'long_slow', 'short_slow']
X_by_model = {
    mk: pd.read_parquet(os.path.join(GENERATED_DIR, f"X_{mk}.parquet")).astype(np.float32)
    for mk in MODEL_KEYS
}
# Representative aliases for index checks / regime gate (long_* per cadence).
X_fast = X_by_model['long_fast']
X_slow = X_by_model['long_slow']
ohlc = pd.read_parquet(os.path.join(GENERATED_DIR, "ohlc.parquet"))
ohlc = ohlc.astype(np.float32)

# Rename columns to standard OHLC names
ohlc = ohlc.rename(columns={
    "m15_open": "open",
    "m15_high": "high", 
    "m15_low": "low",
    "m15_close": "close"
})

for mk in MODEL_KEYS:
    if not X_by_model[mk].index.equals(ohlc.index):
        raise ValueError(f"X_{mk} and OHLC data have mismatched indices!")

# Ensure OHLC index is DatetimeIndex, sorted and unique
ohlc.index = pd.to_datetime(ohlc.index, dayfirst=True, errors="coerce")
if ohlc.index.isnull().any():
    raise ValueError(f"Could not parse timestamps in ohlc.index")

# Remove duplicates
ohlc = datahandling.remove_duplicates(ohlc, name="ohlc", removal_strategy='first')
for mk in MODEL_KEYS:
    X_by_model[mk] = X_by_model[mk].loc[ohlc.index]
X_fast = X_by_model['long_fast']
X_slow = X_by_model['long_slow']

# Optional regime-score sidecar written by advanced_train.py: daily_rgm_* columns
# persisted independently of the per-model matrices, so scores tagged
# `role: helper` in the features config (kept OUT of every model) still reach
# --regime-risk / --regime-source ml / --direction-source trend. X_slow wins
# when it carries a column (role: model); the sidecar is the fallback.
_regime_scores = regime_model.load_regime_scores(GENERATED_DIR, ohlc.index)
if _regime_scores is not None:
    print(f"Regime-score sidecar loaded: {list(_regime_scores.columns)}")


def rgm_column(name):
    """daily_rgm_* series aligned to ohlc.index, or None when unavailable."""
    return regime_model.resolve_rgm_series(name, X_slow, _regime_scores)


# Load pre-computed regime labels (saved by training). Fall back to on-the-fly
# computation only if the file is absent (e.g. standalone backtest on old models).
_regime_parquet = os.path.join(GENERATED_DIR, "regime_analysis", "regime_labels.parquet")
try:
    if os.path.exists(_regime_parquet):
        _regime_labels = pd.read_parquet(_regime_parquet)
        _regime_labels.index = pd.to_datetime(_regime_labels.index)
    else:
        _regime_labels = generate_regime_labels(ohlc)
except Exception as _e:
    print(f"Warning: could not load/compute regime labels for breakdown: {_e}")
    _regime_labels = None

n = len(ohlc)
if n == 0:
    raise ValueError("Empty OHLC data after cleaning.")

print(f"Data loaded: {n} bars")
print(f"Chronologically sorted: {ohlc.index.is_monotonic_increasing}")
print(f"Date range: {ohlc.index[0]} to {ohlc.index[-1]}")

# ============================================================================
# Derive Daily Regime Gate from Slow-Model Features
# ============================================================================
# When daily_regime features are enabled in features.yaml (usedInModel: true),
# daily_adx and daily_price_efficiency are present in X_slow.parquet.
# A bar is "trending" if ADX > 25 OR price efficiency > 0.5 (matching regime.py logic).
# Mode "trending" allows entries only on trending days (default — range-market whipsaws
# are the primary source of drawdown). Mode "ranging" allows entries only on ranging days
# (for strategies tuned to mean-reversion). Mode "off" disables the gate.
REGIME_GATE_MODE = args.regime_gate  # 'off' | 'trending' | 'ranging'

regime_trending_arr = None  # numpy bool array aligned to ohlc.index
_regime_needed = REGIME_GATE_MODE != 'off' or args.trend_aware_exits or CLOSE_ON_RANGING_MIN_PIPS is not None
_regime_source = None
regime_is_trending = None
if _regime_needed:
    # --regime-source ml routes the entry gate and trend-aware exits through the
    # fitted regime model (daily_rgm_trend_score) instead of the ADX/PE rule.
    if REGIME_SOURCE == 'ml':
        _ml_trend = rgm_column('daily_rgm_trend_score')
        if _ml_trend is not None:
            regime_is_trending = _ml_trend.abs() > REGIME_ML_TREND_THRESHOLD
            _regime_source = f'|daily_rgm_trend_score|>{REGIME_ML_TREND_THRESHOLD} (ML)'
        else:
            print("WARNING: --regime-source ml set but 'daily_rgm_trend_score' is neither in "
                  "X_slow nor in regime_scores.parquet (enable daily_rgm_* as model or helper "
                  "features + retrain). Falling back to the rule-based regime.")
    # Prefer the pre-computed binary flag, else raw ADX+PE, else percentile ADX+PE.
    if regime_is_trending is not None:
        pass
    elif 'daily_regime_trend' in X_slow.columns:
        regime_is_trending = X_slow['daily_regime_trend'].astype(bool)
        _regime_source = 'daily_regime_trend'
    elif 'daily_adx' in X_slow.columns and 'daily_price_efficiency' in X_slow.columns:
        regime_is_trending = (X_slow['daily_adx'] > 25) | (X_slow['daily_price_efficiency'] > 0.5)
        _regime_source = 'daily_adx>25 | daily_price_efficiency>0.5'
    elif 'daily_adx_percentile' in X_slow.columns and 'daily_price_efficiency' in X_slow.columns:
        regime_is_trending = (X_slow['daily_adx_percentile'] > 0.7) | (X_slow['daily_price_efficiency'] > 0.5)
        _regime_source = 'daily_adx_percentile>0.7 | daily_price_efficiency>0.5'

if _regime_needed and regime_is_trending is not None:
    regime_trending_arr = regime_is_trending.values  # fast positional lookup in loop
    trending_pct = 100.0 * regime_is_trending.mean()
    print(f"\nRegime source: {_regime_source}")
    if REGIME_GATE_MODE == 'trending':
        print(f"Regime gate ENABLED (mode=trending) — entries allowed on {regime_is_trending.sum()} / {n} bars ({trending_pct:.1f}%)")
        print(f"Ranging bars skipped for entries: {(~regime_is_trending).sum()} ({100-trending_pct:.1f}%)")
    elif REGIME_GATE_MODE == 'ranging':
        print(f"Regime gate ENABLED (mode=ranging) — entries allowed on {(~regime_is_trending).sum()} / {n} bars ({100-trending_pct:.1f}%)")
        print(f"Trending bars skipped for entries: {regime_is_trending.sum()} ({trending_pct:.1f}%)")
    else:  # 'off' but computed for trend-aware exits
        print(f"Daily regime detected: {regime_is_trending.sum()} trending / {(~regime_is_trending).sum()} ranging bars ({trending_pct:.1f}% trending)")
elif _regime_needed:
    print(f"\nRegime info (gate={REGIME_GATE_MODE}, trend-aware-exits={args.trend_aware_exits}): "
          f"no regime features found in X_slow (need daily_regime_trend, or daily_adx+daily_price_efficiency, "
          f"or daily_adx_percentile+daily_price_efficiency) — regime features disabled (retrain required)")

# ML regime-based risk-management arrays (position sizing + stop scaling).
# Aligned positionally to ohlc.index, sourced from X_slow or the regime-score
# sidecar (see rgm_column). Absent columns => arrays stay None and the
# per-entry multipliers fall back to neutral (size ×1, stop ×1).
_rgm_trend_series = rgm_column('daily_rgm_trend_score')
_rgm_vol_series = rgm_column('daily_rgm_vol_score')
_rgm_label_series = rgm_column('daily_rgm_label')
rgm_trend_arr = _rgm_trend_series.astype(float).values if _rgm_trend_series is not None else None
rgm_vol_arr = _rgm_vol_series.astype(float).values if _rgm_vol_series is not None else None
rgm_label_arr = _rgm_label_series.astype(float).values if _rgm_label_series is not None else None
if DIRECTION_SOURCE == 'trend':
    if rgm_trend_arr is None:
        print("WARNING: --direction-source trend set but daily_rgm_trend_score is neither in "
              "X_slow nor in regime_scores.parquet (enable daily_rgm_* as model or helper "
              "features + retrain). Falling back to model direction.")
        DIRECTION_SOURCE = 'model'
    else:
        _ts_valid = ~np.isnan(rgm_trend_arr)
        print(f"Direction source: TREND (sign of daily_rgm_trend_score; "
              f"{int(_ts_valid.sum())}/{len(rgm_trend_arr)} bars with valid score, "
              f"long on {int((rgm_trend_arr[_ts_valid] > 0).sum())}, "
              f"short on {int((rgm_trend_arr[_ts_valid] < 0).sum())}; "
              f"model fallback on the rest)")

if REGIME_RISK:
    if rgm_trend_arr is None and rgm_vol_arr is None:
        print("WARNING: --regime-risk set but daily_rgm_trend_score / daily_rgm_vol_score are "
              "neither in X_slow nor in regime_scores.parquet (enable daily_rgm_* as model or "
              "helper features + retrain). Regime risk management DISABLED.")
        REGIME_RISK = False
    else:
        print(f"Regime risk management ENABLED (source=ML scores): "
              f"size *= clip(1 + {REGIME_SIZE_TREND_W}*aligned_trend - {REGIME_SIZE_VOL_W}*vol, "
              f"{REGIME_SIZE_MIN}, {REGIME_SIZE_MAX}); "
              f"stop_pips *= clip(1 + {REGIME_STOP_VOL_W}*(2*vol-1), {REGIME_STOP_MIN_MULT}, {REGIME_STOP_MAX_MULT})")


def regime_risk_multipliers(i, is_long):
    """Return (size_mult, stop_mult) for an entry at bar i given the ML regime.

    size grows with the direction-aligned trend score and shrinks with the
    volatility score; the stop widens in high-vol regimes and tightens in
    low-vol ones. Neutral (1.0, 1.0) when regime risk is off or scores absent.
    """
    if not REGIME_RISK:
        return 1.0, 1.0
    trend = 0.0
    if rgm_trend_arr is not None and not np.isnan(rgm_trend_arr[i]):
        trend = rgm_trend_arr[i] if is_long else -rgm_trend_arr[i]
    vol = 0.5
    if rgm_vol_arr is not None and not np.isnan(rgm_vol_arr[i]):
        vol = rgm_vol_arr[i]
    size_mult = float(np.clip(1.0 + REGIME_SIZE_TREND_W * trend - REGIME_SIZE_VOL_W * vol,
                              REGIME_SIZE_MIN, REGIME_SIZE_MAX))
    stop_mult = float(np.clip(1.0 + REGIME_STOP_VOL_W * (2.0 * vol - 1.0),
                              REGIME_STOP_MIN_MULT, REGIME_STOP_MAX_MULT))
    return size_mult, stop_mult


def position_notional(equity, stop_pips, price):
    """Per-trade notional under the selected risk model.

    'fixed_notional' (default) returns NOTIONAL on every trade regardless of the stop
    distance — the historical behaviour, kept so old runs stay reproducible. It is not a
    risk model: 1,000,000 EUR against 50,000 EUR of capital risks 6.4% of the account on
    a 35-pip stop and would risk 22% on the 120-pip stop a swing design needs. Measured
    on run honest_20260829 it produced a −39.9% maximum drawdown, which makes every
    drawdown- and Sharpe-derived number in the report a statement about the sizing rule
    rather than about the strategy.

    'fixed_fractional' sizes each trade so the stop always costs the same fraction of
    current equity:

        loss_eur ~= notional * stop_pips * pip_size / price
        => notional = risk_eur * price / (stop_pips * pip_size)

    so a wider stop buys a smaller position and the risk per trade stays constant across
    stop distances, regimes and volatility levels. Capped by --max-leverage because a
    very tight stop would otherwise ask for an unbounded position.
    """
    if RISK_MODEL == 'fixed_notional':
        return float(NOTIONAL)
    return risk_utils.position_notional(
        equity, stop_pips, price,
        risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE, pip_size=pip_size,
    )

# Load per-model scalers
print("Loading scalers...")
scaler_by_model = {
    mk: joblib.load(os.path.join(GENERATED_DIR, f"scaler_{mk}.save"))
    for mk in MODEL_KEYS
}

# Load models with new naming convention
print("Loading models...")
model_target_long_fast_path = os.path.join(GENERATED_DIR, "strategy_model_target_long_fast.pkl")
model_target_short_fast_path = os.path.join(GENERATED_DIR, "strategy_model_target_short_fast.pkl")
model_target_long_slow_path = os.path.join(GENERATED_DIR, "strategy_model_target_long_slow.pkl")
model_target_short_slow_path = os.path.join(GENERATED_DIR, "strategy_model_target_short_slow.pkl")

for path in [model_target_long_fast_path, model_target_short_fast_path,
             model_target_long_slow_path, model_target_short_slow_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

bst_target_long_fast = joblib.load(model_target_long_fast_path)
bst_target_short_fast = joblib.load(model_target_short_fast_path)
bst_target_long_slow = joblib.load(model_target_long_slow_path)
bst_target_short_slow = joblib.load(model_target_short_slow_path)

print("Models loaded successfully")

# ============================================================================
# Load Calibrators (optional — applied if present)
# ============================================================================
calibrators = {}
calibrator_names = [
    'target_long_fast', 'target_short_fast',
    'target_long_slow', 'target_short_slow',
]
for name in calibrator_names:
    cal_path = os.path.join(GENERATED_DIR, f"calibrator_{name}.pkl")
    if os.path.exists(cal_path):
        calibrators[name] = joblib.load(cal_path)

if calibrators:
    print(f"Calibrators loaded: {list(calibrators.keys())}")
else:
    print("No calibrators found — using raw probabilities")

# Print active strategy configuration
print("\n" + "="*70)
print("STRATEGY CONFIGURATION")
print("="*70)
print("Closing Strategies:")
print(f"  - Before weekend:                {CLOSING_BEFORE_WEEKEND}")
print(f"  - Time-based (after {HOLD_BARS} bars): {CLOSING_AFTER_TIME}")
print(f"  - Signal reversal (fast model):  {CLOSING_AFTER_SIGNAL_REVERSAL_FAST}")
print(f"  - Signal reversal (slow model):  {CLOSING_AFTER_SIGNAL_REVERSAL_SLOW}")
print(f"  - PnL exit (>{P_CLOSE_PIP_THRESHOLD} pips): {CLOSING_AFTER_X_PIPS}")
print(f"  - Trailing stop:                 {CLOSING_WITH_TRAILING_STOP}")
print(f"  - Level retest:                  {CLOSING_ON_LEVEL_RETEST} (retrace={LEVEL_RETEST_MIN_RETRACE}p, proximity={LEVEL_RETEST_PROXIMITY}p)")
print(f"  - Trend-aware exits:             {TREND_AWARE_EXITS} (ATR trail mult={ATR_TRAIL_MULT}, activation={ATR_TRAIL_ACTIVATION_MULT}xATR, period={ATR_PERIOD})")
print(f"  - Breakeven SL move:             {BREAKEVEN_TRIGGER_PIPS is not None}"
      + (f" (trigger={BREAKEVEN_TRIGGER_PIPS}p, offset={BREAKEVEN_OFFSET_PIPS}p)" if BREAKEVEN_TRIGGER_PIPS is not None else ""))
print(f"  - Ranging harvest:               {CLOSE_ON_RANGING_MIN_PIPS is not None}"
      + (f" (min pnl={CLOSE_ON_RANGING_MIN_PIPS}p)" if CLOSE_ON_RANGING_MIN_PIPS is not None else ""))
print("\nOpening Requirements:")
print(f"  - Fast signal required:          {OPENING_REQUIRES_FAST_SIGNAL}")
print(f"  - Slow signal required:          {OPENING_REQUIRES_SLOW_SIGNAL}")
if not ENTRY_GATES_ACTIVE:
    print("  ! Both requirements are off -> no entry gate remains; NO trades will be opened.")
print("="*70 + "\n")

# ============================================================================
# Generate Predictions
# ============================================================================
print("Generating predictions...")
# Each model has its own feature set + scaler + DMatrix.
def _dmatrix(mk):
    return xgb.DMatrix(scaler_by_model[mk].transform(X_by_model[mk]).astype(np.float32))

# Fast models (entry quality — fast feature set)
probs_long_fast = bst_target_long_fast.predict(_dmatrix('long_fast'))
probs_short_fast = bst_target_short_fast.predict(_dmatrix('short_fast'))

# Slow models (pip target — slow feature set)
probs_long_slow = bst_target_long_slow.predict(_dmatrix('long_slow'))
probs_short_slow = bst_target_short_slow.predict(_dmatrix('short_slow'))

# Apply calibration if calibrators are available
def apply_calibrator(raw_probs, calibrator):
    """Apply a fitted calibrator (Platt or Isotonic) to raw probabilities."""
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    elif isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(raw_probs)
    else:
        return raw_probs

if 'target_long_fast' in calibrators:
    probs_long_fast = apply_calibrator(probs_long_fast, calibrators['target_long_fast'])
if 'target_short_fast' in calibrators:
    probs_short_fast = apply_calibrator(probs_short_fast, calibrators['target_short_fast'])
if 'target_long_slow' in calibrators:
    probs_long_slow = apply_calibrator(probs_long_slow, calibrators['target_long_slow'])
if 'target_short_slow' in calibrators:
    probs_short_slow = apply_calibrator(probs_short_slow, calibrators['target_short_slow'])

if calibrators:
    print(f"Calibration applied to {len(calibrators)} models")

# ============================================================================
# Externally supplied probabilities (--proba-file)
# ============================================================================
# An alternative learner is scored by replacing the four probability vectors here and
# changing nothing else: same bars, same entry gates, same exits, same costs, same
# sizing. Overriding further downstream would compare a model AND an execution engine
# at once, which measures neither.
if args.proba_file:
    _needed = ['long_fast', 'short_fast', 'long_slow', 'short_slow']
    _proba = pd.read_parquet(args.proba_file)
    _missing = [c for c in _needed if c not in _proba.columns]
    if _missing:
        raise SystemExit(
            f"--proba-file {args.proba_file} is missing column(s) {_missing}; "
            f"expected {_needed} indexed by bar timestamp."
        )
    _proba.index = pd.to_datetime(_proba.index)
    _proba = _proba[~_proba.index.duplicated(keep='last')]
    _target_index = X_by_model['long_slow'].index
    _aligned = _proba.reindex(_target_index)

    _covered = int(_aligned[_needed].notna().all(axis=1).sum())
    if _covered == 0:
        raise SystemExit(
            f"--proba-file {args.proba_file} covers none of the {len(_target_index)} "
            f"backtest bars ({_target_index.min()} .. {_target_index.max()}). Its own "
            f"index spans {_proba.index.min()} .. {_proba.index.max()}."
        )
    # Bars the file does not cover must not trade. 0.0 is below every entry threshold,
    # so an uncovered bar is simply never an entry — never a silent 0.5 coin flip.
    _aligned = _aligned[_needed].astype(float).fillna(0.0)

    probs_long_fast = _aligned['long_fast'].to_numpy()
    probs_short_fast = _aligned['short_fast'].to_numpy()
    probs_long_slow = _aligned['long_slow'].to_numpy()
    probs_short_slow = _aligned['short_slow'].to_numpy()

    print(f"Probabilities taken from {args.proba_file}: {_covered}/{len(_target_index)} "
          f"bars covered ({_covered / len(_target_index):.1%}); model inference and "
          f"calibration bypassed.")

if args.use_trained_threshold:
    # The training run already chose an operating point in-fold: run_time_series_cv
    # picks the threshold that maximises precision subject to --target-recall, on
    # pooled validation predictions from the TRAINING data only. Reuse it instead of a
    # hand-set number — choosing a threshold by whichever backtest looks best would be
    # fitting the test window, the same selection-on-noise trap as picking a winner
    # from a large grid.
    #
    # That threshold lives in RAW probability space (calibration is fitted after CV),
    # while the probabilities above are calibrated. Map it forward through each
    # direction's own calibrator: Platt/isotonic are monotone, so thresholding the
    # calibrated probability at f(thr_raw) is exactly thresholding the raw one at
    # thr_raw — but it has to be done per direction, because long and short get
    # different calibrators and one shared calibrated threshold silently means two
    # different raw operating points.
    _summary_path = os.path.join(GENERATED_DIR, 'training_summary.json')
    if not os.path.exists(_summary_path):
        raise SystemExit(
            f"--use-trained-threshold needs {_summary_path}, which does not exist. "
            f"Run the training for this run-id first, or drop the flag."
        )
    with open(_summary_path, encoding='utf-8') as _f:
        _summary = json.load(_f)
    _cv = _summary.get('cv_metrics', {})
    _raw_thr = _cv.get('slow_final_global_threshold', _cv.get('slow_post_global_threshold'))
    if _raw_thr is None:
        raise SystemExit(
            "--use-trained-threshold found no slow_final_global_threshold in "
            "training_summary.json. The run predates the threshold export."
        )

    def _map_threshold(raw_thr, name):
        cal = calibrators.get(name)
        if cal is None:
            return float(raw_thr)
        return float(apply_calibrator(np.array([raw_thr], dtype=float), cal)[0])

    P_OPEN_SLOW_LONG = _map_threshold(_raw_thr, 'target_long_slow')
    P_OPEN_SLOW_SHORT = _map_threshold(_raw_thr, 'target_short_slow')
    P_OPEN_THRESHOLD_SLOW = (P_OPEN_SLOW_LONG + P_OPEN_SLOW_SHORT) / 2.0  # reporting only
    print(f"Trained entry threshold: {_raw_thr:.4f} raw -> "
          f"long {P_OPEN_SLOW_LONG:.4f} / short {P_OPEN_SLOW_SHORT:.4f} calibrated")

    # Same rule for the FAST threshold. It only gates trades when
    # --opening-requires-fast-signal is on, but the confound is identical: a hand-set
    # --p-open-fast is applied to CALIBRATED probabilities, so the same number means a
    # different raw operating point per direction and per label mode. Summaries from
    # before the fast threshold export keep the hand-set value (with a warning when the
    # gate would actually use it). The fast reversal exit is unaffected — it triggers
    # on P_CLOSE_THRESHOLD, not on the entry threshold.
    _fast_thr = _cv.get('fast_final_global_threshold', _cv.get('fast_post_global_threshold'))
    if _fast_thr is not None:
        P_OPEN_FAST_LONG = _map_threshold(_fast_thr, 'target_long_fast')
        P_OPEN_FAST_SHORT = _map_threshold(_fast_thr, 'target_short_fast')
        P_OPEN_THRESHOLD_FAST = (P_OPEN_FAST_LONG + P_OPEN_FAST_SHORT) / 2.0  # reporting only
        print(f"Trained fast threshold:  {_fast_thr:.4f} raw -> "
              f"long {P_OPEN_FAST_LONG:.4f} / short {P_OPEN_FAST_SHORT:.4f} calibrated")
    elif OPENING_REQUIRES_FAST_SIGNAL:
        print(f"WARNING: --use-trained-threshold found no fast_final_global_threshold in "
              f"training_summary.json; the fast entry gate keeps the hand-set "
              f"--p-open-fast ({P_OPEN_THRESHOLD_FAST}).")

print(f"Predictions generated: {len(probs_long_fast)} bars")

# Extract price arrays (needed by signal stats and backtest loop)
timestamps = ohlc.index
prices_open = ohlc["open"].values
prices_close = ohlc["close"].values
prices_low = ohlc["low"].values
prices_high = ohlc["high"].values

pip_size = forex.pip_value_for_symbol(SYMBOL)

# ============================================================================
# Transaction costs
# ============================================================================
# The cost model only converts a DECISION price into a FILL price — it never moves a
# trigger. With --cost-model none every fill equals its decision price, which is the
# behaviour every result before 2026-08 was produced under.
_cost_csv = args.data_csv or os.path.join(dir_config.DATA_DIR, "eurusd_m15.csv")
cost_model, cost_diagnostics = costs.build_cost_model(
    mode=args.cost_model,
    index=timestamps,
    pip_size=pip_size,
    csv_path=_cost_csv,
    spread_pips=args.spread_pips,
    slippage_pips=args.slippage_pips,
    commission_per_million=args.commission_per_million,
    overnight_long_per_million=args.overnight_long_per_million,
    overnight_short_per_million=args.overnight_short_per_million,
    min_spread_pips=args.min_spread_pips,
)
print(cost_model.describe())

# ATR in pips, used by the trend-aware chandelier trail.
# EWM of True Range — matches labeling/window_labels._compute_atr semantics.
_prev_close = np.roll(prices_close, 1); _prev_close[0] = prices_close[0]
_tr = np.maximum(prices_high - prices_low,
                 np.maximum(np.abs(prices_high - _prev_close),
                            np.abs(prices_low - _prev_close)))
atr_pips = (pd.Series(_tr).ewm(span=ATR_PERIOD, adjust=False).mean().values) / pip_size

# ============================================================================
# Signal Statistics Analysis (per month)
# ============================================================================
print("\n" + "="*70)
print("SIGNAL STATISTICS (per month)")
print("="*70)

# Build a DataFrame with all predictions for analysis
signal_df = pd.DataFrame({
    'timestamp': timestamps,
    'p_long_fast': probs_long_fast,
    'p_short_fast': probs_short_fast,
    'p_long_slow': probs_long_slow,
    'p_short_slow': probs_short_slow,
}, index=timestamps)

# Filter to backtest period only
signal_df = signal_df[(signal_df.index >= timeframes.BACKTEST_START) & 
                      (signal_df.index <= timeframes.BACKTEST_END)]

if len(signal_df) > 0:
    # Derive per-bar signals (matching entry logic)
    signal_df['direction_long'] = signal_df['p_long_slow'] > signal_df['p_short_slow']
    signal_df['fast_match'] = np.where(
        signal_df['direction_long'],
        signal_df['p_long_fast'] > P_OPEN_THRESHOLD_FAST,
        signal_df['p_short_fast'] > P_OPEN_THRESHOLD_FAST
    )
    signal_df['slow_match'] = np.where(
        signal_df['direction_long'],
        signal_df['p_long_slow'] >= P_OPEN_SLOW_LONG,
        signal_df['p_short_slow'] >= P_OPEN_SLOW_SHORT
    )
    signal_df['margin_match'] = (
        (signal_df['p_long_slow'] - signal_df['p_short_slow']).abs() > SLOW_DIRECTION_MARGIN
    )
    fast_gate = signal_df['fast_match'] if OPENING_REQUIRES_FAST_SIGNAL else True
    slow_gate = signal_df['slow_match'] if OPENING_REQUIRES_SLOW_SIGNAL else True
    margin_gate = signal_df['margin_match'] if SLOW_DIRECTION_MARGIN > 0 else True
    # Add regime column to signal stats (True = trending day, entry allowed under mode=trending)
    if regime_trending_arr is not None:
        regime_series = pd.Series(regime_trending_arr, index=ohlc.index, name='regime_trending')
        signal_df['regime_trending'] = regime_series.reindex(signal_df.index, fill_value=False)
        if REGIME_GATE_MODE == 'trending':
            regime_gate = signal_df['regime_trending']
        elif REGIME_GATE_MODE == 'ranging':
            regime_gate = ~signal_df['regime_trending']
        else:
            regime_gate = True
    else:
        signal_df['regime_trending'] = True
        regime_gate = True
    if ENTRY_GATES_ACTIVE:
        signal_df['all_match'] = fast_gate & slow_gate & regime_gate & margin_gate
    else:
        signal_df['all_match'] = False
    signal_df['month'] = signal_df.index.to_period('M')

    # Monthly breakdown
    monthly_stats = signal_df.groupby('month').agg(
        total_bars=('fast_match', 'count'),
        fast_signals=('fast_match', 'sum'),
        slow_signals=('slow_match', 'sum'),
        regime_bars=('regime_trending', 'sum'),
        all_signals=('all_match', 'sum'),
    )
    monthly_stats['fast_pct'] = (monthly_stats['fast_signals'] / monthly_stats['total_bars'] * 100)
    monthly_stats['slow_pct'] = (monthly_stats['slow_signals'] / monthly_stats['total_bars'] * 100)
    monthly_stats['regime_pct'] = (monthly_stats['regime_bars'] / monthly_stats['total_bars'] * 100)
    monthly_stats['all_pct'] = (monthly_stats['all_signals'] / monthly_stats['total_bars'] * 100)

    print(f"\n{'Month':<10} {'Bars':>6} {'Fast':>8} {'Slow':>8} {'Regime':>9} {'All':>8}")
    print(f"{'':10} {'':>6} {'(>' + str(P_OPEN_THRESHOLD_FAST) + ')':>8} {'(>' + str(P_OPEN_THRESHOLD_SLOW) + ')':>8} {'(trend)':>9} {'(combined)':>10}")
    print("-" * 72)
    for month, row in monthly_stats.iterrows():
        print(f"{str(month):<10} {int(row['total_bars']):>6} "
              f"{int(row['fast_signals']):>5} ({row['fast_pct']:4.1f}%) "
              f"{int(row['slow_signals']):>5} ({row['slow_pct']:4.1f}%) "
              f"{int(row['regime_bars']):>6} ({row['regime_pct']:4.1f}%) "
              f"{int(row['all_signals']):>5} ({row['all_pct']:4.1f}%)")

    # Totals
    total_bars = monthly_stats['total_bars'].sum()
    total_fast = monthly_stats['fast_signals'].sum()
    total_slow = monthly_stats['slow_signals'].sum()
    total_regime = monthly_stats['regime_bars'].sum()
    total_all = monthly_stats['all_signals'].sum()
    print("-" * 72)
    print(f"{'TOTAL':<10} {int(total_bars):>6} "
          f"{int(total_fast):>5} ({total_fast/total_bars*100:4.1f}%) "
          f"{int(total_slow):>5} ({total_slow/total_bars*100:4.1f}%) "
          f"{int(total_regime):>6} ({total_regime/total_bars*100:4.1f}%) "
          f"{int(total_all):>5} ({total_all/total_bars*100:4.1f}%)")

    # ========================================================================
    # Probability Distribution (histogram bins 0.0–0.1, 0.1–0.2, … 0.9–1.0)
    # ========================================================================
    print("\n" + "="*70)
    print("PROBABILITY DISTRIBUTIONS (bin counts)")
    print("="*70)

    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{b:.1f}-{b+0.1:.1f}" for b in bins[:-1]]

    prob_columns = {
        'Long Fast':  signal_df['p_long_fast'],
        'Short Fast': signal_df['p_short_fast'],
        'Long Slow':  signal_df['p_long_slow'],
        'Short Slow': signal_df['p_short_slow'],
    }

    # Print header
    header = f"{'Bin':<12}"
    for name in prob_columns:
        header += f" {name:>12}"
    print(header)
    print("-" * (12 + 13 * len(prob_columns)))

    # Compute bin counts
    bin_counts = {}
    for name, series in prob_columns.items():
        counts, _ = np.histogram(series.values, bins=bins)
        bin_counts[name] = counts

    for i, label in enumerate(bin_labels):
        row = f"{label:<12}"
        for name in prob_columns:
            c = bin_counts[name][i]
            pct = c / len(signal_df) * 100
            row += f" {c:>6} ({pct:4.1f}%)"
        print(row)

print("")

# ============================================================================
# Helper Functions (matching Java)
# ============================================================================

def calculate_trailing_stop(pnl_pips):
    """
    Fixed-pip trailing stop: trail TRAILING_DISTANCE_PIPS behind the peak.
    Only activates once profit exceeds TRAILING_ACTIVATION_PIPS.
    Returns the minimum profit level that must be maintained, or None if not yet active.
    """
    if pnl_pips < TRAILING_ACTIVATION_PIPS:
        return None   # not yet active — let the trade breathe
    return pnl_pips - TRAILING_DISTANCE_PIPS

def is_weekend_soon(timestamp):
    """Check if it's close to weekend (matches Java logic)"""
    weekday = timestamp.dayofweek  # Monday=0, Sunday=6
    hour = timestamp.hour
    # Friday after 4pm, or Saturday/Sunday
    return (weekday == 4 and hour >= 16) or weekday == 5 or weekday == 6

# ============================================================================
# Backtest Loop (MATCHING JAVA STRATEGY EXACTLY)
# ============================================================================
print("Running backtest (Java strategy logic)...")
trades = []
position = 0  # 0 = flat, 1 = long, -1 = short
entry_idx = None
entry_price = None
entry_fill = None                 # entry price after crossing the book (P&L only)
planned_exit_idx = None
stop_price = None
entry_notional = NOTIONAL         # per-trade notional (regime-scaled when --regime-risk)
highest_profit_pips = 0
lowest_pnl_pips = 0              # most negative unrealized pnl reached during trade (<= 0)
has_retraced_from_peak = False   # level retest state
equity = START_CAPITAL
trading_suspended = False
suspended_date = None
entry_count = 0  # Debug counter

for i in range(n):
    current_timestamp = timestamps[i]
    
    # Skip if outside backtest timeframe
    if current_timestamp < timeframes.BACKTEST_START or current_timestamp > timeframes.BACKTEST_END:
        continue
    current_date = current_timestamp.date()
    current_price = prices_close[i]  # Java uses askBar.getClose()
    
    p_long_fast = float(np.float32(probs_long_fast[i]))
    p_short_fast = float(np.float32(probs_short_fast[i]))
    p_long_slow = float(np.float32(probs_long_slow[i]))
    p_short_slow = float(np.float32(probs_short_slow[i]))

    if ENABLE_BACKTEST_LOGGING:
        fast_features_float32 = X_fast.iloc[i].values.astype(np.float32)
        slow_features_float32 = X_slow.iloc[i].values.astype(np.float32)

        print(f"{current_timestamp}: Model predictions p_long_fast: {p_long_fast:.3f} | p_short_fast: {p_short_fast:.3f} | p_long_slow: {p_long_slow:.3f} | p_short_slow: {p_short_slow:.3f}")
        print(f"  OHLC: O: {ohlc.iloc[i]['open']:.5f} | H: {ohlc.iloc[i]['high']:.5f} | L: {ohlc.iloc[i]['low']:.5f} | C: {ohlc.iloc[i]['close']:.5f}")

        fast_features_formatted = [f"{float(v):.6f}" for v in fast_features_float32]
        slow_features_formatted = [f"{float(v):.6f}" for v in slow_features_float32]
        print(f"  Fast features: [{', '.join(fast_features_formatted)}]")
        print(f"  Slow features: [{', '.join(slow_features_formatted)}]")

        continue  # Skip trading logic for logging only
    
    # Reset trading suspension on new day (matches Java)
    if trading_suspended and suspended_date != current_date:
        trading_suspended = False
        suspended_date = None
        if DEBUG_MODE:
            print(f"{current_timestamp}: Trading suspension released")
    
    # Skip if suspended and no position open
    if trading_suspended and position == 0:
        continue
    
    weekend_soon = is_weekend_soon(current_timestamp)
    
    # ========================================================================
    # MANAGE EXISTING POSITION (matches Java managePosition)
    # ========================================================================
    if position != 0:
        periods_held = i - entry_idx
        pnl_pips = ((current_price - entry_price) / pip_size) if position == 1 else ((entry_price - current_price) / pip_size)
        
        p_fast = p_long_fast if position == 1 else p_short_fast
        p_slow = p_long_slow if position == 1 else p_short_slow
        
        exit_reason = None
        close_price = None

        # Java exit priority order (EXACT MATCH with configurable flags):

        # 2. Weekend protection (if enabled)
        if exit_reason is None and CLOSING_BEFORE_WEEKEND:
            if weekend_soon and pnl_pips > 0:
                close_price = current_price
                exit_reason = "weekend_close"

        # 2b. Harvest profitable trade when regime turns ranging (if enabled)
        if (exit_reason is None
                and CLOSE_ON_RANGING_MIN_PIPS is not None
                and regime_trending_arr is not None
                and not bool(regime_trending_arr[i])
                and pnl_pips >= CLOSE_ON_RANGING_MIN_PIPS):
            close_price = current_price
            exit_reason = "ranging_harvest"

        # 3. Stop loss (always checked, uses intrabar low/high to match Java)
        if exit_reason is None:
            if position == 1 and prices_low[i] <= stop_price:
                close_price = stop_price
                exit_reason = "stop_loss"
            elif position == -1 and prices_high[i] >= stop_price:
                close_price = stop_price
                exit_reason = "stop_loss"
        
        # 4. Time exit (if enabled)
        if exit_reason is None and CLOSING_AFTER_TIME:
            if periods_held >= HOLD_BARS:
                close_price = current_price
                exit_reason = "time_exit"
        
        # 5. Signal reversal - fast model (if enabled)
        if exit_reason is None and CLOSING_AFTER_SIGNAL_REVERSAL_FAST:
            if p_fast < P_CLOSE_THRESHOLD:
                close_price = current_price
                exit_reason = "signal_reversal_fast"
        
        # 6. Signal reversal - slow model (if enabled)
        if exit_reason is None and CLOSING_AFTER_SIGNAL_REVERSAL_SLOW:
            if p_slow < P_CLOSE_THRESHOLD:
                close_price = current_price
                exit_reason = "signal_reversal_slow"
        
        # 7. Track highest profit (Java does this BEFORE trailing stop check)
        if pnl_pips > highest_profit_pips:
            highest_profit_pips = pnl_pips
            has_retraced_from_peak = False   # reset: new peak invalidates prior retrace flag
        if pnl_pips < lowest_pnl_pips:
            lowest_pnl_pips = pnl_pips

        # 7a. Breakeven stop move — once MFE >= trigger, tighten stop to entry +/- offset.
        # Updates stop_price after the SL check above, so it takes effect from the NEXT bar.
        if BREAKEVEN_TRIGGER_PIPS is not None and highest_profit_pips >= BREAKEVEN_TRIGGER_PIPS:
            if position == 1:
                be_stop = entry_price + BREAKEVEN_OFFSET_PIPS * pip_size
                if be_stop > stop_price:
                    stop_price = be_stop
            else:
                be_stop = entry_price - BREAKEVEN_OFFSET_PIPS * pip_size
                if be_stop < stop_price:
                    stop_price = be_stop

        # 7b. Level retest state machine
        if CLOSING_ON_LEVEL_RETEST and highest_profit_pips >= LEVEL_RETEST_MIN_RETRACE:
            # Mark that price has retraced enough from the peak
            if not has_retraced_from_peak and pnl_pips <= highest_profit_pips - LEVEL_RETEST_MIN_RETRACE:
                has_retraced_from_peak = True

        # Trend-aware exits: in a trending bar, the fixed-pip TP caps trend trades
        # at 1R. Skip it (the chandelier trail at step 8b will manage the exit).
        in_trend_bar = (
            TREND_AWARE_EXITS
            and regime_trending_arr is not None
            and bool(regime_trending_arr[i])
        )

        # 8. Close with realized pips threshold (skipped in trend bars when trend-aware exits on)
        if exit_reason is None and CLOSING_AFTER_X_PIPS and not in_trend_bar:
            if pnl_pips > P_CLOSE_PIP_THRESHOLD:
                close_price = current_price
                exit_reason = "pips_threshold"

        # 8b. Chandelier ATR trail (trend bars only, when trend-aware exits enabled)
        if exit_reason is None and in_trend_bar:
            atr_p = atr_pips[i]
            if atr_p > 0 and highest_profit_pips >= ATR_TRAIL_ACTIVATION_MULT * atr_p:
                trail_level = highest_profit_pips - ATR_TRAIL_MULT * atr_p
                if pnl_pips <= trail_level:
                    close_price = current_price
                    exit_reason = "atr_trail"

        # 9. Level retest exit (if enabled and no other exit triggered)
        if exit_reason is None and CLOSING_ON_LEVEL_RETEST and has_retraced_from_peak:
            if pnl_pips >= highest_profit_pips - LEVEL_RETEST_PROXIMITY:
                close_price = current_price
                exit_reason = "level_retest"

        # 10. Trailing stop (if enabled and no other exit triggered)
        if exit_reason is None and CLOSING_WITH_TRAILING_STOP:
            trail_level = calculate_trailing_stop(highest_profit_pips)
            if trail_level is not None and pnl_pips < trail_level:
                close_price = current_price
                exit_reason = "trailing_stop"
        
        # Close position if exit condition met
        if exit_reason is not None:
            # close_price stays the DECISION price (it is what the triggers above
            # compared against); the fill crosses the book and pays the slippage.
            exit_fill = cost_model.exit_fill(i, close_price, position == 1)
            commission_eur = cost_model.commission_eur(entry_notional)
            rollover_nights = costs.count_rollover_nights(trade["open_time"], current_timestamp)
            overnight_eur = cost_model.overnight_eur(entry_notional, rollover_nights, position == 1)

            gross_pnl_pips = ((close_price - entry_price) / pip_size) if position == 1 else ((entry_price - close_price) / pip_size)
            final_pnl_pips = ((exit_fill - entry_fill) / pip_size) if position == 1 else ((entry_fill - exit_fill) / pip_size)
            pnl_eur = (entry_notional * ((exit_fill - entry_fill) / entry_fill)
                       if position == 1 else
                       entry_notional * ((entry_fill - exit_fill) / entry_fill)) - commission_eur - overnight_eur
            equity += pnl_eur

            trade["close_time"] = current_timestamp
            trade["close_price"] = float(close_price)
            trade["close_fill_price"] = float(exit_fill)
            trade["pnl"] = float(pnl_eur)
            trade["pnl_pips"] = float(final_pnl_pips)
            trade["pnl_pips_gross"] = float(gross_pnl_pips)
            trade["cost_pips"] = float(gross_pnl_pips - final_pnl_pips)
            trade["commission_eur"] = float(commission_eur)
            trade["overnight_eur"] = float(overnight_eur)
            trade["rollover_nights"] = int(rollover_nights)
            trade["duration_bars"] = int(i - entry_idx)
            trade["exit_reason"] = exit_reason
            trade["highest_profit_pips"] = float(highest_profit_pips)
            trade["lowest_pnl_pips"] = float(lowest_pnl_pips)
            
            if position == 1:
                trade["prob_long_fast_close"] = float(p_fast)
                trade["prob_long_slow_close"] = float(p_slow)
            else:
                trade["prob_short_fast_close"] = float(p_fast)
                trade["prob_short_slow_close"] = float(p_slow)

            trades.append(trade)
            
            if DEBUG_MODE and len(trades) <= 10:
                print(f"{current_timestamp}: CLOSE {trade['action']} @ {trade['close_price']} | Reason: {exit_reason} | PnL: {pnl_pips:.1f} pips | Duration: {periods_held} bars")
                print("  OHLC:", ohlc.iloc[i].to_dict())
                print("  Fast features:", X_fast.iloc[i].to_dict())
                print("  Slow features:", X_slow.iloc[i].to_dict())
                print(80 * "=")
                print("\n")
            
            # Suspend trading if stop loss hit (matches Java)
            if exit_reason == "stop_loss":
                trading_suspended = True
                suspended_date = current_date
                if DEBUG_MODE:
                    print(f"{current_timestamp}: Trading suspended until tomorrow")
            
            # Reset position state
            position = 0
            entry_idx = None
            entry_price = None
            entry_fill = None
            planned_exit_idx = None
            stop_price = None
            entry_notional = NOTIONAL
            highest_profit_pips = 0
            lowest_pnl_pips = 0
            has_retraced_from_peak = False
            continue
    
    # ========================================================================
    # ENTRY LOGIC (matches Java openPosition with configurable flags)
    # ========================================================================
    if ENTRY_GATES_ACTIVE and position == 0 and not weekend_soon and not trading_suspended:
        # Regime gate: skip entries on the wrong-side regime per REGIME_GATE_MODE.
        # mode=trending -> skip ranging bars; mode=ranging -> skip trending bars; mode=off -> no gate.
        if regime_trending_arr is not None and REGIME_GATE_MODE != 'off':
            if REGIME_GATE_MODE == 'trending' and not regime_trending_arr[i]:
                continue
            if REGIME_GATE_MODE == 'ranging' and regime_trending_arr[i]:
                continue

        # Slow model determines direction (daily trend); fast model gates entry timing
        if SLOW_DIRECTION_MARGIN > 0 and abs(p_long_slow - p_short_slow) <= SLOW_DIRECTION_MARGIN:
            continue
        direction_is_long = (p_long_slow > p_short_slow)

        # --direction-source trend: follow the ML regime's trend sign instead of the
        # slow-model probabilities (which measure ~0.50 AUC within trend bars). This
        # is momentum-following, not prediction; per-bar fallback to the model
        # direction when the score is NaN or exactly 0.
        if DIRECTION_SOURCE == 'trend':
            _ts = rgm_trend_arr[i]
            if not np.isnan(_ts) and _ts != 0.0:
                direction_is_long = _ts > 0

        # Regime-scaled position size and stop distance (neutral unless --regime-risk).
        # The stop multiplier is resolved FIRST: under fixed_fractional sizing a wider
        # regime stop must buy a smaller position, otherwise the two knobs multiply the
        # risk instead of holding it constant.
        _size_mult, _stop_mult = regime_risk_multipliers(i, direction_is_long)
        eff_stop_pips = STOP_PIPS * _stop_mult
        entry_notional = position_notional(equity, eff_stop_pips, current_price) * _size_mult
        if entry_notional <= 0:
            continue

        if direction_is_long:
            # Try LONG entry (matches Java)
            # Check fast signal requirement
            fast_signal_ok = (not OPENING_REQUIRES_FAST_SIGNAL) or (p_long_fast > P_OPEN_FAST_LONG)

            if fast_signal_ok and current_price > 0:
                # Check slow model requirement. Both requirements are hard AND-gates:
                # an enabled flag can only ever REMOVE entries, and can never be
                # satisfied by the other model's probability.
                prob_slow_boolean = (not OPENING_REQUIRES_SLOW_SIGNAL) or (p_long_slow >= P_OPEN_SLOW_LONG)
                
                if prob_slow_boolean:
                    # Enter LONG
                    entry_idx = i
                    entry_price = current_price
                    entry_fill = cost_model.entry_fill(i, current_price, True)
                    position = 1
                    planned_exit_idx = min(entry_idx + HOLD_BARS, n - 1)
                    stop_price = entry_price - eff_stop_pips * pip_size
                    entry_count += 1

                    trade = {
                        "open_time": current_timestamp,
                        "action": "BUY",
                        "open_price": float(entry_price),
                        "open_fill_price": float(entry_fill),
                        "prob_long_fast_open": float(p_long_fast),
                        "prob_long_slow_open": float(p_long_slow),
                        "stop_price": float(stop_price),
                        "notional": float(entry_notional),
                        "close_time": None,
                        "close_price": None,
                        "pnl": None,
                        "pnl_pips": None,
                        "duration_bars": None,
                        "exit_reason": None,
                        "highest_profit_pips": None,
                        "lowest_pnl_pips": None,
                    }
                    
                    if DEBUG_MODE and entry_count <= 10:
                        print(f"{current_timestamp}: OPEN BUY @ {entry_price:.5f} | P_fast: {p_long_fast:.3f} | P_slow: {p_long_slow:.3f}")
                        print("  OHLC:", ohlc.iloc[i].to_dict())
                        print("  Fast features:", X_fast.iloc[i].to_dict())
                        print("  Slow features:", X_slow.iloc[i].to_dict())
                        print("\n")
        
        else:
            # Try SHORT entry (matches Java)
            # Check fast signal requirement
            fast_signal_ok = (not OPENING_REQUIRES_FAST_SIGNAL) or (p_short_fast > P_OPEN_FAST_SHORT)

            if fast_signal_ok and current_price > 0:
                # Check slow model requirement. Both requirements are hard AND-gates:
                # an enabled flag can only ever REMOVE entries, and can never be
                # satisfied by the other model's probability.
                prob_slow_boolean = (not OPENING_REQUIRES_SLOW_SIGNAL) or (p_short_slow >= P_OPEN_SLOW_SHORT)
                
                if prob_slow_boolean:
                    # Enter SHORT
                    entry_idx = i
                    entry_price = current_price
                    entry_fill = cost_model.entry_fill(i, current_price, False)
                    position = -1
                    planned_exit_idx = min(entry_idx + HOLD_BARS, n - 1)
                    stop_price = entry_price + eff_stop_pips * pip_size
                    entry_count += 1

                    trade = {
                        "open_time": current_timestamp,
                        "action": "SELL",
                        "open_price": float(entry_price),
                        "open_fill_price": float(entry_fill),
                        "prob_short_fast_open": float(p_short_fast),
                        "prob_short_slow_open": float(p_short_slow),
                        "stop_price": float(stop_price),
                        "notional": float(entry_notional),
                        "close_time": None,
                        "close_price": None,
                        "pnl": None,
                        "pnl_pips": None,
                        "duration_bars": None,
                        "exit_reason": None,
                        "highest_profit_pips": None,
                        "lowest_pnl_pips": None,
                    }
                    
                    if DEBUG_MODE and entry_count <= 10:
                        print(f"{current_timestamp}: OPEN SELL @ {entry_price:.5f} | P_fast: {p_short_fast:.3f} | P_slow: {p_short_slow:.3f}")
                        print("  OHLC:", ohlc.iloc[i].to_dict())
                        print("  Fast features:", X_fast.iloc[i].to_dict())
                        print("  Slow features:", X_slow.iloc[i].to_dict())
                        print("\n")

# ============================================================================
# Close any remaining position at end of data
# ============================================================================
if position != 0 and entry_idx is not None:
    last_idx = n - 1
    close_price = prices_close[last_idx]
    current_price = close_price

    # entry_notional, not NOTIONAL: with --regime-risk the position was opened at a
    # scaled size, and settling it at the unscaled one misstates the last trade.
    exit_fill = cost_model.exit_fill(last_idx, close_price, position == 1)
    commission_eur = cost_model.commission_eur(entry_notional)
    rollover_nights = costs.count_rollover_nights(trade["open_time"], timestamps[last_idx])
    overnight_eur = cost_model.overnight_eur(entry_notional, rollover_nights, position == 1)

    if position == 1:
        gross_pnl_pips = (close_price - entry_price) / pip_size
        pnl_pips = (exit_fill - entry_fill) / pip_size
        pnl_eur = entry_notional * ((exit_fill - entry_fill) / entry_fill) - commission_eur - overnight_eur
        trade["prob_long_fast_close"] = float(probs_long_fast[last_idx])
        trade["prob_long_slow_close"] = float(probs_long_slow[last_idx])
    else:
        gross_pnl_pips = (entry_price - close_price) / pip_size
        pnl_pips = (entry_fill - exit_fill) / pip_size
        pnl_eur = entry_notional * ((entry_fill - exit_fill) / entry_fill) - commission_eur - overnight_eur
        trade["prob_short_fast_close"] = float(probs_short_fast[last_idx])
        trade["prob_short_slow_close"] = float(probs_short_slow[last_idx])

    equity += pnl_eur
    trade["close_time"] = timestamps[last_idx]
    trade["close_price"] = float(close_price)
    trade["close_fill_price"] = float(exit_fill)
    trade["pnl"] = float(pnl_eur)
    trade["pnl_pips"] = float(pnl_pips)
    trade["pnl_pips_gross"] = float(gross_pnl_pips)
    trade["cost_pips"] = float(gross_pnl_pips - pnl_pips)
    trade["commission_eur"] = float(commission_eur)
    trade["overnight_eur"] = float(overnight_eur)
    trade["rollover_nights"] = int(rollover_nights)
    trade["duration_bars"] = int(last_idx - entry_idx)
    trade["exit_reason"] = "end_of_data"
    trade["highest_profit_pips"] = float(highest_profit_pips)
    trade["lowest_pnl_pips"] = float(lowest_pnl_pips)
    
    trades.append(trade)

print(f"\nBacktest complete: {len(trades)} trades executed")
print(f"Total entry signals evaluated: {entry_count}")

csv_path = os.path.join(REPORT_DIR, "trade_list.csv")
pdf_path = os.path.join(REPORT_DIR, "backtest_report.pdf")

# ============================================================================
# Generate Report
# ============================================================================
# Clear existing trade_list.csv if it exists
csv_path = os.path.join(REPORT_DIR, "trade_list.csv")
if os.path.exists(csv_path):
    os.remove(csv_path)

# Full measurement provenance for the summary. Measured need (2026-09-12): a run
# with --opening-requires-fast-signal produced 16 trades against 19 for the default
# invocation, and the two summaries were indistinguishable — thresholds, costs and
# risk model were recorded, the entry gates and every other flag were not. The
# summary must let two runs be compared without guessing the command line.
SETTINGS_PROVENANCE = {
    'command_line': ' '.join(sys.argv),
    'args': provenance.sanitize_for_json(vars(args)),
    'entry_gates': {
        'opening_requires_fast_signal': bool(OPENING_REQUIRES_FAST_SIGNAL),
        'opening_requires_slow_signal': bool(OPENING_REQUIRES_SLOW_SIGNAL),
        'entry_gates_active': bool(ENTRY_GATES_ACTIVE),
    },
    'regime_gate': {
        'mode': REGIME_GATE_MODE,
        # None = no feature source found in X_slow -> the gate silently disabled
        # itself (the documented ML-arm pitfall); recording it makes that visible.
        'source': _regime_source,
        'active': regime_trending_arr is not None,
    },
    'exits': {
        'stop_pips': float(STOP_PIPS),
        'hold_bars': HOLD_BARS,
        'closing_before_weekend': bool(CLOSING_BEFORE_WEEKEND),
        'closing_after_time': bool(CLOSING_AFTER_TIME),
        'closing_after_signal_reversal_fast': bool(CLOSING_AFTER_SIGNAL_REVERSAL_FAST),
        'closing_after_signal_reversal_slow': bool(CLOSING_AFTER_SIGNAL_REVERSAL_SLOW),
        'closing_with_trailing_stop': bool(CLOSING_WITH_TRAILING_STOP),
        'closing_on_level_retest': bool(CLOSING_ON_LEVEL_RETEST),
        'trend_aware_exits': bool(TREND_AWARE_EXITS),
        'close_on_ranging_min_pips': CLOSE_ON_RANGING_MIN_PIPS,
    },
}

if len(trades) == 0:
    print("No trades generated. Check thresholds and model outputs.")
    # Create empty CSV with proper headers for metrics extraction
    empty_df = pd.DataFrame(columns=[
        'open_time', 'action', 'open_price', 'open_fill_price',
        'prob_long_fast_open', 'prob_long_slow_open',
        'prob_short_fast_open', 'prob_short_slow_open', 'stop_price',
        'close_time', 'close_price', 'close_fill_price', 'pnl', 'pnl_pips',
        'pnl_pips_gross', 'cost_pips', 'commission_eur', 'overnight_eur',
        'rollover_nights', 'duration_bars', 'exit_reason',
        'highest_profit_pips', 'lowest_pnl_pips', 'prob_long_fast_close', 'prob_long_slow_close',
        'prob_short_fast_close', 'prob_short_slow_close'
    ])
    empty_df.to_csv(csv_path, index=False)
    print(f"Empty trade list saved: {csv_path}")

    summary_path = os.path.join(REPORT_DIR, "backtest_summary.json")
    summary = {
        'period_start': timeframes.BACKTEST_START.strftime('%Y-%m-%d'),
        'period_end': timeframes.BACKTEST_END.strftime('%Y-%m-%d'),
        'thresholds': {
            'fast': float(P_OPEN_THRESHOLD_FAST),
            'slow': float(P_OPEN_THRESHOLD_SLOW),
            'close': float(P_CLOSE_THRESHOLD),
        },
        'total_trades': 0,
        'note': 'No trades generated.',
        'settings': SETTINGS_PROVENANCE,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")
    df_trades = None
else:
    df_trades = pd.DataFrame(trades)
    
    # Compute cumulative equity
    df_trades["cum_pnl"] = df_trades["pnl"].cumsum()
    df_trades["equity_after"] = START_CAPITAL + df_trades["cum_pnl"]
    
    # Save trade list CSV
    df_trades.to_csv(csv_path, index=False)
    print(f"Trade list saved: {csv_path}")
    
    # Print statistics
    total_pnl = df_trades["pnl"].sum()
    total_pips = df_trades["pnl_pips"].sum()

    # Gross vs net. Every result produced before the cost model existed is the GROSS
    # column: both legs filled at the ASK close, so one full spread per round trip was
    # never charged. Reporting both makes the size of that omission visible instead of
    # leaving it in the headline number.
    total_pips_gross = float(df_trades["pnl_pips_gross"].sum())
    total_cost_pips = float(df_trades["cost_pips"].sum())
    total_commission = float(df_trades["commission_eur"].sum())
    total_overnight = float(df_trades["overnight_eur"].sum())
    cost_share = (
        total_cost_pips / total_pips_gross * 100.0
        if total_pips_gross > 0 and total_cost_pips > 0 else None
    )

    win_trades = df_trades[df_trades["pnl"] > 0]
    loss_trades = df_trades[df_trades["pnl"] <= 0]
    win_rate = len(win_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0
    
    # Risk-adjusted metrics: build daily equity curve from trades, then derive
    # Sharpe (annualized, 252 trading days), Sortino (downside-only volatility),
    # Max Drawdown (peak-to-trough on the equity curve), and Calmar (CAGR/MaxDD).
    daily_index = pd.date_range(
        start=pd.to_datetime(timeframes.BACKTEST_START).normalize(),
        end=pd.to_datetime(timeframes.BACKTEST_END).normalize(),
        freq='D',
    )
    equity_events = pd.Series(
        df_trades['equity_after'].values,
        index=pd.to_datetime(df_trades['close_time']),
    ).sort_index()
    # Resample to daily (last close-equity of each day), then align to full
    # backtest daily index. Days before the first trade inherit START_CAPITAL.
    daily_equity = equity_events.resample('1D').last()
    daily_equity = daily_equity.reindex(daily_index).ffill().fillna(START_CAPITAL)
    daily_returns = daily_equity.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        downside = daily_returns[daily_returns < 0]
        sortino_ratio = (
            daily_returns.mean() / downside.std() * np.sqrt(252)
            if len(downside) > 1 and downside.std() > 0 else float('nan')
        )
    else:
        sharpe_ratio = float('nan')
        sortino_ratio = float('nan')

    running_peak = daily_equity.cummax()
    drawdown_series = (daily_equity - running_peak) / running_peak
    max_drawdown_pct = float(drawdown_series.min() * 100) if len(drawdown_series) > 0 else 0.0
    max_drawdown_eur = float((daily_equity - running_peak).min()) if len(daily_equity) > 0 else 0.0

    period_days = max((daily_index[-1] - daily_index[0]).days, 1)

    def _annualize(final_value, base_value):
        """
        Annualized growth rate in percent, or NaN when it is undefined.

        A wiped-out account gives a non-positive final value; raising that negative
        base to a fractional power yields a complex number in Python, so guard it
        instead of letting float() blow up the whole backtest.
        """
        if base_value <= 0 or final_value <= 0:
            return float('nan')
        return ((final_value / base_value) ** (365.0 / period_days) - 1) * 100

    # CAGR on margin: leverage-aware, "how fast does my account grow".
    cagr = _annualize(float(daily_equity.iloc[-1]), START_CAPITAL)
    # CAGR on notional: leverage-stripped, "what's the per-position-size edge".
    # Same PnL, but base = full position size instead of margin. Under fixed_fractional
    # sizing the notional varies per trade, so the base is the mean traded notional.
    mean_notional = float(df_trades['notional'].mean()) if 'notional' in df_trades else float(NOTIONAL)
    cagr_on_notional = _annualize(mean_notional + total_pnl, mean_notional)
    leverage = mean_notional / START_CAPITAL if START_CAPITAL > 0 else float('nan')
    calmar_ratio = cagr / abs(max_drawdown_pct) if max_drawdown_pct < 0 else float('nan')

    profit_factor = (
        win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum())
        if len(loss_trades) > 0 and loss_trades['pnl'].sum() != 0 else float('nan')
    )

    avg_win_pnl = win_trades['pnl'].mean() if len(win_trades) > 0 else 0.0
    avg_win_pips = win_trades['pnl_pips'].mean() if len(win_trades) > 0 else 0.0
    avg_loss_pnl = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0.0
    avg_loss_pips = loss_trades['pnl_pips'].mean() if len(loss_trades) > 0 else 0.0

    print("\n" + "="*70)
    print("BACKTEST SUMMARY (Java Strategy Logic)")
    print(f"Period: {timeframes.BACKTEST_START.strftime('%Y-%m-%d')} to {timeframes.BACKTEST_END.strftime('%Y-%m-%d')}")
    print("="*70)
    print(f"Total Trades:        {len(df_trades)}")
    print(f"Winning Trades:      {len(win_trades)} ({win_rate:.1f}%)")
    print(f"Losing Trades:       {len(loss_trades)}")
    print(f"Total PnL:           EUR {total_pnl:,.2f} ({total_pips:.1f} pips)")
    print(f"Avg PnL/Trade:       EUR {df_trades['pnl'].mean():,.2f} ({df_trades['pnl_pips'].mean():.1f} pips)")
    print(f"Avg Win PnL/Trade:   EUR {avg_win_pnl:,.2f} ({avg_win_pips:.1f} pips)")
    print(f"Avg Loss PnL/Trade:  EUR {avg_loss_pnl:,.2f} ({avg_loss_pips:.1f} pips)")
    print(f"Best Trade:          EUR {df_trades['pnl'].max():,.2f} ({df_trades['pnl_pips'].max():.1f} pips)")
    print(f"Worst Trade:         EUR {df_trades['pnl'].min():,.2f} ({df_trades['pnl_pips'].min():.1f} pips)")
    print(f"Start Capital:       EUR {START_CAPITAL:,.2f}")
    print(f"Final Equity:        EUR {df_trades['equity_after'].iloc[-1]:,.2f}")
    print(f"Return:              {(total_pnl / START_CAPITAL * 100):.2f}%")
    if RISK_MODEL == 'fixed_fractional':
        print(f"Notional/Trade:      EUR {mean_notional:,.0f} mean "
              f"(min {df_trades['notional'].min():,.0f} / max {df_trades['notional'].max():,.0f})"
              f"  — fixed_fractional, {RISK_PCT:.2f}% equity risk, max {MAX_LEVERAGE:.0f}x")
    else:
        print(f"Notional/Trade:      EUR {NOTIONAL:,.0f}  (Leverage {leverage:.1f}x on "
              f"{START_CAPITAL:,.0f} margin) — fixed_notional: risk per trade scales with "
              f"the stop, so drawdown figures describe the sizing rule, not the strategy")
    print(f"CAGR (on margin):    {cagr:.2f}%" if not np.isnan(cagr)
          else "CAGR (on margin):    N/A (equity wiped out)")
    print(f"CAGR (on notional):  {cagr_on_notional:.2f}%   (leverage-stripped)"
          if not np.isnan(cagr_on_notional)
          else "CAGR (on notional):  N/A (equity wiped out)")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "Sharpe Ratio:        N/A")
    print(f"Sortino Ratio:       {sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "Sortino Ratio:       N/A")
    print(f"Max Drawdown:        {max_drawdown_pct:.2f}% (EUR {max_drawdown_eur:,.2f})")
    print(f"Calmar Ratio:        {calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "Calmar Ratio:        N/A")
    print(f"Profit Factor:       {profit_factor:.2f}" if not np.isnan(profit_factor) else "Profit Factor:       N/A")
    print(f"Avg Duration:        {df_trades['duration_bars'].mean():.1f} bars")
    print("-"*70)
    print(cost_model.describe())
    print(f"Gross PnL:           {total_pips_gross:.1f} pips "
          f"({total_pips_gross / len(df_trades):.2f} pips/trade)")
    print(f"Execution cost:      {total_cost_pips:.1f} pips "
          f"({total_cost_pips / len(df_trades):.2f} pips/trade)"
          + (f" + EUR {total_commission:,.2f} commission" if total_commission else "")
          + (f" + EUR {total_overnight:,.2f} overnight" if total_overnight else ""))
    print(f"Net PnL:             {total_pips:.1f} pips "
          f"({total_pips / len(df_trades):.2f} pips/trade)"
          + (f"  — {cost_share:.0f}% of the gross edge consumed" if cost_share is not None else ""))
    print("="*70)

    # Persist summary as JSON for downstream comparison / threshold sweeps
    summary_path = os.path.join(REPORT_DIR, "backtest_summary.json")
    summary = {
        'period_start': timeframes.BACKTEST_START.strftime('%Y-%m-%d'),
        'period_end': timeframes.BACKTEST_END.strftime('%Y-%m-%d'),
        'period_days': int(period_days),
        'thresholds': {
            'fast': float(P_OPEN_THRESHOLD_FAST),
            'slow': float(P_OPEN_THRESHOLD_SLOW),
            'close': float(P_CLOSE_THRESHOLD),
        },
        'total_trades': int(len(df_trades)),
        'winning_trades': int(len(win_trades)),
        'losing_trades': int(len(loss_trades)),
        'win_rate_pct': float(win_rate),
        'total_pnl_eur': float(total_pnl),
        'total_pnl_pips': float(total_pips),
        'avg_pnl_per_trade_eur': float(df_trades['pnl'].mean()),
        'avg_pnl_per_trade_pips': float(df_trades['pnl_pips'].mean()),
        'avg_win_eur': float(avg_win_pnl),
        'avg_win_pips': float(avg_win_pips),
        'avg_loss_eur': float(avg_loss_pnl),
        'avg_loss_pips': float(avg_loss_pips),
        'best_trade_eur': float(df_trades['pnl'].max()),
        'best_trade_pips': float(df_trades['pnl_pips'].max()),
        'worst_trade_eur': float(df_trades['pnl'].min()),
        'worst_trade_pips': float(df_trades['pnl_pips'].min()),
        'start_capital_eur': float(START_CAPITAL),
        'final_equity_eur': float(df_trades['equity_after'].iloc[-1]),
        'return_pct': float(total_pnl / START_CAPITAL * 100),
        'notional_per_trade_eur': mean_notional,
        'leverage': float(leverage),
        'risk_model': {
            'mode': RISK_MODEL,
            'risk_pct': float(RISK_PCT),
            'max_leverage': float(MAX_LEVERAGE),
            'notional_min_eur': float(df_trades['notional'].min()) if 'notional' in df_trades else float(NOTIONAL),
            'notional_max_eur': float(df_trades['notional'].max()) if 'notional' in df_trades else float(NOTIONAL),
        },
        'cagr_pct': None if np.isnan(cagr) else float(cagr),
        'cagr_on_notional_pct': None if np.isnan(cagr_on_notional) else float(cagr_on_notional),
        'sharpe_ratio': None if np.isnan(sharpe_ratio) else float(sharpe_ratio),
        'sortino_ratio': None if np.isnan(sortino_ratio) else float(sortino_ratio),
        'max_drawdown_pct': float(max_drawdown_pct),
        'max_drawdown_eur': float(max_drawdown_eur),
        'calmar_ratio': None if np.isnan(calmar_ratio) else float(calmar_ratio),
        'profit_factor': None if np.isnan(profit_factor) else float(profit_factor),
        'avg_duration_bars': float(df_trades['duration_bars'].mean()),
        'holdout_unsealed': bool(args.unseal_holdout),
        # Provenance: a run scored from an external probability file is not a run
        # of the models in this directory, and must never be read as one.
        'proba_file': args.proba_file,
        'costs': {
            'mode': args.cost_model,
            'description': cost_model.describe(),
            'spread_pips': float(args.spread_pips),
            'slippage_pips': float(args.slippage_pips),
            'commission_per_million': float(args.commission_per_million),
            'overnight_long_per_million': float(args.overnight_long_per_million),
            'overnight_short_per_million': float(args.overnight_short_per_million),
            'total_pnl_pips_gross': total_pips_gross,
            'total_cost_pips': total_cost_pips,
            'total_commission_eur': total_commission,
            'total_overnight_eur': total_overnight,
            'avg_cost_per_trade_pips': total_cost_pips / len(df_trades),
            'cost_share_of_gross_pct': cost_share,
            'spread_diagnostics': cost_diagnostics or None,
        },
        'settings': SETTINGS_PROVENANCE,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")
    
    # Exit reason breakdown — counts, win rate, avg/total PnL per reason.
    # `pips_threshold` (fixed +50 pip TP) and `atr_trail` (trend-aware chandelier)
    # are reported as separate lines so the two exit policies can be compared.
    print("\nExit Reason Breakdown:")
    print(f"  {'reason':<24s} {'n':>4s} {'%':>6s} {'win%':>6s} {'avg_pips':>9s} {'pnl_eur':>12s}")
    for reason, count in df_trades["exit_reason"].value_counts().items():
        sub = df_trades[df_trades["exit_reason"] == reason]
        win_rate = (sub["pnl"] > 0).mean() * 100
        avg_pips = sub["pnl_pips"].mean()
        pnl_eur = sub["pnl"].sum()
        print(f"  {reason:<24s} {count:>4d} {count/len(df_trades)*100:>5.1f}% "
              f"{win_rate:>5.1f}% {avg_pips:>9.1f} {pnl_eur:>12,.2f}")

    # Stop-loss MFE quintile breakdown — distinguishes bad entry (low MFE, price
    # never went our way) from bad exit (high MFE, profit given back to SL).
    sl_trades = df_trades[df_trades["exit_reason"] == "stop_loss"]
    if len(sl_trades) >= 5 and "highest_profit_pips" in sl_trades.columns:
        print("\nStop-Loss MFE Quintile Breakdown (highest_profit_pips reached before SL hit):")
        print(f"  {'quintile':<10s} {'n':>4s} {'mfe_range':>16s} {'avg_mfe':>9s} {'avg_pnl':>9s} {'avg_bars':>9s}")
        try:
            quintiles = pd.qcut(sl_trades["highest_profit_pips"], q=5, labels=False, duplicates='drop')
        except ValueError:
            quintiles = None
        if quintiles is not None:
            sl_with_q = sl_trades.assign(_q=quintiles.values)
            for q in sorted(sl_with_q["_q"].dropna().unique()):
                bucket = sl_with_q[sl_with_q["_q"] == q]
                lo, hi = bucket["highest_profit_pips"].min(), bucket["highest_profit_pips"].max()
                avg_mfe = bucket["highest_profit_pips"].mean()
                avg_pnl_pips = bucket["pnl_pips"].mean()
                avg_bars = bucket["duration_bars"].mean()
                print(f"  Q{int(q)+1:<9d} {len(bucket):>4d} "
                      f"{lo:>6.1f}..{hi:>6.1f} {avg_mfe:>9.1f} {avg_pnl_pips:>9.1f} {avg_bars:>9.1f}")

    # Winning-trade MAE quintile breakdown — distinguishes clean entries (low drawdown
    # before turning profitable) from lucky escapes (deep drawdown that recovered).
    win_trades = df_trades[df_trades["pnl"] > 0]
    if len(win_trades) >= 5 and "lowest_pnl_pips" in win_trades.columns:
        print("\nWinning Trade MAE Quintile Breakdown (deepest unrealized loss before close, in pips):")
        print(f"  {'quintile':<10s} {'n':>4s} {'mae_range':>16s} {'avg_mae':>9s} {'avg_pnl':>9s} {'avg_bars':>9s}")
        # qcut on the absolute drawdown so Q1 = smallest drawdown, Q5 = deepest
        mae_abs = (-win_trades["lowest_pnl_pips"]).abs()  # also normalizes -0.0 → 0.0
        try:
            quintiles_w = pd.qcut(mae_abs, q=5, labels=False, duplicates='drop')
        except ValueError:
            quintiles_w = None
        if quintiles_w is not None:
            win_with_q = win_trades.assign(_q=quintiles_w.values, _mae=mae_abs.values)
            for q in sorted(win_with_q["_q"].dropna().unique()):
                bucket = win_with_q[win_with_q["_q"] == q]
                lo, hi = bucket["_mae"].min(), bucket["_mae"].max()
                avg_mae = bucket["_mae"].mean()
                avg_pnl_pips = bucket["pnl_pips"].mean()
                avg_bars = bucket["duration_bars"].mean()
                print(f"  Q{int(q)+1:<9d} {len(bucket):>4d} "
                      f"{lo:>6.1f}..{hi:>6.1f} {avg_mae:>9.1f} {avg_pnl_pips:>9.1f} {avg_bars:>9.1f}")

    # Direction breakdown
    print("\nDirection Breakdown:")
    for action, count in df_trades["action"].value_counts().items():
        action_trades = df_trades[df_trades["action"] == action]
        action_pnl = action_trades["pnl"].sum()
        action_win_rate = (action_trades["pnl"] > 0).sum() / len(action_trades) * 100
        print(f"  {action:10s}: {count:3d} trades | Win Rate: {action_win_rate:.1f}% | PnL: EUR {action_pnl:,.2f}")

    # Regime breakdown
    if args.regime_breakdown and _regime_labels is not None and 'regime_combined' in _regime_labels.columns:
        open_times = pd.to_datetime(df_trades["open_time"])
        df_trades_regime = df_trades.copy()
        df_trades_regime["regime"] = (
            _regime_labels["regime_combined"]
            .reindex(open_times)
            .values
        )
        df_trades_regime["regime"] = df_trades_regime["regime"].fillna("UNKNOWN")
        print("\nRegime Breakdown:")
        for regime_name in sorted(df_trades_regime["regime"].unique()):
            r_trades = df_trades_regime[df_trades_regime["regime"] == regime_name]
            r_pnl = r_trades["pnl"].sum()
            r_wr = (r_trades["pnl"] > 0).sum() / len(r_trades) * 100 if len(r_trades) else 0.0
            r_avg = r_trades["pnl"].mean()
            print(f"  {regime_name:<22}: {len(r_trades):3d} trades | Win Rate: {r_wr:.1f}% | Avg PnL: EUR {r_avg:,.2f} | Total PnL: EUR {r_pnl:,.2f}")

    # ML regime-state breakdown — P&L by the fitted regime model's discrete label
    # at entry (requires daily_rgm_label enabled + retrained).
    if args.regime_breakdown and rgm_label_arr is not None:
        label_series = pd.Series(rgm_label_arr, index=ohlc.index)
        open_times = pd.to_datetime(df_trades["open_time"])
        df_trades_ml = df_trades.copy()
        df_trades_ml["rgm_state"] = label_series.reindex(open_times).values
        print("\nML Regime-State Breakdown (daily_rgm_label at entry):")
        for state in sorted(df_trades_ml["rgm_state"].dropna().unique()):
            s_trades = df_trades_ml[df_trades_ml["rgm_state"] == state]
            s_pnl = s_trades["pnl"].sum()
            s_wr = (s_trades["pnl"] > 0).sum() / len(s_trades) * 100 if len(s_trades) else 0.0
            s_avg = s_trades["pnl"].mean()
            print(f"  state s{int(state):<14d}: {len(s_trades):3d} trades | Win Rate: {s_wr:.1f}% | "
                  f"Avg PnL: EUR {s_avg:,.2f} | Total PnL: EUR {s_pnl:,.2f}")

    # Build equity curve — anchor at BACKTEST_START, not the full OHLC start
    # (otherwise the plot stretches across the training window before any trade exists).
    eq_times = [pd.to_datetime(timeframes.BACKTEST_START)]
    eq_vals = [START_CAPITAL]
    for _, row in df_trades.iterrows():
        eq_times.append(pd.to_datetime(row["close_time"]))
        eq_vals.append(float(row["equity_after"]))
    
    eq_df_plot = pd.DataFrame({"equity": eq_vals}, index=pd.to_datetime(eq_times))
    
    # Resample for smoother plotting
    if not eq_df_plot.empty:
        eq_df_plot = eq_df_plot.resample("1h").ffill().dropna()
    
    # Compute plot limits
    y_min = min(START_CAPITAL, float(eq_df_plot["equity"].min()))
    y_max = max(START_CAPITAL, float(eq_df_plot["equity"].max()))
    
    # Generate PDF report
    with PdfPages(pdf_path) as pdf:
        # 1) Equity curve
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.step(eq_df_plot.index, eq_df_plot["equity"], where="post", 
                label=f"Equity (EUR) | Final: €{df_trades['equity_after'].iloc[-1]:,.0f}", linewidth=2)
        ax.axhline(START_CAPITAL, color='gray', linestyle='--', alpha=0.7, label=f"Start Capital: €{START_CAPITAL:,.0f}")
        ax.set_title("Equity Curve", fontsize=14, fontweight='bold')
        ax.set_ylabel("Equity (EUR)", fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylim(bottom=y_min * 0.995, top=y_max * 1.005)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 2) Trade PnL distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_trades["pnl"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax.set_title("Trade PnL Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("PnL (EUR)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 3) Cumulative PnL
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(df_trades)), df_trades["cum_pnl"], linewidth=2, color='darkgreen')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_title("Cumulative PnL", fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative PnL (EUR)", fontsize=12)
        ax.set_xlabel("Trade Number", fontsize=12)
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # 4) PnL in pips
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_trades["pnl_pips"], bins=50, color="orange", edgecolor="black", alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax.set_title("Trade PnL Distribution (Pips)", fontsize=14, fontweight='bold')
        ax.set_xlabel("PnL (Pips)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # 5) Signal threshold match rates per month
        if len(signal_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Monthly Signal Match Rates", fontsize=14, fontweight='bold')

            month_labels = [str(m) for m in monthly_stats.index]
            x_pos = range(len(month_labels))

            plot_specs = [
                ('fast_pct', f'Fast Model (>{P_OPEN_THRESHOLD_FAST})', 'tab:blue'),
                ('slow_pct', f'Slow Model (>{P_OPEN_THRESHOLD_SLOW})', 'tab:orange'),
                ('all_pct', 'All Combined', 'tab:red'),
            ]
            for ax, (col, title, color) in zip(axes.flat, plot_specs):
                ax.bar(x_pos, monthly_stats[col].values, color=color, alpha=0.7, edgecolor='black')
                ax.set_title(title, fontsize=11)
                ax.set_ylabel("Match Rate (%)")
                ax.set_xticks(list(x_pos))
                ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
                # Add value labels on bars
                for xi, val in zip(x_pos, monthly_stats[col].values):
                    ax.text(xi, val + 0.3, f"{val:.1f}%", ha='center', fontsize=7)
            # Hide unused subplots
            for ax in list(axes.flat)[len(plot_specs):]:
                ax.set_visible(False)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # 6) Probability distribution histograms
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle("Model Output Distributions (Backtest Period)", fontsize=14, fontweight='bold')

            for ax, (name, series) in zip(axes.flat[:4], prob_columns.items()):
                ax.hist(series.values, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
                threshold = P_OPEN_THRESHOLD_FAST if 'Fast' in name else P_OPEN_THRESHOLD_SLOW
                ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                          label=f'Threshold ({threshold})')
                ax.set_title(name, fontsize=11)
                ax.set_xlabel("Probability")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            axes.flat[4].set_visible(False)
            axes.flat[5].set_visible(False)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"\nBacktest report saved: {pdf_path}")

# ============================================================================
# W&B experiment tracking (opt-in via --wandb)
# ============================================================================
# Group = the model's run_id, job_type = 'backtest' — so every backtest of the
# same trained model lands in the training run's W&B group, and multiple
# backtests with different parameters stay distinguishable via config + name.
if experiment_tracking.wandb_enabled(args):
    # The raw argparse namespace is NOT this run's configuration and logging it is
    # actively misleading. Every entry gate and exit rule is a store_true flag whose
    # default lives in a module constant, and the ones that default to ON are turned
    # off by a --no-* partner writing a SEPARATE dest — so on a default run
    # vars(args)['opening_requires_slow_signal'] is False while the backtest required
    # the slow signal, and 'no_closing_after_signal_reversal_slow' False says nothing
    # about the reversal exit that ran. The --p-*/--stop-pips/--hold-bars knobs read
    # None whenever their constant applied, and the regime gate disables ITSELF when
    # X_slow carries no regime feature, so even a string arg can contradict the run.
    #
    # Each entry is (config key, the value the run used, the argparse dests it
    # supersedes) — the superseded dests are dropped so no misleading key survives.
    _wandb_effective = [
        ('run_id', RUN_ID, ()),
        ('backtest_start', timeframes.BACKTEST_START.strftime('%Y-%m-%d'), ()),
        ('backtest_end', timeframes.BACKTEST_END.strftime('%Y-%m-%d'), ()),
        ('start_capital_eur', float(START_CAPITAL), ()),
        # Thresholds / geometry: args hold None whenever the constant applied.
        ('p_open_threshold_fast', float(P_OPEN_THRESHOLD_FAST), ('p_open_fast',)),
        ('p_open_threshold_slow', float(P_OPEN_THRESHOLD_SLOW), ('p_open_slow',)),
        # Per-direction operating points; they differ from the scalar above only
        # under --use-trained-threshold, which maps one raw threshold through two
        # calibrators.
        ('p_open_slow_long', float(P_OPEN_SLOW_LONG), ()),
        ('p_open_slow_short', float(P_OPEN_SLOW_SHORT), ()),
        ('slow_direction_margin', float(SLOW_DIRECTION_MARGIN), ('slow_direction_margin',)),
        ('p_close_threshold', float(P_CLOSE_THRESHOLD), ('p_close_threshold',)),
        ('p_close_pip_threshold', float(P_CLOSE_PIP_THRESHOLD), ('p_close_pip_threshold',)),
        ('stop_pips', float(STOP_PIPS), ('stop_pips',)),
        ('hold_bars', int(HOLD_BARS), ('hold_bars',)),
        # Entry gates. Both --x/--no-x pairs write two different dests.
        ('opening_requires_fast_signal', bool(OPENING_REQUIRES_FAST_SIGNAL),
         ('opening_requires_fast_signal', 'no_opening_requires_fast_signal')),
        ('opening_requires_slow_signal', bool(OPENING_REQUIRES_SLOW_SIGNAL),
         ('opening_requires_slow_signal', 'no_opening_requires_slow_signal')),
        # Derived: with both gates off the entry is disabled entirely and the run
        # reports zero trades — the one config key that explains an empty backtest.
        ('entry_gates_active', bool(ENTRY_GATES_ACTIVE), ()),
        # Exit rules.
        ('closing_before_weekend', bool(CLOSING_BEFORE_WEEKEND), ('no_closing_before_weekend',)),
        ('closing_after_time', bool(CLOSING_AFTER_TIME), ('closing_after_time',)),
        ('closing_after_signal_reversal_fast', bool(CLOSING_AFTER_SIGNAL_REVERSAL_FAST),
         ('closing_after_signal_reversal_fast',)),
        ('closing_after_signal_reversal_slow', bool(CLOSING_AFTER_SIGNAL_REVERSAL_SLOW),
         ('no_closing_after_signal_reversal_slow',)),
        ('closing_after_x_pips', bool(CLOSING_AFTER_X_PIPS), ('closing_after_x_pips',)),
        ('closing_with_trailing_stop', bool(CLOSING_WITH_TRAILING_STOP), ('closing_with_trailing_stop',)),
        ('closing_on_level_retest', bool(CLOSING_ON_LEVEL_RETEST), ('closing_on_level_retest',)),
        ('trend_aware_exits', bool(TREND_AWARE_EXITS), ('trend_aware_exits',)),
        ('atr_trail_mult', float(ATR_TRAIL_MULT), ('atr_trail_mult',)),
        ('atr_trail_activation_mult', float(ATR_TRAIL_ACTIVATION_MULT), ('atr_trail_activation_mult',)),
        ('atr_period', int(ATR_PERIOD), ('atr_period',)),
        ('breakeven_trigger_pips',
         None if BREAKEVEN_TRIGGER_PIPS is None else float(BREAKEVEN_TRIGGER_PIPS),
         ('breakeven_trigger_pips',)),
        ('breakeven_offset_pips', float(BREAKEVEN_OFFSET_PIPS), ('breakeven_offset_pips',)),
        ('close_on_ranging_min_pips',
         None if CLOSE_ON_RANGING_MIN_PIPS is None else float(CLOSE_ON_RANGING_MIN_PIPS),
         ('close_on_ranging_min_pips',)),
        # Regime gate: the requested mode stays as regime_gate; these two record what
        # it resolved to. --regime-source ml falls back to the rule when the ML column
        # is missing, and with no regime feature at all the gate is inert while
        # --regime-gate still reads 'trending'.
        ('regime_source_effective', _regime_source, ()),
        ('regime_gate_active', bool(REGIME_GATE_MODE != 'off' and regime_trending_arr is not None), ()),
    ]
    _wandb_config = experiment_tracking.effective_config(args, _wandb_effective)
    # Provenance of the MODEL being backtested (label_mode above all) — it lives in
    # the training run's args, not in this CLI, so without it a backtest run cannot be
    # grouped or filtered by label mode alongside its training run. Absent for runs
    # without a training_summary.json (e.g. --proba-file), which is not an error.
    _wandb_config.update(experiment_tracking.training_provenance_config(GENERATED_DIR))
    _wandb_run = experiment_tracking.init_wandb_run(
        args, job_type='backtest', run_id=RUN_ID,
        name=args.wandb_run_name or experiment_tracking.derive_backtest_run_name(
            RUN_ID, p_open_slow=P_OPEN_THRESHOLD_SLOW, stop_pips=STOP_PIPS,
            cost_model=args.cost_model, regime_gate=args.regime_gate),
        config=_wandb_config,
    )
    if _wandb_run is not None:
        experiment_tracking.log_metrics(experiment_tracking.backtest_payload(summary))
        if df_trades is not None and len(df_trades) > 0:
            _wandb_regime_rows = experiment_tracking.regime_breakdown_rows(df_trades, _regime_labels)
            experiment_tracking.log_metrics(
                experiment_tracking.breakdown_payload(_wandb_regime_rows, 'regime'))
            experiment_tracking.log_table('regime_breakdown', _wandb_regime_rows)
            _wandb_exit_rows = experiment_tracking.exit_reason_rows(df_trades)
            experiment_tracking.log_metrics(
                experiment_tracking.breakdown_payload(_wandb_exit_rows, 'exit_reason'))
            experiment_tracking.log_table('exit_reason_breakdown', _wandb_exit_rows)
            experiment_tracking.log_trades(df_trades)
            experiment_tracking.log_equity_curve(
                df_trades, timeframes.BACKTEST_START, timeframes.BACKTEST_END)
            experiment_tracking.log_table('trade_list', df_trades)
        experiment_tracking.finish_wandb_run()
