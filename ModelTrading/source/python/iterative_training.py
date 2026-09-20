"""
Iterative Training Script with Parallel Execution

This script orchestrates multiple parallel training and backtesting runs to test
different hyperparameter configurations or data splits.

System Specs (for reference):
- RAM: 64 GB
- CPU: AMD Ryzen 9 9955HX 16-Core Processor (32 logical processors)
- Recommended max parallel jobs: 4-8 (to avoid memory exhaustion)

Usage:
    python iterative_training.py --parallel-jobs 4
    python iterative_training.py --parallel-jobs 4 --plot  # Enable plotting
"""

import copy
import os
import random
import sys
import subprocess
import uuid
import json
import time
import argparse
import statistics
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config
import ModelTrading.config.timeframes as timeframes
import pandas as pd

from ModelTrading.source.python.labeling.regime_filter import (
    add_regime_filter_args, validate_regime_args, VALID_REGIMES,
)
from ModelTrading.source.python.labeling.sampling import (
    add_sampling_args, validate_sampling_args,
)
import ModelTrading.source.python.utils.costs as costs
import ModelTrading.source.python.utils.experiment_tracking as experiment_tracking
from ModelTrading.source.python.utils.stats import (
    clustered_se as _clustered_se,
    MIN_CLUSTERS_FOR_SIGNIFICANCE,
)


# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Configuration for a single training/backtest run"""

    def __init__(self, name, params=None, train_start=None, train_end=None,
                 backtest_start=None, backtest_end=None,
                 label_mode='static', atr_multiplier=2.5, atr_stop_multiplier=0.875,
                 # XGBoost hyperparameters (shared / slow defaults)
                 max_depth=6, eta=0.1, num_boost_round=200,
                 subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
                 # None inherits advanced_train's default (1.0 = XGBoost default)
                 colsample_bynode=None,
                 # Label config (static mode)
                 pip_target=None, stop_pips=None, horizon_min=None, horizon_max=None,
                 # Label-mode specific parameters (forwarded only for the owning mode)
                 daily_vol_span=None, slow_hysteresis_multiplier=None,
                 lookahead_horizon=None, lookahead_pct=None, lookahead_lookback=None,
                 lookahead_min_pips=None, lookahead_stop_pips=None,
                 direction_horizon=None, direction_dead_zone_pips=None,
                 timing_entry=None, direction_aware_regime=False,
                 # Which regime DEFINITION drives regime_conditional / trend_only labels:
                 # 'rule' = ADX/price-efficiency, 'ml' = the fitted regime model via
                 # data/regime_daily.csv. None inherits advanced_train's default ('rule').
                 regime_label_source=None, regime_label_trend_threshold=None,
                 # Shared training knobs
                 target_recall=None, features_config=None,
                 # Backtest regime gate ('off' | 'trending' | 'ranging'); None = backtest default
                 regime_gate=None,
                 # Backtest thresholds
                 p_open_fast=0.6, p_open_slow=0.55,
                 # Exit strategy
                 p_close_pip_threshold=None, closing_after_x_pips=False,
                 # Regime filtering
                 regime_filter=False, regime_type=None,
                 # Training-data sampling
                 training_sampling=True, sampling_stride=True, sampling_context=True,
                 sampling_stride_x=10, sampling_hours_before=4.0, sampling_hours_after=4.0,
                 # Multi-seed support
                 seed=None, base_name=None,
                 # Fast model specific hyperparameters
                 fast_mfe_threshold=None, fast_mfe_horizon=None,
                 fast_max_depth=None, fast_num_boost_round=None,
                 fast_min_child_weight=None, fast_spw_factor=1.0,
                 fast_lambda=1.0, fast_mi_threshold=None,
                 fast_train_start=None, fast_use_slow_label=False,
                 fast_conditional_on_setup=False,
                 # Slow model specific hyperparameters
                 slow_spw_factor=1.0, slow_mi_threshold=0,
                 slow_max_depth=3, slow_num_boost_round=200,
                 slow_min_child_weight=3, slow_lambda=5.0,
                 # MI scoring: None inherits advanced_train's default (20 permutations,
                 # noise floor gates selection); 0 reports the floor without filtering.
                 mi_permutations=None, legacy_mi=False,
                 # Walk-forward: per-config training window (months). None = use the
                 # run-wide --wf-train-months. Test windows stay identical either way,
                 # so configs with different history lengths remain comparable.
                 wf_train_months=None,
                 # Backtest at the operating point the training chose in-fold instead of
                 # a hand-set p_open_slow. Required for comparing label modes.
                 use_trained_threshold=False,
                 # Backtest entry/exit gates that backtest.py defaults differently from
                 # what a config may need (None = leave the backtest default alone):
                 #   opening_requires_fast_signal   backtest default False
                 #   closing_after_signal_reversal_slow  backtest default True
                 opening_requires_fast_signal=None,
                 closing_after_signal_reversal_slow=None,
                 # Multiple backtests against the SAME trained model: list of
                 # (suffix, {config-field overrides}) pairs, e.g.
                 #   [('slowonly', {'opening_requires_fast_signal': False}),
                 #    ('fastslow', {'opening_requires_fast_signal': True})]
                 # Each variant becomes its own result row whose name/base_name carry
                 # '__<suffix>', so aggregate_walk_forward pools the arms separately —
                 # paired by construction, because they share every trained model.
                 # None = the historical single post-training backtest.
                 backtest_variants=None):
        self.name = name
        self.base_name = base_name or name
        self.seed = seed
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.effective_run_id = self.run_id  # overridden to scenarios/{name}/{run_id} when --scenario-name is used
        self.params = params or {}
        self.train_start = train_start
        self.train_end = train_end
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        # Dynamic label generation parameters
        self.label_mode = label_mode
        self.atr_multiplier = atr_multiplier
        self.atr_stop_multiplier = atr_stop_multiplier
        # XGBoost hyperparameters
        self.max_depth = max_depth
        self.eta = eta
        self.num_boost_round = num_boost_round
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.min_child_weight = min_child_weight
        # Label config
        self.pip_target = pip_target
        self.stop_pips = stop_pips
        self.horizon_min = horizon_min
        self.horizon_max = horizon_max
        # Label-mode specific
        self.daily_vol_span = daily_vol_span
        self.slow_hysteresis_multiplier = slow_hysteresis_multiplier
        self.lookahead_horizon = lookahead_horizon
        self.lookahead_pct = lookahead_pct
        self.lookahead_lookback = lookahead_lookback
        self.lookahead_min_pips = lookahead_min_pips
        self.lookahead_stop_pips = lookahead_stop_pips
        self.direction_horizon = direction_horizon
        self.direction_dead_zone_pips = direction_dead_zone_pips
        self.timing_entry = timing_entry
        self.direction_aware_regime = direction_aware_regime
        self.regime_label_source = regime_label_source
        self.regime_label_trend_threshold = regime_label_trend_threshold
        # Shared training knobs
        self.target_recall = target_recall
        self.features_config = features_config
        # Backtest regime gate
        self.regime_gate = regime_gate
        # Backtest thresholds
        self.p_open_fast = p_open_fast
        self.p_open_slow = p_open_slow
        self.p_close_pip_threshold = p_close_pip_threshold
        self.closing_after_x_pips = closing_after_x_pips
        # Regime filtering
        self.regime_filter = regime_filter
        self.regime_type = regime_type
        # Training-data sampling
        self.training_sampling = training_sampling
        self.sampling_stride = sampling_stride
        self.sampling_context = sampling_context
        self.sampling_stride_x = sampling_stride_x
        self.sampling_hours_before = sampling_hours_before
        self.sampling_hours_after = sampling_hours_after
        # Fast model specific
        self.fast_mfe_threshold = fast_mfe_threshold
        self.fast_mfe_horizon = fast_mfe_horizon
        self.fast_max_depth = fast_max_depth
        self.fast_num_boost_round = fast_num_boost_round
        self.fast_min_child_weight = fast_min_child_weight
        self.fast_spw_factor = fast_spw_factor
        self.fast_lambda = fast_lambda
        self.fast_mi_threshold = fast_mi_threshold
        self.fast_train_start = fast_train_start
        self.fast_use_slow_label = fast_use_slow_label
        # Meta-labelling: train each fast model only on bars where its matching slow
        # label is 1. Mutually exclusive with fast_use_slow_label (advanced_train
        # rejects the pair: together every remaining row would be a positive).
        self.fast_conditional_on_setup = fast_conditional_on_setup
        # Slow model specific
        self.slow_spw_factor = slow_spw_factor
        self.slow_mi_threshold = slow_mi_threshold
        self.slow_max_depth = slow_max_depth
        self.slow_num_boost_round = slow_num_boost_round
        self.slow_min_child_weight = slow_min_child_weight
        self.slow_lambda = slow_lambda
        self.mi_permutations = mi_permutations
        self.legacy_mi = legacy_mi
        self.wf_train_months = wf_train_months
        self.use_trained_threshold = use_trained_threshold
        self.opening_requires_fast_signal = opening_requires_fast_signal
        self.closing_after_signal_reversal_slow = closing_after_signal_reversal_slow
        self.backtest_variants = backtest_variants

    def to_dict(self):
        return {
            'name': self.name,
            'base_name': self.base_name,
            'seed': self.seed,
            'run_id': self.run_id,
            'effective_run_id': self.effective_run_id,
            'train_start': str(self.train_start) if self.train_start else None,
            'train_end': str(self.train_end) if self.train_end else None,
            'backtest_start': str(self.backtest_start) if self.backtest_start else None,
            'backtest_end': str(self.backtest_end) if self.backtest_end else None,
            'label_mode': self.label_mode,
            'atr_multiplier': self.atr_multiplier,
            'atr_stop_multiplier': self.atr_stop_multiplier,
            'max_depth': self.max_depth,
            'eta': self.eta,
            'num_boost_round': self.num_boost_round,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bynode': self.colsample_bynode,
            'min_child_weight': self.min_child_weight,
            'pip_target': self.pip_target,
            'stop_pips': self.stop_pips,
            'horizon_min': self.horizon_min,
            'horizon_max': self.horizon_max,
            'daily_vol_span': self.daily_vol_span,
            'slow_hysteresis_multiplier': self.slow_hysteresis_multiplier,
            'lookahead_horizon': self.lookahead_horizon,
            'lookahead_pct': self.lookahead_pct,
            'lookahead_lookback': self.lookahead_lookback,
            'lookahead_min_pips': self.lookahead_min_pips,
            'lookahead_stop_pips': self.lookahead_stop_pips,
            'direction_horizon': self.direction_horizon,
            'direction_dead_zone_pips': self.direction_dead_zone_pips,
            'timing_entry': self.timing_entry,
            'direction_aware_regime': self.direction_aware_regime,
            'regime_label_source': self.regime_label_source,
            'regime_label_trend_threshold': self.regime_label_trend_threshold,
            'target_recall': self.target_recall,
            'features_config': self.features_config,
            'regime_gate': self.regime_gate,
            'p_open_fast': self.p_open_fast,
            'p_open_slow': self.p_open_slow,
            'p_close_pip_threshold': self.p_close_pip_threshold,
            'closing_after_x_pips': self.closing_after_x_pips,
            'regime_filter': self.regime_filter,
            'regime_type': self.regime_type,
            'training_sampling': self.training_sampling,
            'sampling_stride': self.sampling_stride,
            'sampling_context': self.sampling_context,
            'sampling_stride_x': self.sampling_stride_x,
            'sampling_hours_before': self.sampling_hours_before,
            'sampling_hours_after': self.sampling_hours_after,
            'fast_mfe_threshold': self.fast_mfe_threshold,
            'fast_mfe_horizon': self.fast_mfe_horizon,
            'fast_max_depth': self.fast_max_depth,
            'fast_num_boost_round': self.fast_num_boost_round,
            'fast_min_child_weight': self.fast_min_child_weight,
            'fast_spw_factor': self.fast_spw_factor,
            'fast_lambda': self.fast_lambda,
            'fast_mi_threshold': self.fast_mi_threshold,
            'fast_train_start': str(self.fast_train_start) if self.fast_train_start else None,
            'fast_use_slow_label': self.fast_use_slow_label,
            'fast_conditional_on_setup': self.fast_conditional_on_setup,
            'slow_spw_factor': self.slow_spw_factor,
            'slow_mi_threshold': self.slow_mi_threshold,
            'mi_permutations': self.mi_permutations,
            'legacy_mi': self.legacy_mi,
            'wf_train_months': self.wf_train_months,
            'use_trained_threshold': self.use_trained_threshold,
            'opening_requires_fast_signal': self.opening_requires_fast_signal,
            'closing_after_signal_reversal_slow': self.closing_after_signal_reversal_slow,
            'backtest_variants': self.backtest_variants,
            'slow_max_depth': self.slow_max_depth,
            'slow_num_boost_round': self.slow_num_boost_round,
            'slow_min_child_weight': self.slow_min_child_weight,
            'slow_lambda': self.slow_lambda,
        }


class BacktestSweepConfig:
    """
    Configuration for a backtest-only parameter sweep.

    Does not run any training — reuses the already-trained model from
    dir_config.GENERATED_DIR (or a specific run_id subdirectory).
    Each job writes to a unique temp report directory to allow parallel execution.

    Flag semantics:
      None  = use backtest.py default (don't pass the flag)
      True  = explicitly enable (pass --flag)
      False = explicitly disable (pass --no-flag, only for flags that default True)
    """

    def __init__(self, name,
                 backtest_start=None, backtest_end=None,
                 model_run_id=None,
                 # Entry thresholds
                 p_open_fast=0.45, p_open_slow=0.35,
                 # Exit thresholds
                 p_close_threshold=None, p_close_pip_threshold=None,
                 closing_after_x_pips=False,
                 # Stop / hold
                 stop_pips=None, hold_bars=None,
                 # Daily regime entry gate ('off' | 'trending' | 'ranging'); None = backtest default
                 regime_gate=None,
                 # Closing flags (None = backtest.py default)
                 closing_before_weekend=None,
                 closing_after_time=None,
                 closing_after_signal_reversal_fast=None,
                 closing_after_signal_reversal_slow=None,
                 closing_with_trailing_stop=None,
                 closing_on_level_retest=None,
                 # Opening flags (None = backtest.py default)
                 opening_requires_fast_signal=None,
                 opening_requires_slow_signal=None):
        self.name = name
        self.model_run_id = model_run_id  # None → default GENERATED_DIR
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.p_open_fast = p_open_fast
        self.p_open_slow = p_open_slow
        self.p_close_threshold = p_close_threshold
        self.p_close_pip_threshold = p_close_pip_threshold
        self.closing_after_x_pips = closing_after_x_pips
        self.stop_pips = stop_pips
        self.hold_bars = hold_bars
        self.regime_gate = regime_gate
        self.closing_before_weekend = closing_before_weekend
        self.closing_after_time = closing_after_time
        self.closing_after_signal_reversal_fast = closing_after_signal_reversal_fast
        self.closing_after_signal_reversal_slow = closing_after_signal_reversal_slow
        self.closing_with_trailing_stop = closing_with_trailing_stop
        self.closing_on_level_retest = closing_on_level_retest
        self.opening_requires_fast_signal = opening_requires_fast_signal
        self.opening_requires_slow_signal = opening_requires_slow_signal

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# Exit-strategy sweep table, shared by the round-8 sweep and the label-study sweep.
# (name_suffix, pip_threshold, after_x_pips, reversal_slow, after_time, trailing)
#   reversal_slow=False → pass --no-closing-after-signal-reversal-slow
#   reversal_slow=None  → keep backtest.py default (True)
EXIT_STRATEGY_COMBOS = [
    ("B_base",        50,  False, None,  None,  None),   # defaults only
    ("B_pip50x",      50,  True,  None,  None,  None),   # close at +50 pips
    ("B_pip80x",      80,  True,  None,  None,  None),   # close at +80 pips
    ("B_pip100x",     100, True,  None,  None,  None),   # close at +100 pips
    ("B_norev",       50,  False, False, None,  None),   # no slow reversal exit
    ("B_pip80x_norev",80,  True,  False, None,  None),   # +80 pips, no reversal
    ("B_trail",       50,  False, None,  None,  True),   # trailing stop
    ("B_time",        50,  False, None,  True,  None),   # time-based exit
]


def create_backtest_sweep_configs():
    """
    Round 8: Backtest parameter sweep on the trained 96m model.

    Model: trained on 2017-04-01 → 2025-03-31 (set in timeframes.py TRAIN_START/TRAIN_END)
    Backtest window: 2025-04-01 → 2026-04-12

    Phase A — Threshold grid (25 configs):
      p_open_fast [0.40, 0.45, 0.50, 0.55, 0.60] ×
      p_open_slow [0.10, 0.15, 0.20, 0.25, 0.30]
      Goal: find which threshold combo produces trades and at what quality.

    Phase B — Exit strategy (8 configs, fixed thresholds 0.45/0.35):
      Varies: p_close_pip_threshold, closing_after_x_pips,
              closing_after_signal_reversal_slow, closing_with_trailing_stop
      Goal: find optimal trade management once entry thresholds are set.
    """
    configs = []

    BT_START = "2025-04-01"
    BT_END   = "2026-04-12"

    # -------------------------------------------------------------------------
    # Phase A: Threshold grid
    # -------------------------------------------------------------------------
    fast_vals = [0.40, 0.45, 0.50, 0.55, 0.60]
    slow_vals = [0.10, 0.15, 0.20, 0.25, 0.30]
    for pf in fast_vals:
        for ps in slow_vals:
            configs.append(BacktestSweepConfig(
                name=f"A_f{int(pf*100)}_s{int(ps*100)}",
                backtest_start=BT_START, backtest_end=BT_END,
                p_open_fast=pf, p_open_slow=ps,
            ))

    # -------------------------------------------------------------------------
    # Phase B: Exit strategy combos
    # Using known working thresholds (fast=0.45, slow=0.35)
    # -------------------------------------------------------------------------
    BASE_FAST = 0.45
    BASE_SLOW = 0.35

    for suffix, pip_th, after_pips, rev_slow, after_time, trailing in EXIT_STRATEGY_COMBOS:
        configs.append(BacktestSweepConfig(
            name=suffix,
            backtest_start=BT_START, backtest_end=BT_END,
            p_open_fast=BASE_FAST, p_open_slow=BASE_SLOW,
            p_close_pip_threshold=pip_th,
            closing_after_x_pips=after_pips,
            closing_after_signal_reversal_slow=rev_slow,
            closing_after_time=after_time,
            closing_with_trailing_stop=trailing,
        ))

    return configs  # 33 total (25 threshold grid + 8 exit strategy)


def build_sweep_backtest_args(config, report_dir, backtest_script=None, extra_args=None):
    """
    Build the backtest.py command line for a BacktestSweepConfig.

    Pure function (no subprocess, no filesystem) so the flag forwarding is unit-testable.

    Args:
        config (BacktestSweepConfig): Sweep parameter configuration.
        report_dir (str): Unique report directory for this job.
        backtest_script (str, optional): Path to backtest.py. Defaults to the copy
            next to this file.
        extra_args (list, optional): Additional flags appended verbatim (e.g. the
            forwarded --wandb* flags). Must be handed down from the parent process
            — see build_train_args for the Windows-spawn rationale.

    Returns:
        list[str]: Full argv list, starting with the Python executable.
    """
    if backtest_script is None:
        backtest_script = os.path.join(os.path.dirname(__file__), 'backtest.py')

    args = [sys.executable, backtest_script, '--report-dir', report_dir]

    # Model source: if model_run_id given, point to that run's generated dir
    if config.model_run_id:
        args.extend(['--run-id', config.model_run_id])

    # Backtest window
    if config.backtest_start:
        args.extend(['--backtest-start', config.backtest_start])
    if config.backtest_end:
        args.extend(['--backtest-end', config.backtest_end])

    # Entry thresholds
    args.extend(['--p-open-fast', str(config.p_open_fast)])
    args.extend(['--p-open-slow', str(config.p_open_slow)])

    # Exit thresholds
    if config.p_close_threshold is not None:
        args.extend(['--p-close-threshold', str(config.p_close_threshold)])
    if config.p_close_pip_threshold is not None:
        args.extend(['--p-close-pip-threshold', str(config.p_close_pip_threshold)])
    if config.closing_after_x_pips:
        args.append('--closing-after-x-pips')

    # Stop / hold
    if config.stop_pips is not None:
        args.extend(['--stop-pips', str(config.stop_pips)])
    if config.hold_bars is not None:
        args.extend(['--hold-bars', str(config.hold_bars)])

    # Daily regime entry gate
    if config.regime_gate is not None:
        args.extend(['--regime-gate', config.regime_gate])

    # Closing flags — True: add flag; False: add --no- flag; None: skip (default)
    if config.closing_before_weekend is False:
        args.append('--no-closing-before-weekend')
    if config.closing_after_time:
        args.append('--closing-after-time')
    if config.closing_after_signal_reversal_fast:
        args.append('--closing-after-signal-reversal-fast')
    if config.closing_after_signal_reversal_slow is False:
        args.append('--no-closing-after-signal-reversal-slow')
    if config.closing_with_trailing_stop:
        args.append('--closing-with-trailing-stop')
    if config.closing_on_level_retest:
        args.append('--closing-on-level-retest')

    # Opening flags
    if config.opening_requires_fast_signal:
        args.append('--opening-requires-fast-signal')
    elif config.opening_requires_fast_signal is False:
        args.append('--no-opening-requires-fast-signal')
    if config.opening_requires_slow_signal:
        args.append('--opening-requires-slow-signal')
    elif config.opening_requires_slow_signal is False:
        args.append('--no-opening-requires-slow-signal')

    if extra_args:
        args.extend(extra_args)

    return args


def run_backtest_only_job(config, enable_plotting=False, wandb_args=None):
    """
    Run a single backtest against the already-trained model in GENERATED_DIR.

    Each job writes to a unique temp report directory so parallel execution is safe.
    Temp directory is cleaned up after metrics are extracted.

    Args:
        config (BacktestSweepConfig): Backtest parameter configuration.
        enable_plotting (bool): Unused — kept for uniform executor.submit signature.
        wandb_args (list, optional): Forwarded --wandb* flags from the parent (must
            come as an argument — the module globals are empty in a spawned worker).
            Each sweep cell becomes its own W&B run named after the config, grouped
            under the model_run_id it backtests.

    Returns:
        dict: Results with status, timing, and backtest_metrics.
    """
    import shutil
    import tempfile
    del enable_plotting  # parameter exists only for uniform executor.submit signature

    result = {
        'config': config.to_dict(),
        'status': 'failed',
        'backtest_status': None,
        'backtest_time': None,
        'train_time': 0.0,   # no training — keep key for uniform summary reporting
        'error': None,
    }

    temp_report_dir = None
    try:
        # Unique temp directory so parallel jobs don't overwrite each other
        temp_report_dir = tempfile.mkdtemp(prefix=f"bt_sweep_{config.name}_")

        extra_args = list(wandb_args) + ['--wandb-run-name', f'{config.name}__bt'] if wandb_args else None
        backtest_args = build_sweep_backtest_args(config, temp_report_dir, extra_args=extra_args)

        print(f"\n{'='*60}")
        print(f"Backtest sweep: {config.name}")
        print(f"  Entry: fast={config.p_open_fast}, slow={config.p_open_slow}")
        if config.p_close_pip_threshold is not None:
            print(f"  Exit:  pip_threshold={config.p_close_pip_threshold}, after_x_pips={config.closing_after_x_pips}")
        print(f"{'='*60}")

        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning,ignore::FutureWarning'
        env['PYTHONIOENCODING'] = 'utf-8'

        bt_start = time.time()
        process = subprocess.run(backtest_args, capture_output=True, text=True, env=env)
        elapsed = time.time() - bt_start
        result['backtest_time'] = elapsed

        stderr = process.stderr or ""
        has_error = (
            process.returncode != 0 or
            'Traceback' in stderr or
            'Error:' in stderr or
            'FileNotFoundError' in stderr
        )

        if has_error:
            result['backtest_status'] = 'failed'
            result['error'] = f"Backtest failed: {stderr[:500]}"
            print(f"FAILED: {config.name} — {stderr[:200]}")
            return result

        result['backtest_status'] = 'success'
        result['status'] = 'success'

        # Extract metrics from the temp report dir
        trade_csv = os.path.join(temp_report_dir, "trade_list.csv")
        metrics = _extract_metrics_from_csv(trade_csv)
        if metrics:
            result['backtest_metrics'] = metrics
            print(f"OK: {config.name} in {elapsed:.1f}s — "
                  f"PnL={metrics['total_pnl']:.2f}, Trades={metrics['total_trades']}, "
                  f"WR={metrics['win_rate']:.1%}")
        else:
            result['backtest_metrics'] = {'total_trades': 0, 'total_pnl': 0,
                                          'win_rate': 0, 'avg_pnl_per_trade': 0,
                                          'avg_win_pnl_per_trade': 0, 'avg_loss_pnl_per_trade': 0,
                                          'max_pnl': 0, 'min_pnl': 0, 'total_pips': 0}
            print(f"OK: {config.name} in {elapsed:.1f}s — 0 trades")

    except Exception as e:
        result['error'] = str(e)
        print(f"FAILED: Exception in {config.name}: {e}")

    finally:
        # Always clean up temp dir
        if temp_report_dir and os.path.exists(temp_report_dir):
            try:
                shutil.rmtree(temp_report_dir)
            except Exception:
                pass

    return result


def _extract_metrics_from_csv(csv_path):
    """
    Extract backtest metrics directly from a trade_list.csv path.

    Returns dict of metrics or None if the file is missing or empty.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        total_trades = len(df)
        total_pnl = df['pnl'].sum()
        winning = (df['pnl'] > 0).sum()
        win_df = df[df['pnl'] > 0]
        loss_df = df[df['pnl'] <= 0]
        basic = {
            'total_trades': int(total_trades),
            'total_pnl': float(total_pnl),
            'win_rate': float(winning / total_trades),
            'avg_pnl_per_trade': float(df['pnl'].mean()),
            'avg_win_pnl_per_trade': float(win_df['pnl'].mean()) if len(win_df) > 0 else 0.0,
            'avg_loss_pnl_per_trade': float(loss_df['pnl'].mean()) if len(loss_df) > 0 else 0.0,
            'max_pnl': float(df['pnl'].max()),
            'min_pnl': float(df['pnl'].min()),
            'total_pips': float(df['pnl_pips'].sum()),
        }
        # Sweep results are ranked on risk-adjusted terms too, so they need the same
        # extended metrics (profit factor, max drawdown, Sharpe) as training runs.
        return {**basic, **compute_extended_backtest_metrics(df)}
    except Exception as e:
        print(f"  ERROR reading {csv_path}: {e}")
        return None


# =============================================================================
# Label-mode study (see plans/in-my-project-i-cryptic-mango.md)
# =============================================================================

# Training windows: all end the day before the fixed backtest window starts.
LABEL_STUDY_WINDOWS = {
    'w03':  "2025-07-01",   # 3 months
    'w06':  "2025-04-01",   # 6 months
    'w12':  "2024-10-01",   # 1 year
    'w18':  "2024-04-01",   # 18 months
    'w24':  "2023-10-01",   # 2 years
    'w36':  "2022-10-01",   # 3 years
    'w60':  "2020-10-01",   # 5 years
    'w120': "2015-10-01",   # 10 years
}

# Phase 1 screens these four window lengths.
LABEL_STUDY_PHASE1_WINDOWS = ['w06', 'w18', 'w36', 'w120']

# Label modes under test, with a short tag and the parameters that mode consumes.
# atr_scaled is excluded: documented as stuck near AUC 0.54 across volatility
# regimes (see the comment above --label-mode in advanced_train.py).
LABEL_STUDY_MODES = {
    'static':            ('static',   {}),
    'daily_vol_scaled':  ('dvol',     dict(daily_vol_span=100,
                                           # multipliers chosen so mean target/stop lands
                                           # near the 35/35 pips of the static baseline
                                           atr_multiplier=0.65, atr_stop_multiplier=0.65)),
    'lookahead':         ('look',     dict(lookahead_horizon=96, lookahead_pct=20.0,
                                           lookahead_lookback=2880, lookahead_min_pips=30.0,
                                           lookahead_stop_pips=65.0)),
    'regime_conditional':('regcond',  {}),
    'trend_only':        ('trendonly', {}),
    'direction_horizon': ('dirhz',    dict(direction_horizon=192,
                                           direction_dead_zone_pips=12.0)),
    'window_cascade':    ('wcasc',    dict(timing_entry='fhl')),
}

# Fast-model label policy. window_cascade is excluded from both: its fast model IS
# the timing model, which brings its own labels.
LABEL_STUDY_FAST_POLICIES = {
    'sl':  dict(fast_use_slow_label=True),
    'mfe': dict(fast_use_slow_label=False, fast_mfe_threshold=35, fast_mfe_horizon=24),
}

# Phase-1 finalists, as (label_mode, fast_policy, [window tags]).
#
# PHASE 1 RESULTS (52 configs x 16-template sweep, backtest 2025-10-01..2026-04-19):
#   Kept:
#     trend_only         AUC_slow 0.61-0.84 (best at every window), best P&L of the study
#                        (w06: 67.7k, 4.2 trades/mo, WR 53.6%), ECE 0.05-0.14
#     regime_conditional AUC_slow 0.60-0.76, the only mode profitable at ALL four windows
#                        (17k-29k), ECE 0.09-0.21
#     window_cascade     AUC_slow 0.65-0.81, profitable at all windows, and the only mode
#                        clearing the trade-volume gate comfortably (8.5 trades/mo at w06)
#   Rejected:
#     static             AUC_slow 0.46-0.53 (chance), ECE 0.34 (worst calibration),
#                        positive-prediction rate up to 87% = near-collapse, P&L negative
#                        at 3 of 4 windows
#     daily_vol_scaled   AUC_slow 0.47-0.54, ECE 0.27-0.36, negative at 3 of 4 windows
#     direction_horizon  AUC_slow 0.49-0.56, ECE 0.23-0.28, negative at 3 of 4 windows,
#                        and wiped out the account at w36
#     lookahead          Good calibration (ECE 0.02-0.08) but degenerate coverage: ZERO
#                        trades at w18 and w120, 5 trades at w36 — unusable as a strategy
#
#   The three symmetric +/-35-pip barrier modes (static, daily_vol_scaled,
#   direction_horizon) all land at chance AUC with ~50% label rates: a symmetric
#   barrier race over a 96h horizon is close to a coin flip, and the features cannot
#   predict it. The modes that condition on regime (trend_only, regime_conditional,
#   window_cascade) are the ones carrying signal.
#
# Windows: w120 is dropped — best AUC but only 1.5-1.7 trades/month, far under the
# 4/month gate. w06/w18/w36 bracket the observed P&L optimum.
LABEL_STUDY_PHASE2_FINALISTS = [
    ('trend_only',         'sl',  ['w06', 'w18', 'w36']),
    ('regime_conditional', 'sl',  ['w06', 'w18', 'w36']),
    ('window_cascade',     'wc',  ['w06', 'w18', 'w36']),
]


def _label_study_base():
    """
    Hyperparameters held fixed across every label-study run, so the label mode is
    the only variable. Mirrors the documented best command in CLAUDE.md, except
    --fast-train-start is deliberately omitted: it would override the training
    window under test.
    """
    return dict(
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=5,               # advanced_train.py default
        min_child_weight=3,
        num_boost_round=200,
        target_recall=0.3,
        training_sampling=False,
        # Slow model (documented best)
        slow_spw_factor=0.10,
        slow_mi_threshold=0,
        slow_max_depth=3,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_lambda=5.0,
        # Fast model (documented best)
        fast_max_depth=3,
        fast_min_child_weight=3,
        fast_lambda=5.0,
        fast_spw_factor=0.5,
    )


def _label_study_config(label_mode, policy, window_tag, prefix):
    """Build one label-study TrainingConfig for a (mode, fast policy, window) triple."""
    mode_tag, mode_params = LABEL_STUDY_MODES[label_mode]
    policy_params = LABEL_STUDY_FAST_POLICIES[policy] if policy in LABEL_STUDY_FAST_POLICIES else {}
    return TrainingConfig(
        name=f"{prefix}_{mode_tag}_{policy}_{window_tag}",
        label_mode=label_mode,
        train_start=LABEL_STUDY_WINDOWS[window_tag],
        **mode_params,
        **policy_params,
        **_label_study_base(),
    )


def create_label_study_configs(phase='1'):
    """
    Build the label-mode bake-off configurations.

    The question: which label mode + training-window length + fast-label policy
    produces the best out-of-sample result on the fixed backtest window
    2025-10-01 → 2026-04-19? Modes define different prediction targets, so AUC/F1
    are NOT comparable across them — ranking comes from the backtest.

    Args:
        phase (str): 'smoke' → one run per mode at the shortest phase-1 window
                     (7 configs) to prove every mode executes end-to-end;
                     '1' → 6 modes x 2 fast policies + window_cascade, x 4 windows
                     (52 configs, 1 seed);
                     '2' → the LABEL_STUDY_PHASE2_FINALISTS refinement grid
                     (run with --num-seeds 3).

    Returns:
        list[TrainingConfig]
    """
    configs = []

    if phase == 'smoke':
        for label_mode in LABEL_STUDY_MODES:
            policy = 'wc' if label_mode == 'window_cascade' else 'sl'
            configs.append(_label_study_config(label_mode, policy, 'w06', 'smoke'))
        return configs

    if phase == '1':
        for label_mode in LABEL_STUDY_MODES:
            policies = ['wc'] if label_mode == 'window_cascade' else list(LABEL_STUDY_FAST_POLICIES)
            for policy in policies:
                for window_tag in LABEL_STUDY_PHASE1_WINDOWS:
                    configs.append(_label_study_config(label_mode, policy, window_tag, 'p1'))
        return configs

    if phase == '2':
        if not LABEL_STUDY_PHASE2_FINALISTS:
            raise ValueError(
                "Phase 2 needs LABEL_STUDY_PHASE2_FINALISTS filled in with the top-3 "
                "(label_mode, fast_policy, [window tags]) from the phase-1 /eval-models "
                "ranking. Edit iterative_training.py before running phase 2."
            )
        for label_mode, policy, window_tags in LABEL_STUDY_PHASE2_FINALISTS:
            for window_tag in window_tags:
                configs.append(_label_study_config(label_mode, policy, window_tag, 'p2'))
        return configs

    raise ValueError(f"Unknown label-study phase: {phase!r} (expected 'smoke', '1' or '2')")


def create_label_study_sweep_configs(full=False):
    """
    Backtest sweep templates for the label-mode study.

    Two things must be swept rather than left at their defaults:

    * The **regime gate** defaults to 'trending', which structurally suppresses the
      range-regime arm of regime_conditional — that would penalise the mode for
      reasons unrelated to its labels.
    * The **fast-signal entry gate** defaults OFF (in Python and Java alike), so the
      fast model has no influence on entries at all. With it off, p_open_fast is
      inert; emitting several p_open_fast values would produce identical jobs. The
      grid therefore varies p_open_fast only when the fast gate is on.

    Args:
        full (bool): Also emit the exit-strategy combos (phase 2). False → the entry
            grid only, used for the phase-1 screen.

    Returns:
        list[BacktestSweepConfig]: 16 templates (entry grid) or 32 (full).
    """
    configs = []

    for gate in ('off', 'trending'):
        for p_slow in (0.35, 0.55):
            # Fast gate off: p_open_fast is inert, so pin it to one value.
            configs.append(BacktestSweepConfig(
                name=f"E_{gate}_nofast_s{int(p_slow*100)}",
                p_open_fast=0.45, p_open_slow=p_slow,
                regime_gate=gate,
                opening_requires_fast_signal=False,
            ))
            # Fast gate on: the fast model actually gates entries, so sweep its threshold.
            for p_fast in (0.45, 0.55, 0.65):
                configs.append(BacktestSweepConfig(
                    name=f"E_{gate}_f{int(p_fast*100)}_s{int(p_slow*100)}",
                    p_open_fast=p_fast, p_open_slow=p_slow,
                    regime_gate=gate,
                    opening_requires_fast_signal=True,
                ))

    if full:
        for gate in ('off', 'trending'):
            for suffix, pip_th, after_pips, rev_slow, after_time, trailing in EXIT_STRATEGY_COMBOS:
                configs.append(BacktestSweepConfig(
                    name=f"{suffix}_{gate}",
                    p_open_fast=0.55, p_open_slow=0.35,
                    regime_gate=gate,
                    p_close_pip_threshold=pip_th,
                    closing_after_x_pips=after_pips,
                    closing_after_signal_reversal_slow=rev_slow,
                    closing_after_time=after_time,
                    closing_with_trailing_stop=trailing,
                ))

    return configs


def create_label_study_walk_forward_configs():
    """
    Label-mode bake-off, rebuilt for --walk-forward. Three configurations.

    **Why the 2026-08 study needs redoing.** It ranked 52 configurations on ONE fixed
    backtest window with ONE seed. That window carries ~34 trades, where the standard
    error of the total is +/-303 pips — a ranking built on it is dominated by noise, and
    picking the best of 52 draws maximises the luck component. The measured proof:
    in the regularisation A/B, a configuration that reached t=2.42 on a single seed fell
    to t=1.38 over three, with its per-trade estimate halved. Two further changes since
    then also move the numbers — the MI tie-inflation fix (feature order now follows
    features.yaml instead of MI score, which acts like a reseed under colsample_bytree)
    and the MI noise-floor gate, worth ~123k EUR in an A/B when left at its default.

    **Scope is deliberately small.** With 27 month-clusters the evaluation resolves
    roughly a factor-2 difference, so a 52-cell grid would again return a lucky winner.
    Only the three modes that phase 1 showed carry signal are kept:

      trend_only          best P&L of the study, AUC_slow 0.61-0.84
      regime_conditional  the only mode profitable at all four windows
      window_cascade      the only mode comfortably clearing the trade-volume gate

    The four rejected modes stay out: static, daily_vol_scaled and direction_horizon
    all sat at chance AUC with ~50% label rates (a symmetric +/-35-pip barrier race over
    96h is close to a coin flip), and lookahead produced zero trades at two of four
    windows. Those are label-quality failures that no feature-selection change repairs.

    Everything except the label mode is held at the best currently known settings:
    subsample 0.7 / colsample 0.6 (see create_regularisation_configs) and
    mi_permutations=0. The training window is set by --wf-train-months, so the old
    w06/w18/w36 axis is gone from this grid — window length is testable by running the
    same set again at a different --wf-train-months, which keeps the test months
    identical and therefore directly comparable.

    Suggested invocation (108 runs, ~25 min at --parallel-jobs 4):
        --walk-forward --config-set label-walk-forward --num-seeds 3
        --wf-test-start 2023-01-01 --wf-test-end 2026-04-19
        --wf-train-months 18 --wf-test-months 6 --wf-step-months 3
    """
    base = _label_study_base()
    # Fold geometry supplies the dates; drop the study's fixed ones so nothing implies
    # otherwise if these configs are ever inspected before expansion.
    for key in ('train_end', 'backtest_start', 'backtest_end'):
        base.pop(key, None)
    base.update(
        subsample=0.7, colsample_bytree=0.6, mi_permutations=0,
        # Backtest each model at the operating point its own training chose in-fold.
        # A fixed --p-open-slow is not neutral across label modes: it is applied to
        # CALIBRATED probabilities, and every mode fits its own Platt coefficients, so
        # the same number lands on a different raw threshold for each. Measured on run
        # feature_eval, one single model: 0.55 calibrated meant 0.206 raw for long_slow
        # but 0.260 for short_slow — a 37% difference in operating point between the two
        # directions of the SAME model, before any mode comparison even starts.
        use_trained_threshold=True,
    )

    modes = ['trend_only', 'regime_conditional', 'window_cascade']
    # Training-window length is a free parameter with no evidence behind any single
    # value — the 2026-08 study found 6 months best for trend_only while everything
    # measured since used 18. It may well interact with the label mode, so it is a
    # second axis here rather than an assumption. Test windows are unaffected.
    train_windows = [6, 18, 36]

    configs = []
    for label_mode in modes:
        mode_tag, mode_params = LABEL_STUDY_MODES[label_mode]
        # window_cascade brings its own fast labels; the others reuse the slow ones.
        policy = 'wc' if label_mode == 'window_cascade' else 'sl'
        policy_params = LABEL_STUDY_FAST_POLICIES.get(policy, {})
        for months in train_windows:
            configs.append(TrainingConfig(
                name=f"{mode_tag}_w{months:02d}", label_mode=label_mode,
                wf_train_months=months,
                **mode_params, **policy_params, **base,
            ))
    return configs


def create_winner_configs():
    """
    The single validated best configuration: trend_only, 18-month training window.

    Phase 3 of the label study (2026-08-08) ranked this first of nine cells over 12
    rolling folds x 3 seeds: 878 trades, +480,032 EUR, 546.7 EUR/trade, t=2.17, 25/36
    folds won. Settings are the ones each study landed on — subsample 0.7 / colsample 0.6
    from the regularisation A/B, mi_permutations=0 because the MI gate costs ~123k EUR,
    and use_trained_threshold so the backtest runs at the operating point the training
    chose in-fold rather than a hand-set number.

    **This is a baseline, not a confirmation.** Re-running the winner on the same folds
    cannot rescue its significance: the p=0.039 became p=0.354 under Bonferroni because
    nine cells were searched, and repeating the search winner on the same data does not
    undo that. Only periods not used in the selection can. What a single cell buys is
    speed (36 runs instead of 324) and a clean reference to measure the next change
    against — transaction costs, the unvalidated exit knobs, a feature change.

    **Known weakness, worth watching in any run of this.** The most recent window is by
    far its worst: 2025-10..2026-03 lost 70,862 EUR across the three seeds, tightly
    clustered (-21.5k / -23.9k / -25.4k), on 157 trades. The same window with a fixed
    --p-open-slow 0.55 and no trained threshold made +44,727 on 34 trades. Whether that
    is the threshold mechanism or the market phase is unresolved — compare
    use_trained_threshold True vs False on that window to settle it.
    """
    base = _label_study_base()
    for key in ('train_end', 'backtest_start', 'backtest_end'):
        base.pop(key, None)
    base.update(subsample=0.7, colsample_bytree=0.6, mi_permutations=0,
                use_trained_threshold=True)

    mode_tag, mode_params = LABEL_STUDY_MODES['trend_only']
    return [TrainingConfig(
        name=f'{mode_tag}_w18', label_mode='trend_only', wf_train_months=18,
        **mode_params, **LABEL_STUDY_FAST_POLICIES['sl'], **base,
    )]


def create_current_setup_configs():
    """
    The hand-run command at the documented regularisation, as one walk-forward cell.

    Reproduces this pair exactly, once per fold::

        advanced_train.py --label-mode trend_only --max-depth 5 --eta 0.05
          --num-boost-round 200 --subsample 0.7 --colsample-bytree 0.6
          --min-child-weight 3 --pip-target 35 --stop-pips 35 --target-recall 0.3
          --skip-shap --skip-pfi --fast-max-depth 3 --fast-min-child-weight 3
          --fast-spw-factor 0.5 --fast-lambda 5.0 --fast-use-slow-label
          --slow-spw-factor 0.1 --slow-mi-threshold 0 --slow-max-depth 3
          --slow-num-boost-round 200 --slow-min-child-weight 3 --slow-lambda 5.0
          --no-training-sampling --mi-permutations 0
        backtest.py --p-open-slow 0.55 --regime-gate off

    Every hyperparameter comes from ``_label_study_base()`` + the ``sl`` fast policy;
    only the dates are replaced per fold. The entry threshold is the fixed 0.55, not
    the trained operating point — the trained one collapsed in two of twelve windows
    in the label study (395 and 424 trades against a normal 9-121).

    ``--regime-gate off`` is **not** the backtest default (``trending``), so it is set
    explicitly. ``--regime-breakdown`` is deliberately not forwarded: backtest stdout
    is discarded for successful runs, so the breakdown would be computed and thrown
    away. Read regimes from a single re-run of a kept run directory instead.

    Two settings override the hand-run command, both on measured evidence:

    * ``subsample`` / ``colsample_bytree`` 0.7/0.6 instead of 0.9/0.9 — the
      regularisation A/B put subagg ahead of base on every measure over 12 folds
      (410 vs 270 EUR/trade, 35.1% vs 32.7% win rate).
    * ``mi_permutations=0`` — at ``advanced_train``'s default of 20 the MI noise floor
      **gates** feature selection and strips ~27 of 37 slow features including
      ``daily_adx``. Measured over 12 walk-forward folds: -4,374 EUR gated vs
      +119,185 EUR ungated.

    That makes this cell the production command of CLAUDE.md with a fixed entry
    threshold and the regime gate off, rather than the trained threshold of
    ``create_winner_configs()`` — the two differ only in those last two knobs, so a
    run of both on the same folds and seeds isolates the threshold mechanism.

    Window geometry is left to the CLI (``--wf-train-months`` / ``--wf-test-months``
    / ``--wf-test-start`` / ``--wf-test-end``), so the same cell can be re-measured on
    a different span without editing this function.
    """
    base = _label_study_base()
    for key in ('train_start', 'train_end', 'backtest_start', 'backtest_end'):
        base.pop(key, None)
    base.update(subsample=0.7, colsample_bytree=0.6, mi_permutations=0) #, regime_label_source='ml', regime_label_trend_threshold=0.5433)

    mode_tag, mode_params = LABEL_STUDY_MODES['trend_only']
    return [TrainingConfig(
        name=f'{mode_tag}_p055',
        label_mode='trend_only',
        use_trained_threshold=False,
        p_open_slow=0.55,
        regime_gate='off',
        **mode_params, **LABEL_STUDY_FAST_POLICIES['sl'], **base,
    )]


def create_fast_gate_configs():
    """
    Does a conditionally-trained fast model earn the entry gate it needs to matter?

    Three cells on identical folds and paired seeds, all otherwise the production
    command of ``create_current_setup_configs`` (trend_only, subsample 0.7/0.6,
    mi_permutations 0, fixed --p-open-slow 0.55, --regime-gate off):

    ==============  ================================  =============================
    cell            fast training label               entry gate
    ==============  ================================  =============================
    ``sl_nogate``   ``--fast-use-slow-label``         off  (= the current baseline)
    ``sl_gate``     ``--fast-use-slow-label``         ``--opening-requires-fast-signal``
    ``cond_gate``   ``--fast-conditional-on-setup``   ``--opening-requires-fast-signal``
    ==============  ================================  =============================

    ``sl_gate`` exists to keep the comparison interpretable. The requested change is a
    *bundle*: the fast model only reaches the P&L through the entry gate (with the gate
    off the fast probability affects nothing but the signal-reversal exit, itself off by
    default), so the training change cannot be measured without also switching the gate
    on. Enabling the gate is not free — measured 2026-08-07 it *lowered* the share of
    profitable configurations, 78% -> 64%. Without the middle cell a loss in
    ``cond_gate`` could not be attributed to either half of the bundle.

    **Threshold caveat.** All three cells pass the same ``--p-open-fast 0.6``.
    ``--fast-conditional-on-setup`` trains on a different row set (only bars where the
    matching slow label is 1) with a much higher positive rate, so its probabilities do
    not live on the same scale as the ``sl`` arm's. A difference between ``sl_gate`` and
    ``cond_gate`` may therefore be a difference in score distribution rather than in
    timing skill. Read the trade counts first: if the two gated arms differ by a large
    factor, the next experiment is a ``--p-open-fast`` sweep per arm, not a verdict.

    Window geometry is left to the CLI, exactly as in create_current_setup_configs().
    """
    base = _label_study_base()
    for key in ('train_start', 'train_end', 'backtest_start', 'backtest_end'):
        base.pop(key, None)
    base.update(subsample=0.7, colsample_bytree=0.6, mi_permutations=0)

    mode_tag, mode_params = LABEL_STUDY_MODES['trend_only']
    common = dict(
        label_mode='trend_only',
        use_trained_threshold=False,
        p_open_slow=0.55,
        p_open_fast=0.6,
        regime_gate='off',
        **mode_params, **base,
    )
    return [
        TrainingConfig(name='sl_nogate', opening_requires_fast_signal=False,
                       **LABEL_STUDY_FAST_POLICIES['sl'], **common),
        TrainingConfig(name='sl_gate', opening_requires_fast_signal=True,
                       **LABEL_STUDY_FAST_POLICIES['sl'], **common),
        TrainingConfig(name='cond_gate', opening_requires_fast_signal=True,
                       fast_conditional_on_setup=True, **common),
    ]


def _rule_trending_share():
    """Share of daily bars the rule regime (ADX/price-efficiency) calls trending.

    The reference the ML arms are matched to: an absolute |rgm_trend_score|
    threshold means a different *amount* of filtering for every algorithm, so
    matching the share is what makes the arms differ in regime DEFINITION rather
    than in how selective they happen to be.
    """
    from pathlib import Path
    import ModelTrading.config.directories as _dir
    from ModelTrading.source.python.utils import csv as _csv
    from ModelTrading.source.python.features.config import get_feature_config
    from ModelTrading.source.python.labeling.regime import generate_regime_labels

    params = get_feature_config().get_parameters()
    df = _csv.load_csv(str(Path(_dir.DATA_DIR) / 'eurusd_daily.csv'),
                       start_date=None, end_date=None, filter_weekends_flag=True)
    labels = generate_regime_labels(
        df[['high', 'low', 'close']].copy(),
        adx_threshold=params.get('adx_threshold', 25.0),
        efficiency_threshold=params.get('price_efficiency_threshold', 0.5),
        adx_period=params.get('adx_period', 14),
        volatility_period=params.get('volatility_period', 14),
    )
    return float(labels['regime_trend'].astype(bool).mean())


def _rgm_threshold_for_share(algo, share):
    """|rgm_trend_score| threshold at which ``algo`` calls ``share`` of bars trending.

    Derived from the algorithm's own precomputed CSV rather than hard-coded,
    because the score scale is a property of the fit: ``_semantic_scores``
    normalises the direction by ``max|centred return|``, so exactly one state
    lands at +/-1 and the rest at model-specific values. A number that matched
    one fit silently means something else after a refit — measured on the same
    data, 0.42 selected 75.7% of bars under hmm, 65.5% under gmm and 20.3% under
    kmeans.
    """
    import pandas as pd
    from ModelTrading.source.python.features.regime_model import regime_csv_path

    path = regime_csv_path('daily', algo)
    if not path.exists():
        raise FileNotFoundError(
            f"Regime CSV for algo '{algo}' not found: {path}. "
            f"Run: python data/update_regime_model_data.py --algo {algo} "
            "--train-end <day before the first test window>"
        )
    scores = pd.read_csv(path, parse_dates=['date']).set_index('date')['rgm_trend_score']
    return float(scores.dropna().abs().quantile(1.0 - share))


def create_regime_algo_study_configs():
    """
    Rule regime vs hmm / gmm / kmeans — four arms, one walk-forward run.

    **The question.** Does a fitted regime model define trend regimes better than
    the ADX/price-efficiency rule? An earlier 5-window run answered "hmm wins"
    (713,168 vs 468,586 EUR), but the whole margin sat in a single window
    (2024-10..2025-04: hmm +114,326 vs rule +28,670 mean over 5 seeds). Drop that
    window and hmm is the worst of the four (141,536 vs rule 325,238), winning 2
    of 5 windows with a median paired difference of -6,877. Five windows cannot
    separate "captures trends" from "caught one trend", which is what this run is
    for: ~20 windows instead of 5.

    **What is held identical across the arms.** Everything from
    ``create_current_setup_configs()`` — trend_only labels, the fixed 0.55 entry
    threshold, regime gate off, subsample 0.7 / colsample 0.6,
    ``mi_permutations=0``. The arms differ in exactly one thing: which regime
    definition produces the labels and the daily macro-context feature.

    **Per-arm feature config.** ``features-rgm-{algo}.yaml`` swaps
    ``daily_regime_trend`` out for ``daily_rgm_trend_score`` + ``daily_rgm_conf``
    and pins ``parameters.rgm_algo``, which is what routes the arm to
    ``data/regime_daily_{algo}.csv`` for both its features and its labels. Without
    the per-algo files all three ML arms read the same CSV — that is how an
    earlier hmm/gmm pair ended up training on byte-identical labels.

    **Thresholds are derived, not fixed:** each arm's
    ``regime_label_trend_threshold`` is set so it calls the same share of bars
    trending as the rule arm, so the arms differ in regime definition and not in
    selectivity. This reads the CSVs, so they must exist before the run.

    Prerequisite (the ``--train-end`` must precede the FIRST test window, or the
    regime fit has seen the test data and the ML arms get a look-ahead the rule
    arm cannot have)::

        python data/update_regime_model_data.py --algo hmm    --train-end 2015-12-31
        python data/update_regime_model_data.py --algo gmm    --train-end 2015-12-31
        python data/update_regime_model_data.py --algo kmeans --train-end 2015-12-31
    """
    base = _label_study_base()
    for key in ('train_start', 'train_end', 'backtest_start', 'backtest_end'):
        base.pop(key, None)
    base.update(subsample=0.7, colsample_bytree=0.6, mi_permutations=0)

    mode_tag, mode_params = LABEL_STUDY_MODES['trend_only']
    common = dict(label_mode='trend_only', use_trained_threshold=False,
                  p_open_slow=0.55, regime_gate='off',
                  **mode_params, **LABEL_STUDY_FAST_POLICIES['sl'], **base)

    share = _rule_trending_share()
    print(f"[regime-algo-study] rule trending share: {share:.4f} "
          f"- ML thresholds matched to it")

    configs = [TrainingConfig(name='rule', **common)]
    for algo in ('hmm', 'gmm', 'kmeans'):
        threshold = _rgm_threshold_for_share(algo, share)
        print(f"[regime-algo-study] {algo:7} threshold {threshold:.4f}")
        configs.append(TrainingConfig(
            name=algo,
            features_config=f'features-rgm-{algo}.yaml',
            regime_label_source='ml',
            regime_label_trend_threshold=round(threshold, 4),
            **common,
        ))
    return configs


def create_threshold_configs():
    """
    Trained entry threshold vs a fixed one — two cells, everything else identical.

    **The observation this exists to explain.** In the label-study walk-forward the
    winning cell (trend_only, 18m) lost 70,862 EUR in its most recent window,
    2025-10..2026-03, tightly clustered across the three seeds (-21.5k / -23.9k /
    -25.4k) on 157 trades. The same window trained the same way but backtested at a
    fixed --p-open-slow 0.55 made +44,727 on 34 trades. Roughly five times the trades
    and the sign flips. Either the trained operating point is too permissive in the
    current market phase, or that phase is simply unprofitable and the fixed threshold
    only avoided it by trading less. Those have opposite consequences and the run
    distinguishes them.

    **Why only two cells.** A third value would add a multiple-comparison penalty to a
    question that is already at the edge of what 27 month-clusters can resolve. The
    documented best backtest uses --p-open-slow 0.35; test it separately if this run
    makes it interesting.

    Note that ``trained`` reproduces the label study's ``trendonly_w18`` exactly when
    run with the same seeds (34362,51386,99827), so its numbers can be checked against
    that run rather than taken on trust.
    """
    base = _label_study_base()
    for key in ('train_end', 'backtest_start', 'backtest_end'):
        base.pop(key, None)
    base.update(subsample=0.7, colsample_bytree=0.6, mi_permutations=0)

    mode_tag, mode_params = LABEL_STUDY_MODES['trend_only']
    common = dict(label_mode='trend_only', wf_train_months=18,
                  **mode_params, **LABEL_STUDY_FAST_POLICIES['sl'], **base)

    return [
        TrainingConfig(name='trained', use_trained_threshold=True, **common),
        TrainingConfig(name='fixed055', use_trained_threshold=False,
                       p_open_slow=0.55, **common),
    ]


def create_regime_training_configs():
    """
    Train on all bars vs train on trend bars only — two cells.

    **The finding this exists to act on.** Measured 2026-08-09 with
    analytics/feature_ab.py, which splits test AUC by regime: the slow model's headline
    AUC of 0.745 is almost entirely trend-vs-range separation. Inside the trend bars —
    the only population the strategy opens positions in — it scores **0.4256**, below
    chance, and the finer splits are worse (TREND_HIGH_VOL 0.249, TREND_LOW_VOL 0.352,
    TREND_MED_VOL 0.366). Not a window artefact: 0.4256 / 0.4102 / 0.4615 across three
    different train/test splits.

    The suspected mechanism is the training signal itself. Under trend_only every range
    bar is a forced 0, and range bars are roughly half the data, so "is this a range bar"
    is the easiest and most rewarded thing to learn. The model spends its capacity there
    — which is also why daily_regime_range carries 20-80% of the gain — and has little
    left for discriminating winners from losers inside a trend, the thing that actually
    decides a trade.

      allbars  — current behaviour: range bars kept as negatives
      trendfit — --regime-filter --regime-type trend: range bars removed from training

    If the hypothesis holds, `trendfit` should raise the TREND AUC even if its overall
    AUC falls (it loses the free separation), and that trade is worth making: the live
    regime gate already discards range bars, so overall AUC buys nothing.

    **Confound to keep in mind:** filtering roughly halves the training set. A drop in
    trend AUC could be less data rather than a wrong hypothesis, so read the result
    together with the overfitting gap.

    Entry threshold is the fixed 0.55, not the trained one: the trained operating point
    collapsed in two of twelve windows (395 and 424 trades against a normal 9-121) and
    lost the comparison in create_threshold_configs.
    """
    base = _label_study_base()
    for key in ('train_end', 'backtest_start', 'backtest_end'):
        base.pop(key, None)
    base.update(subsample=0.7, colsample_bytree=0.6, mi_permutations=0,
                use_trained_threshold=False, p_open_slow=0.55)

    mode_tag, mode_params = LABEL_STUDY_MODES['trend_only']
    common = dict(label_mode='trend_only', wf_train_months=18,
                  **mode_params, **LABEL_STUDY_FAST_POLICIES['sl'], **base)

    return [
        TrainingConfig(name='allbars', **common),
        TrainingConfig(name='trendfit', regime_filter=True, regime_type='trend', **common),
    ]


def create_window_cascade_wf_configs():
    """
    Walk-forward for the window_cascade / 150-pip configuration.

    **What is being tested.** The single-window run of this configuration
    (`generated/window_cascade`, test period 2025-10..2026-04) produced 25 trades at
    2,766 EUR/trade — the first result in this project above the 1,500 EUR/trade gate,
    against a previous best of 546.7 (`trendonly_w18`, phase 3 of the label study).
    That number rests on ONE window with 25 trades and t = 1.77, and its execution
    knobs were picked on that same window. This config-set moves it onto rolling folds.

    **Why the two cells.** A threshold grid on the single window (4 fast x 5 slow, all
    20 cells profitable, spread 315..3,881 EUR/trade) put the in-sample optimum at
    `p_open_slow` 0.40 (3,881) and the trained threshold mid-field (2,766). The gap is
    four dropped trades out of 25 — exactly the kind of edge that does not survive a
    new window. The two cells differ ONLY in that knob, so the folds answer it:

      * ``trained``  — ``--use-trained-threshold``: the operating point each fold's own
        training picked at target-recall 0.3, which never sees the test window.
      * ``slow040``  — a fixed ``--p-open-slow 0.40``, the single-window optimum. Note
        that a fixed value applies to BOTH directions, while the trained threshold maps
        one raw operating point through each direction's calibrator (long 0.3167 /
        short 0.2864 on the reference run) — so this cell is not merely "the same rule
        at a different number".

    Everything else is held at the reference run: fast gate on at 0.65 (the best row at
    every slow setting in the grid; the trained fast thresholds of 0.79/0.86 produce
    ZERO trades), take-profit 150 pips, stop 35, slow-reversal exit OFF (with it on the
    same models lose money — 93% of trades exit within ~6 bars), regime gate off.

    **Deliberately left as-is:** ``pip_target=150`` does not actually bind — window
    validation accepts ``mfe >= pip_target OR mfe >= 2.5 * ATR``, and the ATR branch
    binds at ~23 pips. The short-horizon label is what supplies the entry timing;
    replacing it with a real 150-pip/20-day label was measured and lost badly
    (-1,570 EUR/trade). The flag stays because it is part of the configuration that
    produced the reference numbers.

    ``stop_pips=35`` matches backtest.py's own default, so the 35-pip stop applies to
    both the labels and the trades even though build_backtest_args() does not forward
    it.

    Usage:
        --walk-forward --config-set window-cascade-wf --num-seeds 3
        --wf-test-start 2023-01-01 --wf-test-end 2026-04-19
        --wf-test-months 6 --wf-step-months 3
    """
    common = dict(
        # --- training: the reference command, verbatim -----------------------
        label_mode='window_cascade',
        wf_train_months=18,
        pip_target=150, stop_pips=35,
        max_depth=3, eta=0.05, subsample=0.7, colsample_bytree=0.6,
        min_child_weight=3,
        slow_max_depth=3, slow_min_child_weight=3, slow_lambda=5.0,
        slow_spw_factor=0.1, slow_mi_threshold=0, slow_num_boost_round=200,
        fast_max_depth=3, fast_min_child_weight=3, fast_lambda=5.0,
        fast_spw_factor=0.5, fast_use_slow_label=True,
        target_recall=0.3, mi_permutations=0, training_sampling=False,
        # --- backtest: the verified reproduction of the reference run --------
        p_open_fast=0.65,
        opening_requires_fast_signal=True,
        p_close_pip_threshold=150, closing_after_x_pips=True,
        closing_after_signal_reversal_slow=False,
        regime_gate='off',
    )

    return [
        TrainingConfig(name='wc150_trained', use_trained_threshold=True, **common),
        TrainingConfig(name='wc150_slow040', use_trained_threshold=False,
                       p_open_slow=0.40, **common),
    ]


def create_pruned_configs():
    """
    Does the pruned feature set hold up out of sample, for every label mode?

    `analytics/feature_pruning.py --emit-config` produced one config per label mode. Their
    CONTENT is identical — 14 features disabled, 2 narrowed to fast-only — which is the
    expected result and worth stating: the redundancy clustering is **label-free**, so the
    cluster structure does not move when the labels do. What does move is the verification,
    because that is scored against each mode's own labels, and there the verdicts differ:

    ==================== ============================================================
    label mode           step-3 verdict per model (3 paired seeds)
    ==================== ============================================================
    trend_only           long_fast REJECT (-0.0126), other three ACCEPT
    regime_conditional   all four ACCEPT
    window_cascade       short_slow REJECT (-0.0309), other three ACCEPT
    ==================== ============================================================

    So the pruned set is not uniformly safe, and a single-window AUC verdict is not enough
    to decide it. That is what this config-set is for: six cells, full vs pruned for each
    of the three label modes the bake-off kept, on identical folds.

    **Three seeds is not a stable verdict.** The same `trend_only` / `long_fast` comparison
    read -0.0091 [-0.0136, -0.0046] at 8 seeds and -0.0126 [-0.0183, -0.0070] at 3 — an
    ACCEPT and a REJECT of the same change. Read these cells as a screen, not a decision.

    Modes deliberately not included: `lookahead`, `direction_horizon`, `daily_vol_scaled`,
    `static` and `atr_scaled` were rejected on measured evidence in the label bake-off
    (chance AUC, near-collapse positive-prediction rates, or zero trades at two of four
    windows); `file` needs an external label parquet.
    """
    shared = dict(
        wf_train_months=18,
        max_depth=3, eta=0.05, subsample=0.7, colsample_bytree=0.6,
        min_child_weight=3,
        slow_max_depth=3, slow_min_child_weight=3, slow_lambda=5.0,
        slow_spw_factor=0.1, slow_mi_threshold=0, slow_num_boost_round=200,
        fast_max_depth=3, fast_min_child_weight=3, fast_lambda=5.0,
        fast_spw_factor=0.5, fast_use_slow_label=True,
        target_recall=0.3, mi_permutations=0, training_sampling=False,
        use_trained_threshold=True,
    )
    # Per-mode label geometry and execution settings, each the configuration that mode was
    # actually measured with — mixing them would confound the feature comparison with a
    # change of trading rules.
    modes = {
        'trend_only': dict(
            label_mode='trend_only', pip_target=35, stop_pips=35,
            p_open_fast=0.5, regime_gate='off',
        ),
        'regcond': dict(
            label_mode='regime_conditional', pip_target=35, stop_pips=35,
            p_open_fast=0.5, regime_gate='off',
        ),
        'wcasc': dict(
            label_mode='window_cascade', pip_target=150, stop_pips=35,
            p_open_fast=0.65, opening_requires_fast_signal=True,
            p_close_pip_threshold=150, closing_after_x_pips=True,
            closing_after_signal_reversal_slow=False, regime_gate='off',
        ),
    }
    config_for = {
        'trend_only': 'features-trend_only.yaml',
        'regcond': 'features-regime_conditional.yaml',
        'wcasc': 'features-window_cascade.yaml',
    }

    out = []
    for tag, mode_params in modes.items():
        common = {**shared, **mode_params}
        out.append(TrainingConfig(name=f'{tag}_full',
                                  features_config='features.yaml', **common))
        out.append(TrainingConfig(name=f'{tag}_pruned',
                                  features_config=config_for[tag], **common))
    return out


def create_labelmode_pruned_configs():
    """
    Label-mode bake-off, each mode running on the feature config validated FOR that mode.

    The original bake-off (2026-08-07/08) ranked label modes on a shared feature set and,
    more importantly, was measured with a chain that is now known to be broken: no
    transaction costs, fixed 1M notional, no CV or walk-forward embargo, no uniqueness
    weights. Every number it produced describes a measurement that no longer exists. This
    re-runs the question with the Stage-0 stack in force and with each mode carrying the
    feature set its own step-3 verification accepted.

    Per-mode feature counts (fast / slow), from
    `analytics/feature_pruning.py --emit-config`, which keeps the full set for any model
    whose verification rejected the pruning:

    ==================== ========== ==========
    label mode           fast       slow
    ==================== ========== ==========
    trend_only           17         22 / 22
    regime_conditional   15         19 / 19
    window_cascade       15         19 / 30
    atr_scaled           15         30 / 30
    ==================== ========== ==========

    (`trend_only` was re-selected 2026-09-06 from the full 65/134 screening catalogue
    via the within-TREND label-association screen; the other three are the mechanical
    --emit-config output.)

    **Consolidated 2026-09-06 to the four living modes.** The original eight-mode grid
    ran twice with the honest measurement chain — labelmode-pruned 2026-08-30
    (docs/results/labelmode_pruned_walkforward_2026-08-30.json) and labelmode-gate
    2026-09-06 (docs/results/labelmode_gate_walkforward_2026-09-06.json) — and both
    agree: `lookahead` and `daily_vol_scaled` lose 1.5-4k EUR/month *reliably*
    (|t| 7-12; the loss is turnover x spread, not prediction error), while `static`
    and `direction_horizon` sit at chance-level AUC with noise P&L. Their per-mode
    feature configs were deleted with the modes; `feature_pruning.py --emit-config`
    regenerates any of them if a mode is ever resurrected. Renamed 2026-09-06: the
    canonical per-mode configs are `features-<mode>.yaml` (formerly
    `features-pruned-<mode>.yaml`).

    **This compares BUNDLES, not labels in isolation.** Each mode keeps the label geometry
    and the execution settings it was actually measured with — `window_cascade` its 150-pip
    take-profit and fast gate, the rest the 35/35 barrier at default exits. Stripping those
    out would put `window_cascade` on an operating point it is known to lose money at (93 %
    of its trades exit within ~6 bars with the slow-reversal exit on), which would compare
    something nobody would run.

    **`--use-trained-threshold` is deliberate and has a consequence.** Each mode picks its
    own operating point in-fold, which is the only fair choice — one mode's threshold on
    another mode's score distribution is meaningless. But it means turnover varies wildly
    between cells, and a mode that trades 5x more at a fifth of the expectancy is not
    obviously worse. **Read trades/month next to EUR/trade**, never EUR/trade alone.

    **This ranks, it does not prove.** 39 month-clusters against the ~680 a significant
    per-trade difference would need. And the arm-to-arm seed component is real: measured
    2026-08-30, an identical comparison read +19.3 EUR/trade on one seed triple and -19.8 on
    another. Five seeds narrows that; it does not remove it.

    Usage:
        --walk-forward --config-set labelmode-pruned --num-seeds 5
        --wf-test-start 2023-01-01 --wf-test-end 2026-04-19
        --wf-test-months 6 --wf-step-months 3 --wf-embargo-days 5
        --cost-model data --risk-model fixed_fractional --cv-gap -1
        --sample-weight uniqueness
    """
    shared = dict(
        wf_train_months=18,
        max_depth=3, eta=0.05, subsample=0.7, colsample_bytree=0.6,
        min_child_weight=3,
        slow_max_depth=3, slow_min_child_weight=3, slow_lambda=5.0,
        slow_spw_factor=0.1, slow_mi_threshold=0, slow_num_boost_round=200,
        fast_max_depth=3, fast_min_child_weight=3, fast_lambda=5.0,
        fast_spw_factor=0.5, fast_use_slow_label=True,
        target_recall=0.3, mi_permutations=0, training_sampling=False,
        use_trained_threshold=True,
        regime_gate='off',
    )
    # The barrier modes share 35/35 at default exits. window_cascade keeps the exit family
    # that defines it.
    barrier = dict(pip_target=35, stop_pips=35, p_open_fast=0.5)
    cascade = dict(pip_target=150, stop_pips=35, p_open_fast=0.65,
                   opening_requires_fast_signal=True,
                   p_close_pip_threshold=150, closing_after_x_pips=True,
                   closing_after_signal_reversal_slow=False)

    modes = [
        ('trendonly', 'trend_only', barrier),
        ('regcond', 'regime_conditional', barrier),
        ('wcasc', 'window_cascade', cascade),
        ('atrscaled', 'atr_scaled', barrier),
    ]
    return [
        TrainingConfig(name=tag, label_mode=mode,
                       features_config=f'features-{mode}.yaml',
                       **{**shared, **execution})
        for tag, mode, execution in modes
    ]


def create_labelmode_gate_configs():
    """
    Label-mode bake-off with the entry gate as a second, training-free axis.

    Identical to ``create_labelmode_pruned_configs`` — the four living label modes,
    each on the pruned feature config validated FOR that mode, trained thresholds,
    regime gate off — but every trained model is backtested TWICE via
    ``backtest_variants``:

    ==============  =============================================================
    ``__slowonly``  entry on the slow signal alone (``--no-opening-requires-fast-signal``)
    ``__fastslow``  entry requires fast AND slow  (``--opening-requires-fast-signal``)
    ==============  =============================================================

    The two arms share every trained model, every fold and every seed, so the gate
    comparison is paired by construction — the seed and window spread cancels out of
    the difference instead of being added twice. Training cost is HALF of running two
    config cells: 4 modes x F folds trainings, 2x that many backtests.

    Both arms run at the operating points the training chose in-fold
    (``--use-trained-threshold`` now resolves the FAST threshold too, per direction
    through each calibrator): a hand-set --p-open-fast would land on a different raw
    operating point per label mode and turn the gate comparison into a threshold
    comparison. The per-config ``p_open_fast`` remains only as the fallback for
    summaries that predate the fast-threshold export.

    Note on window_cascade: its documented bundle has the fast gate ON (the fast
    model IS its timing model), so its ``__slowonly`` arm deliberately deviates from
    the bundle — that is the measurement, not an oversight. Measured 2026-08-07 on
    the shared feature set, enabling the gate LOWERED the share of profitable
    configurations (78% -> 64%); this asks the same question per mode, on each
    mode's own pruned features, paired.

    Usage (5 folds x 4 modes = 20 trainings, 40 backtests):
        --walk-forward --config-set labelmode-gate
        --wf-train-months 18 --wf-test-months 6 --wf-step-months 6
        --wf-test-start 2023-10-01 --wf-test-end 2026-04-19 --wf-embargo-days 5
        --cost-model data --slippage-pips 0.2 --risk-model fixed_fractional
        --risk-pct 1.0 --cv-gap -1 --sample-weight uniqueness

    **Measured 2026-09-06** (docs/results/labelmode_gate_walkforward_2026-09-06.json,
    then still the 8-mode grid, 3 seeds): the fast gate helps NO living mode — paired
    delta `atr_scaled` -2,699 EUR/fold (t=-2.09), `trend_only` -673 (t=-1.08),
    `window_cascade`/`regime_conditional` ~0. Where it read positive (`lookahead`
    +5,431, t=2.99) it did so by halving the turnover of a mode that loses ~2.4k
    EUR/month — bleed reduction, not edge. The default (gate off) stands, now
    per-mode and paired.
    """
    configs = create_labelmode_pruned_configs()
    for cfg in configs:
        cfg.backtest_variants = [
            ('slowonly', dict(opening_requires_fast_signal=False)),
            ('fastslow', dict(opening_requires_fast_signal=True)),
        ]
    return configs


def create_label_geometry_configs():
    """
    Label-geometry sweep (Amendment A15, round-3 B5 Leg A): TP:SL ratio x vertical
    horizon on the two living barrier modes, each on its canonical feature config.

    **The question is the TARGET side, not the features.** Ten feature families
    (A5-A10) found nothing against the fixed 35/35/96h geometry; this sweeps the
    geometry itself with the ML model in the loop — the variant H5's model-free
    rejection of asymmetric trend geometry could not test. The 6-of-18 label-empty
    months of the documented trend_only window are the motivating symptom: a binding
    target geometry shows up as label coverage, which is why coverage/degenerate
    folds are a pre-registered gate, not a nuisance print.

    Grid (12 cells, frozen in docs/preregistration.md A15 before any run):

      trend_only          TP {35, 70, 105, 150} x horizon_max {192, 384}   8 cells
      regime_conditional  TP {35, 70, 105, 150} x horizon_max {384}        4 cells

    SL fixed 35 (the stop dimension is closed, H5 follow-up); TP=35/h384 is each
    mode's living baseline, so the sweep carries its own control arm. In
    regime_conditional only the trend leg sees the sweep — range bars keep their BB
    mean-reversion labels by construction.

    Every trained model is backtested TWICE via ``backtest_variants`` (paired on
    identical models, the labelmode-gate machinery):

      ``__deflt``  the living default exits — isolates the label-geometry question
      ``__tpal``   take-profit exit aligned at the cell's label TP
                   (--closing-after-x-pips at TP) — the bundle-coherence arm: a
                   150-pip label whose execution force-exits on defaults cannot
                   harvest its own target.

    Usage (12 cells x 6 folds x 2 seeds = 144 trainings, 288 backtests):
        --walk-forward --config-set label-geometry --seeds 34362,51386
        --wf-train-months 18 --wf-test-months 6 --wf-step-months 6
        --wf-test-start 2023-01-01 --wf-test-end 2026-04-19 --wf-embargo-days 5
        --cost-model data --slippage-pips 0.2 --risk-model fixed_fractional
        --risk-pct 1.0 --cv-gap -1 --sample-weight uniqueness
        --keep-diagnostics ModelTrading/generated/diag_label_geometry
        --parallel-jobs 2
    """
    shared = dict(
        wf_train_months=18,
        max_depth=3, eta=0.05, subsample=0.7, colsample_bytree=0.6,
        min_child_weight=3,
        slow_max_depth=3, slow_min_child_weight=3, slow_lambda=5.0,
        slow_spw_factor=0.1, slow_mi_threshold=0, slow_num_boost_round=200,
        fast_max_depth=3, fast_min_child_weight=3, fast_lambda=5.0,
        fast_spw_factor=0.5, fast_use_slow_label=True,
        target_recall=0.3, mi_permutations=0, training_sampling=False,
        use_trained_threshold=True,
        regime_gate='off',
        stop_pips=35, p_open_fast=0.5,
    )

    grid = [
        ('trendonly', 'trend_only', (35, 70, 105, 150), (192, 384)),
        ('regcond', 'regime_conditional', (35, 70, 105, 150), (384,)),
    ]

    configs = []
    for tag, mode, targets, horizons in grid:
        for tp in targets:
            for hmax in horizons:
                configs.append(TrainingConfig(
                    name=f"{tag}_tp{tp:03d}_h{hmax:03d}",
                    label_mode=mode,
                    features_config=f'features-{mode}.yaml',
                    pip_target=tp,
                    horizon_max=hmax,
                    backtest_variants=[
                        ('deflt', {}),
                        # Exactly one delta vs __deflt (the registered A15 arm):
                        # the TP exit. Nothing else moves, so the pair isolates it.
                        ('tpal', dict(closing_after_x_pips=True,
                                      p_close_pip_threshold=tp)),
                    ],
                    **shared,
                ))
    return configs


# create_ratediff_configs / create_ccy_configs (and features-ratediff.yaml /
# features-ccy.yaml / features-paper.yaml) were removed 2026-09-06: all three encode
# CLOSED null/negative experiments — ratediff showed no improvement over 36 fold-runs
# (docs/ratediff_experiment_results.md), the ccy family failed its audit 0/240 and its
# frozen replication cell flipped sign out of sample (docs/cross_asset_family_results.md),
# the paper features failed the redundancy screen (docs/paper_features_screen.md).
# To re-run such a family A/B after a PASSED audit: copy the canonical config, flip the
# family's `enabled:` lines (every config carries all catalogue lines — see
# docs/feature_set_consolidation.md), and pair it against features.yaml with everything
# else identical (slow_mi_threshold=0, mi_permutations=0, fast arms bit-identical).


def create_training_configs():
    """
    Fast-model optimisation study — 40 iterations across 5 batches.

    ROOT CAUSE: Current fast model auc_fast≈0.56, f1_fast_post≈0.107 — nearly random.
    Hypothesis: MFE-before-MAE at 10-pip threshold in 6h is ~coin-flip for EUR/USD M15;
    the label is too noisy for the model to learn anything meaningful.

    BATCH 1 (runs 1–8): MFE threshold × horizon sweep
      Goal: find the threshold/horizon that produces a discriminative, learnable label
      while keeping label rate in a reasonable range (15–40%).
      All configs share: best slow model params, 2015-2025 training window,
      regime_conditional labels, no training sampling.

    Subsequent batches update this function after each analysis pass.
    """
    # -------------------------------------------------------------------------
    # Shared base params (best slow model settings from CLAUDE.md)
    # -------------------------------------------------------------------------
    BASE = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        # Slow model (locked to best known config)
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        # No training sampling (improves calibration, per slow model study)
        training_sampling=False,
    )

    configs = []

    # =========================================================================
    # BATCH 1 RESULTS (8 runs):
    #   mfe_threshold↑ → AUC↑ (10pip=0.590 → 35pip=0.675), F1↓ (0.357→0.214)
    #   Horizon 24 > 48 at same threshold. Best: mfe=35, horizon=24.
    #
    # BATCH 2 RESULTS (8 runs):
    #   SPW factor barely moves F1 at mfe=35 (range 0.208–0.218).
    #   Surprising: spw=0.2 gives BEST AUC=0.682 — model fires only when confident.
    #   30-pip label gives F1=0.245 but AUC=0.665 (−0.02 vs 35-pip).
    #   Lock in: mfe=35, horizon=24, spw=0.2. Now sweep XGBoost architecture.
    # =========================================================================

    # =========================================================================
    # BATCH 3 RESULTS (8 runs, mfe=35, spw=0.2):
    #   Shallower trees WIN: depth=4, mcw=3 → AUC=0.687, F1=0.219, R=0.409 (#1)
    #   depth=3, mcw=1 → AUC=0.685; depth=5, mcw=3 → AUC=0.684
    #   Deeper trees (depth=8) overfit: AUC=0.673. More rounds (400) also overfits.
    #   Best so far: mfe=35, h=24, spw=0.2, depth=4, mcw=3 → AUC=0.687, F1=0.219
    # =========================================================================

    # -------------------------------------------------------------------------
    # BATCH 4: Training window + label sweep at best architecture
    # Best arch: depth=4, mcw=3, mfe=35, spw=0.2
    # Vary: (a) training start date, (b) mfe threshold, (c) training sampling
    # Goal: more M15 data helps? Slightly easier label recovers F1?
    # -------------------------------------------------------------------------
    # Lock in best fast model architecture
    FAST_BEST = dict(
        fast_mfe_threshold=35,
        fast_mfe_horizon=24,
        fast_spw_factor=0.2,
        fast_max_depth=4,
        fast_min_child_weight=3,
    )

    # (a) Training window: vary start date, keep end=2025-09-30
    for start_year, name in [("2012-01-01", "B4_2012"),
                              ("2014-01-01", "B4_2014"),
                              ("2015-01-01", "B4_2015"),  # reference
                              ("2017-01-01", "B4_2017"),
                              ("2019-01-01", "B4_2019"),]:
        cfg = dict(**BASE)
        cfg['train_start'] = start_year
        configs.append(TrainingConfig(name=name, **FAST_BEST, **cfg))

    # (b) Easier labels with best architecture: can F1 recover without losing AUC?
    for mfe_thr, name in [(25, "B4_mfe25"), (30, "B4_mfe30")]:
        cfg = dict(**BASE)
        cfg_fast = dict(**FAST_BEST)
        cfg_fast['fast_mfe_threshold'] = mfe_thr
        configs.append(TrainingConfig(name=name, **cfg_fast, **cfg))

    # (c) Training sampling ON (default was OFF for best slow model)
    # Does focusing on bars near label events help M15 fast model?
    cfg = dict(**BASE)
    cfg['training_sampling'] = True   # override: re-enable sampling
    configs.append(TrainingConfig(name="B4_sampling", **FAST_BEST, **cfg))

    # =========================================================================
    # BATCH 4 RESULTS (8 runs):
    #   2017 start → fast_AUC=0.693 (best!) but composite dragged down (slow to 0.717)
    #   2015 reference: fast_AUC=0.687, slow_AUC=0.734 → composite=0.711
    #   2012/2014/2019: all worse than 2015/2017
    #   mfe=25/30: lower AUC (0.670/0.660) confirming 35-pip is optimal label
    #   sampling=ON: no improvement (consistent with slow model finding)
    #   KEY INSIGHT: decoupling M15 fast window (2017) from slow window (2015)
    #   via --fast-train-start should combine best of both: fast_AUC≈0.693 + slow_AUC≈0.734
    # =========================================================================

    # =========================================================================
    # BATCH 5: Per-scope training window decoupling + post-decoupling arch search
    # Uses new --fast-train-start arg to allow M15 fast scope from 2017
    # while keeping 4h/daily slow scope from 2015.
    # Goal: confirm decoupled window beats both 2015 and 2017 baselines.
    #       Then test arch variations on top of the decoupled winner.
    # =========================================================================
    configs = []  # reset: batch 4 already ran; only emit batch 5 configs
    FAST_BEST5 = dict(
        fast_mfe_threshold=35,
        fast_mfe_horizon=24,
        fast_spw_factor=0.2,
        fast_max_depth=4,
        fast_min_child_weight=3,
    )

    # (a) Core decoupled window: fast=2017, slow=2015 — the key test of the new feature
    configs.append(TrainingConfig(name="B5_fts2017",
        fast_train_start="2017-01-01", **FAST_BEST5, **BASE))

    # (b) SPW sensitivity post-decoupling: maybe optimal SPW shifts with 2017 data
    for spw, name in [(1.0, "B5_fts2017_spw1"), (0.5, "B5_fts2017_spw05")]:
        cfg_fast = dict(**FAST_BEST5)
        cfg_fast['fast_spw_factor'] = spw
        configs.append(TrainingConfig(name=name, fast_train_start="2017-01-01", **cfg_fast, **BASE))

    # (c) Depth variations at 2017 start: depth=3 (simpler), depth=5 (more capacity)
    for depth, name in [(3, "B5_fts2017_depth3"), (5, "B5_fts2017_depth5")]:
        cfg_fast = dict(**FAST_BEST5)
        cfg_fast['fast_max_depth'] = depth
        configs.append(TrainingConfig(name=name, fast_train_start="2017-01-01", **cfg_fast, **BASE))

    # (d) Even more recent window: fast=2019, slow=2015
    configs.append(TrainingConfig(name="B5_fts2019",
        fast_train_start="2019-01-01", **FAST_BEST5, **BASE))

    # (e) More boost rounds at 2017 start (eta=0.05 → 300 rounds may find better minimum)
    cfg_fast = dict(**FAST_BEST5)
    cfg_fast['fast_num_boost_round'] = 300
    configs.append(TrainingConfig(name="B5_fts2017_nbr300",
        fast_train_start="2017-01-01", **cfg_fast, **BASE))

    # (f) Lower min_child_weight: less regularization, allow smaller leaf splits
    cfg_fast = dict(**FAST_BEST5)
    cfg_fast['fast_min_child_weight'] = 1
    configs.append(TrainingConfig(name="B5_fts2017_mcw1",
        fast_train_start="2017-01-01", **cfg_fast, **BASE))

    # =========================================================================
    # BATCH 5 RESULTS (8 runs):
    #   fts2017_depth3: fast_AUC=0.695, composite=0.7149 (new best)
    #   fts2017 (depth4): 0.694, fts2017_mcw1: 0.694, fts2017_spw05: 0.693
    #   fts2019: fast_AUC=0.657 (too little data)
    #   depth=3 confirms as best for fast scope at 2017 window
    #   SPW barely matters at mfe=35; all variants F1≈0.21
    #   KEY BOTTLENECK: at mfe=35 (~20% label rate), max achievable F1≈0.21
    #     F1>0.5 requires either much higher AUC OR higher label rate
    #     Slow model: AUC=0.734, label_rate=26%, F1=0.514
    #     Fast model: AUC=0.695, label_rate=20%, F1=0.213 → ceiling
    # =========================================================================

    # =========================================================================
    # BATCH 6: Push fast F1 > 0.5
    # Strategy:
    #   Group A: use slow-style regime_conditional labels for fast scope
    #            → same label as slow model (F1=0.514), different features (M15 only)
    #            → tests whether M15 features can predict medium-term direction
    #   Group B: lower MFE threshold (10, 15 pips) with best arch (2017, depth=3)
    #            → higher label rate → higher F1 achievable; tests precision recovery
    # =========================================================================
    configs = []  # reset: batch 5 done; emit batch 6 only

    BASE6 = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
        fast_train_start="2017-01-01",
    )

    # --- Group A: slow-style labels for fast scope ---
    # mfe args intentionally omitted so fast model gets long_slow substituted in
    SLOW_LABEL_ARCH = dict(fast_max_depth=3, fast_min_child_weight=3, fast_spw_factor=0.2)

    # A1: baseline — slow label, depth=3, spw=0.2
    configs.append(TrainingConfig(name="B6_slowlabel",
        fast_use_slow_label=True, **SLOW_LABEL_ARCH, **BASE6))

    # A2: depth=4 — more capacity for medium-term prediction from M15
    configs.append(TrainingConfig(name="B6_slowlabel_d4",
        fast_use_slow_label=True, **{**SLOW_LABEL_ARCH, 'fast_max_depth': 4}, **BASE6))

    # A3: spw=1.0 — natural class weight (slow model uses spw=0.2, fast may differ)
    configs.append(TrainingConfig(name="B6_slowlabel_spw1",
        fast_use_slow_label=True, **{**SLOW_LABEL_ARCH, 'fast_spw_factor': 1.0}, **BASE6))

    # A4: train_start=2015 for fast too (more data, slow label requires longer history)
    cfg_base6_2015 = dict(**BASE6)
    cfg_base6_2015['fast_train_start'] = "2015-01-01"
    configs.append(TrainingConfig(name="B6_slowlabel_2015",
        fast_use_slow_label=True, **SLOW_LABEL_ARCH, **cfg_base6_2015))

    # --- Group B: MFE with lower threshold using best 2017 arch ---
    BEST_ARCH = dict(fast_max_depth=3, fast_min_child_weight=3, fast_spw_factor=0.2)

    # B1: mfe=10 + best arch (CMD1 threshold but with 2017 window + depth=3)
    configs.append(TrainingConfig(name="B6_mfe10",
        fast_mfe_threshold=10, fast_mfe_horizon=24, **BEST_ARCH, **BASE6))

    # B2: mfe=10, horizon=12 (2h: ultra-short entry quality)
    configs.append(TrainingConfig(name="B6_mfe10_h12",
        fast_mfe_threshold=10, fast_mfe_horizon=12, **BEST_ARCH, **BASE6))

    # B3: mfe=15, horizon=16 (4h: moderate threshold)
    configs.append(TrainingConfig(name="B6_mfe15_h16",
        fast_mfe_threshold=15, fast_mfe_horizon=16, **BEST_ARCH, **BASE6))

    # B4: mfe=10, spw=2.0 (push recall — trades precision for coverage)
    configs.append(TrainingConfig(name="B6_mfe10_spw2",
        fast_mfe_threshold=10, fast_mfe_horizon=24,
        **{**BEST_ARCH, 'fast_spw_factor': 2.0}, **BASE6))

    # =========================================================================
    # BATCH 6 RESULTS (8 runs):
    #   slow-label approach: F1 jumps from 0.21 → 0.42 (regime_conditional labels on M15)
    #   B6_slowlabel: fast_AUC=0.626, F1=0.420, P=0.325, R=0.782
    #   Bottleneck: precision too low (0.325 vs slow model's 0.385)
    #     Slow model: AUC=0.734, uses lambda=5.0 and 4h resampling
    #     Fast model (slow label): AUC=0.626, uses lambda=1.0, no resampling
    #   mfe=10/15 configs: F1=0.31-0.35 — slow label is strictly better
    #   KEY INSIGHT: need to close precision gap; try slow model's exact hyperparams
    #     (lambda=5.0, possibly 4h resampling of M15 for slow label training)
    # =========================================================================

    # =========================================================================
    # BATCH 7: Push slow-label fast model precision to close gap vs slow model
    # Slow model (P=0.385, R=0.776, F1=0.514): lambda=5.0, depth=3, 4h-resampled
    # Fast model best (P=0.325, R=0.782, F1=0.420): lambda=1.0, depth=3, raw M15
    # Hypothesis: higher lambda forces more conservative splits → higher precision
    # Also try: longer training, shallower trees, SPW rebalancing
    # =========================================================================
    configs = []  # reset: batch 6 done; emit batch 7 only

    BASE7 = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
        fast_train_start="2017-01-01",
        fast_use_slow_label=True,
        fast_max_depth=3,
        fast_min_child_weight=3,
    )

    # A: lambda sweep — slow model uses lambda=5.0; current fast uses lambda=1.0
    # Higher lambda → more conservative → higher precision
    for lam, name in [(2.0, "B7_sl_lam2"), (5.0, "B7_sl_lam5"), (10.0, "B7_sl_lam10")]:
        configs.append(TrainingConfig(name=name, fast_spw_factor=0.2,
            fast_lambda=lam, **BASE7))

    # B: depth × lambda cross: depth=2 forces very simple model, might avoid false positives
    configs.append(TrainingConfig(name="B7_sl_d2_lam5",
        fast_spw_factor=0.2, fast_max_depth=2, fast_lambda=5.0, **{k: v for k, v in BASE7.items() if k != 'fast_max_depth'}))

    # C: more boost rounds with high lambda — let the conservative model learn longer
    configs.append(TrainingConfig(name="B7_sl_lam5_nbr400",
        fast_spw_factor=0.2, fast_lambda=5.0, fast_num_boost_round=400, **BASE7))

    # D: SPW tuning with high lambda — spw=0.1 to strongly favour precision
    configs.append(TrainingConfig(name="B7_sl_lam5_spw01",
        fast_spw_factor=0.1, fast_lambda=5.0, **BASE7))

    # E: slower learning rate — same as slow model's eta=0.05 but 500 rounds
    configs.append(TrainingConfig(name="B7_sl_lam5_nbr500",
        fast_spw_factor=0.2, fast_lambda=5.0, fast_num_boost_round=500, **BASE7))

    # F: MFE=35, horizon=96 (24h) — extend MFE horizon to increase label rate
    # Tests whether longer-horizon MFE label can recover F1 while keeping high AUC
    configs.append(TrainingConfig(name="B7_mfe35_h96",
        fast_use_slow_label=False, fast_mfe_threshold=35, fast_mfe_horizon=96,
        fast_spw_factor=0.2, fast_max_depth=3, fast_min_child_weight=3, fast_lambda=1.0,
        **{k: v for k, v in BASE7.items() if k not in ('fast_use_slow_label', 'fast_max_depth', 'fast_min_child_weight')}))

    # =========================================================================
    # BATCH 7 RESULTS (8 runs):
    #   All slow-label configs plateau at F1≈0.42, P≈0.32, R≈0.78 regardless of:
    #   lambda (1→10), depth (2→3), rounds (200→500), spw (0.1→0.2)
    #   mfe35_h96: WORSE (AUC=0.599) — longer MFE horizon adds noise not signal
    #   HARD CEILING: M15 features alone cannot predict slow-style label beyond AUC=0.626
    #   The 0.108 AUC gap vs slow model (0.734) is from missing macro context.
    #   SOLUTION: add 4h/daily context features to fast scope to give macro awareness.
    # =========================================================================

    # =========================================================================
    # BATCH 8: 4h context features injected into fast scope
    # Hypothesis: 3-5 key 4h/daily context features will push fast_AUC from 0.626 → 0.68+
    # which should push fast F1 from 0.42 → 0.45-0.52 with slow-style labels
    #
    # NOTE (historical): batches 8-10 selected these context sets with the
    # --fast-4h-context CLI flag, which no longer exists — feature→model assignment
    # now lives in features.yaml (`models:` tags). The CTX_* constants below are kept
    # only so the RESULTS comments stay readable; the configs no longer inject them.
    # =========================================================================
    configs = []  # reset: batch 7 done; emit batch 8 only

    BASE8 = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
        fast_train_start="2017-01-01",
        fast_use_slow_label=True,
        fast_max_depth=3,
        fast_min_child_weight=3,
        fast_spw_factor=0.2,
        fast_lambda=1.0,
    )

    CTX_TREND_4H    = "4hours_adx_percentile,4hours_price_efficiency,4hours_price_above_cloud"
    CTX_TREND_DAILY = "daily_adx_percentile,daily_price_efficiency,daily_price_above_cloud"
    CTX_BOTH_3      = CTX_TREND_4H + "," + CTX_TREND_DAILY
    CTX_5_4H        = CTX_TREND_4H + ",4hours_rsi,4hours_sma_slope_80"
    CTX_MACRO_DAILY = "daily_vix_level,daily_dxy_zscore_20,daily_volatility_percentile,daily_rsi"
    CTX_ALL_4H      = ",".join([
        "4hours_adx_percentile","4hours_bb_deviation","4hours_bb_percent",
        "4hours_cloud_thickness","4hours_macd_atr","4hours_macd_diff_atr",
        "4hours_momentum_medium","4hours_price_above_cloud","4hours_price_efficiency",
        "4hours_rsi","4hours_sma_30_cross_over_sma_80","4hours_sma_slope_80",
        "4hours_stoch_k","4hours_tenkan_kijun_diff","4hours_volatility_percentile",
    ])

    # A: 4h context only (trend quality: do 4h bars confirm direction?)
    configs.append(TrainingConfig(name="B8_sl_ctx3_4h",
        **BASE8))

    # B: daily context only (macro: is the daily market trending?)
    configs.append(TrainingConfig(name="B8_sl_ctx3_daily",
        **BASE8))

    # C: both 4h + daily trend context (6 features)
    configs.append(TrainingConfig(name="B8_sl_ctx6_both",
        **BASE8))

    # D: 5 key 4h features (trend + momentum + rsi)
    configs.append(TrainingConfig(name="B8_sl_ctx5_4h",
        **BASE8))

    # E: macro daily (VIX, DXY, volatility, RSI) — external macro context
    configs.append(TrainingConfig(name="B8_sl_ctx_macro",
        **BASE8))

    # F: all 4h features injected into fast scope (~15 features, near-full 4h context)
    configs.append(TrainingConfig(name="B8_sl_ctxALL4h",
        **BASE8))

    # G: best context (6 both) + lambda=5.0 (precision boost like slow model)
    configs.append(TrainingConfig(name="B8_sl_ctx6_lam5",
        fast_lambda=5.0, **{k: v for k, v in BASE8.items() if k != 'fast_lambda'}))

    # H: MFE label + 3 4h context (does macro context help MFE-based fast model too?)
    configs.append(TrainingConfig(name="B8_mfe_ctx3_4h",
        fast_use_slow_label=False,
        fast_mfe_threshold=35, fast_mfe_horizon=24,
        **{k: v for k, v in BASE8.items() if k != 'fast_use_slow_label'}))

    # =========================================================================
    # BATCH 8 RESULTS (8 runs):
    #   BREAKTHROUGH: daily context features push fast_AUC 0.626 → 0.810!
    #   B8_sl_ctx6_lam5 (3 4h + 3 daily trend + lambda=5): F1=0.5806 ← F1>0.5 achieved
    #   B8_sl_ctx6_both (3 4h + 3 daily, lambda=1):        F1=0.5791
    #   B8_sl_ctx3_daily (3 daily only, lambda=1):          F1=0.5736
    #   B8_sl_ctx_macro (VIX/DXY):                          F1=0.445  — external macro insufficient
    #   B8_sl_ctxALL4h (all 15 4h features):               F1=0.416  — too many 4h features overfits
    #   B8_sl_ctx3_4h (4h trend only):                      F1=0.415  — 4h alone insufficient
    #   KEY: 3 daily trend features (adx_percentile, price_efficiency, price_above_cloud)
    #        are the critical missing piece. Composite AUC jumps to ~0.77 from 0.715.
    #   Next: more daily features? SPW/lambda tuning? Depth for richer feature set?
    # =========================================================================

    # =========================================================================
    # BATCH 9: Refine context feature set + architecture for F1 > 0.58
    # Winner: ctx6 (3 4h + 3 daily trend) + lambda=5 → F1=0.5806
    # Questions:
    #   (a) More daily features beyond the 3 trend ones → F1 gain?
    #   (b) Does depth=4 help with a richer feature set?
    #   (c) SPW=0.1 → more precision at the cost of recall?
    #   (d) Ablation: is 4h context worth it vs daily-only?
    # =========================================================================
    configs = []  # reset: batch 8 done; emit batch 9 only

    BASE9 = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
        fast_train_start="2017-01-01",
        fast_use_slow_label=True,
        fast_max_depth=3,
        fast_min_child_weight=3,
        fast_spw_factor=0.2,
        fast_lambda=5.0,  # bake in winner from B8
    )

    CTX_3D  = "daily_adx_percentile,daily_price_efficiency,daily_price_above_cloud"
    CTX_4H3 = "4hours_adx_percentile,4hours_price_efficiency,4hours_price_above_cloud"
    CTX_6   = CTX_3D + "," + CTX_4H3
    # Extended daily context: add momentum + volatility dimension
    CTX_5D  = CTX_3D + ",daily_sma_slope_80,daily_rsi"
    CTX_7D  = CTX_3D + ",daily_sma_slope_80,daily_rsi,daily_macd_atr,daily_bb_deviation"
    # 4h extended: trend + momentum
    CTX_5_4H = CTX_4H3 + ",4hours_rsi,4hours_sma_slope_80"

    # A: ablation — 3 daily trend + lam5, no 4h (how much do 4h features add?)
    configs.append(TrainingConfig(name="B9_ctx3d_lam5",
        **BASE9))

    # B: 5 daily features (+ sma_slope_80 + rsi for momentum context) + lam5
    configs.append(TrainingConfig(name="B9_ctx5d_lam5",
        **BASE9))

    # C: 7 daily features (full stationary daily set) + lam5
    configs.append(TrainingConfig(name="B9_ctx7d_lam5",
        **BASE9))

    # D: ctx6 winner + depth=4 (richer feature set may benefit from more depth)
    configs.append(TrainingConfig(name="B9_ctx6_d4_lam5",
        fast_max_depth=4,
        **{k: v for k, v in BASE9.items() if k != 'fast_max_depth'}))

    # E: ctx6 + lam5 + spw=0.1 (aggressive precision focus)
    configs.append(TrainingConfig(name="B9_ctx6_spw01",
        fast_spw_factor=0.1,
        **{k: v for k, v in BASE9.items() if k != 'fast_spw_factor'}))

    # F: ctx6 + lam5 + spw=0.5 (moderate precision/recall balance)
    configs.append(TrainingConfig(name="B9_ctx6_spw05",
        fast_spw_factor=0.5,
        **{k: v for k, v in BASE9.items() if k != 'fast_spw_factor'}))

    # G: 5 daily + 5 4h (richer context on both timeframes)
    configs.append(TrainingConfig(name="B9_ctx10_lam5",
        **BASE9))

    # H: ctx3d + spw=0.1 (simplest winning combo, push precision)
    configs.append(TrainingConfig(name="B9_ctx3d_spw01",
        fast_spw_factor=0.1,
        **{k: v for k, v in BASE9.items() if k != 'fast_spw_factor'}))

    # =========================================================================
    # BATCH 9 RESULTS (8 runs):
    #   New composite AUC record: 0.7734 (B9_ctx6_spw05, fast=0.8125)
    #   SPW=0.5 beats SPW=0.2 and SPW=0.1 — 0.5 is new optimal for fast scope
    #   More context features consistently hurt: 7d/10 configs score below 6-feat winner
    #   6 features (3 4h + 3 daily trend) confirmed as sweet spot
    #   depth=4 is worse than depth=3 for the enriched feature set
    #   F1 plateau at 0.574–0.584 — very tight across all 8 configs (new ceiling ≈0.58)
    #   ctx3d (3 daily only) near-ties ctx6 on AUC while generating more trades (87 vs 66)
    #   Backtest P&L note: ctx3d_lam5 highest at EUR 17,956 (87 trades) but with WR=47%;
    #     ctx6_spw05 better WR=56% suggesting better trade quality; backtest too noisy to pick
    # =========================================================================

    # =========================================================================
    # BATCH 10: SPW fine-grid + lambda tuning at winning context config
    # Winner: ctx6 (3 4h + 3 daily trend) + lambda=5 + spw=0.5 → composite=0.7734
    # Questions:
    #   (a) Is spw=0.5 really the optimum or just best of {0.1, 0.2, 0.5}? Test 0.3, 0.4, 0.6, 0.7
    #   (b) ctx3d (3 daily only) with spw=0.5 — can simpler model match/beat ctx6?
    #   (c) Lambda sensitivity at spw=0.5: lam=3, lam=8 vs current lam=5
    #   (d) Higher min_child_weight for further regularization
    # =========================================================================
    configs = []  # reset: batch 9 done; emit batch 10 only

    BASE10 = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
        fast_train_start="2017-01-01",
        fast_use_slow_label=True,
        fast_max_depth=3,
        fast_min_child_weight=3,
        fast_lambda=5.0,
        fast_spw_factor=0.5,  # bake in B9 winner
    )

    CTX_3D = "daily_adx_percentile,daily_price_efficiency,daily_price_above_cloud"
    CTX_6  = CTX_3D + ",4hours_adx_percentile,4hours_price_efficiency,4hours_price_above_cloud"

    # (a) SPW fine grid around 0.5
    for spw, name in [(0.3, "B10_ctx6_spw03"),
                      (0.4, "B10_ctx6_spw04"),
                      (0.6, "B10_ctx6_spw06"),
                      (0.7, "B10_ctx6_spw07")]:
        configs.append(TrainingConfig(name=name,
            fast_spw_factor=spw,
            **{k: v for k, v in BASE10.items() if k != 'fast_spw_factor'}))

    # (b) ctx3d (simpler: 3 daily only) with spw=0.5 and spw=0.4
    configs.append(TrainingConfig(name="B10_ctx3d_spw05",
        **BASE10))
    configs.append(TrainingConfig(name="B10_ctx3d_spw04",
        fast_spw_factor=0.4,
        **{k: v for k, v in BASE10.items() if k != 'fast_spw_factor'}))

    # (c) Lambda sensitivity at spw=0.5
    configs.append(TrainingConfig(name="B10_ctx6_lam3",
        fast_lambda=3.0,
        **{k: v for k, v in BASE10.items() if k != 'fast_lambda'}))
    configs.append(TrainingConfig(name="B10_ctx6_lam8",
        fast_lambda=8.0,
        **{k: v for k, v in BASE10.items() if k != 'fast_lambda'}))

    # =========================================================================
    # BATCH 10 RESULTS (8 runs) — HARD CEILING CONFIRMED:
    #   AUC range across all 8 configs: 0.7718–0.7728 (delta = 0.001, noise)
    #   F1 range: 0.574–0.584. Lambda 3→8, SPW 0.3→0.7: no meaningful effect.
    #   B10_ctx6_lam8: composite=0.7728, fast=0.8112, F1=0.584 (#1 AUC, marginal)
    #   B10_ctx3d_spw04: composite=0.7724, F1=0.574, 82 trades, P&L=19,620 (#1 P&L)
    #   CONCLUSION: Model is fully converged. Hyperparameter search exhausted.
    #   The architectural choices are locked: slow-style labels + 6 ctx features + lam5
    #
    # FINAL BEST CONFIG (composite AUC): B9_ctx6_spw05
    #   fast_4h_context = 3 4h trend + 3 daily trend (6 features)
    #   fast_spw_factor=0.5, fast_lambda=5.0, fast_max_depth=3, fast_min_child_weight=3
    #   fast_use_slow_label=True, fast_train_start=2017-01-01
    #   composite_AUC=0.7734, fast_AUC=0.8125, fast_F1=0.581, fast_P=0.441, fast_R=0.859
    #
    # FINAL BEST CONFIG (P&L/volume): B10_ctx3d_spw04
    #   fast_4h_context = 3 daily trend only (simpler model)
    #   fast_spw_factor=0.4, fast_lambda=5.0
    #   composite_AUC=0.7724, 82 trades, P&L=19,620, WR=54.9%
    # =========================================================================

    return configs  # batch 10 was the final batch — study complete


def create_regularisation_configs():
    """
    Baseline vs aggressive subsampling, for --walk-forward. Two configurations only,
    differing in one dimension:

      base    — today's production command (subsample 0.9 / colsample 0.9)
      subagg  — subsampling heuristic:     subsample 0.7 / colsample 0.6

    **Walk-forward result, 2026-08-08** (12 folds, test windows 2023-01..2026-03,
    18m train / 6m test / 3m step, 27 month-clusters, mi_permutations=0):

      1 seed                                        3 seeds (36 fold-runs)
      config   EUR/trade   95% CI            t      EUR/trade   95% CI          t
      subagg         867   [ +165, +1,570]   2.42         410   [-174, +993]  1.38
      base           499   [ -264, +1,262]   1.28         270   [-375, +916]  0.82

    **The single-seed significance did not survive reseeding.** Both point estimates
    roughly halved once two more seeds were added: the first run had simply drawn a
    favourable seed, exactly the regression the p=0.068 Bonferroni figure warned about.
    Neither configuration demonstrates a positive expectancy over 2023-2026.

    subagg still leads base on every measure across all 36 fold-runs — per trade
    (410 vs 270), win rate (35.1% vs 32.7%), mean per fold (8,067 vs 4,892) — and both
    win 23/36 folds. That makes subagg the better default, not a proven improvement:
    the gap between them is far smaller than either one's uncertainty.

    The binding constraint is calendar coverage, not trades. At this effect size the
    interval needs roughly twice the 27 available month-clusters. More seeds and more
    overlapping folds add trades inside the same months and barely move it.

    **A third configuration was dropped.** ``lam50`` (slow_lambda 50 + 50 slow rounds)
    came from a single-window AUC study where it looked best by a wide margin (test AUC
    0.7526 vs 0.7110, overfitting gap 0.146 vs 0.257). Over the same 12 walk-forward
    folds it produced -42,870 EUR, won only 6/12 folds and generated the fewest trades
    of the three — the worst of the set, not merely neutral. It is kept out so the
    multiple-comparison penalty stays small. The episode is the clearest case yet of
    this project's AUC/P&L disconnect (see label_mode_comparison.md, r = -0.415):
    do not promote a configuration on AUC evidence alone.

    Everything except the lever under test is held at the current production values
    (see CLAUDE.md).
    """
    shared = dict(
        label_mode='trend_only',
        max_depth=5, eta=0.05, min_child_weight=3,
        pip_target=35, stop_pips=35, target_recall=0.3,
        fast_max_depth=3, fast_min_child_weight=3, fast_spw_factor=0.5,
        fast_lambda=5.0, fast_use_slow_label=True,
        slow_spw_factor=0.1, slow_mi_threshold=0,
        slow_max_depth=3, slow_min_child_weight=3,
        training_sampling=False,
        # Match the production command: report the MI noise floor but do not let it
        # gate selection. At the default of 20 permutations the gate strips ~27 of the
        # 37 slow features — including daily_adx, the largest contributor to the
        # model's within-trend SHAP spread — which is a different experiment.
        mi_permutations=0,
    )

    return [
        TrainingConfig(name='base', subsample=0.9, colsample_bytree=0.9,
                       slow_lambda=5.0, slow_num_boost_round=200, **shared),
        TrainingConfig(name='subagg', subsample=0.7, colsample_bytree=0.6,
                       slow_lambda=5.0, slow_num_boost_round=200, **shared),
    ]


def create_training_configs_batch2(best_mfe_threshold, best_mfe_horizon):
    """
    BATCH 2 (runs 9–16): XGBoost hyperparameter sweep for fast model.
    Fix MFE label at the batch-1 winner and vary fast model architecture.
    Call this after batch 1 analysis to get the next set of configs.
    """
    BASE = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        fast_mfe_threshold=best_mfe_threshold,
        fast_mfe_horizon=best_mfe_horizon,
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
    )

    configs = []
    hyper_sweep = [
        # (name, fast_max_depth, fast_min_child_weight, fast_num_boost_round, fast_lambda)
        ("B2_d3_mcw1",   3, 1, None, 1.0),
        ("B2_d4_mcw3",   4, 3, None, 1.0),
        ("B2_d5_mcw3",   5, 3, None, 1.0),   # default
        ("B2_d6_mcw3",   6, 3, None, 1.0),
        ("B2_d8_mcw3",   8, 3, None, 1.0),
        ("B2_d5_mcw1",   5, 1, None, 1.0),
        ("B2_d5_mcw5",   5, 5, None, 1.0),
        ("B2_d5_r400",   5, 3, 400,  1.0),   # more rounds + low eta
        ("B2_d6_lam2",   6, 3, None, 2.0),   # more L2 regularisation
        ("B2_d8_lam3",   8, 3, None, 3.0),   # stronger regularisation
    ]
    for name, md, mcw, nbr, lam in hyper_sweep:
        configs.append(TrainingConfig(
            name=name,
            fast_max_depth=md,
            fast_min_child_weight=mcw,
            fast_num_boost_round=nbr,
            fast_lambda=lam,
            **BASE,
        ))
    return configs


def create_training_configs_batch3(best_mfe_threshold, best_mfe_horizon,
                                    best_fast_max_depth, best_fast_mcw,
                                    best_fast_nbr, best_fast_lambda):
    """
    BATCH 3 (runs 17–24): SPW factor + MI threshold for fast model.
    Fix MFE label and best XGBoost params; vary class-weight and feature selection.
    """
    BASE = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        fast_mfe_threshold=best_mfe_threshold,
        fast_mfe_horizon=best_mfe_horizon,
        fast_max_depth=best_fast_max_depth,
        fast_min_child_weight=best_fast_mcw,
        fast_num_boost_round=best_fast_nbr,
        fast_lambda=best_fast_lambda,
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
    )

    configs = []
    spw_mi_sweep = [
        ("B3_spw05_mi0",    0.5,  0.0,    None),   # favor precision, all features
        ("B3_spw07_mi0",    0.7,  0.0,    None),
        ("B3_spw10_mi0",    1.0,  0.0,    None),   # natural balance, all features
        ("B3_spw12_mi0",    1.2,  0.0,    None),   # slight recall boost
        ("B3_spw15_mi0",    1.5,  0.0,    None),   # favor recall
        ("B3_spw10_mi001",  1.0,  None,   None),   # default MI threshold (0.001)
        ("B3_spw10_mi005",  1.0,  0.0,    0.005),  # stricter MI
        ("B3_spw07_mi005",  0.7,  0.0,    0.005),  # combined best
    ]
    for name, spw, slow_mi, fast_mi in spw_mi_sweep:
        cfg = dict(**BASE)
        cfg['fast_spw_factor'] = spw
        if fast_mi is not None:
            cfg['fast_mi_threshold'] = fast_mi
        if slow_mi is not None:
            cfg['slow_mi_threshold'] = slow_mi
        configs.append(TrainingConfig(name=name, **cfg))
    return configs


def create_fast_diagnostic_configs():
    """
    Diagnostic study for the fast-model regression (fast_AUC ~0.65 vs documented 0.81).

    Plan: see plans/lass-uns-evaluieren-was-elegant-moon.md

    Two tracks, run as one batch:
      Track A — feature ablation: vary --fast-4h-context and --fast-spw-factor
      Track B — target/label sweep: vary pip-target/stop, label-mode, MFE labels

    BL (= A4 = B2) is the documented "best command" baseline. A4 / B2 are not
    duplicated here — interpret them as BL when reading results.
    """
    # Documented best command (CLAUDE.md, 2026-04-27)
    BASE = dict(
        train_start="2015-01-01",
        train_end="2025-09-30",
        backtest_start="2025-10-01",
        backtest_end="2026-04-19",
        label_mode='regime_conditional',
        pip_target=35,
        stop_pips=35,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        # Fast model best
        fast_max_depth=3,
        fast_min_child_weight=3,
        fast_lambda=5.0,
        fast_train_start="2017-01-01",
        fast_use_slow_label=True,
        fast_spw_factor=0.5,
        # Slow model locked to documented best
        slow_spw_factor=0.20,
        slow_mi_threshold=0,
        slow_min_child_weight=3,
        slow_num_boost_round=200,
        slow_max_depth=3,
        slow_lambda=5.0,
        training_sampling=False,
    )

    # Context feature sets
    CTX_4H_TREND  = "4hours_adx_percentile,4hours_price_efficiency,4hours_price_above_cloud"
    CTX_DAILY_3   = "daily_adx_percentile,daily_price_efficiency,daily_price_above_cloud"
    CTX_6_BEST    = CTX_4H_TREND + "," + CTX_DAILY_3
    CTX_2_CURRENT = "4hours_price_efficiency,daily_price_efficiency"

    configs = []

    # --- Phase 0: Baseline reproduction (BL) -------------------------------
    configs.append(TrainingConfig(
        name="BL_ctx6_spw05",
        **BASE,
    ))

    # --- Phase 1: Track A — feature ablation (A1, A2, A3, A5) --------------
    # A4 is identical to BL and is omitted; treat BL as A4.
    configs.append(TrainingConfig(
        name="A1_ctx0_spw05",
        **BASE,
    ))
    configs.append(TrainingConfig(
        name="A2_ctx2_spw05",
        **BASE,
    ))
    configs.append(TrainingConfig(
        name="A3_ctx3daily_spw05",
        **BASE,
    ))
    configs.append(TrainingConfig(
        name="A5_ctx2_spw030",
        **{**BASE, 'fast_spw_factor': 0.30},
    ))

    # --- Phase 2: Track B — target/label sweep (B1, B3, B4, B5, B6) --------
    # B2 is identical to BL and is omitted; treat BL as B2.
    configs.append(TrainingConfig(
        name="B1_pt25",
        **{**BASE, 'pip_target': 25, 'stop_pips': 25},
    ))
    configs.append(TrainingConfig(
        name="B3_pt45",
        **{**BASE, 'pip_target': 45, 'stop_pips': 45},
    ))
    configs.append(TrainingConfig(
        name="B4_static",
        **{**BASE, 'label_mode': 'static'},
    ))
    # B5 / B6: MFE labels (override fast_use_slow_label=False)
    configs.append(TrainingConfig(
        name="B5_mfe15h24",
        fast_mfe_threshold=15,
        fast_mfe_horizon=24,
        **{**BASE, 'fast_use_slow_label': False},
    ))
    configs.append(TrainingConfig(
        name="B6_mfe25h48",
        fast_mfe_threshold=25,
        fast_mfe_horizon=48,
        **{**BASE, 'fast_use_slow_label': False},
    ))

    return configs


def create_training_configs_old():
    """
    Round 9: Normalized backtest study — fixed period 2025-10-01 → 2026-04-12.

    Round 8 findings (train+backtest-sweep on 150 models × 33 sweep params = 4950 jobs):
    - 96m window wins: 95% of sweep configs profitable, +22.6k avg PnL
    - T_96m s54668 / A_f50_s10: best balance — 61% WR, 90 trades, +38.5k (+6k/month)
    - T_96m s68928 / A_f55_s10: highest PnL — 45.2k (+7.1k/month), ~57% WR
    - T_96m + trailing stop (B_trail): 65-66% WR but only 1.6k/month — WR boost kills PnL
    - p_slow=0.10 is optimal; higher p_slow paradoxically reduces WR
    - T_216m: best AUC (0.636) but only 38% profitable — do NOT deploy
    - T_120m: 80% profitable but lower than 96m
    - 2024H1/2025H1 dead zones for WV models; Oct25-Apr26 is 96m territory

    All configs use the same fixed backtest: 2025-10-01 → 2026-04-12.
    Thresholds are handled by the sweep (create_backtest_sweep_configs).

    Group A: Window × Train-end matrix — 6 windows × 5 end dates = 30 configs
      Purpose: find optimal training history length and recency for Oct25-Apr26 trading.
      Windows: 48m, 72m, 96m, 120m, 144m, 216m
      Train ends: Sep 2025, Sep 2024, Sep 2023, Sep 2022, Sep 2021

    Group B: Hyperparameter search on 96m_2025 (Round 8 winner) — 12 configs
      Purpose: can better hyperparams squeeze more out of the winning window?

    Group C: Hyperparameter variants on 72m and 120m (2025-09) — 8 configs
      Purpose: cross-validate that H3 is optimal for adjacent windows too.

    Total: 50 configs
    Run with: --mode train+backtest-sweep --parallel-jobs 8 --num-seeds 3
    (3 seeds × 50 = 150 jobs; sweep adds 150 × 33 = 4950 backtest jobs)
    """
    configs = []
    BT_START = "2025-10-01"
    BT_END   = "2026-04-12"
    H3  = dict(max_depth=8, subsample=0.75, colsample_bytree=0.75)

    # -------------------------------------------------------------------------
    # GROUP A: Window × Train-end matrix (6 × 5 = 30 configs)
    # All use H3 hyperparams.
    # Naming: A_{window}m_{end_year} e.g. A_96m_2025
    # -------------------------------------------------------------------------
    windows_months = [48, 72, 96, 120, 144, 216]
    end_dates = [
        "2025-09-30",
        "2024-09-30",
        "2023-09-30",
        "2022-09-30",
        "2021-09-30",
    ]
    for wm in windows_months:
        for end_str in end_dates:
            end_dt   = pd.to_datetime(end_str)
            start_dt = end_dt - pd.DateOffset(months=wm)
            configs.append(TrainingConfig(
                name=f"A_{wm}m_{end_str[:4]}",
                train_start=start_dt.strftime("%Y-%m-%d"),
                train_end=end_str,
                backtest_start=BT_START,
                backtest_end=BT_END,
                **H3,
            ))

    # -------------------------------------------------------------------------
    # GROUP B: Hyperparameter search on 96m_2025 (12 configs)
    # Base: train 2017-10-01 → 2025-09-30, H3.
    # Each variant changes one or two hyperparameters relative to H3.
    # -------------------------------------------------------------------------
    hyper_b = [
        ("d6",     dict(max_depth=6,  subsample=0.75, colsample_bytree=0.75)),
        ("d7",     dict(max_depth=7,  subsample=0.75, colsample_bytree=0.75)),
        ("d10",    dict(max_depth=10, subsample=0.75, colsample_bytree=0.75)),
        ("eta05",  dict(max_depth=8,  subsample=0.75, colsample_bytree=0.75, eta=0.05, num_boost_round=400)),
        ("r300",   dict(max_depth=8,  subsample=0.75, colsample_bytree=0.75, num_boost_round=300)),
        ("r150",   dict(max_depth=8,  subsample=0.75, colsample_bytree=0.75, num_boost_round=150)),
        ("sub08",  dict(max_depth=8,  subsample=0.8,  colsample_bytree=0.8)),
        ("sub06",  dict(max_depth=8,  subsample=0.6,  colsample_bytree=0.6)),
        ("mcw3",   dict(max_depth=8,  subsample=0.75, colsample_bytree=0.75, min_child_weight=3)),
        ("mcw5",   dict(max_depth=8,  subsample=0.75, colsample_bytree=0.75, min_child_weight=5)),
        ("d8sub07",dict(max_depth=8,  subsample=0.7,  colsample_bytree=0.7)),
    ]
    for suffix, hparams in hyper_b:
        configs.append(TrainingConfig(
            name=f"B_96m_{suffix}",
            train_start="2017-10-01",
            train_end="2025-09-30",
            backtest_start=BT_START,
            backtest_end=BT_END,
            **hparams,
        ))

    # -------------------------------------------------------------------------
    # GROUP C: Hyperparam variants on 72m and 120m (2025-09) — 8 configs
    # 4 variants × 2 windows = 8 configs
    # -------------------------------------------------------------------------
    hyper_c = [
        ("d6",   dict(max_depth=6,  subsample=0.75, colsample_bytree=0.75)),
        ("d10",  dict(max_depth=10, subsample=0.75, colsample_bytree=0.75)),
        ("eta05",dict(max_depth=8,  subsample=0.75, colsample_bytree=0.75, eta=0.05, num_boost_round=400)),
        ("mcw3", dict(max_depth=8,  subsample=0.75, colsample_bytree=0.75, min_child_weight=3)),
    ]
    window_c = [
        (72,  "2019-10-01"),
        (120, "2015-10-01"),
    ]
    for wm, ts in window_c:
        for suffix, hparams in hyper_c:
            configs.append(TrainingConfig(
                name=f"C_{wm}m_{suffix}",
                train_start=ts,
                train_end="2025-09-30",
                backtest_start=BT_START,
                backtest_end=BT_END,
                **REG, **hparams,
            ))

    return configs  # 30 + 12 + 8 = 50 configs


def generate_rolling_window_configs(
    train_window_months=12,
    step_months=3,
    data_start=None,
    data_end=None,
    backtest_start=None,
    backtest_end=None
):
    """
    Generate configurations for rolling window training.

    This creates multiple training configurations by rolling a fixed-size
    training window across your data. All configurations use the SAME backtest
    period (from timeframes.BACKTEST_START/END) for fair comparison.

    Args:
        train_window_months (int): Size of training window in months
        step_months (int): How many months to step forward between windows
        data_start (str/datetime, optional): Start of available data for training.
            Defaults to timeframes.DATA_AVAILABLE_START
        data_end (str/datetime, optional): End of available data for training.
            Defaults to timeframes.BACKTEST_START - 1 day (train up to backtest period)
        backtest_start (str/datetime, optional): Fixed backtest start date.
            Defaults to timeframes.BACKTEST_START
        backtest_end (str/datetime, optional): Fixed backtest end date.
            Defaults to timeframes.BACKTEST_END

    Returns:
        list[TrainingConfig]: List of rolling window configurations

    Example:
        # Train on rolling 12-month windows, all tested on same backtest period
        configs = generate_rolling_window_configs(
            train_window_months=12,
            step_months=3
        )
        # Window 1: Train 2020-01 to 2020-12 -> Test 2025-10 to 2025-11
        # Window 2: Train 2020-04 to 2021-03 -> Test 2025-10 to 2025-11
        # Window 3: Train 2020-07 to 2021-06 -> Test 2025-10 to 2025-11
        # ... all windows test on the same period for fair comparison
    """
    from dateutil.relativedelta import relativedelta

    # Use timeframes config if not provided
    if data_start is None:
        data_start = timeframes.DATA_AVAILABLE_START
    else:
        data_start = pd.to_datetime(data_start)

    # Fixed backtest period (same for all windows)
    if backtest_start is None:
        backtest_start = timeframes.BACKTEST_START
    else:
        backtest_start = pd.to_datetime(backtest_start)

    if backtest_end is None:
        backtest_end = timeframes.BACKTEST_END
    else:
        backtest_end = pd.to_datetime(backtest_end)

    # Training data should end before backtest period starts
    if data_end is None:
        data_end = backtest_start - pd.Timedelta(days=1)
    else:
        data_end = pd.to_datetime(data_end)

    configs = []
    current_train_start = data_start

    window_num = 1
    while True:
        train_end = current_train_start + relativedelta(months=train_window_months) - pd.Timedelta(days=1)

        # Stop if training window extends beyond available training data
        if train_end > data_end:
            break
        
        
        configs.append(TrainingConfig(
            name=f"rolling_window_{window_num:02d}",
            train_start=current_train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            backtest_start=backtest_start.strftime("%Y-%m-%d"),
            backtest_end=backtest_end.strftime("%Y-%m-%d")
        ))
        

        # Step forward
        current_train_start += relativedelta(months=step_months)
        window_num += 1

    return configs


# =============================================================================
# Walk-Forward Evaluation
# =============================================================================
#
# Why this exists: a single backtest cannot rank configurations here. Measured on
# run feature_eval (2025-10..2026-04): 34 trades, +15.4 pips mean, 51.9 pips spread
# per trade -> standard error of the total +/-303 pips, so a +525 pip result has a
# 95% interval of -68..+1119. It does not even establish that the strategy is
# profitable, let alone that config A beats config B. Walk-forward fixes the sample
# size by testing on every period in the history instead of one, and pools the
# trades so the interval shrinks with the square root of their number.

def generate_walk_forward_folds(data_start=None, data_end=None, train_months=18,
                                test_months=6, step_months=None,
                                test_start=None, test_end=None, embargo_days=0):
    """
    Build the (train_start, train_end, test_start, test_end) windows.

    Each fold trains on ``train_months`` and tests on the ``test_months``
    immediately after — never on data the fold has seen.

    EMBARGO
    -------
    "Never on data the fold has seen" is about the *features*; the *labels* reach further.
    A barrier label at bar t is resolved from bars up to t + horizon, so with
    ``train_end = test_start - 1 day`` the last `horizon` bars of the training labels are
    computed from price action inside the test window. At the default 384-bar (96h) label
    horizon that is the final ~4 trading days of every training window leaking into its
    own evaluation — worse than a CV fold seam, because this is the out-of-sample
    measurement itself.

    ``embargo_days`` ends the training window that many days earlier. Default 0 keeps the
    historical geometry so existing walk-forward results stay comparable; set it to at
    least the label horizon in days (96h labels -> 4, plus a margin) for anything that
    will be believed. The training window keeps its full ``train_months`` length — the
    embargo moves both ends back, it does not shorten the window, so folds stay
    comparable in sample size.

    Two ways to anchor the sequence:

    * ``test_start`` / ``test_end`` (preferred): the test windows are confined to
      that range and each fold's training window is the ``train_months`` directly
      before its own test window. Use this to ask "is the model stable across the
      recent regime" rather than "would it have worked a decade ago".
    * ``data_start`` / ``data_end``: the first fold starts training at data_start
      and folds roll forward through the whole history.

    ``step_months`` defaults to ``test_months``, giving contiguous, non-overlapping
    test windows. A smaller step deliberately overlaps them, which is the right
    design for a stability read (neighbouring windows share most of their data, so
    differences between them isolate the effect of shifting the window) — but the
    trades are then no longer independent. ``aggregate_walk_forward`` handles that
    by clustering the confidence interval on calendar months.

    Returns:
        list[tuple[Timestamp, Timestamp, Timestamp, Timestamp]], chronological.
    """
    from dateutil.relativedelta import relativedelta

    if train_months < 1 or test_months < 1:
        raise ValueError("train_months and test_months must be >= 1")
    step_months = test_months if step_months is None else step_months
    if step_months < 1:
        raise ValueError("step_months must be >= 1")
    if test_start is None and data_start is None:
        raise ValueError("provide either test_start/test_end or data_start/data_end")

    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")

    day = pd.Timedelta(days=1)
    embargo = pd.Timedelta(days=int(embargo_days))
    folds = []

    if test_start is not None:
        # Test-anchored: derive each training window backwards from its test window.
        cursor = pd.to_datetime(test_start)
        last = pd.to_datetime(test_end)
        floor = pd.to_datetime(data_start) if data_start is not None else None
        while True:
            te_s = cursor
            te_e = te_s + relativedelta(months=test_months) - day
            if te_e > last:
                break
            # Both ends move back by the embargo so the window keeps its full length.
            tr_e = te_s - day - embargo
            tr_s = tr_e + day - relativedelta(months=train_months)
            if floor is not None and tr_s < floor:
                raise ValueError(
                    f"Fold testing from {te_s.date()} needs training data from "
                    f"{tr_s.date()}, before the available start {floor.date()}. "
                    f"Shorten --wf-train-months or move --wf-test-start later."
                )
            folds.append((tr_s, tr_e, te_s, te_e))
            cursor += relativedelta(months=step_months)
        return folds

    cursor = pd.to_datetime(data_start)
    end = pd.to_datetime(data_end)
    while True:
        train_end = cursor + relativedelta(months=train_months) - day
        first_test = train_end + day + embargo
        last_test = first_test + relativedelta(months=test_months) - day
        if last_test > end:
            break
        folds.append((cursor, train_end, first_test, last_test))
        cursor += relativedelta(months=step_months)

    return folds


def expand_to_walk_forward(configs, train_months=18, test_months=6, step_months=None,
                           data_start=None, data_end=None, test_start=None, test_end=None,
                           embargo_days=0, verbose=True):
    """
    Turn each configuration into one run per walk-forward fold.

    This is the grid-search entry point: build the grid exactly as before (any of
    the ``create_*_configs`` functions), then expand it here. Every fold of a given
    configuration keeps the same ``base_name``, which is what ``aggregate_walk_forward``
    groups on — so a grid of N configurations over F folds becomes N*F runs that
    collapse back into N rows, each backed by F times the trades.

    Args:
        configs: list[TrainingConfig] — the grid
        train_months/test_months/step_months: window geometry, see generate_walk_forward_folds
        embargo_days: gap between each training window and its test window, see
            generate_walk_forward_folds — without it the last `label_horizon` bars of the
            training labels are resolved from bars inside the test window
        data_start: first training bar; defaults to timeframes.DATA_AVAILABLE_START
        data_end: last usable bar; defaults to timeframes.BACKTEST_END
        verbose: print the fold plan

    Returns:
        list[TrainingConfig] of length len(configs) * n_folds
    """
    data_start = timeframes.DATA_AVAILABLE_START if data_start is None else data_start
    if test_start is None and data_end is None:
        data_end = timeframes.BACKTEST_END

    def _folds_for(months):
        return generate_walk_forward_folds(
            data_start=data_start, data_end=data_end, train_months=months,
            test_months=test_months, step_months=step_months,
            test_start=test_start, test_end=test_end, embargo_days=embargo_days,
        )

    # A config may carry its own training-window length. Only the TRAINING side moves —
    # the test windows are anchored by test_start/step and stay identical across configs,
    # so month clusters line up and the comparison remains valid.
    folds = _folds_for(train_months)
    per_config_months = sorted({m for m in (getattr(c, 'wf_train_months', None) for c in configs)
                                if m is not None and m != train_months})
    fold_cache = {train_months: folds}
    for m in per_config_months:
        fold_cache[m] = _folds_for(m)
    if not folds:
        bounds = (f"test range {pd.to_datetime(test_start).date()}..{pd.to_datetime(test_end).date()}"
                  if test_start is not None else
                  f"data range {pd.to_datetime(data_start).date()}..{pd.to_datetime(data_end).date()}")
        raise ValueError(
            f"No walk-forward fold fits in the {bounds} with train={train_months}m + "
            f"test={test_months}m. Shorten the windows or widen the range."
        )

    step = test_months if step_months is None else step_months
    if verbose:
        covered = sum((d - c).days + 1 for _, _, c, d in folds)
        span = (max(d for *_, d in folds) - min(c for *_, c, _ in folds)).days + 1
        print(f"\n{'='*80}\nWALK-FORWARD PLAN\n{'='*80}")
        print(f"  {len(folds)} folds x {len(configs)} configs = {len(folds)*len(configs)} runs")
        print(f"  train {train_months}m -> test {test_months}m, step {step}m")
        for i, (a, b, c, d) in enumerate(folds, 1):
            print(f"    fold {i:02d}: train {a.date()}..{b.date()}  ->  test {c.date()}..{d.date()}")
        if step < test_months:
            print(f"\n  Test windows overlap by design (step {step}m < test {test_months}m):")
            print(f"    {covered} fold-days over {span} distinct calendar days "
                  f"= each day counted {covered/span:.2f}x on average.")
            print("    Good for reading stability across shifted windows. The pooled confidence")
            print("    interval is clustered on calendar months so the overlap does not inflate")
            print("    the apparent evidence.")
        print(f"{'='*80}\n")

    expanded = []
    for cfg in configs:
        base = cfg.base_name or cfg.name
        cfg_folds = fold_cache[getattr(cfg, 'wf_train_months', None) or train_months]
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(cfg_folds, 1):
            c = copy.deepcopy(cfg)
            c.base_name = base
            # Derive from the PARENT name, not from base: with --num-seeds the parent is
            # already "<base>_s<seed>", and naming folds after base alone would give every
            # seed the same fold names. base_name still groups them all back together.
            c.name = f"{cfg.name}_wf{i:02d}"
            # deepcopy carried the parent's run_id; every run needs its own directory
            c.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            parent_eff = getattr(cfg, 'effective_run_id', None) or cfg.run_id
            prefix = os.path.dirname(parent_eff)          # '' or 'scenarios/<name>'
            c.effective_run_id = os.path.join(prefix, c.run_id) if prefix else c.run_id
            c.train_start = tr_s.strftime('%Y-%m-%d')
            c.train_end = tr_e.strftime('%Y-%m-%d')
            c.backtest_start = te_s.strftime('%Y-%m-%d')
            c.backtest_end = te_e.strftime('%Y-%m-%d')
            c.walk_forward_fold = i
            # An absolute fast-model start date would give each fold a different amount
            # of history and break the comparison between folds.
            if getattr(c, 'fast_train_start', None):
                c.fast_train_start = None
            expanded.append(c)

    return expanded


def _seed_breakdown(runs):
    """
    Split one configuration's fold runs by seed and total each seed separately.

    Pooling everything answers "does this configuration have an edge". This answers a
    different question: **is that answer the same on every random draw**. The seed
    changes the row and column subsamples and the tree structure, not the data, so a
    configuration whose seeds disagree in sign has not been measured yet, whatever the
    pooled total says. This project has already been bitten by exactly that — the
    regularisation A/B's single-seed t=2.42 fell to t=1.38 when two more seeds were
    added, because the first run had drawn a favourable one.

    Each seed sees the *same* calendar months, so the per-seed intervals are not
    independent evidence to be combined — read them as a reproducibility check on the
    pooled row above, not as three separate studies.

    Args:
        runs: the successful result dicts of a single base_name.

    Returns:
        list[dict] sorted by seed, or [] when the runs carry no seed distinction
        (a single-seed run has nothing to break down).
    """
    from collections import defaultdict

    by_seed = defaultdict(list)
    for r in runs:
        by_seed[r['config'].get('seed')].append(r)
    if len(by_seed) < 2:
        return []

    out = []
    for seed in sorted(by_seed, key=lambda s: (s is None, s)):
        seed_runs = by_seed[seed]
        fold_pnls, pnl, pips, months = [], [], [], []
        for r in seed_runs:
            trades = r.get('trades') or []
            fold_pnls.append(sum(float(t.get('pnl', 0.0)) for t in trades))
            for t in trades:
                pnl.append(float(t.get('pnl', 0.0)))
                pips.append(float(t.get('pnl_pips', 0.0)))
                months.append(str(t.get('open_time', ''))[:7])

        n = len(pnl)
        mean = sum(pnl) / n if n else 0.0
        se, n_clusters = _clustered_se(pnl, months if any(months) else None)
        out.append({
            'seed': seed,
            'n_folds': len(seed_runs),
            'n_trades': n,
            'n_month_clusters': n_clusters,
            'total_pnl': sum(pnl),
            'total_pips': sum(pips),
            'mean_pnl_per_trade': mean,
            'mean_pnl_per_fold': statistics.mean(fold_pnls) if fold_pnls else 0.0,
            'std_pnl_per_fold': statistics.stdev(fold_pnls) if len(fold_pnls) > 1 else 0.0,
            't_stat': mean / se if n > 1 and se == se and se > 0 else 0.0,
            'win_rate': (sum(1 for p in pnl if p > 0) / n) if n else 0.0,
            'folds_profitable': sum(1 for p in fold_pnls if p > 0),
            'fold_pnl_min': min(fold_pnls) if fold_pnls else 0.0,
            'fold_pnl_max': max(fold_pnls) if fold_pnls else 0.0,
            'fold_pnls': fold_pnls,
        })
    return out


def aggregate_walk_forward(results):
    """
    Pool the trades of every fold per configuration and quantify the uncertainty.

    Pooling trades is the point — averaging each fold's P&L would weight a 3-trade
    fold like a 40-trade one and would throw away the per-trade spread the interval
    is built from.

    Two questions are answered separately:

    * **Evidence** — is the per-trade expectancy positive? Answered by the pooled
      mean and its clustered 95% interval (see ``_clustered_se``).
    * **Stability** — does it hold up as the window shifts? Answered by the spread
      of the per-fold results, which pooling deliberately hides. A configuration
      that is significant overall but loses in a third of the folds is a different
      proposition from one that wins in nearly all of them.

    Returns:
        list[dict] sorted by t_stat descending.
    """
    from collections import defaultdict
    import math

    groups = defaultdict(list)
    for r in results:
        if r.get('status') != 'success':
            continue
        groups[r['config'].get('base_name') or r['config']['name']].append(r)

    out = []
    for base, runs in groups.items():
        # run_parallel_training appends results as futures COMPLETE, so the incoming
        # order is whichever job happened to finish first. Sort before aggregating,
        # otherwise fold_pnls is an unlabelled list in nondeterministic order and no
        # entry can be traced back to the period that produced it.
        runs = sorted(runs, key=lambda r: (r['config'].get('backtest_start') or '',
                                           str(r['config'].get('seed'))))
        fold_pnls, fold_details, pnl, pips, months = [], [], [], [], []
        for r in runs:
            trades = r.get('trades') or []
            fold_pnl = sum(float(t.get('pnl', 0.0)) for t in trades)
            fold_pnls.append(fold_pnl)
            fold_details.append({
                'test_start': r['config'].get('backtest_start'),
                'test_end': r['config'].get('backtest_end'),
                'train_start': r['config'].get('train_start'),
                'seed': r['config'].get('seed'),
                'pnl': fold_pnl,
                'n_trades': len(trades),
            })
            for t in trades:
                pnl.append(float(t.get('pnl', 0.0)))
                pips.append(float(t.get('pnl_pips', 0.0)))
                months.append(str(t.get('open_time', ''))[:7])   # 'YYYY-MM'

        n = len(pnl)
        mean = sum(pnl) / n if n else 0.0
        std = statistics.stdev(pnl) if n > 1 else 0.0
        naive_se = std / math.sqrt(n) if n > 1 else float('nan')
        se, n_clusters = _clustered_se(pnl, months if any(months) else None)
        ci = 1.96 * se if n > 1 and se == se else float('nan')
        t_stat = mean / se if n > 1 and se == se and se > 0 else 0.0

        n_folds = len(runs)
        fold_mean = statistics.mean(fold_pnls) if fold_pnls else 0.0
        fold_std = statistics.stdev(fold_pnls) if len(fold_pnls) > 1 else 0.0

        out.append({
            'base_name': base,
            # n_folds counts fold RUNS: with S seeds every test window is run S times,
            # so it is n_windows * n_seeds, not the number of distinct windows.
            'n_folds': n_folds,
            'n_windows': len({r['config'].get('backtest_start') for r in runs}),
            'n_seeds': len({r['config'].get('seed') for r in runs}),
            'n_trades': n,
            'n_month_clusters': n_clusters,
            'total_pnl': sum(pnl),
            'total_pips': sum(pips),
            'mean_pnl_per_trade': mean,
            'std_pnl_per_trade': std,
            'se_pnl_per_trade': se,
            'se_naive': naive_se,
            'ci95_low': mean - ci if ci == ci else float('nan'),
            'ci95_high': mean + ci if ci == ci else float('nan'),
            't_stat': t_stat,
            # Cluster-robust inference needs a decent number of clusters; below about a
            # year of distinct months the interval itself is too unreliable to license a
            # claim, however wide or narrow it comes out.
            'clusters_reliable': n_clusters >= MIN_CLUSTERS_FOR_SIGNIFICANCE,
            # The interval excluding 0 is the only claim the data supports: this
            # configuration has a positive per-trade expectancy.
            'significant': bool(n > 1 and ci == ci and (mean - ci) > 0
                                and n_clusters >= MIN_CLUSTERS_FOR_SIGNIFICANCE),
            'win_rate': (sum(1 for p in pnl if p > 0) / n) if n else 0.0,
            # --- stability across shifted windows ---
            'folds_profitable': sum(1 for p in fold_pnls if p > 0),
            'fold_win_ratio': (sum(1 for p in fold_pnls if p > 0) / n_folds) if n_folds else 0.0,
            'fold_pnl_mean': fold_mean,
            'fold_pnl_std': fold_std,
            'fold_pnl_min': min(fold_pnls) if fold_pnls else 0.0,
            'fold_pnl_max': max(fold_pnls) if fold_pnls else 0.0,
            'fold_pnls': fold_pnls,
            # Same order as fold_pnls, but labelled: which window, which seed, how many
            # trades. Without this the summary cannot answer "when did this money come
            # from", which is the first question anyone asks of a walk-forward.
            'fold_details': fold_details,
            # --- reproducibility across random draws ---
            # Empty for a single-seed run. See _seed_breakdown for why this is read
            # separately from the pooled row rather than added to it.
            'per_seed': _seed_breakdown(runs),
            # What is actually missing, in the unit that binds. Trades are NOT the
            # limiting resource once the interval is clustered: extra seeds and extra
            # overlapping folds add trades inside the SAME calendar months and move the
            # clustered interval hardly at all. The independent unit is the month, so
            # report the calendar coverage the current effect size would need. Scaling:
            # SE ~ 1/sqrt(G), so reaching |t| = 1.96 needs G * (1.96/t)^2 clusters.
            'months_needed_for_significance': (
                int(math.ceil(n_clusters * (1.96 / abs(t_stat)) ** 2))
                if n_clusters and t_stat and abs(t_stat) > 0 else None
            ),
        })

    out.sort(key=lambda x: x['t_stat'], reverse=True)
    return out


def _print_per_seed_tables(aggregates):
    """
    One block per configuration: what each seed contributed on its own.

    Silent for single-seed runs — ``per_seed`` is empty there and a table of one row
    would only suggest a comparison that was not made.
    """
    with_seeds = [a for a in aggregates if a.get('per_seed')]
    if not with_seeds:
        return

    print(f"\n{'='*120}")
    print("WALK-FORWARD — PER SEED  (same windows, different random draw)")
    print(f"{'='*120}")
    for a in with_seeds:
        print(f"  {a['base_name']}   ({a['n_windows']} windows x {a['n_seeds']} seeds)")
        print(f"    {'Seed':>8} {'Won':>8} {'Trades':>7} {'Total EUR':>12} {'Mean/Wnd':>12} "
              f"{'Std/Wnd':>12} {'EUR/Trade':>10} {'t':>6} {'WR':>6} {'Worst':>12} {'Best':>12}")
        print(f"    {'-'*8} {'-'*8} {'-'*7} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*6} "
              f"{'-'*6} {'-'*12} {'-'*12}")
        for s in a['per_seed']:
            seed = 'n/a' if s['seed'] is None else str(s['seed'])
            print(f"    {seed:>8} {s['folds_profitable']:>3}/{s['n_folds']:<4} "
                  f"{s['n_trades']:>7} {s['total_pnl']:>12,.0f} "
                  f"{s['mean_pnl_per_fold']:>12,.0f} {s['std_pnl_per_fold']:>12,.0f} "
                  f"{s['mean_pnl_per_trade']:>10,.1f} {s['t_stat']:>6.2f} "
                  f"{s['win_rate']:>6.1%} {s['fold_pnl_min']:>12,.0f} {s['fold_pnl_max']:>12,.0f}")
        totals = [s['total_pnl'] for s in a['per_seed']]
        spread = max(totals) - min(totals)
        signs = {t > 0 for t in totals}
        print(f"    {'spread':>8} {'':>8} {'':>7} {spread:>12,.0f}"
              f"   <- max-min across seeds"
              + ("" if len(signs) > 1 else "  (all seeds agree in sign)"))
        if len(signs) > 1:
            print("    Seeds disagree in sign: the pooled total is a property of the draw, not")
            print("    of the configuration. Add seeds before reading anything into it.")
    print(f"{'='*120}")
    print("  The seed changes the row/column subsample and the tree structure, not the data or")
    print("  the windows. Every seed therefore covers the SAME calendar months — these rows are")
    print("  a reproducibility check on the pooled result, not independent evidence to add to it.")


def print_walk_forward_summary(results, top_n=20):
    """
    Three tables, because they answer three different questions.

    EVIDENCE ranks by t-statistic — is the per-trade expectancy positive at all.
    STABILITY shows the spread across fold runs — does it survive shifting the window,
    which is what pooling deliberately averages away.
    PER SEED shows the same total split by random draw — is the pooled number a
    property of the configuration or of the seed. Printed only for multi-seed runs.
    """
    agg = aggregate_walk_forward(results)
    if not agg:
        print("\nNo successful walk-forward runs to aggregate.")
        return agg

    print(f"\n{'='*120}")
    print("WALK-FORWARD — EVIDENCE  (trades pooled across folds, interval clustered on calendar months)")
    print(f"{'='*120}")
    print(f"  {'#':<3} {'Config':<24} {'Runs':>5} {'Wnd':>4} {'Sd':>3} {'Trades':>7} {'Months':>7} "
          f"{'Total EUR':>12} {'EUR/Trade':>10} {'95% CI':>20} {'t':>6} {'WR':>6}")
    print(f"  {'-'*3} {'-'*24} {'-'*5} {'-'*4} {'-'*3} {'-'*7} {'-'*7} {'-'*12} {'-'*10} "
          f"{'-'*20} {'-'*6} {'-'*6}")
    for i, a in enumerate(agg[:top_n], 1):
        ci = (f"[{a['ci95_low']:>8,.0f},{a['ci95_high']:>8,.0f}]"
              if a['ci95_low'] == a['ci95_low'] else f"{'n/a':>20}")
        mark = '*' if a['significant'] else ' '
        print(f"  {i:<3} {a['base_name']:<24} {a['n_folds']:>5} {a['n_windows']:>4} "
              f"{a['n_seeds']:>3} {a['n_trades']:>7} "
              f"{a['n_month_clusters']:>7} {a['total_pnl']:>12,.0f} "
              f"{a['mean_pnl_per_trade']:>10,.1f} {ci:>20} {a['t_stat']:>6.2f} "
              f"{a['win_rate']:>5.1%}{mark}")
    print(f"{'='*120}")
    print("  Runs = Wnd x Sd: every test window (Wnd) is run once per seed (Sd). Months counts")
    print("  DISTINCT calendar months, which is what the interval is clustered on — reseeding")
    print("  multiplies the trades but not the months.")
    print("  * = the 95% interval excludes zero: a positive per-trade expectancy is actually")
    print("      demonstrated. Rows without it are not distinguishable from break-even,")
    print("      however large the total looks.")

    print(f"\n{'='*120}")
    print("WALK-FORWARD — STABILITY  (per-run results; one shifted window at one seed)")
    print(f"{'='*120}")
    print(f"  {'#':<3} {'Config':<24} {'Runs won':>10} {'Mean/Run':>12} {'Std/Run':>12} "
          f"{'Worst':>12} {'Best':>12}")
    print(f"  {'-'*3} {'-'*24} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i, a in enumerate(agg[:top_n], 1):
        print(f"  {i:<3} {a['base_name']:<24} "
              f"{a['folds_profitable']:>4}/{a['n_folds']:<5} {a['fold_pnl_mean']:>12,.0f} "
              f"{a['fold_pnl_std']:>12,.0f} {a['fold_pnl_min']:>12,.0f} {a['fold_pnl_max']:>12,.0f}")
    print(f"{'='*120}")
    print("  A config that is significant overall but wins only two thirds of its runs is a")
    print("  different proposition from one that wins nearly all of them. Read both tables.")

    _print_per_seed_tables(agg[:top_n])

    if not any(a['significant'] for a in agg):
        best = agg[0]
        need = best['months_needed_for_significance']
        print(f"\n  No configuration reached significance. Best evidence: {best['base_name']} "
              f"(t={best['t_stat']:.2f}, {best['n_trades']} trades over "
              f"{best['n_month_clusters']} months).")
        if need:
            print(f"  At this effect size it would take ~{need} months of distinct calendar "
                  f"coverage ({best['n_month_clusters']} available). More seeds or more "
                  f"overlapping folds will NOT get you there — they add trades inside the")
            print("  same months, and the clustered interval barely moves. Only a wider test "
                  "range, or a larger per-trade edge, will.")
    print()
    return agg


# =============================================================================
# Execution Functions
# =============================================================================

# Campaign-wide measurement settings, set once by main() from the CLI and appended to
# EVERY train/backtest invocation. They belong at campaign level, not per config: a
# comparison in which cells are charged different costs or sized differently is not a
# comparison. Empty = the child scripts' own defaults, i.e. exactly the pre-2026-08-29
# behaviour, so existing walk-forward results stay reproducible.
CAMPAIGN_TRAIN_ARGS: list = []
CAMPAIGN_BACKTEST_ARGS: list = []
# --wandb* flags for every child, kept separately for the backtest-sweep path
# (which deliberately does NOT inherit the measurement flags above).
CAMPAIGN_WANDB_ARGS: list = []


def set_campaign_args(args):
    """Derive the campaign-wide train/backtest flags from the parsed CLI."""
    global CAMPAIGN_TRAIN_ARGS, CAMPAIGN_BACKTEST_ARGS, CAMPAIGN_WANDB_ARGS
    train, bt = [], []

    if getattr(args, 'cv_gap', 0):
        train += ['--cv-gap', str(args.cv_gap)]
    if getattr(args, 'sample_weight', 'none') != 'none':
        train += ['--sample-weight', args.sample_weight]
    if getattr(args, 'fail_on_degenerate_folds', False):
        train.append('--fail-on-degenerate-folds')

    if getattr(args, 'cost_model', 'none') != 'none':
        bt += ['--cost-model', args.cost_model]
        if args.cost_model == 'fixed':
            bt += ['--spread-pips', str(args.spread_pips)]
        # Commission and overnight financing are forwarded EXPLICITLY, including an
        # explicit 0: backtest.py now defaults to the Dukascopy schedule (18 /
        # 63.65 / 28.65 per M), so a truthiness check would silently turn a
        # campaign's "commission-free" request into the full fee schedule. Under
        # --cost-model none the child zeroes all three anyway, so nothing is
        # forwarded and old campaigns stay byte-identical.
        bt += ['--commission-per-million',
               str(getattr(args, 'commission_per_million',
                           costs.DEFAULT_COMMISSION_PER_MILLION))]
        bt += ['--overnight-long-per-million',
               str(getattr(args, 'overnight_long_per_million',
                           costs.DEFAULT_OVERNIGHT_LONG_PER_MILLION))]
        bt += ['--overnight-short-per-million',
               str(getattr(args, 'overnight_short_per_million',
                           costs.DEFAULT_OVERNIGHT_SHORT_PER_MILLION))]
    if getattr(args, 'slippage_pips', 0.0):
        bt += ['--slippage-pips', str(args.slippage_pips)]
    if getattr(args, 'risk_model', 'fixed_notional') != 'fixed_notional':
        bt += ['--risk-model', args.risk_model, '--risk-pct', str(args.risk_pct)]

    # W&B tracking: forward the flags to EVERY child so each training and each
    # backtest becomes its own W&B run (grouped by run_id in the child). Campaign
    # coordinates travel as tags so a grid can be filtered in the UI.
    campaign_tags = []
    if getattr(args, 'scenario_name', None):
        campaign_tags.append(f'scenario:{args.scenario_name}')
    if getattr(args, 'config_set', None):
        campaign_tags.append(f'config-set:{args.config_set}')
    wandb_flags = experiment_tracking.wandb_cli_flags(args, extra_tags=campaign_tags)
    train += wandb_flags
    bt += wandb_flags

    CAMPAIGN_TRAIN_ARGS, CAMPAIGN_BACKTEST_ARGS = train, bt
    CAMPAIGN_WANDB_ARGS = wandb_flags
    if train or bt:
        print(f"Campaign measurement settings — train: {' '.join(train) or '(defaults)'} | "
              f"backtest: {' '.join(bt) or '(defaults)'}")
    return train, bt


def build_train_args(config, enable_plotting=False, train_script=None,
                     campaign_args=None, diagnostics_full=False):
    """
    Build the advanced_train.py command line for a TrainingConfig.

    Kept as a pure function (no subprocess, no filesystem) so the argument
    forwarding — especially the label-mode specific flags — is unit-testable.

    Args:
        config (TrainingConfig): Training configuration.
        enable_plotting (bool): Append --plot.
        train_script (str, optional): Path to advanced_train.py. Defaults to the
            copy next to this file.

    Returns:
        list[str]: Full argv list, starting with the Python executable.
    """
    if train_script is None:
        train_script = os.path.join(os.path.dirname(__file__), 'advanced_train.py')

    args = [sys.executable, train_script, '--run-id', config.effective_run_id]
    if enable_plotting:
        args.append('--plot')

    if config.features_config:
        args.extend(['--features-config', config.features_config])

    # Timeframes
    if config.train_start:
        args.extend(['--train-start', config.train_start])
    if config.train_end:
        args.extend(['--train-end', config.train_end])
    if config.backtest_start:
        args.extend(['--backtest-start', config.backtest_start])
    if config.backtest_end:
        args.extend(['--backtest-end', config.backtest_end])

    # Label mode + the parameters that mode actually consumes.
    # advanced_train.py rejects unknown flags, so each block stays mode-scoped.
    mode = config.label_mode
    if mode != 'static':
        args.extend(['--label-mode', mode])

    if mode in ('atr_scaled', 'daily_vol_scaled'):
        # daily_vol_scaled reuses the ATR multipliers as volatility multipliers
        args.extend(['--atr-multiplier', str(config.atr_multiplier)])
        args.extend(['--atr-stop-multiplier', str(config.atr_stop_multiplier)])
    if mode == 'daily_vol_scaled':
        if config.daily_vol_span is not None:
            args.extend(['--daily-vol-span', str(config.daily_vol_span)])
        if config.slow_hysteresis_multiplier is not None:
            args.extend(['--slow-hysteresis-multiplier', str(config.slow_hysteresis_multiplier)])
    if mode == 'lookahead':
        for flag, value in [('--lookahead-horizon', config.lookahead_horizon),
                            ('--lookahead-pct', config.lookahead_pct),
                            ('--lookahead-lookback', config.lookahead_lookback),
                            ('--lookahead-min-pips', config.lookahead_min_pips),
                            ('--lookahead-stop-pips', config.lookahead_stop_pips)]:
            if value is not None:
                args.extend([flag, str(value)])
    if mode == 'direction_horizon':
        if config.direction_horizon is not None:
            args.extend(['--direction-horizon', str(config.direction_horizon)])
        if config.direction_dead_zone_pips is not None:
            args.extend(['--direction-dead-zone-pips', str(config.direction_dead_zone_pips)])
    if mode == 'window_cascade' and config.timing_entry is not None:
        args.extend(['--timing-entry', str(config.timing_entry)])
    if mode in ('regime_conditional', 'trend_only') and config.direction_aware_regime:
        args.append('--direction-aware-regime')
    if mode in ('regime_conditional', 'trend_only'):
        # The regime definition behind the labels. Only these two modes read
        # regime_labels, so the flags stay mode-scoped like every block above.
        if config.regime_label_source is not None:
            args.extend(['--regime-label-source', str(config.regime_label_source)])
        if config.regime_label_trend_threshold is not None:
            args.extend(['--regime-label-trend-threshold',
                         str(config.regime_label_trend_threshold)])

    # XGBoost hyperparameters
    args.extend(['--max-depth', str(config.max_depth)])
    args.extend(['--eta', str(config.eta)])
    args.extend(['--num-boost-round', str(config.num_boost_round)])
    args.extend(['--subsample', str(config.subsample)])
    args.extend(['--colsample-bytree', str(config.colsample_bytree)])
    if config.colsample_bynode is not None:
        args.extend(['--colsample-bynode', str(config.colsample_bynode)])
    args.extend(['--min-child-weight', str(config.min_child_weight)])

    # Label config overrides
    if config.pip_target is not None:
        args.extend(['--pip-target', str(config.pip_target)])
    if config.stop_pips is not None:
        args.extend(['--stop-pips', str(config.stop_pips)])
    if config.horizon_min is not None:
        args.extend(['--horizon-min', str(config.horizon_min)])
    if config.horizon_max is not None:
        args.extend(['--horizon-max', str(config.horizon_max)])

    # Decision-threshold rule
    if config.target_recall is not None:
        args.extend(['--target-recall', str(config.target_recall)])

    # Speed flags: skip SHAP and PFI for parallel runs. A diagnostics run wants them,
    # because SHAP and PFI are precisely the artefacts the feature chapter reports.
    if not diagnostics_full:
        args.extend(['--skip-shap', '--skip-pfi'])

    # Fast model specific params
    if config.fast_mfe_threshold is not None:
        args.extend(['--mfe-threshold', str(config.fast_mfe_threshold)])
    if config.fast_mfe_horizon is not None:
        args.extend(['--mfe-horizon', str(config.fast_mfe_horizon)])
    if config.fast_max_depth is not None:
        args.extend(['--fast-max-depth', str(config.fast_max_depth)])
    if config.fast_num_boost_round is not None:
        args.extend(['--fast-num-boost-round', str(config.fast_num_boost_round)])
    if config.fast_min_child_weight is not None:
        args.extend(['--fast-min-child-weight', str(config.fast_min_child_weight)])
    if config.fast_spw_factor != 1.0:
        args.extend(['--fast-spw-factor', str(config.fast_spw_factor)])
    if config.fast_lambda != 1.0:
        args.extend(['--fast-lambda', str(config.fast_lambda)])
    if config.fast_mi_threshold is not None:
        args.extend(['--fast-mi-threshold', str(config.fast_mi_threshold)])
    if config.fast_train_start is not None:
        args.extend(['--fast-train-start', str(config.fast_train_start)])
    if config.fast_use_slow_label:
        args.append('--fast-use-slow-label')
    if getattr(config, 'fast_conditional_on_setup', False):
        args.append('--fast-conditional-on-setup')

    # Slow model specific params
    args.extend(['--slow-spw-factor', str(config.slow_spw_factor)])
    args.extend(['--slow-mi-threshold', str(config.slow_mi_threshold)])
    args.extend(['--slow-max-depth', str(config.slow_max_depth)])
    args.extend(['--slow-num-boost-round', str(config.slow_num_boost_round)])
    args.extend(['--slow-min-child-weight', str(config.slow_min_child_weight)])
    args.extend(['--slow-lambda', str(config.slow_lambda)])

    # MI scoring mode. Left unset this inherits advanced_train's default of 20
    # permutations, which turns the noise-floor test into an active feature gate and
    # strips most slow features — a large, silent difference from a command that
    # passed --mi-permutations 0. Forward it so a config states its own MI regime.
    if getattr(config, 'mi_permutations', None) is not None:
        args.extend(['--mi-permutations', str(config.mi_permutations)])
    if getattr(config, 'legacy_mi', False):
        args.append('--legacy-mi')

    # Seed for reproducibility
    if config.seed is not None:
        args.extend(['--seed', str(config.seed)])

    # Regime filtering
    if config.regime_filter and config.regime_type:
        args.append('--regime-filter')
        args.extend(['--regime-type', config.regime_type])

    # Training-data sampling (default on; forward overrides + numeric params)
    if not config.training_sampling:
        args.append('--no-training-sampling')
    if not config.sampling_stride:
        args.append('--no-sampling-stride')
    if not config.sampling_context:
        args.append('--no-sampling-context')
    args.extend(['--sampling-stride-x', str(config.sampling_stride_x)])
    args.extend(['--sampling-hours-before', str(config.sampling_hours_before)])
    args.extend(['--sampling-hours-after', str(config.sampling_hours_after)])

    # Campaign-wide measurement settings last — see set_campaign_args.
    # `campaign_args` must be passed explicitly from the parent when this runs inside a
    # worker process: on Windows the pool starts with spawn, the child re-imports this
    # module, and the module-level list is back to empty. Falling back to the global
    # keeps serial callers and the unit tests working unchanged.
    campaign = CAMPAIGN_TRAIN_ARGS if campaign_args is None else list(campaign_args)
    args.extend(campaign)

    # W&B display name + family tag per child. A campaign child's on-disk run_id is a
    # timestamp+uuid, so without these the W&B run list is unreadable. --wandb-tags is
    # action='append' in the child, so this adds to (not overrides) the campaign tags.
    if '--wandb' in campaign:
        args.extend(['--wandb-run-name', config.name])
        args.extend(['--wandb-tags', f'base:{getattr(config, "base_name", None) or config.name}'])

    return args


def build_backtest_args(config, backtest_script=None, campaign_args=None,
                        report_dir=None):
    """
    Build the backtest.py command line for a TrainingConfig (single post-training backtest).

    Args:
        config (TrainingConfig): Training configuration.
        backtest_script (str, optional): Path to backtest.py. Defaults to the copy
            next to this file.
        report_dir (str, optional): Override backtest.py's report directory. Used by
            the backtest-variant path so two backtests of the same trained model do
            not overwrite each other's trade list.

    Returns:
        list[str]: Full argv list, starting with the Python executable.
    """
    if backtest_script is None:
        backtest_script = os.path.join(os.path.dirname(__file__), 'backtest.py')

    args = [sys.executable, backtest_script, '--run-id', config.effective_run_id]
    if report_dir is not None:
        args.extend(['--report-dir', report_dir])

    if config.features_config:
        args.extend(['--features-config', config.features_config])
    if config.backtest_start:
        args.extend(['--backtest-start', config.backtest_start])
    if config.backtest_end:
        args.extend(['--backtest-end', config.backtest_end])

    args.extend(['--p-open-fast', str(config.p_open_fast)])
    if getattr(config, 'use_trained_threshold', False):
        # The in-fold operating point supersedes p_open_slow; passing both would leave
        # it ambiguous which one applies.
        args.append('--use-trained-threshold')
    else:
        args.extend(['--p-open-slow', str(config.p_open_slow)])
    if config.p_close_pip_threshold is not None:
        args.extend(['--p-close-pip-threshold', str(config.p_close_pip_threshold)])
    if config.closing_after_x_pips:
        args.append('--closing-after-x-pips')
    if config.regime_gate is not None:
        args.extend(['--regime-gate', config.regime_gate])

    # Tri-state gates: None leaves backtest.py's own default in place, so existing
    # configs keep behaving exactly as before this was forwarded.
    if getattr(config, 'opening_requires_fast_signal', None) is True:
        args.append('--opening-requires-fast-signal')
    elif getattr(config, 'opening_requires_fast_signal', None) is False:
        args.append('--no-opening-requires-fast-signal')
    # backtest.py exposes only the negative flag here (the default is True).
    if getattr(config, 'closing_after_signal_reversal_slow', None) is False:
        args.append('--no-closing-after-signal-reversal-slow')

    # Campaign-wide measurement settings last, so they cannot be overridden by a
    # per-config value and every cell of a comparison is charged identically. See
    # build_train_args for why the parent has to hand these down explicitly.
    campaign = CAMPAIGN_BACKTEST_ARGS if campaign_args is None else list(campaign_args)
    args.extend(campaign)

    # Same naming rule as build_train_args; '__bt' keeps the training run and its
    # backtest adjacent in the W&B run list while job_type separates them.
    if '--wandb' in campaign:
        args.extend(['--wandb-run-name', f'{config.name}__bt'])
        args.extend(['--wandb-tags', f'base:{getattr(config, "base_name", None) or config.name}'])

    return args


def derive_variant_config(config, suffix, overrides):
    """
    Build the per-variant TrainingConfig for one backtest variant.

    Pure function (no subprocess, no filesystem) so the naming and override rules are
    unit-testable. The suffix lands in BOTH name and base_name: name keeps the per-run
    identity (fold + seed), base_name is what aggregate_walk_forward pools on — so each
    variant aggregates as its own configuration across all folds and seeds while
    sharing every trained model with its sibling variants.

    Args:
        config (TrainingConfig): the trained configuration (fold/seed already applied).
        suffix (str): variant tag, e.g. 'slowonly' or 'fastslow'.
        overrides (dict): config fields the variant changes, e.g.
            {'opening_requires_fast_signal': True}. Unknown fields raise — a typo here
            would otherwise silently backtest the wrong thing.

    Returns:
        TrainingConfig: a deep copy with the overrides applied and
        backtest_variants cleared (the variant is resolved, not recursive).
    """
    vcfg = copy.deepcopy(config)
    vcfg.backtest_variants = None
    vcfg.name = f"{config.name}__{suffix}"
    vcfg.base_name = f"{config.base_name or config.name}__{suffix}"
    for key, value in (overrides or {}).items():
        if not hasattr(vcfg, key):
            raise ValueError(
                f"backtest variant '{suffix}' overrides unknown config field '{key}'")
        setattr(vcfg, key, value)
    return vcfg


def _run_backtest_variants(config, base_result, env, campaign_bt,
                           keep_artifacts=False, diagnostics_dir=None):
    """
    Run one backtest per declared variant against the SAME trained model.

    This is what lets a study compare execution settings (e.g. slow-only entry vs
    fast+slow entry) without training every model twice: the training runs once, and
    each variant backtests it with its own flags into its own report directory
    (report_<suffix>/ inside the run directory), so the trade lists coexist.

    Args:
        config (TrainingConfig): the trained configuration, carrying backtest_variants.
        base_result (dict): the result dict of run_training_job after the successful
            training step (train_status/train_time filled in).
        env, campaign_bt: exactly as in run_training_job.
        keep_artifacts / diagnostics_dir: run-directory lifecycle, applied ONCE after
            all variants ran.

    Returns:
        list[dict]: one result per variant, each shaped exactly like a normal
        run_training_job result. The training wallclock is charged to the first
        variant only so summed timings stay truthful.
    """
    results = []
    training_metrics = extract_training_metrics(config.effective_run_id, dir_config.GENERATED_DIR)

    for vi, (suffix, overrides) in enumerate(config.backtest_variants):
        vcfg = derive_variant_config(config, suffix, overrides)
        report_dirname = f'report_{suffix}'
        report_dir = os.path.join(dir_config.GENERATED_DIR, config.effective_run_id,
                                  report_dirname)
        backtest_args = build_backtest_args(vcfg, campaign_args=campaign_bt,
                                            report_dir=report_dir)

        bt_start = time.time()
        process = subprocess.run(backtest_args, capture_output=True, text=True, env=env)
        elapsed = time.time() - bt_start

        vres = copy.deepcopy(base_result)
        vres['config'] = vcfg.to_dict()
        vres['config']['backtest_variant'] = suffix
        if vi > 0:
            vres['train_time'] = 0.0
        vres['backtest_time'] = elapsed

        stderr = process.stderr or ""
        has_error = (
            process.returncode != 0 or
            'Traceback' in stderr or
            'Error:' in stderr or
            'FileNotFoundError' in stderr
        )
        if has_error:
            vres['backtest_status'] = 'failed'
            vres['error'] = f"Backtest failed: {stderr[:1000]}"
            print(f"FAILED: Backtest FAILED for {vcfg.name}")
            print(f"  Return code: {process.returncode}")
            print(f"  Error: {stderr[:500]}")
            results.append(vres)
            continue

        vres['backtest_status'] = 'success'
        vres['status'] = 'success'
        vres['output_dir'] = os.path.join(dir_config.GENERATED_DIR, config.effective_run_id)

        metrics = extract_backtest_metrics(config.effective_run_id, dir_config.GENERATED_DIR,
                                           report_dirname=report_dirname)
        if metrics:
            vres['backtest_metrics'] = metrics
            print(f"OK: Backtest {vcfg.name} in {elapsed:.1f}s — "
                  f"PnL={metrics['total_pnl']:.2f}, Trades={metrics['total_trades']}, "
                  f"WR={metrics['win_rate']:.1%}")
        else:
            print(f"  WARNING: Could not extract metrics for {vcfg.name} "
                  f"(trade_list.csv may be missing)")
        if training_metrics:
            vres['training_metrics'] = training_metrics
        vres['trades'] = extract_trade_records(config.effective_run_id, dir_config.GENERATED_DIR,
                                               report_dirname=report_dirname)
        results.append(vres)

    # Diagnostics + cleanup once — the variants share one run directory.
    if diagnostics_dir:
        dest = save_run_diagnostics(config, dir_config.GENERATED_DIR, diagnostics_dir)
        for vres in results:
            vres['diagnostics_dir'] = dest
    if not keep_artifacts:
        cleanup_training_artifacts(config.effective_run_id, dir_config.GENERATED_DIR)

    return results


def run_training_job(config, enable_plotting=False, keep_artifacts=False,
                     campaign=None, diagnostics_dir=None, diagnostics_full=False):
    """
    Execute a single training + backtest job.

    Args:
        config (TrainingConfig): Training configuration
        enable_plotting (bool): Whether to enable plotting
        keep_artifacts (bool): Skip cleanup so a subsequent sweep phase can reuse the run dir
        campaign (tuple|None): (train_args, backtest_args) handed down from the parent.
            This CANNOT come from the module globals here — the pool spawns on Windows,
            so a worker re-imports the module and sees them empty. None = read the
            globals anyway, which is correct for a serial caller.
        diagnostics_dir (str|None): copy the per-run diagnostics here before cleanup
            deletes the run directory. None = the historical behaviour, nothing kept.
        diagnostics_full (bool): let the run compute SHAP and PFI.

    Returns:
        dict: Results summary with status, timing, and metrics — or, when the config
        declares backtest_variants, list[dict] with one such summary per variant
        (see _run_backtest_variants). run_parallel_training flattens the list.
    """
    start_time = time.time()
    result = {
        'config': config.to_dict(),
        'status': 'failed',
        'train_status': None,
        'backtest_status': None,
        'train_time': None,
        'backtest_time': None,
        'error': None,
        'output_dir': None
    }

    try:
        campaign_train, campaign_bt = campaign if campaign is not None else (None, None)
        train_args = build_train_args(config, enable_plotting=enable_plotting,
                                      campaign_args=campaign_train,
                                      diagnostics_full=diagnostics_full)

        print(f"\n{'='*80}")
        print(f"Starting training job: {config.name} (run_id: {config.run_id})")
        if config.train_start and config.train_end:
            print(f"  Training period: {config.train_start} to {config.train_end}")
        if config.backtest_start and config.backtest_end:
            print(f"  Backtest period: {config.backtest_start} to {config.backtest_end}")
        _coln = f", coln={config.colsample_bynode}" if config.colsample_bynode is not None else ""
        print(f"  XGBoost: depth={config.max_depth}, eta={config.eta}, rounds={config.num_boost_round}, sub={config.subsample}, col={config.colsample_bytree}{_coln}, mcw={config.min_child_weight}")
        if config.label_mode == 'atr_scaled':
            print(f"  Label mode: {config.label_mode} (ATR mult: {config.atr_multiplier}, stop mult: {config.atr_stop_multiplier})")
        elif config.label_mode != 'static':
            print(f"  Label mode: {config.label_mode}")
        if config.regime_filter:
            print(f"  Regime filter: {config.regime_type}")
        print(f"{'='*80}")

        # Run training with environment variable to suppress warnings
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning,ignore::FutureWarning'
        env['PYTHONIOENCODING'] = 'utf-8'

        train_start = time.time()
        train_process = subprocess.run(
            train_args,
            capture_output=True,
            text=True,
            env=env
        )
        train_elapsed = time.time() - train_start
        result['train_time'] = train_elapsed

        # More lenient error checking - only fail on actual errors
        if train_process.returncode != 0:
            stderr = train_process.stderr or ""
            stdout = train_process.stdout or ""

            # Check if training actually completed successfully despite non-zero return code
            success_indicators = [
                "OK: All models saved",
                "OK: All data saved",
                "OK: DATA SAVED",
                "All ONNX models saved",
                "MODELS SAVED & EXPORTED"
            ]

            training_succeeded = any(indicator in stdout for indicator in success_indicators)

            # Only treat as failure if there's a real error and no success indicators
            if not training_succeeded:
                # Check if there's a real Python error (traceback, exception, error)
                has_real_error = any(keyword in stderr for keyword in [
                    'Traceback (most recent call last)',
                    'Error:',
                    'Exception:',
                    'ImportError',
                    'ValueError',
                    'KeyError',
                    'FileNotFoundError',
                    'RuntimeError'
                ])

                # Check if it's just warnings (no real error indicators)
                is_only_warnings = (
                    ('UserWarning' in stderr or
                     'DeprecationWarning' in stderr or
                     'FutureWarning' in stderr) and
                    not has_real_error and
                    len(stderr) < 2000
                )

                if is_only_warnings:
                    # Likely just warnings, treat as success
                    print(f"WARNING: Training for {config.name} completed with warnings (ignored)")
                else:
                    result['train_status'] = 'failed'
                    result['error'] = f"Training failed: {stderr[:2000]}"
                    print(f"FAILED: Training FAILED for {config.name}")
                    print(f"  Return code: {train_process.returncode}")
                    print(f"  Stderr (last 3000 chars):\n{stderr[-3000:]}")
                    if stdout:
                        print(f"  Stdout (last 2000 chars):\n{stdout[-2000:]}")
                    return result
            else:
                print(f"WARNING: Training for {config.name} completed with warnings but succeeded")

        result['train_status'] = 'success'
        print(f"OK: Training completed for {config.name} in {train_elapsed:.1f}s")

        # Multiple backtests against this one trained model? Each variant becomes its
        # own result row; the single-backtest path below stays byte-identical.
        if getattr(config, 'backtest_variants', None):
            return _run_backtest_variants(config, result, env, campaign_bt,
                                          keep_artifacts=keep_artifacts,
                                          diagnostics_dir=diagnostics_dir)

        # Run backtest
        backtest_args = build_backtest_args(config, campaign_args=campaign_bt)

        backtest_start = time.time()
        backtest_process = subprocess.run(
            backtest_args,
            capture_output=True,
            text=True,
            env=env
        )
        backtest_elapsed = time.time() - backtest_start
        result['backtest_time'] = backtest_elapsed

        # Check for backtest errors (both return code and stderr content)
        backtest_stderr = backtest_process.stderr or ""
        backtest_stdout = backtest_process.stdout or ""

        has_backtest_error = (
            backtest_process.returncode != 0 or
            'Traceback' in backtest_stderr or
            'Error:' in backtest_stderr or
            'FileNotFoundError' in backtest_stderr
        )

        if has_backtest_error:
            result['backtest_status'] = 'failed'
            result['error'] = f"Backtest failed: {backtest_stderr[:1000]}"
            print(f"FAILED: Backtest FAILED for {config.name}")
            print(f"  Return code: {backtest_process.returncode}")
            print(f"  Error: {backtest_stderr[:500]}")
            if backtest_stdout:
                print(f"  Last stdout: {backtest_stdout[-500:]}")
            return result

        result['backtest_status'] = 'success'
        result['status'] = 'success'
        result['output_dir'] = os.path.join(dir_config.GENERATED_DIR, config.effective_run_id)

        total_elapsed = time.time() - start_time
        print(f"OK: Backtest completed for {config.name} in {backtest_elapsed:.1f}s")
        print(f"OK: Total time for {config.name}: {total_elapsed:.1f}s")

        # Extract metrics BEFORE cleanup
        metrics = extract_backtest_metrics(config.effective_run_id, dir_config.GENERATED_DIR)
        if metrics:
            result['backtest_metrics'] = metrics
            print(f"  Extracted metrics: PnL={metrics['total_pnl']:.2f}, Trades={metrics['total_trades']}, WinRate={metrics['win_rate']:.1%}")
        else:
            print(f"  WARNING: Could not extract metrics for {config.name} (trade_list.csv may be missing)")

        training_metrics = extract_training_metrics(config.effective_run_id, dir_config.GENERATED_DIR)
        if training_metrics:
            result['training_metrics'] = training_metrics
            print(f"  Extracted AUC: fast={training_metrics['auc_fast']:.4f}, slow={training_metrics['auc_slow']:.4f}, composite={training_metrics['auc_composite']:.4f}")
        else:
            print(f"  WARNING: Could not extract training metrics for {config.name} (training_summary.json may be missing)")

        # Individual trades, captured BEFORE cleanup deletes the run directory.
        # Walk-forward pools these across folds: a confidence interval needs the
        # per-trade spread, which no aggregate in backtest_metrics preserves.
        result['trades'] = extract_trade_records(config.effective_run_id, dir_config.GENERATED_DIR)

        # Rescue the per-fold diagnostics BEFORE the cleanup below deletes them. Without
        # this a campaign leaves no learning curve, no MI table, no selected-feature list
        # and no out-of-fold scores for any of its folds — only the aggregate snapshots
        # inside the summary JSON — so feature-selection stability and per-fold model
        # quality are not measurable over a walk-forward at all.
        if diagnostics_dir:
            result['diagnostics_dir'] = save_run_diagnostics(
                config, dir_config.GENERATED_DIR, diagnostics_dir)

        # Clean up training artifacts after successful backtest and metric extraction
        if not keep_artifacts:
            cleanup_training_artifacts(config.effective_run_id, dir_config.GENERATED_DIR)

    except Exception as e:
        result['error'] = str(e)
        print(f"FAILED: Exception in job {config.name}: {e}")

    return result


# What survives a campaign. Everything else in a run directory is either huge (the
# parquet design matrices), reproducible from these (the report PDF), or a deployment
# artefact with no analytical value (the ONNX models and scalers).
DIAGNOSTIC_FILES = (
    'training_summary.json',                              # CV gates, boost rounds, coverage
    'oof_predictions.parquet',                            # the model-quality figures
    os.path.join('training_output', 'visualization', 'learning_curves.json'),
    os.path.join('report', 'backtest_summary.json'),      # risk-adjusted + cost block
    os.path.join('report', 'trade_list.csv'),
)
# Per-model selection artefacts, resolved under feature_selection/{cadence}/{model}/.
DIAGNOSTIC_SELECTION_FILES = ('mi_scores.csv', 'pfi_scores.csv', 'selected_features.txt')
# SHAP writes one directory per model. Only the machine-readable tables are kept: the
# beeswarm and dependence PNGs are large, and the CSV is what a comparison of gain, PFI
# and SHAP rankings actually needs.
DIAGNOSTIC_SHAP_GLOBS = ('shap_importance_*.csv', 'shap_regime_comparison_*.csv')


def resolve_diagnostics_dir(path):
    """Anchor a relative --keep-diagnostics DIR at the repository root, not the CWD.

    The documented working directory is ModelTrading/source/python, so a
    repo-relative path like 'ModelTrading/generated/diag' would otherwise be
    created as a stray ModelTrading/generated tree inside the source folder
    (happened 2026-09-05 with --keep-diagnostics ModelTrading/generated/diag_handrun).
    Absolute paths are returned unchanged.
    """
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(dir_config.BASE_DIR)
    return os.path.join(repo_root, path)


def diagnostics_slot(config):
    """Directory name for one run's diagnostics: '<base_name>/wf03_s42'.

    Named from the walk-forward fold and the seed rather than the run_id, because the
    run_id is a timestamp+uuid that says nothing about which window or seed it was —
    and comparing folds is the entire point of keeping these.
    """
    base = getattr(config, 'base_name', None) or getattr(config, 'name', 'run')
    fold = getattr(config, 'walk_forward_fold', None)
    seed = getattr(config, 'seed', None)
    leaf = 'wf' + (format(int(fold), '02d') if fold is not None else '00')
    if seed is not None:
        leaf += '_s' + str(seed)
    return os.path.join(str(base), leaf)


def save_run_diagnostics(config, generated_dir, dest_root):
    """Copy the diagnostic whitelist out of a run directory before it is deleted.

    Returns the destination directory, or None when the run directory is already gone.
    Never raises: losing a diagnostic copy must not fail an otherwise good job.
    """
    import shutil

    run_dir = os.path.join(generated_dir, config.effective_run_id)
    if not os.path.exists(run_dir):
        return None
    dest = os.path.join(dest_root, diagnostics_slot(config))

    import glob as _glob

    copied = 0
    try:
        os.makedirs(dest, exist_ok=True)
        for rel in DIAGNOSTIC_FILES:
            src = os.path.join(run_dir, rel)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, os.path.basename(rel)))
                copied += 1
        # Backtest variants write report_<suffix>/ beside the default report/. Keep
        # each variant's trade list and summary under a suffixed name so all arms of
        # the comparison survive the cleanup, not just the last one written.
        for rep_dir in sorted(_glob.glob(os.path.join(run_dir, 'report_*'))):
            tag = os.path.basename(rep_dir)[len('report_'):]
            for fname in ('trade_list.csv', 'backtest_summary.json'):
                src = os.path.join(rep_dir, fname)
                if os.path.exists(src):
                    root, ext = os.path.splitext(fname)
                    shutil.copy2(src, os.path.join(dest, f'{root}_{tag}{ext}'))
                    copied += 1
        # feature_selection/{fast,slow}/{model}/*.csv — flattened to {model}_{file}
        fs_root = os.path.join(run_dir, 'feature_selection')
        for cadence in ('fast', 'slow'):
            cadence_dir = os.path.join(fs_root, cadence)
            if not os.path.isdir(cadence_dir):
                continue
            for model in sorted(os.listdir(cadence_dir)):
                for name in DIAGNOSTIC_SELECTION_FILES:
                    src = os.path.join(cadence_dir, model, name)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(dest, model + '_' + name))
                        copied += 1
        # shap_analysis/{model}/shap_importance_{model}.csv — only produced under
        # --diagnostics-full, so absent from a normal grid run.
        shap_root = os.path.join(run_dir, 'shap_analysis')
        if os.path.isdir(shap_root):
            for pattern in DIAGNOSTIC_SHAP_GLOBS:
                for src in sorted(_glob.glob(os.path.join(shap_root, '*', pattern))):
                    shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
                    copied += 1

        # The window this run actually covered — the diagnostics are meaningless
        # without it, and the run directory that carried it is about to disappear.
        with open(os.path.join(dest, 'run_context.json'), 'w', encoding='utf-8') as fh:
            json.dump({
                'name': getattr(config, 'name', None),
                'base_name': getattr(config, 'base_name', None),
                'seed': getattr(config, 'seed', None),
                'walk_forward_fold': getattr(config, 'walk_forward_fold', None),
                'run_id': config.effective_run_id,
                'train_start': getattr(config, 'train_start', None),
                'train_end': getattr(config, 'train_end', None),
                'backtest_start': getattr(config, 'backtest_start', None),
                'backtest_end': getattr(config, 'backtest_end', None),
            }, fh, indent=2)
        print(f"  Kept {copied} diagnostic file(s) -> {dest}")
        return dest
    except Exception as e:
        print(f"  Warning: could not keep diagnostics for {config.effective_run_id}: {e}")
        return None


def cleanup_training_artifacts(run_id, generated_dir):
    """
    Clean up training artifacts after successful backtest completion.
    Deletes the entire run directory to save disk space.

    Args:
        run_id (str): Run identifier
        generated_dir (str): Base generated directory
    """
    import shutil

    run_dir = os.path.join(generated_dir, run_id)
    if not os.path.exists(run_dir):
        return

    try:
        # Delete the entire run directory (including backtest reports)
        shutil.rmtree(run_dir)
        print(f"  Cleaned up all artifacts for {run_id} (deleted entire folder)")

    except Exception as e:
        print(f"  Warning: Could not clean up artifacts for {run_id}: {e}")


def run_parallel_training(configs, max_workers=4, enable_plotting=False,
                          keep_artifacts=False, diagnostics_dir=None,
                          diagnostics_full=False):
    """
    Run multiple training jobs in parallel.

    The campaign measurement settings are read HERE, in the parent, and handed to every
    job as an argument. They used to be read inside the worker from the module globals,
    which silently produced default measurement on any spawn-based platform (Windows):
    the parent logged "Campaign measurement settings — --cost-model data ..." while every
    child ran gross, unsized, un-embargoed and without the degenerate-fold guard.

    Args:
        configs (list[TrainingConfig]): List of training configurations
        max_workers (int): Maximum number of parallel jobs
        enable_plotting (bool): Whether to enable plotting
        keep_artifacts (bool): Skip the per-run cleanup entirely
        diagnostics_dir (str|None): keep a whitelist of per-run diagnostics here
        diagnostics_full (bool): let each run compute SHAP and PFI

    Returns:
        list[dict]: Results from all jobs
    """
    campaign = (list(CAMPAIGN_TRAIN_ARGS), list(CAMPAIGN_BACKTEST_ARGS))
    print(f"\n{'='*80}")
    print(f"PARALLEL TRAINING ORCHESTRATOR")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Parallel workers: {max_workers}")
    print(f"Plotting enabled: {enable_plotting}")
    print(f"{'='*80}\n")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_training_job, config, enable_plotting, keep_artifacts,
                            campaign, diagnostics_dir, diagnostics_full): config
            for config in configs
        }

        # Collect results as they complete. A job with backtest variants returns a
        # LIST of results (one per variant); flatten so every downstream consumer
        # keeps seeing one flat list of result dicts.
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except Exception as e:
                print(f"FAILED: Job {config.name} generated an exception: {e}")
                results.append({
                    'config': config.to_dict(),
                    'status': 'exception',
                    'error': str(e)
                })

    return results


def extract_backtest_metrics(run_id, generated_dir, report_dirname="report"):
    """
    Extract key metrics from backtest results.

    Args:
        run_id (str): Run identifier
        generated_dir (str): Base generated directory
        report_dirname (str): Report directory inside the run directory. Backtest
            variants write to 'report_<suffix>' instead of the default 'report'.

    Returns:
        dict: Backtest metrics or None if unavailable
    """
    trade_list_path = os.path.join(generated_dir, run_id, report_dirname, "trade_list.csv")

    if not os.path.exists(trade_list_path):
        return None

    try:
        df = pd.read_csv(trade_list_path)
        if len(df) == 0:
            return {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl_per_trade': 0,
                    'avg_win_pnl_per_trade': 0, 'avg_loss_pnl_per_trade': 0,
                    'max_pnl': 0, 'min_pnl': 0, 'total_pips': 0}

        total_trades = len(df)
        total_pnl = df['pnl'].sum()
        winning_trades = (df['pnl'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        win_df = df[df['pnl'] > 0]
        loss_df = df[df['pnl'] <= 0]

        basic = {
            'total_trades': int(total_trades),
            'total_pnl': float(total_pnl),
            'win_rate': float(win_rate),
            'avg_pnl_per_trade': float(df['pnl'].mean()),
            'avg_win_pnl_per_trade': float(win_df['pnl'].mean()) if len(win_df) > 0 else 0.0,
            'avg_loss_pnl_per_trade': float(loss_df['pnl'].mean()) if len(loss_df) > 0 else 0.0,
            'max_pnl': float(df['pnl'].max()),
            'min_pnl': float(df['pnl'].min()),
            'total_pips': float(df['pnl_pips'].sum()),
        }
        extended = compute_extended_backtest_metrics(df)
        return {**basic, **extended}
    except Exception as e:
        print(f"  ERROR extracting metrics for {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_trade_records(run_id, generated_dir, report_dirname="report"):
    """
    Read the individual trades of one run as a compact list of dicts.

    Aggregates cannot be pooled into a confidence interval — that needs the
    per-trade values. Keep the payload small: it crosses a process boundary and is
    held for every fold of every configuration.

    Returns:
        list[dict] with pnl, pnl_pips, open_time, action, exit_reason; [] if absent.
    """
    path = os.path.join(generated_dir, run_id, report_dirname, "trade_list.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return []
        cols = [c for c in ('pnl', 'pnl_pips', 'open_time', 'action', 'exit_reason')
                if c in df.columns]
        return df[cols].to_dict('records')
    except Exception as e:
        print(f"  WARNING: could not read trades for {run_id}: {e}")
        return []


def extract_training_metrics(run_id, generated_dir):
    """
    Extract model quality metrics from training_summary.json.

    Reads the AUC-ROC values for the fast and slow models (post feature-selection)
    that advanced_train.py saves before the run directory is cleaned up.

    Args:
        run_id (str): Run identifier
        generated_dir (str): Base generated directory

    Returns:
        dict: Training metrics with auc_fast, auc_slow, auc_composite, supporting CV
              F1 scores, label base rates, positive prediction rates and calibration
              error — or None if unavailable.
    """
    summary_path = os.path.join(generated_dir, run_id, "training_summary.json")

    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path) as f:
            summary = json.load(f)

        cv = summary.get('cv_metrics', {})

        def _safe(key):
            val = cv.get(key)
            return float(val) if val is not None else float('nan')

        auc_fast = _safe('fast_post_global_val_auc_roc')
        auc_slow = _safe('slow_post_global_val_auc_roc')

        # Composite: average of fast and slow AUC, ignoring NaN
        valid = [v for v in [auc_fast, auc_slow] if not (v != v)]  # NaN-safe
        auc_composite = float(sum(valid) / len(valid)) if valid else float('nan')

        return {
            'auc_fast': auc_fast,
            'auc_slow': auc_slow,
            'auc_composite': auc_composite,
            'f1_fast_post': summary.get('cv_global_val_f1_fast_post'),
            'f1_slow_post': summary.get('cv_global_val_f1_slow_post'),
            'precision_fast_post': _safe('fast_post_global_val_precision'),
            'recall_fast_post': _safe('fast_post_global_val_recall'),
            'precision_slow_post': _safe('slow_post_global_val_precision'),
            'recall_slow_post': _safe('slow_post_global_val_recall'),
            # Collapse check: share of validation bars predicted positive
            'pos_pred_rate_fast': _positive_prediction_rate(
                cv.get('fast_post_global_val_confusion_matrix')),
            'pos_pred_rate_slow': _positive_prediction_rate(
                cv.get('slow_post_global_val_confusion_matrix')),
            # Calibration error (probabilities feed fixed entry thresholds, so this matters)
            'ece_fast_post': _safe('fast_post_global_val_ece'),
            'ece_slow_post': _safe('slow_post_global_val_ece'),
            # Class distribution of the labels themselves — comparable across label modes
            'label_stats': summary.get('label_stats'),
            # CV trustworthiness: folds with no positive labels yield no AUC, so a pooled
            # AUC built mostly on such folds must not be compared against a healthy one.
            'single_class_folds_fast': cv.get('fast_post_n_single_class_val_folds'),
            'single_class_folds_slow': cv.get('slow_post_n_single_class_val_folds'),
            'n_cv_folds': cv.get('slow_post_n_cv_folds'),
            # Models that actually got a calibrator (skipped when the tail is single-class)
            'n_calibrated_models': len(summary.get('calibration_params') or {}),
            'n_features_fast': summary.get('n_features_fast'),
            'n_features_slow': summary.get('n_features_slow'),
            'n_training_samples': summary.get('n_training_samples'),
        }
    except Exception as e:
        print(f"  ERROR extracting training metrics for {run_id}: {e}")
        return None


def _is_nan(val):
    return val != val  # NaN != NaN is True


def _positive_prediction_rate(confusion_matrix):
    """
    Share of validation samples predicted positive, from a pooled CV confusion matrix.

    Detects majority-class collapse: a rate of 0.0 means the model never fires.

    Args:
        confusion_matrix: [[tn, fp], [fn, tp]] as written by _serialize_cv_metrics,
            or None when the metric is unavailable.

    Returns:
        float: predicted-positive rate, or NaN if the matrix is missing/degenerate.
    """
    try:
        (tn, fp), (fn, tp) = confusion_matrix
    except (TypeError, ValueError):
        return float('nan')

    total = tn + fp + fn + tp
    if total <= 0:
        return float('nan')
    return float(fp + tp) / float(total)


def compute_extended_backtest_metrics(df):
    """
    Compute risk-adjusted backtest metrics from a trade list DataFrame.

    Requires columns: pnl_pips, action (BUY/SELL), open_time.
    Optional column: duration_bars.

    Returns a dict with: profit_factor, max_drawdown_pips, sharpe_ratio,
    sortino_ratio, recovery_factor, calmar_ratio, long_win_rate, short_win_rate,
    avg_pips_per_trade, avg_trade_duration_bars.
    """
    null_result = {
        'profit_factor': None, 'max_drawdown_pips': None, 'sharpe_ratio': None,
        'sortino_ratio': None, 'recovery_factor': None, 'calmar_ratio': None,
        'long_win_rate': None, 'short_win_rate': None, 'avg_pips_per_trade': None,
        'avg_trade_duration_bars': None,
    }

    if df is None or len(df) == 0 or 'pnl_pips' not in df.columns:
        return null_result

    try:
        pips = df['pnl_pips'].astype(float)

        # Profit factor
        wins_sum = float(pips[pips > 0].sum())
        loss_sum = float(abs(pips[pips < 0].sum()))
        if loss_sum > 0:
            profit_factor = wins_sum / loss_sum
        elif wins_sum > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0

        # Max drawdown (peak-to-trough on cumulative pip curve, starting from 0)
        cum = pd.concat([pd.Series([0.0]), pips.cumsum()]).reset_index(drop=True)
        drawdowns = cum.cummax() - cum
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        total_pips = float(pips.sum())
        recovery_factor = (total_pips / max_dd) if max_dd > 0 else (float('inf') if total_pips > 0 else 0.0)

        avg_pips_per_trade = float(pips.mean())

        # Daily aggregation for Sharpe / Sortino / Calmar
        sharpe = sortino = calmar = None
        if 'open_time' in df.columns:
            try:
                df2 = df[['open_time', 'pnl_pips']].copy()
                df2['open_time'] = pd.to_datetime(df2['open_time'])
                df2['date'] = df2['open_time'].dt.date
                daily = df2.groupby('date')['pnl_pips'].sum()
                if len(daily) > 1:
                    mean_d = float(daily.mean())
                    std_d = float(daily.std())
                    if std_d > 0:
                        sharpe = mean_d / std_d * (252 ** 0.5)
                    down = daily[daily < 0]
                    if len(down) > 0:
                        down_std = float(down.std())
                        if down_std > 0:
                            sortino = mean_d / down_std * (252 ** 0.5)
                    ann_pips = mean_d * 252
                    if max_dd > 0:
                        calmar = ann_pips / max_dd
            except Exception:
                pass

        # Direction win rates
        long_wr = short_wr = None
        if 'action' in df.columns:
            longs = df[df['action'] == 'BUY']
            shorts = df[df['action'] == 'SELL']
            if len(longs) > 0:
                long_wr = float((longs['pnl_pips'] > 0).sum() / len(longs))
            if len(shorts) > 0:
                short_wr = float((shorts['pnl_pips'] > 0).sum() / len(shorts))

        avg_dur = None
        if 'duration_bars' in df.columns:
            avg_dur = float(df['duration_bars'].mean())

        return {
            'profit_factor': round(profit_factor, 4) if not _is_nan(profit_factor) else None,
            'max_drawdown_pips': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 4) if sharpe is not None else None,
            'sortino_ratio': round(sortino, 4) if sortino is not None else None,
            'recovery_factor': round(recovery_factor, 4) if not _is_nan(recovery_factor) else None,
            'calmar_ratio': round(calmar, 4) if calmar is not None else None,
            'long_win_rate': round(long_wr, 4) if long_wr is not None else None,
            'short_win_rate': round(short_wr, 4) if short_wr is not None else None,
            'avg_pips_per_trade': round(avg_pips_per_trade, 4),
            'avg_trade_duration_bars': round(avg_dur, 2) if avg_dur is not None else None,
        }
    except Exception as e:
        print(f"  WARNING: compute_extended_backtest_metrics failed: {e}")
        return null_result


def save_scenario_manifest(results, scenario_name, scenario_description, scenario_dir,
                            command_line):
    """
    Save scenario_manifest.json with raw per-run metrics for later comparison.

    The manifest stores all_runs with their individual metrics. The Top-K
    aggregation (mean ± std) is intentionally deferred to compare_scenarios.py
    so the user can choose K freely at comparison time without re-running training.

    Also writes best_run_metrics.json (the #1 run by composite AUC) for quick reference.

    Args:
        results: list of run result dicts from run_parallel_training
        scenario_name: human-readable scenario identifier
        scenario_description: free-text architecture description
        scenario_dir: output folder (created if missing)
        command_line: full command string for reproducibility
    """
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    successful = [
        r for r in results
        if r['status'] == 'success' and r.get('training_metrics')
    ]
    ranked = sorted(
        [r for r in successful if not _is_nan(r['training_metrics']['auc_composite'])],
        key=lambda x: x['training_metrics']['auc_composite'],
        reverse=True,
    )

    manifest = {
        'scenario_name': scenario_name,
        'description': scenario_description or '',
        'timestamp': datetime.now().isoformat(),
        'git_hash': git_hash,
        'command_line': command_line,
        'total_runs': len(results),
        'successful_runs': len(successful),
        'best_run_id': ranked[0]['config']['run_id'] if ranked else None,
        'all_runs': [
            {
                'run_id': r['config']['run_id'],
                'name': r['config']['name'],
                'status': r['status'],
                'training_metrics': r.get('training_metrics'),
                'backtest_metrics': r.get('backtest_metrics'),
            }
            for r in results
        ],
    }

    os.makedirs(scenario_dir, exist_ok=True)
    manifest_path = os.path.join(scenario_dir, 'scenario_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nScenario manifest saved: {manifest_path}")

    if ranked:
        best = ranked[0]
        best_path = os.path.join(scenario_dir, 'best_run_metrics.json')
        with open(best_path, 'w') as f:
            json.dump({
                'run_id': best['config']['run_id'],
                'training_metrics': best.get('training_metrics'),
                'backtest_metrics': best.get('backtest_metrics'),
            }, f, indent=2, default=str)

    return manifest_path


def save_summary_report(results, output_path):
    """
    Save a summary report of all training runs with backtest metrics.

    Args:
        results (list[dict]): Results from all jobs
        output_path (str): Path to save the summary
    """
    # Metrics already extracted in run_training_job before cleanup

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_jobs': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] != 'success'),
        'results': results
    }

    # Rank by composite AUC (avg of fast + slow post-selection AUC-ROC)
    successful_with_metrics = [
        r for r in results
        if r['status'] == 'success' and r.get('training_metrics')
        and not (r['training_metrics']['auc_composite'] != r['training_metrics']['auc_composite'])  # exclude NaN
    ]

    if successful_with_metrics:
        ranked = sorted(
            successful_with_metrics,
            key=lambda x: x['training_metrics']['auc_composite'],
            reverse=True
        )
        summary['top_performers'] = [
            {
                'rank': i + 1,
                'name': r['config']['name'],
                'run_id': r['config']['run_id'],
                'auc_composite': r['training_metrics']['auc_composite'],
                'auc_fast': r['training_metrics']['auc_fast'],
                'auc_slow': r['training_metrics']['auc_slow'],
                'f1_fast_post': r['training_metrics'].get('f1_fast_post'),
                'f1_slow_post': r['training_metrics'].get('f1_slow_post'),
                'precision_fast_post': r['training_metrics'].get('precision_fast_post'),
                'recall_fast_post': r['training_metrics'].get('recall_fast_post'),
                'total_pnl': r['backtest_metrics']['total_pnl'] if r.get('backtest_metrics') else None,
                'total_pips': r['backtest_metrics']['total_pips'] if r.get('backtest_metrics') else None,
                'win_rate': r['backtest_metrics']['win_rate'] if r.get('backtest_metrics') else None,
                'total_trades': r['backtest_metrics']['total_trades'] if r.get('backtest_metrics') else None,
                'train_period': f"{r['config'].get('train_start', 'default')} to {r['config'].get('train_end', 'default')}",
                'backtest_period': f"{r['config'].get('backtest_start', 'default')} to {r['config'].get('backtest_end', 'default')}"
            }
            for i, r in enumerate(ranked[:10])  # Top 10
        ]

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary report saved to: {output_path}")


def _aggregate_seed_results(results):
    """
    Group successful results by base_name and average metrics across seeds.

    Returns list of dicts sorted by avg total_pnl descending.
    Each dict has: base_name, n_seeds, avg/std for pnl, trades, win_rate, pips, auc.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        if r['status'] != 'success':
            continue
        base = r['config'].get('base_name') or r['config']['name']
        groups[base].append(r)

    # Only return groups that actually have multiple seeds
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        return []

    aggregated = []
    for base_name, runs in multi.items():
        bm_list = [r['backtest_metrics'] for r in runs if r.get('backtest_metrics')]
        tm_list = [r['training_metrics'] for r in runs if r.get('training_metrics')]

        if not bm_list:
            continue

        pnls            = [m['total_pnl']              for m in bm_list]
        trades          = [m['total_trades']            for m in bm_list]
        win_rates       = [m['win_rate']                for m in bm_list]
        pips            = [m['total_pips']              for m in bm_list]
        avg_pnl_pt      = [m['avg_pnl_per_trade']      for m in bm_list]
        avg_win_pnl_pt  = [m.get('avg_win_pnl_per_trade', 0)  for m in bm_list]
        avg_loss_pnl_pt = [m.get('avg_loss_pnl_per_trade', 0) for m in bm_list]
        aucs            = [m['auc_composite']           for m in tm_list
                           if m.get('auc_composite') == m.get('auc_composite')]  # NaN-safe

        def _mean(lst): return statistics.mean(lst) if lst else 0.0
        def _std(lst):  return statistics.stdev(lst) if len(lst) > 1 else 0.0

        aggregated.append({
            'base_name':              base_name,
            # n_seeds is the number of averaged RUNS, kept under its historical name.
            # Under --walk-forward a config's runs are windows x seeds, so the two
            # counts below are what actually describe the group.
            'n_seeds':                len(bm_list),
            'n_distinct_seeds':       len({r['config'].get('seed') for r in runs}),
            'n_windows':              len({r['config'].get('backtest_start') for r in runs}),
            'avg_pnl':                _mean(pnls),
            'std_pnl':                _std(pnls),
            'avg_trades':             _mean(trades),
            'avg_win_rate':           _mean(win_rates),
            'std_win_rate':           _std(win_rates),
            'avg_pips':               _mean(pips),
            'avg_pnl_per_trade':      _mean(avg_pnl_pt),
            'avg_win_pnl_per_trade':  _mean(avg_win_pnl_pt),
            'avg_loss_pnl_per_trade': _mean(avg_loss_pnl_pt),
            'avg_auc':                _mean(aucs) if aucs else None,
            'std_auc':                _std(aucs)  if aucs else None,
        })

    aggregated.sort(key=lambda x: x['avg_pnl'], reverse=True)
    return aggregated


def print_seed_averaged_summary(results, top_n=10):
    """Print leaderboard of averaged backtest results across seeds, grouped by base config."""
    aggregated = _aggregate_seed_results(results)
    if not aggregated:
        return

    head = aggregated[0]
    # Under --walk-forward a group is windows x seeds, and every row here is an average
    # over RUNS. Calling that "N seeds" reads as N independent repetitions of the same
    # backtest, which it is not: the spread mixes the seed draw with the window, and the
    # PnL is one window's worth, not the campaign total. Say what is being averaged.
    if head['n_windows'] > 1:
        label = (f"RUN-AVERAGED RESULTS  ({head['n_seeds']} runs per config = "
                 f"{head['n_windows']} windows x {head['n_distinct_seeds']} seeds; "
                 f"figures are PER RUN, see the walk-forward tables for the totals)")
    else:
        label = f"SEED-AVERAGED RESULTS  ({head['n_distinct_seeds']} seeds per config)"

    print(f"\n{'='*80}")
    print(label)
    print(f"{'='*80}")
    print(f"  {'#':<3} {'Config':<30} {'Avg PnL':>12} {'±':>1} {'StdDev':>10}  "
          f"{'Trades':>6}  {'Avg WR':>7}  {'Avg Pips':>9}  {'AUC':>6}")
    print(f"  {'-'*3} {'-'*30} {'-'*12} {'-'*1} {'-'*10}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*6}")

    for i, a in enumerate(aggregated[:top_n], 1):
        auc_str = f"{a['avg_auc']:.4f}" if a['avg_auc'] is not None else "   N/A"
        print(f"  {i:<3} {a['base_name']:<30} "
              f"EUR {a['avg_pnl']:>8,.0f} ± {a['std_pnl']:>8,.0f}  "
              f"{a['avg_trades']:>6.0f}  "
              f"{a['avg_win_rate']:>6.1%}  "
              f"{a['avg_pips']:>9.0f}  "
              f"{auc_str:>6}")

    print(f"{'='*80}")


def print_final_summary(results, top_n=5):
    """
    Print a human-readable summary of all results.

    Order:
      1. Job counts
      2. All successful runs (with top-performer tags)
      3. Failed runs
      4. Top-N leaderboards: AUC, P&L, Win Rate, Avg PnL/Trade
    """
    MIN_TRADES_FOR_RATE = 5  # Minimum trades to qualify for win-rate / avg-PnL rankings

    def _fmt_auc(val):
        return f"{val:.4f}" if val is not None and val == val else "N/A"  # NaN-safe

    successful = [r for r in results if r['status'] == 'success']
    failed     = [r for r in results if r['status'] != 'success']

    # ------------------------------------------------------------------
    # Pre-compute top-N sets so we can tag each run in the listing below
    # ------------------------------------------------------------------
    def _run_id(r):
        # BacktestSweepConfig has no run_id; fall back to name (always unique per sweep)
        return r['config'].get('run_id') or r['config']['name']

    # By composite AUC
    ranked_auc = sorted(
        [r for r in successful if r.get('training_metrics')
         and not (r['training_metrics']['auc_composite'] != r['training_metrics']['auc_composite'])],
        key=lambda r: r['training_metrics']['auc_composite'], reverse=True
    )
    top_auc_ids = {_run_id(r) for r in ranked_auc[:top_n]}

    # By total P&L
    ranked_pnl = sorted(
        [r for r in successful if r.get('backtest_metrics')],
        key=lambda r: r['backtest_metrics']['total_pnl'], reverse=True
    )
    top_pnl_ids = {_run_id(r) for r in ranked_pnl[:top_n]}

    # By win rate (min trades filter)
    ranked_wr = sorted(
        [r for r in successful if r.get('backtest_metrics')
         and r['backtest_metrics']['total_trades'] >= MIN_TRADES_FOR_RATE],
        key=lambda r: r['backtest_metrics']['win_rate'], reverse=True
    )
    top_wr_ids = {_run_id(r) for r in ranked_wr[:top_n]}

    # By avg PnL per trade (min trades filter)
    ranked_avg = sorted(
        [r for r in successful if r.get('backtest_metrics')
         and r['backtest_metrics']['total_trades'] >= MIN_TRADES_FOR_RATE],
        key=lambda r: r['backtest_metrics']['avg_pnl_per_trade'], reverse=True
    )
    top_avg_ids = {_run_id(r) for r in ranked_avg[:top_n]}

    # ------------------------------------------------------------------
    # 1. Job summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total jobs:  {len(results)}")
    print(f"Successful:  {len(successful)}")
    print(f"Failed:      {len(failed)}")

    # ------------------------------------------------------------------
    # 2. All successful runs with top-performer tags
    # ------------------------------------------------------------------
    if successful:
        print(f"\n{'='*80}")
        print("ALL SUCCESSFUL RUNS:")
        print(f"{'='*80}")
        for r in successful:
            run_id       = _run_id(r)
            config_name  = r['config']['name']
            train_time   = r.get('train_time', 0)
            backtest_time = r.get('backtest_time', 0)
            tm = r.get('training_metrics')
            bm = r.get('backtest_metrics')

            # Collect which leaderboards this run appears in
            tags = []
            if run_id in top_auc_ids:
                rank = next(i+1 for i, x in enumerate(ranked_auc) if _run_id(x) == run_id)
                tags.append(f"#{rank} AUC")
            if run_id in top_pnl_ids:
                rank = next(i+1 for i, x in enumerate(ranked_pnl) if _run_id(x) == run_id)
                tags.append(f"#{rank} P&L")
            if run_id in top_wr_ids:
                rank = next(i+1 for i, x in enumerate(ranked_wr) if _run_id(x) == run_id)
                tags.append(f"#{rank} WinRate")
            if run_id in top_avg_ids:
                rank = next(i+1 for i, x in enumerate(ranked_avg) if _run_id(x) == run_id)
                tags.append(f"#{rank} AvgPnL/Trade")

            cfg = r['config']
            has_training = bool(cfg.get('train_start'))
            # For training runs show run_id; for backtest-sweep runs it's redundant
            header = f"{config_name} ({run_id})" if has_training else config_name
            print(f"\n  {header}")
            if has_training:
                print(f"    Training time:  {train_time:.1f}s | Backtest time: {backtest_time:.1f}s | Total: {train_time+backtest_time:.1f}s")
            else:
                print(f"    Backtest time: {backtest_time:.1f}s")
            if tm:
                pf = tm.get('precision_fast_post', float('nan'))
                rf = tm.get('recall_fast_post', float('nan'))
                f1f = tm.get('f1_fast_post') or float('nan')
                pf_s = f"{pf:.3f}" if pf == pf else "N/A"
                rf_s = f"{rf:.3f}" if rf == rf else "N/A"
                f1f_s = f"{f1f:.3f}" if f1f == f1f else "N/A"
                print(f"    AUC: composite={_fmt_auc(tm['auc_composite'])} fast={_fmt_auc(tm['auc_fast'])} slow={_fmt_auc(tm['auc_slow'])}")
                print(f"    Fast: F1={f1f_s} P={pf_s} R={rf_s}  |  n_feat_fast={tm.get('n_features_fast')}")
                label_stats = tm.get('label_stats') or {}
                if label_stats:
                    rates = " ".join(
                        f"{name}={stats['rate']:.1%}"
                        for name, stats in label_stats.items() if stats.get('rate') is not None
                    )
                    print(f"    Label rates: {rates}")
                scf, scs = tm.get('single_class_folds_fast'), tm.get('single_class_folds_slow')
                if scf or scs:
                    print(f"    WARNING: single-class CV folds — fast {scf}/{tm.get('n_cv_folds')}, "
                          f"slow {scs}/{tm.get('n_cv_folds')} — AUC above is unreliable")
                ppr_f = tm.get('pos_pred_rate_fast', float('nan'))
                ppr_s = tm.get('pos_pred_rate_slow', float('nan'))
                if ppr_f == ppr_f or ppr_s == ppr_s:  # NaN-safe
                    ppr_f_s = f"{ppr_f:.1%}" if ppr_f == ppr_f else "N/A"
                    ppr_s_s = f"{ppr_s:.1%}" if ppr_s == ppr_s else "N/A"
                    print(f"    Positive prediction rate: fast={ppr_f_s} slow={ppr_s_s}")
            if bm:
                print(f"    PnL: EUR {bm['total_pnl']:,.2f} | Trades: {bm['total_trades']} | Win Rate: {bm['win_rate']:.1%}")
            if tags:
                print(f"    *** Top {top_n}: {', '.join(tags)}")

    # ------------------------------------------------------------------
    # 3. Failed runs
    # ------------------------------------------------------------------
    if failed:
        print(f"\n{'='*80}")
        print("FAILED RUNS:")
        print(f"{'='*80}")
        for r in failed:
            print(f"\n  {r['config']['name']}")
            print(f"    Error: {r.get('error', 'Unknown error')[:200]}")

    # ------------------------------------------------------------------
    # 4. Top-N leaderboards
    # ------------------------------------------------------------------
    def _print_leaderboard(title, ranked_list):
        if not ranked_list:
            return
        print(f"\n{'='*80}")
        print(f"TOP {top_n} — {title}")
        print(f"{'='*80}")
        for i, r in enumerate(ranked_list[:top_n], 1):
            tm = r.get('training_metrics')
            bm = r.get('backtest_metrics')
            cfg = r['config']
            print(f"\n  #{i}: {cfg['name']}")
            if tm:
                print(f"      AUC: composite={_fmt_auc(tm['auc_composite'])} "
                      f"fast={_fmt_auc(tm['auc_fast'])} slow={_fmt_auc(tm['auc_slow'])}")
            if bm:
                print(f"      PnL: EUR {bm['total_pnl']:,.2f} ({bm['total_pips']:.1f} pips) | "
                      f"Trades: {bm['total_trades']} | WR: {bm['win_rate']:.1%} | "
                      f"Avg: EUR {bm['avg_pnl_per_trade']:,.2f}/trade")
            if cfg.get('train_start'):
                print(f"      Train:    {cfg['train_start']} to {cfg['train_end']}")
            if cfg.get('backtest_start'):
                print(f"      Backtest: {cfg['backtest_start']} to {cfg['backtest_end']}")

    _print_leaderboard("Composite AUC-ROC (model quality)", ranked_auc)
    _print_leaderboard("Total P&L", ranked_pnl)
    _print_leaderboard(f"Win Rate (min {MIN_TRADES_FOR_RATE} trades)", ranked_wr)
    _print_leaderboard(f"Avg PnL per Trade (min {MIN_TRADES_FOR_RATE} trades)", ranked_avg)

    print_seed_averaged_summary(results, top_n=top_n)

    print(f"\n{'='*80}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_parallel_backtest_sweep(configs, max_workers=8):
    """
    Run multiple backtest-only jobs in parallel.

    Higher parallelism than training is safe here (no GPU/RAM contention from XGBoost).
    Each job uses a unique temp report dir so there are no write conflicts.

    Args:
        configs (list[BacktestSweepConfig]): Sweep configurations.
        max_workers (int): Maximum parallel jobs (default 8).

    Returns:
        list[dict]: Results from all jobs.
    """
    print(f"\n{'='*80}")
    print(f"BACKTEST PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Parallel workers: {max_workers}")
    print(f"{'='*80}\n")

    # Read in the PARENT and hand down as an argument — a spawned worker (Windows)
    # re-imports this module with the global back to empty. Same defect class as the
    # measurement settings; see run_parallel_training.
    wandb_args = list(CAMPAIGN_WANDB_ARGS)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {
            executor.submit(run_backtest_only_job, config, False, wandb_args): config
            for config in configs
        }
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"FAILED: Job {config.name} raised exception: {e}")
                results.append({
                    'config': config.to_dict(),
                    'status': 'exception',
                    'error': str(e),
                })
    return results


def _save_scenario_manifest_if_named(args, results):
    """Write the scenario manifest when --scenario-name was given (no-op otherwise)."""
    scenario_name = getattr(args, 'scenario_name', None)
    if not scenario_name:
        return
    save_scenario_manifest(
        results=results,
        scenario_name=scenario_name,
        scenario_description=getattr(args, 'scenario_description', None),
        scenario_dir=dir_config.get_scenario_dir(scenario_name),
        command_line=' '.join(sys.argv),
    )


def build_sweep_configs_for_runs(successful_runs, sweep_templates):
    """
    Expand sweep templates across every successfully trained model.

    Each sweep job must point at the directory the model was actually written to:
    that is the *effective* run id, which under --scenario-name is
    scenarios/<name>/<run_id> rather than the bare run_id.

    Args:
        successful_runs (list[dict]): Results from run_parallel_training with status success.
        sweep_templates (list[BacktestSweepConfig]): Templates to apply to each model.

    Returns:
        list[BacktestSweepConfig]: One config per (model, template) pair.
    """
    sweep_configs = []
    for run in successful_runs:
        cfg = run['config']
        run_id = cfg.get('effective_run_id') or cfg['run_id']
        model_name = cfg['name']
        bt_start = cfg.get('backtest_start')
        bt_end = cfg.get('backtest_end')
        for template in sweep_templates:
            sweep_config = copy.copy(template)
            sweep_config.model_run_id = run_id
            sweep_config.name = f"{model_name}__{template.name}"
            if bt_start:
                sweep_config.backtest_start = bt_start
            if bt_end:
                sweep_config.backtest_end = bt_end
            sweep_configs.append(sweep_config)
    return sweep_configs


def _select_sweep_configs(sweep_set):
    """Return the backtest-sweep templates for the given --sweep-set value."""
    if sweep_set == 'label-study':
        return create_label_study_sweep_configs(full=False)
    if sweep_set == 'label-study-full':
        return create_label_study_sweep_configs(full=True)
    return create_backtest_sweep_configs()


def main():
    parser = argparse.ArgumentParser(
        description='Run parallel training and backtesting jobs'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'backtest-sweep', 'train+backtest-sweep'],
        help=(
            'Execution mode: '
            '"train" runs full training + single backtest per config (default); '
            '"backtest-sweep" only runs parameter sweeps on the existing trained model '
            'in GENERATED_DIR; '
            '"train+backtest-sweep" trains all configs then immediately runs the full '
            'parameter sweep against each trained model.'
        )
    )
    parser.add_argument(
        '--parallel-jobs',
        type=int,
        default=4,
        help='Number of parallel jobs (default: 4 for train, 8 for backtest-sweep).'
    )
    parser.add_argument(
        '--plot',
        '-p',
        action='store_true',
        help='Enable plotting in training scripts (train mode only)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top performers to show in each leaderboard (default: 5)'
    )
    parser.add_argument(
        '--label-mode',
        type=str,
        default=None,
        choices=['static', 'atr_scaled', 'daily_vol_scaled', 'lookahead',
                 'regime_conditional', 'trend_only', 'window_cascade', 'direction_horizon'],
        help='Override label mode for all configs (train mode only)'
    )
    parser.add_argument(
        '--features-config',
        type=str,
        default=None,
        help=(
            'Features YAML for all configs, e.g. "features-rgm.yaml" (filename inside '
            'ModelTrading/config/, or an absolute path). Forwarded to advanced_train.py '
            'AND backtest.py so both sides read the same feature set. Without it every '
            'run reads whatever features.yaml happens to contain at the time, which makes '
            'runs from different days non-comparable — pass a frozen copy for anything '
            'you intend to compare later. Recorded per run in '
            'parallel_training_summary_*.json.'
        )
    )
    parser.add_argument(
        '--config-set',
        type=str,
        default='default',
        choices=['default', 'fast-diagnostic', 'label-study', 'label-walk-forward',
                 'regularisation', 'winner', 'current', 'threshold', 'regime-training',
                 'regime-algo-study', 'window-cascade-wf', 'fast-gate',
                 'pruned', 'labelmode-pruned', 'labelmode-gate', 'label-geometry'],
        help=(
            'Which training-config builder to use. "default" calls '
            'create_training_configs(); "fast-diagnostic" calls '
            'create_fast_diagnostic_configs() — the BL/A1-A5/B1-B6 ablation '
            'plan for the fast-model regression study; "label-study" calls '
            'create_label_study_configs(--study-phase) — the label-mode bake-off on a '
            'fixed window; "label-walk-forward" calls '
            'create_label_study_walk_forward_configs() — the three signal-carrying '
            'label modes, built for --walk-forward; "window-cascade-wf" calls '
            'create_window_cascade_wf_configs() — the window_cascade/150-pip reference '
            'configuration as two cells that differ only in the slow entry threshold '
            '(trained vs the single-window optimum 0.40), built for --walk-forward; '
            '"regularisation" calls create_regularisation_configs() — baseline vs '
            'aggressive subsampling, built for --walk-forward; '
            '"winner" calls create_winner_configs() — the single validated best cell '
            '(trend_only, 18m), as a fast reference baseline for the next change; '
            '"current" calls create_current_setup_configs() — the production command '
            '(trend_only, subsample 0.7/0.6, mi_permutations 0) with a fixed '
            '--p-open-slow 0.55 and --regime-gate off, as one walk-forward cell; '
            '"fast-gate" calls create_fast_gate_configs() — fast-label policy '
            '(slow-label vs conditional-on-setup) x entry gate, three cells; '
            '"threshold" calls create_threshold_configs() — trained entry threshold vs '
            'a fixed 0.55, to explain the loss in the most recent window; '
            '"regime-training" calls create_regime_training_configs() — train on all '
            'bars vs on trend bars only, testing why the within-trend AUC is sub-chance; '
            '"labelmode-pruned" calls create_labelmode_pruned_configs() — the four '
            'living label modes (trend_only, regime_conditional, window_cascade, '
            'atr_scaled), each on its own pruned feature config, built for '
            '--walk-forward; "labelmode-gate" calls create_labelmode_gate_configs() — '
            'the same four cells, but every trained model is backtested twice '
            '(slow-only entry vs fast+slow entry), paired on identical models; '
            '"label-geometry" calls create_label_geometry_configs() — the A15 '
            'TP:SL x horizon sweep on the two living barrier modes, each model '
            'backtested twice (default exits vs TP-aligned exit), built for '
            '--walk-forward.'
        ),
    )
    parser.add_argument(
        '--study-phase',
        type=str,
        default='1',
        choices=['smoke', '1', '2'],
        help=(
            'Phase of the label-mode study (only used with --config-set label-study): '
            '"smoke" = 1 run per mode at the 6-month window; "1" = full screen '
            '(52 configs); "2" = refinement of the phase-1 finalists.'
        ),
    )
    parser.add_argument(
        '--sweep-set',
        type=str,
        default='default',
        choices=['default', 'label-study', 'label-study-full'],
        help=(
            'Which backtest-sweep templates to use in backtest-sweep / '
            'train+backtest-sweep mode. "default" = the 33-template round-8 sweep; '
            '"label-study" = compact 12-template entry-threshold x regime-gate grid; '
            '"label-study-full" = the 12 entry templates plus exit-strategy combos (28).'
        ),
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=1,
        help='Train each config N times with different random seeds and average results (default: 1 = no multi-seed)'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help=(
            'Explicit seeds, e.g. "34362,51386,99827". Overrides --num-seeds and makes the '
            'run reproducible. Use the SAME seeds as a previous run to pair the comparison: '
            'both sides then see identical random draws and only the treatment differs, '
            'which removes the seed spread from the difference instead of adding it twice. '
            'Seeds actually used are recorded per run in parallel_training_summary_*.json.'
        )
    )
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help=(
            'Evaluate every config across rolling train/test folds instead of one fixed '
            'backtest period, and pool the trades so the result carries a confidence '
            'interval. Combines with --config-set for grid search and with --num-seeds.'
        )
    )
    parser.add_argument('--wf-train-months', type=int, default=18,
                        help='Walk-forward: training window length in months (default: 18)')
    parser.add_argument('--wf-test-months', type=int, default=6,
                        help='Walk-forward: out-of-sample window per fold in months (default: 6)')
    parser.add_argument('--wf-step-months', type=int, default=None,
                        help='Walk-forward: months to roll between folds (default: = --wf-test-months, '
                             'which keeps test periods non-overlapping. Smaller values reuse market '
                             'days across folds and make the pooled interval too narrow.)')
    # --- campaign-wide measurement settings (see set_campaign_args) ----------
    parser.add_argument('--cost-model', type=str, default='none',
                        choices=['none', 'fixed', 'data'],
                        help='Transaction costs for EVERY backtest in this campaign. '
                             'Default none = gross, the pre-2026-08-29 behaviour. Use '
                             '"data" for anything that will be believed: without it, '
                             'configurations that trade more are flattered in proportion.')
    parser.add_argument('--spread-pips', type=float, default=0.4,
                        help='Full spread for --cost-model fixed (default 0.4).')
    parser.add_argument('--slippage-pips', type=float, default=0.0,
                        help='Slippage per leg for every backtest (round trip pays twice).')
    parser.add_argument('--commission-per-million', type=float,
                        default=costs.DEFAULT_COMMISSION_PER_MILLION,
                        help='Trade commission per 1M per side for every backtest '
                             '(default 18.0, the Dukascopy net-deposit > 50k tier). '
                             'Only reaches the children when --cost-model is not none.')
    parser.add_argument('--overnight-long-per-million', type=float,
                        default=costs.DEFAULT_OVERNIGHT_LONG_PER_MILLION,
                        help='Overnight financing per 1M per settlement night, LONG '
                             'side, for every backtest (default 63.65).')
    parser.add_argument('--overnight-short-per-million', type=float,
                        default=costs.DEFAULT_OVERNIGHT_SHORT_PER_MILLION,
                        help='Overnight financing per 1M per settlement night, SHORT '
                             'side, for every backtest (default 28.65).')
    parser.add_argument('--risk-model', type=str, default='fixed_notional',
                        choices=['fixed_notional', 'fixed_fractional'],
                        help='Position sizing for every backtest. fixed_notional (default) '
                             'trades 1M regardless of the stop, which makes drawdown a '
                             'statement about the sizing rule rather than the strategy.')
    parser.add_argument('--risk-pct', type=float, default=1.0,
                        help='Equity risked per trade under --risk-model fixed_fractional.')
    parser.add_argument('--cv-gap', type=int, default=0,
                        help='CV embargo in M15 bars for every training in this campaign; '
                             '-1 auto-derives it from the label horizon.')
    parser.add_argument('--sample-weight', type=str, default='none',
                        choices=['none', 'uniqueness'],
                        help='Label-uniqueness training weights for every training.')
    parser.add_argument('--fail-on-degenerate-folds', action='store_true',
                        help='Abort a cell whose CV folds would train on zero positives '
                             'instead of recording it and continuing.')
    parser.add_argument('--keep-diagnostics', type=str, default=None, metavar='DIR',
                        help='Copy the per-run diagnostics (training_summary.json, '
                             'oof_predictions.parquet, learning_curves.json, MI/PFI/selected '
                             'features, trade list, backtest summary) into DIR before the run '
                             'directory is deleted. Without it a campaign leaves no per-fold '
                             'artefacts at all, so feature-selection stability and per-fold '
                             'model quality cannot be measured over a walk-forward. A relative '
                             'DIR is resolved against the repository root, not the CWD.')
    parser.add_argument('--diagnostics-full', action='store_true',
                        help='Also compute SHAP and PFI in every run. Slow — meant for a '
                             'small diagnostics campaign, not for a grid.')

    parser.add_argument('--wf-embargo-days', type=int, default=0,
                        help='Walk-forward: days of embargo between each training window and '
                             'its test window. Without it the last `label_horizon` bars of the '
                             'training labels are resolved from price action INSIDE the test '
                             'window — a leak into the out-of-sample measurement itself. Set to '
                             'at least the label horizon in days (the default 384-bar/96h label '
                             'needs 4, use 5-6 for margin). Default 0 = historical geometry, '
                             'kept so existing walk-forward results stay comparable.')
    parser.add_argument('--wf-test-start', type=str, default=None,
                        help='Walk-forward: confine the TEST windows to start here; each fold '
                             'trains on the --wf-train-months directly before its own test window. '
                             'Use this to measure stability within a recent regime instead of '
                             'replaying the whole history.')
    parser.add_argument('--wf-test-end', type=str, default=None,
                        help='Walk-forward: last day any test window may cover (with --wf-test-start)')
    parser.add_argument('--wf-data-start', type=str, default=None,
                        help='Walk-forward: first training bar (default: timeframes.DATA_AVAILABLE_START)')
    parser.add_argument('--wf-data-end', type=str, default=None,
                        help='Walk-forward: last usable bar (default: timeframes.BACKTEST_END)')
    parser.add_argument(
        '--scenario-name',
        type=str,
        default=None,
        help=(
            'Name for this scenario run (e.g. "regime_conditional_v2"). '
            'All run outputs are placed under generated/scenarios/<name>/<run_id>/. '
            'A scenario_manifest.json with Top-K aggregated metrics is written at the end.'
        )
    )
    parser.add_argument(
        '--scenario-description',
        type=str,
        default=None,
        help='Free-text description of the architecture / hypothesis for this scenario.'
    )
    add_regime_filter_args(parser)
    add_sampling_args(parser)
    # W&B tracking, forwarded to every child train/backtest (see set_campaign_args)
    experiment_tracking.add_wandb_args(parser)

    args = parser.parse_args()
    validate_regime_args(args)
    validate_sampling_args(args)
    set_campaign_args(args)
    if args.keep_diagnostics:
        args.keep_diagnostics = resolve_diagnostics_dir(args.keep_diagnostics)

    start_time = time.time()

    if args.mode == 'backtest-sweep':
        # ----------------------------------------------------------------
        # Backtest-only sweep: reuse existing trained model
        # ----------------------------------------------------------------
        configs = _select_sweep_configs(args.sweep_set)
        if not configs:
            print("No backtest sweep configurations defined. "
                  "Edit create_backtest_sweep_configs() to add configurations.")
            return

        max_workers = args.parallel_jobs if args.parallel_jobs != 4 else 8
        results = run_parallel_backtest_sweep(configs, max_workers=max_workers)

    else:
        # ----------------------------------------------------------------
        # Full training + (optional) backtest sweep mode
        # ----------------------------------------------------------------
        if args.config_set == 'fast-diagnostic':
            configs = create_fast_diagnostic_configs()
        elif args.config_set == 'label-study':
            configs = create_label_study_configs(args.study_phase)
        elif args.config_set == 'label-walk-forward':
            configs = create_label_study_walk_forward_configs()
        elif args.config_set == 'fast-gate':
            configs = create_fast_gate_configs()
        elif args.config_set == 'window-cascade-wf':
            configs = create_window_cascade_wf_configs()
        elif args.config_set == 'pruned':
            configs = create_pruned_configs()
        elif args.config_set == 'labelmode-pruned':
            configs = create_labelmode_pruned_configs()
        elif args.config_set == 'labelmode-gate':
            configs = create_labelmode_gate_configs()
        elif args.config_set == 'label-geometry':
            configs = create_label_geometry_configs()
        elif args.config_set == 'regularisation':
            configs = create_regularisation_configs()
        elif args.config_set == 'winner':
            configs = create_winner_configs()
        elif args.config_set == 'current':
            configs = create_current_setup_configs()
        elif args.config_set == 'threshold':
            configs = create_threshold_configs()
        elif args.config_set == 'regime-training':
            configs = create_regime_training_configs()
        elif args.config_set == 'regime-algo-study':
            configs = create_regime_algo_study_configs()
        else:
            configs = create_training_configs()

        if args.scenario_name:
            for cfg in configs:
                cfg.effective_run_id = os.path.join("scenarios", args.scenario_name, cfg.run_id)

        if args.label_mode is not None:
            for cfg in configs:
                cfg.label_mode = args.label_mode

        if getattr(args, 'features_config', None):
            for cfg in configs:
                cfg.features_config = args.features_config

        if getattr(args, 'regime_filter', False):
            for cfg in configs:
                cfg.regime_filter = True
                cfg.regime_type = args.regime_type

        # Propagate sampling overrides from CLI to all configs.
        # Only values the user actually passed override the config: an unconditional
        # copy would silently reset every builder's training_sampling=False back to
        # the argparse default (True) and change what the study measures.
        for attr in ('training_sampling', 'sampling_stride', 'sampling_context',
                     'sampling_stride_x', 'sampling_hours_before', 'sampling_hours_after'):
            value = getattr(args, attr, None)
            if value is None or value == parser.get_default(attr):
                continue
            for cfg in configs:
                setattr(cfg, attr, value)

        explicit_seeds = None
        if getattr(args, 'seeds', None):
            explicit_seeds = [int(s) for s in args.seeds.replace(',', ' ').split()]

        if explicit_seeds or args.num_seeds > 1:
            if explicit_seeds:
                seeds = explicit_seeds
                print(f"Multi-seed mode: fixed seeds = {seeds}")
            else:
                # Drawn fresh every invocation, so two runs never share seeds. That is
                # fine for a standalone measurement and wrong for a comparison: the seed
                # spread then sits on BOTH sides of it. Pass --seeds to pair them.
                seeds = random.sample(range(1, 100_000), args.num_seeds)
                print(f"Multi-seed mode: {args.num_seeds} random seeds per config = {seeds}")
                print("  (pass --seeds to reuse these in a later run and pair the comparison)")
            expanded = []
            for cfg in configs:
                for seed in seeds:
                    c = copy.copy(cfg)
                    c.seed = seed
                    c.base_name = cfg.name
                    c.name = f"{cfg.name}_s{seed}"
                    c.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                    if args.scenario_name:
                        c.effective_run_id = os.path.join("scenarios", args.scenario_name, c.run_id)
                    else:
                        c.effective_run_id = c.run_id
                    expanded.append(c)
            configs = expanded
            print(f"Expanded to {len(configs)} total jobs ({len(expanded) // args.num_seeds} configs × {args.num_seeds} seeds)\n")

        # Walk-forward last: it overwrites train/backtest dates, so it must run after
        # every other config mutation. Applied after the seed expansion so N configs x
        # S seeds x F folds all collapse back onto N base_names when aggregating.
        if getattr(args, 'walk_forward', False):
            configs = expand_to_walk_forward(
                configs,
                train_months=args.wf_train_months,
                test_months=args.wf_test_months,
                step_months=args.wf_step_months,
                data_start=args.wf_data_start,
                data_end=args.wf_data_end,
                test_start=args.wf_test_start,
                test_end=args.wf_test_end,
                embargo_days=args.wf_embargo_days,
            )

        if not configs:
            print("No training configurations defined. "
                  "Edit create_training_configs() to add configurations.")
            return

        train_results = run_parallel_training(
            configs,
            max_workers=args.parallel_jobs,
            enable_plotting=args.plot,
            keep_artifacts=(args.mode == 'train+backtest-sweep'),
            diagnostics_dir=args.keep_diagnostics,
            diagnostics_full=args.diagnostics_full,
        )

        if args.mode != 'train+backtest-sweep':
            results = train_results
        else:
            # ----------------------------------------------------------------
            # Phase 2: backtest sweep for every successfully trained model
            # ----------------------------------------------------------------
            sweep_templates = _select_sweep_configs(args.sweep_set)
            successful_runs = [r for r in train_results if r['status'] == 'success']
            print(f"\n{'='*80}")
            print(f"BACKTEST SWEEP PHASE")
            print(f"{'='*80}")
            print(f"Successful training runs: {len(successful_runs)}")
            print(f"Sweep configs per model:  {len(sweep_templates)}")
            print(f"Total sweep jobs:         {len(successful_runs) * len(sweep_templates)}")
            print(f"{'='*80}\n")

            all_sweep_configs = build_sweep_configs_for_runs(successful_runs, sweep_templates)

            sweep_workers = max(args.parallel_jobs, 8)
            sweep_results = run_parallel_backtest_sweep(all_sweep_configs, max_workers=sweep_workers)

            # Clean up run dirs now that the sweep is done
            for r in successful_runs:
                eff = r['config'].get('effective_run_id', r['config']['run_id'])
                cleanup_training_artifacts(eff, dir_config.GENERATED_DIR)

            total_time = time.time() - start_time

            # Save and print training summary first, then sweep summary
            train_summary_path = os.path.join(
                dir_config.GENERATED_DIR,
                f"parallel_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            save_summary_report(train_results, train_summary_path)
            print(f"\n{'='*80}")
            print("TRAINING RESULTS")
            print(f"{'='*80}")
            print_final_summary(train_results, top_n=args.top_n)

            sweep_summary_path = os.path.join(
                dir_config.GENERATED_DIR,
                f"backtest_sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            save_summary_report(sweep_results, sweep_summary_path)
            print(f"\n{'='*80}")
            print("BACKTEST SWEEP RESULTS")
            print(f"{'='*80}")
            print_final_summary(sweep_results, top_n=args.top_n)

            # The manifest is what compare_scenarios.py reads, so it must be written
            # in this mode too — not only on the plain train path below.
            _save_scenario_manifest_if_named(args, train_results)

            print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            return

    total_time = time.time() - start_time

    # Save and print summary (works for both modes)
    summary_path = os.path.join(
        dir_config.GENERATED_DIR,
        f"parallel_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_summary_report(results, summary_path)
    print_final_summary(results, top_n=args.top_n)

    if getattr(args, 'walk_forward', False):
        wf = print_walk_forward_summary(results, top_n=args.top_n)
        wf_path = os.path.join(
            dir_config.GENERATED_DIR,
            f"walk_forward_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(wf_path, 'w', encoding='utf-8') as f:
            json.dump({
                'measurement': {'cost_model': args.cost_model,
                                'spread_pips': args.spread_pips,
                                'slippage_pips': args.slippage_pips,
                                'commission_per_million': args.commission_per_million,
                                'overnight_long_per_million': args.overnight_long_per_million,
                                'overnight_short_per_million': args.overnight_short_per_million,
                                'risk_model': args.risk_model,
                                'risk_pct': args.risk_pct,
                                'cv_gap': args.cv_gap,
                                'sample_weight': args.sample_weight},
                'window': {'train_months': args.wf_train_months,
                           'test_months': args.wf_test_months,
                           'step_months': args.wf_step_months or args.wf_test_months,
                           'embargo_days': args.wf_embargo_days,
                           'test_start': args.wf_test_start,
                           'test_end': args.wf_test_end},
                'configs': wf,
            }, f, indent=2, default=str)
        print(f"Walk-forward summary saved to: {wf_path}")

    _save_scenario_manifest_if_named(args, results)

    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == '__main__':
    main()
