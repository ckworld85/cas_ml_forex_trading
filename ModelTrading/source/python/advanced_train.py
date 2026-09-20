"""
Advanced Training Script with Feature Selection and SHAP Analysis

This script extends the standard training pipeline with:
1. Regime label generation (Trend/Range, High/Low Volatility)
2. Mutual Information pre-filtering
3. Time-series cross-validation
4. Permutation Feature Importance
5. SHAP analysis (global, local, regime-specific)

Usage:
    python advanced_train.py [--plot] [--mi-threshold 0.001] [--pfi-threshold 0.0]
    python advanced_train.py --run-id <run_id> --train-start 2024-01-01 --train-end 2024-12-31
"""

import copy
import os
import sys
import shlex
import subprocess
import warnings
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, matthews_corrcoef
)
import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import argparse

# Suppress common warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config
import ModelTrading.config.timeframes as timeframes
import ModelTrading.source.python.utils.csv as csv
import ModelTrading.source.python.utils.datahandling as datahandling
import ModelTrading.source.python.export.export_scaler as export_scaler
import ModelTrading.source.python.utils.forex as forex
import ModelTrading.source.python.features.config as feature_config
from ModelTrading.source.python.features import indicators as features
from ModelTrading.source.python.features import regime_model as regime_model_mod
from ModelTrading.source.python.labeling.regime import (
    generate_regime_labels,
    generate_regime_labels_from_scores,
    print_regime_statistics
)
from ModelTrading.source.python.labeling.regime_filter import (
    add_regime_filter_args, apply_regime_filter, validate_regime_args,
)
from ModelTrading.source.python.labeling.sampling import (
    add_sampling_args, compute_sampling_index, validate_sampling_args,
)
import ModelTrading.source.python.training.sample_weights as sample_weights_mod
from ModelTrading.source.python.training.feature_selection import (
    MutualInformationFilter,
    PermutationImportanceFilter,
    FeatureSelectionPipeline
)
from ModelTrading.source.python.training.shap_analysis import (
    SHAPAnalyzer,
    RegimeSHAPAnalyzer,
    run_full_shap_analysis,
    plot_scope_comparison_by_regime,
    _is_combined_regime_key,
)
from ModelTrading.source.python.labeling.dynamic import (
    LabelConfig, calculate_dynamic_targets, generate_dynamic_labels,
    calculate_volatility_scaled_targets,
    generate_mean_reversion_labels, generate_regime_conditional_labels,
    generate_trend_only_labels,
    print_label_distribution, print_target_statistics, validate_label_distribution,
    analyze_weekly_labels, print_weekly_label_summary, denoise_labels, print_denoise_statistics,
    mask_fast_labels_by_slow
)
from ModelTrading.source.python.labeling.lookahead import (
    generate_lookahead_slow_labels
)
from ModelTrading.source.python.labeling.direction import (
    generate_direction_horizon_labels
)
from ModelTrading.source.python.labeling.file_labels import (
    load_labels_from_file
)
from ModelTrading.source.python.labeling.window_labels import (
    compute_regime_windows,
    generate_setup_labels,
    generate_timing_labels_fhl,
    generate_timing_labels_rebound,
    print_window_label_stats,
)
from ModelTrading.source.python.features.fingerprint import (
    calculate_indicators as fp_calculate_indicators,
    calculate_current_fingerprint,
    calculate_historical_fingerprints,
    find_similar_windows,
    print_fingerprint,
    print_similar_windows
)
from ModelTrading.source.python.features.fingerprint_config import get_fingerprint_config
from ModelTrading.source.python.analytics.model_report import generate_report as generate_health_report
import ModelTrading.source.python.utils.plot as plot
import ModelTrading.source.python.utils.experiment_tracking as experiment_tracking


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced training with feature selection and SHAP analysis'
    )
    parser.add_argument('--run-id', type=str, default=None,
                        help='Unique run ID for parallel execution')
    parser.add_argument('--features-config', type=str, default=None,
                        help='Features YAML to use (filename inside ModelTrading/config/, '
                             'or an absolute path). Defaults to features.yaml.')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Enable plotting of training visualizations')
    parser.add_argument('--plot-last-weeks', type=int, default=None,
                        help='Limit label distribution plots to the most recent N weeks '
                             '(default: plot all weeks).')
    parser.add_argument('--train-start', type=str, default=None,
                        help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--fast-train-start', type=str, default=None,
                            help='Training start date for fast (M15) scope only (YYYY-MM-DD). '
                                 'Overrides --train-start for M15 fast scope; slow scope keeps --train-start.')
    parser.add_argument('--train-end', type=str, default=None,
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--backtest-start', type=str, default=None,
                        help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--backtest-end', type=str, default=None,
                        help='Backtest end date (YYYY-MM-DD)')
    
    # Feature selection parameters
    parser.add_argument('--mi-threshold', type=float, default=0.0,
                        help='Mutual Information threshold for feature selection (default: 0.0)')
    parser.add_argument('--fast-mi-threshold', type=float, default=None,
                        help='MI threshold for fast (M15) scope only — overrides --mi-threshold for fast scope')
    parser.add_argument('--slow-mi-threshold', type=float, default=None,
                        help='MI threshold for slow (daily) scope only — overrides --mi-threshold for slow scope')
    parser.add_argument('--pfi-threshold', type=float, default=0.0,
                        help='Permutation Feature Importance threshold (default: 0.0)')
    parser.add_argument('--target-recall', type=float, default=None,
                        help='Decision-threshold rule: pick the threshold that maximises precision '
                             'subject to recall >= target-recall. Falls back to F1-optimal if no '
                             'threshold reaches the target. Default: 0.5 — unless a regime-scoped '
                             'floor (--target-recall-trend/-range) is given without this flag, in '
                             'which case the overall floor is dropped (0.0) so the regime floor '
                             'alone constrains the threshold')
    parser.add_argument('--fast-target-recall', type=float, default=None,
                        help='Override --target-recall for the fast (M15) scope only')
    parser.add_argument('--slow-target-recall', type=float, default=None,
                        help='Override --target-recall for the slow (4h) scope only')
    parser.add_argument('--target-recall-trend', type=float, default=None,
                        help='Recall floor evaluated over the TREND bars only '
                             '(regime_trend != 0, up- and downtrends pooled). Given alone, '
                             'it REPLACES the overall floor (which drops to 0.0); given '
                             'together with an explicit --target-recall, both floors must '
                             'hold. If no threshold satisfies all floors the selection '
                             'falls back to the overall-only rule with a warning. '
                             'Default: off')
    parser.add_argument('--target-recall-range', type=float, default=None,
                        help='Recall floor evaluated over the RANGE bars only '
                             '(regime_trend == 0), combinable with --target-recall and '
                             '--target-recall-trend — every GIVEN floor must hold, and '
                             'like --target-recall-trend it drops the implicit overall '
                             '0.5 default when --target-recall is not passed explicitly. '
                             'Default: off')
    parser.add_argument('--mi-permutations', type=int, default=20,
                        help='Shuffled-label repeats for the per-feature MI noise floor '
                             '(p95). A feature is kept when its MI beats its own floor — '
                             'self-calibrating across effective sample sizes. 0 disables '
                             'the test and falls back to --mi-threshold (default: 20)')
    parser.add_argument('--legacy-mi', action='store_true',
                        help='Score MI the pre-fix way: no run-length dedup and no noise '
                             'floor. Tie-inflated for features carried from a coarser '
                             'timeframe (daily values repeat 6x on 4h, 96x on M15) — only '
                             'for reproducing runs made before the fix.')
    parser.add_argument('--skip-mi', action='store_true',
                        help='Skip Mutual Information filtering')
    parser.add_argument('--skip-pfi', action='store_true',
                        help='Skip Permutation Feature Importance filtering (both scopes)')
    parser.add_argument('--skip-pfi-slow', action='store_true', default=True,
                        help='Skip PFI for slow (daily) scope only — keeps all MI-selected daily features (default: True)')
    parser.add_argument('--run-pfi-slow', dest='skip_pfi_slow', action='store_false',
                        help='Enable PFI for slow (daily) scope (overrides --skip-pfi-slow default)')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP analysis')
    
    # Time-series CV parameters
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='Number of time-series cross-validation splits (default: 5)')
    parser.add_argument('--cv-gap', type=int, default=0,
                        help='Embargo between CV train and validation folds, in M15 bars '
                             '(slow cadence receives ceil(gap/16) 4h bars). Prevents '
                             'label-horizon leakage across the fold seam: a barrier label at '
                             'bar t peeks up to horizon_max bars ahead, so set this to '
                             'horizon_max for long-horizon labels (e.g. 480 for a 120h label). '
                             '-1 = auto: embargo exactly the label horizon the run actually '
                             'used. Default 0 = exact historical behaviour (no gap), keeping '
                             'CV metrics comparable with all previous runs; a warning is '
                             'printed because the seam then leaks.')
    parser.add_argument('--sample-weight', type=str, default='none',
                        choices=['none', 'uniqueness'],
                        help="Training sample weights. 'uniqueness' weights each bar by "
                             "its average label uniqueness (Lopez de Prado ch. 4): barrier "
                             "labels are emitted every bar but decided over the following "
                             "horizon, so up to 24 consecutive 4h rows share one outcome "
                             "and quiet stretches contribute near-copies of the same "
                             "observation. 'none' (default) keeps the historical "
                             "behaviour. n_eff is measured and reported either way.")
    parser.add_argument('--unseal-holdout', action='store_true',
                        help='Allow this run to train or backtest on bars from the SEALED '
                             'HOLD-OUT (timeframes.HOLDOUT_START onward). Reserved for the '
                             'single final evaluation of the configuration the '
                             'pre-registered protocol nominates (docs/preregistration.md). '
                             'Recorded in training_summary.json as holdout_unsealed.')
    parser.add_argument('--fail-on-degenerate-folds', action='store_true',
                        help='Abort when a CV fold would train on zero positive labels. '
                             'Such a fold produces a constant model whose AUC of exactly '
                             '0.500 is pooled into the headline number — a property of '
                             'the training window, not of the model. Off by default so a '
                             'parallel grid does not die on one bad cell; the coverage '
                             'report is printed and recorded either way.')

    # Calibration parameters
    parser.add_argument('--calibration-method', type=str, default='platt',
                        choices=['platt', 'isotonic', 'none'],
                        help='Probability calibration method for slow models (default: platt)')
    parser.add_argument('--calibration-fraction', type=float, default=0.2,
                        help='Fraction of training data (tail) used for calibration (default: 0.2)')

    # XGBoost hyperparameters
    parser.add_argument('--max-depth', type=int, default=3,
                        help='XGBoost max_depth (default: 3)')
    parser.add_argument('--slow-max-depth', type=int, default=None,
                                help='XGBoost max_depth for slow (daily) models (default: 3)')
    parser.add_argument('--fast-max-depth', type=int, default=None,
                    help='XGBoost max_depth for fast (M15) models. '
                            'Defaults to --max-depth when not set (default: 3).')
    
    parser.add_argument('--eta', type=float, default=0.05,
                        help='XGBoost learning rate eta (default: 0.05)')
    
    parser.add_argument('--num-boost-round', type=int, default=300,
                        help='XGBoost num_boost_round (default: 300)')
    parser.add_argument('--slow-num-boost-round', type=int, default=300,
                            help='XGBoost num_boost_round for slow (daily) models (default: 300). '
                                 'This is a CAP — the trained count comes from the per-fold CV vote '
                                 'unless --no-cv-boost-rounds is set.')
    parser.add_argument('--fast-num-boost-round', type=int, default=None,
                            help='XGBoost num_boost_round for fast (M15) models. '
                                 'Defaults to --num-boost-round when not set.')
    
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='XGBoost subsample fraction (default: 0.8)')
    parser.add_argument('--colsample-bytree', type=float, default=0.8,
                        help='XGBoost colsample_bytree fraction (default: 0.8)')
    parser.add_argument('--colsample-bynode', type=float, default=1.0,
                        help='XGBoost colsample_bynode fraction, applied per split on top '
                             'of --colsample-bytree (default: 1.0 = XGBoost default, '
                             'keeps existing commands unchanged)')
    
    parser.add_argument('--min-child-weight', type=int, default=3,
                        help='XGBoost min_child_weight for fast models (default: 3)')
    parser.add_argument('--slow-min-child-weight', type=int, default=10,
                            help='XGBoost min_child_weight for slow (daily) models (default: 10)')
    parser.add_argument('--fast-min-child-weight', type=int, default=None,
                            help='XGBoost min_child_weight for fast (M15) models. '
                                'Defaults to --min-child-weight when not set.')

    parser.add_argument('--slow-lambda', type=float, default=5.0,
                            help='XGBoost L2 regularization (lambda) for slow (daily) models (default: 5.0)')    
    parser.add_argument('--fast-lambda', type=float, default=1.0,
                        help='XGBoost L2 regularization (lambda) for fast models (default: 1.0)')
    
    parser.add_argument('--slow-spw-factor', type=float, default=1.0,
                        help='Multiplier on auto scale_pos_weight for slow models. '
                             '<1.0 reduces positive weight → higher precision, lower recall. '
                             'Default 1.0 = natural neg/pos ratio.')
    parser.add_argument('--fast-spw-factor', type=float, default=1.0,
                        help='Multiplier on auto scale_pos_weight for fast models. '
                             '<1.0 reduces positive weight → higher precision, lower recall. '
                             'Default 1.0 = natural neg/pos ratio.')

    parser.add_argument('--fast-use-slow-label', action='store_true', default=False,
                        help='Train fast (M15) model on slow-style directional labels (long_slow/short_slow) '
                             'instead of MFE-before-MAE labels. Tests whether M15 features predict '
                             'the same medium-term direction as 4h/daily features.')
    parser.add_argument('--fast-conditional-on-setup', action='store_true', default=False,
                        help='Train each fast model only on bars where its matching slow label is 1, '
                             'instead of on every M15 bar (meta-labelling). The fast model then answers '
                             '"is THIS the bar to enter" given a setup, rather than re-learning the '
                             'direction question on a near-empty label. Raises the fast positive rate '
                             'from the global one (0.4%% for window_cascade timing labels) to the '
                             'within-setup one (~7%%), which is also the rate the live entry gate sees. '
                             'Mutually exclusive with --fast-use-slow-label, which would make the '
                             'conditioning set identical to the label.')
    parser.add_argument('--seed', type=int, default=None,
                        help='XGBoost random seed for reproducibility (default: None = XGBoost default 0)')
    parser.add_argument('--early-stopping-rounds', type=int, default=0,
                        help='Patience for the HOLDOUT early-stopping fallback, used only for models '
                             'whose CV learning curve yields no usable AUC (or when '
                             '--no-cv-boost-rounds is set). The optimal round count is found on a '
                             'temporal holdout, then the model is refit on full data for that many '
                             'rounds. Default 0 disables the fallback (fixed --num-boost-round); '
                             'a positive int sets the patience, -1 auto-derives it as '
                             'max(20, cap // 4) from each model\'s num_boost_round.')
    parser.add_argument('--es-val-frac', type=float, default=0.15,
                        help='Fraction of the (chronologically last) training data held out as the '
                             'early-stopping validation set (default: 0.15).')
    parser.add_argument('--no-cv-boost-rounds', action='store_true', default=False,
                        help='Do NOT derive num_boost_round from the CV folds. By default every '
                             'CV fold votes for a round count (argmax of its own val AUC, only on '
                             'a real post-peak decline, else the cap) and the MEDIAN vote wins — '
                             'a rank statistic so one degenerate fold cannot set the capacity. '
                             'Set this to always use the fixed --num-boost-round / '
                             '--slow-num-boost-round.')

    # Label config overrides (static mode)
    parser.add_argument('--pip-target', type=int, default=None,
                        help='Static pip target for slow labels (default: 80 pips)')
    parser.add_argument('--stop-pips', type=int, default=None,
                        help='Static stop pips for slow labels (default: uses LabelConfig default of 35)')
    parser.add_argument('--horizon-min', type=int, default=None,
                        help='Min horizon bars for slow labels (default: uses LabelConfig default of 96)')
    parser.add_argument('--horizon-max', type=int, default=None,
                        help='Max horizon bars for slow labels (default: uses LabelConfig default of 384)')

    # Dynamic label generation parameters
    # ATR-scaled:   With ATR-scaled (even smoothed), the question shifts: in 2022 it's "does price move 175 pips before 58?", in 2024 it's "does price move 83 pips before 27?". These require different features to answer. The model can't learn one rule that works across all volatility regimes — hence permanently stuck near 0.54.
    #               Recommendation: revert to static mode. The February dead zone isn't a problem to fix — it's the model correctly refusing to trade a choppy, non-directional market. Over a longer backtest period, months like March 2026 (strong trend, 22 combined signals) will dominate the P&L anyway.
    parser.add_argument('--label-mode', type=str, default='static',
                        choices=['static', 'atr_scaled', 'daily_vol_scaled', 'lookahead', 'regime_conditional', 'trend_only',
                                 'window_cascade', 'direction_horizon', 'file'],
                        help='Label generation mode: static (fixed pips), atr_scaled (ATR-adaptive), '
                            'daily_vol_scaled (daily return-volatility scaled levels applied on M15), '
                             'lookahead (percentile rank of forward return — regime-stable), '
                             'regime_conditional (trend bars: target/stop, range bars: BB mean-reversion), '
                             'window_cascade (regime-gated windows: setup model + FHL timing model), '
                             'trend_only (trend bars: target/stop, range bars: label=0 — no entry), '
                             'direction_horizon (fixed-horizon sign-of-return with dead zone — '
                             'simple direction classification, no TP/SL race), '
                             'file (no label computation at all — read pre-computed labels from '
                             '--label-file)')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Parquet file with pre-computed labels (required for --label-mode file). '
                             'Index = bar timestamps. Either per-model columns '
                             '(long_slow/short_slow[/long_fast/short_fast], optionally prefixed with '
                             'target_ / y_ / y_target_, values 0/1) or a single signed direction column '
                             '(+1 long / -1 short / 0 none). Missing fast columns mirror the slow labels; '
                             'bars missing from the file are dropped as NaN targets.')
    parser.add_argument('--label-column', type=str, default=None,
                        help='Name of the signed (+1/0/-1) direction column in --label-file. '
                             'Only needed when auto-detection (label/signal/direction/y/target) fails '
                             'or the wrong column would be picked.')
    parser.add_argument('--daily-vol-span', type=int, default=100,
                        help='EWM span for daily log-return volatility used in daily_vol_scaled mode (default: 100)')
    parser.add_argument('--slow-hysteresis-multiplier', type=float, default=0.0,
                        help='Volatility-relative arming move for slow labels in daily_vol_scaled mode. '
                             'Threshold = close * multiplier * daily_vol; 0 disables. '
                             'Barrier checks start only after price first leaves this zone, and the arming bar itself is ignored.')
    parser.add_argument('--timing-entry', type=str, default='fhl',
                        choices=['fhl', 'rebound'],
                        help='Timing entry strategy for window_cascade fast model: '
                             'fhl = First Higher Low (default, ~306 long labels 2017-2025), '
                             'rebound = relaxed rebound from SMA50 (~331 long labels 2017-2025)')
    parser.add_argument('--window-atr-mult', type=float, default=2.5,
                        help='window_cascade only: a setup window is validated when its MFE '
                             'reaches --pip-target OR this multiple of ATR(14) at the trigger '
                             'bar (default 2.5). EURUSD M15 ATR(14) has a median of 9.2 pips, '
                             'so the default validates at ~23 pips and silently overrides a '
                             'larger --pip-target. Pass a large value to disable the ATR '
                             'alternative and let --pip-target bind.')
    parser.add_argument('--atr-multiplier', type=float, default=1.5,
                        help='ATR multiplier for target (only used in atr_scaled mode, default: 1.5)')
    parser.add_argument('--atr-smoothing-period', type=int, default=1920,
                        help='Bars to smooth ATR over in atr_scaled mode (default: 1920 = ~1 month M15)')
    parser.add_argument('--atr-stop-multiplier', type=float, default=0.5,
                        help='ATR multiplier for stop (only used in atr_scaled mode, default: 0.5)')
    parser.add_argument('--use-regime-labels', action='store_true',
                        help='Use regime labels for dynamic target adjustments (only in atr_scaled mode)')

    # Lookahead label mode parameters
    parser.add_argument('--lookahead-horizon', type=int, default=96,
                        help='Forward bars for lookahead slow labels (default: 96 = 24h M15)')
    parser.add_argument('--lookahead-pct', type=float, default=20.0,
                        help='Top/bottom percentile threshold for lookahead labels (default: 20.0)')
    parser.add_argument('--lookahead-lookback', type=int, default=2880,
                        help='Backward-looking window for percentile calculation (default: 2880 ~= 30 days M15)')
    parser.add_argument('--lookahead-min-pips', type=float, default=30.0,
                        help='Minimum pip value the percentile threshold must reach for a label to be issued '
                             '(default: 30). Bars where the rolling threshold is below this are skipped.')
    parser.add_argument('--lookahead-stop-pips', type=float, default=65.0,
                        help='Max adverse excursion in pips during the lookahead horizon before label is voided '
                             '(default: 65). Higher values allow more labels through for shorter horizons.')

    # Direction-horizon label mode parameters
    parser.add_argument('--direction-horizon', type=int, default=192,
                        help='Forward bars for direction_horizon slow labels (default: 192 = 48h M15)')
    parser.add_argument('--direction-dead-zone-pips', type=float, default=12.0,
                        help='Dead-zone half-width in pips around zero forward move; bars inside '
                             'get label 0/0 (default: 12). Live stop-loss is 35 pips, so this should '
                             'stay meaningfully smaller.')

    # Fast model: MFE-before-MAE entry quality labels
    parser.add_argument('--mfe-threshold', type=float, default=10.0,
                        help='MFE-before-MAE threshold in pips for fast model labels (default: 10)')
    parser.add_argument('--mfe-horizon', type=int, default=24,
                        help='Lookahead bars for MFE-before-MAE fast labels (default: 24 = 6h M15)')

    # Label denoising parameters
    parser.add_argument('--direction-aware-regime', action='store_true', default=False,
                        help='Split trend regime into UPTREND/DOWNTREND using SMA direction. '
                             'Long labels only in uptrend bars; short labels only in downtrend bars. '
                             'Default OFF — original direction-agnostic TREND/RANGE behaviour.')
    parser.add_argument('--regime-label-source', type=str, default='rule',
                        choices=['rule', 'ml'],
                        help='Which regime definition drives regime labels (regime_conditional '
                             'labelling, --regime-filter, per-regime CV metrics): "rule" = ADX/'
                             'price-efficiency (default), "ml" = the fitted regime model via '
                             'data/regime_daily.csv. Use "ml" together with the daily_rgm_* '
                             'features so the ML arm is labelled by its OWN regime definition — '
                             'labelling it with the rule makes the comparison structurally unfair.')
    parser.add_argument('--regime-label-trend-threshold', type=float, default=0.15,
                        help='With --regime-label-source ml: |rgm_trend_score| above which a bar '
                             'counts as trending (default 0.15).')
    parser.add_argument('--denoise', action='store_true', default=False,
                        help='Enable label denoising to suppress isolated counter-trend signals (default: disabled). '
                             'See "What Did NOT Work" in CLAUDE.md — denoise removes valid TP-before-SL winners.')
    parser.add_argument('--no-denoise', dest='denoise', action='store_false',
                        help='Disable label denoising (default behavior; flag retained for compatibility)')
    parser.add_argument('--denoise-window-fraction', type=float, default=0.5,
                        help='Window size as fraction of label horizon (default: 0.5)')
    parser.add_argument('--denoise-threshold', type=float, default=0.65,
                        help='Dominance threshold for suppressing a label (default: 0.65)')
    parser.add_argument('--denoise-trend-aggression', type=float, default=0.8,
                        help='Threshold multiplier for counter-trend labels in trending regimes '
                             '(default: 0.8, lower=more aggressive)')

    # Fingerprint-based training window selection
    parser.add_argument('--use-fingerprint', action='store_true',
                        help='Auto-select training window using fingerprint similarity analysis')
    parser.add_argument('--fingerprint-top-n', type=int, default=1,
                        help='Use Nth most similar historical window (default: 1 = best match)')

    # Backtesting
    parser.add_argument('--run-backtest', action='store_true',
                        help='Run backtesting after training completes')

    # Model health report generation
    parser.add_argument('--generate-report', action='store_true',
                        help='Generate model health report with Claude analysis after training')
    parser.add_argument('--skip-claude-analysis', action='store_true',
                        help='Skip Claude analysis in report (faster, for testing)')

    # Threshold sweep / PR-curve diagnostic
    parser.add_argument('--threshold-sweep', action='store_true',
                        help='Print precision/recall/F1 vs threshold table for fast and slow scopes after feature selection')
    parser.add_argument('--sweep-thresholds', type=str, default=None,
                        help='Comma-separated thresholds for --threshold-sweep (default: 0.30..0.80 step 0.05)')

    # Regime filtering
    add_regime_filter_args(parser)

    # Training-data sampling
    add_sampling_args(parser)

    # Experiment tracking (opt-in)
    experiment_tracking.add_wandb_args(parser)

    args = parser.parse_args()

    if args.label_mode == 'file' and not args.label_file:
        parser.error("--label-mode file requires --label-file <path to parquet>")
    if args.label_file and args.label_mode != 'file':
        parser.error(f"--label-file is only used with --label-mode file (got '{args.label_mode}')")
    if args.fast_conditional_on_setup and args.fast_use_slow_label:
        parser.error(
            "--fast-conditional-on-setup and --fast-use-slow-label are mutually exclusive: "
            "the former restricts the fast rows to slow_label == 1, the latter sets the fast "
            "label TO the slow label — together every remaining row would be a positive."
        )

    return args


def setup_directories(run_id=None):
    """Setup output directories."""
    run_dirs = dir_config.get_run_dirs(run_id)
    
    # Create standard directories
    for dir_name in ['generated_dir', 'java_config_dir', 'feature_map_dir', 
                     'visualization_dir', 'label_review_dir']:
        os.makedirs(run_dirs[dir_name], exist_ok=True)
    
    # Create additional directories for advanced analysis
    shap_dir = os.path.join(run_dirs['generated_dir'], 'shap_analysis')
    feature_selection_dir = os.path.join(run_dirs['generated_dir'], 'feature_selection')
    regime_dir = os.path.join(run_dirs['generated_dir'], 'regime_analysis')
    
    os.makedirs(shap_dir, exist_ok=True)
    os.makedirs(feature_selection_dir, exist_ok=True)
    os.makedirs(regime_dir, exist_ok=True)
    
    run_dirs['shap_dir'] = shap_dir
    run_dirs['feature_selection_dir'] = feature_selection_dir
    run_dirs['regime_dir'] = regime_dir

    return run_dirs


def run_fingerprint_analysis(top_n_rank: int = 1) -> tuple:
    """
    Run fingerprint analysis to find the best historical training window.

    Args:
        top_n_rank: Which rank to use (1 = best match, 2 = second best, etc.)

    Returns:
        Tuple of (train_start, train_end) as strings in YYYY-MM-DD format
    """
    print("\n" + "=" * 80)
    print("FINGERPRINT ANALYSIS - AUTO-SELECTING TRAINING WINDOW")
    print("=" * 80 + "\n")

    # Load fingerprint configuration
    fp_config = get_fingerprint_config()
    window_size = fp_config.get_window_size()
    window_step = fp_config.get_window_step()
    top_n = max(top_n_rank, fp_config.get_top_n())  # Ensure we get enough results
    params = fp_config.get_parameters()
    rsi_thresholds = fp_config.get_rsi_thresholds()

    print(f"Fingerprint settings:")
    print(f"  Window size: {window_size} days")
    print(f"  Window step: {window_step} days")
    print(f"  Selecting rank: #{top_n_rank}")

    # Load daily data for fingerprint analysis
    data_file = os.path.join(dir_config.DATA_DIR, "eurusd_daily.csv")
    print(f"\nLoading daily data from: {data_file}")

    import ModelTrading.source.python.utils.csv as csv_utils
    df_daily = csv_utils.load_csv(data_file, filter_weekends_flag=True)
    print(f"Loaded {len(df_daily)} daily bars from {df_daily.index.min()} to {df_daily.index.max()}")

    # Calculate indicators
    print("\nCalculating technical indicators...")
    df_daily = fp_calculate_indicators(df_daily, params)

    # Drop NaN values (indicator warmup)
    initial_len = len(df_daily)
    df_daily = df_daily.dropna()
    print(f"Removed {initial_len - len(df_daily)} rows (indicator warmup)")

    # Check minimum data requirement
    min_required = window_size + window_step + 1
    if len(df_daily) < min_required:
        raise ValueError(f"Not enough data: have {len(df_daily)} rows, need at least {min_required}")

    # Calculate current fingerprint
    print("\nCalculating current market fingerprint...")
    current_fp = calculate_current_fingerprint(df_daily, window_size, rsi_thresholds)
    print_fingerprint(current_fp, "CURRENT MARKET FINGERPRINT")

    # Calculate historical fingerprints
    print("\nCalculating historical fingerprints...")
    historical_fps = calculate_historical_fingerprints(
        df_daily, window_size, window_step, rsi_thresholds, parallel_jobs=1
    )
    print(f"Calculated {len(historical_fps)} historical fingerprints")

    # Find similar windows
    print("\nFinding similar historical windows...")
    similar = find_similar_windows(current_fp, historical_fps, top_n)
    print_similar_windows(similar)

    # Select the requested rank
    if top_n_rank > len(similar):
        print(f"\nWarning: Requested rank #{top_n_rank} but only {len(similar)} results available")
        top_n_rank = len(similar)

    selected = similar[top_n_rank - 1]  # 0-indexed
    train_start = selected.fingerprint.window_start
    train_end = selected.fingerprint.window_end

    print(f"\n{'=' * 80}")
    print(f"SELECTED TRAINING WINDOW (Rank #{top_n_rank}, Distance: {selected.distance:.4f})")
    print(f"  Start: {train_start}")
    print(f"  End:   {train_end}")
    print(f"{'=' * 80}\n")

    return train_start, train_end


def run_backtest(run_id=None, backtest_start=None, backtest_end=None, extra_args=None):
    """
    Run backtesting via subprocess call to backtest.py.

    Args:
        run_id: Optional run ID (loads models from generated/{run_id}/)
        backtest_start: Optional start date (YYYY-MM-DD)
        backtest_end: Optional end date (YYYY-MM-DD)
        extra_args: Optional extra CLI flags (e.g. the forwarded --wandb* flags,
            so the backtest lands in the same W&B group as this training)

    Returns:
        True if backtest succeeded, False otherwise
    """
    print("\n" + "=" * 80)
    print("RUNNING BACKTEST")
    print("=" * 80 + "\n")

    backtest_script = os.path.join(
        os.path.dirname(__file__), "backtest.py"
    )

    cmd = [sys.executable, backtest_script]

    if run_id:
        cmd.extend(['--run-id', run_id])
    if backtest_start:
        cmd.extend(['--backtest-start', backtest_start])
    if backtest_end:
        cmd.extend(['--backtest-end', backtest_end])
    cmd.append('--regime-breakdown')
    cmd.append('--use-trained-threshold')
    if extra_args:
        cmd.extend(extra_args)

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"Backtest FAILED (exit code {result.returncode})")
        return False

    print("Backtest completed successfully.")
    return True


def load_and_prepare_data(args):
    """Load data and calculate features."""
    print("=" * 80)
    print("LOADING DATA FROM CSV FILES")
    print("=" * 80 + "\n")

    df_m15 = csv.load_csv(
        os.path.join(dir_config.DATA_DIR, "eurusd_m15.csv"),
        start_date=timeframes.DATALOAD_START,
        end_date=timeframes.DATALOAD_END,
        ## Without filtering weekends some indicators like ADX will just produce Nan values for all bars after the first weekend, which breaks the whole training pipeline. This is because ADX (and some other indicators) require a continuous series of bars to calculate their values, and the gap caused by weekends leads to NaNs that propagate forward. By filtering out weekends, we ensure that the indicators can be calculated correctly and produce valid feature values for all bars.
        filter_weekends_flag=True
    )
    df_4hours = csv.load_csv(
        os.path.join(dir_config.DATA_DIR, "eurusd_4hours.csv"),
        start_date=timeframes.DATALOAD_START,
        end_date=timeframes.DATALOAD_END,
        ## Without filtering weekends some indicators like ADX will just produce Nan values for all bars after the first weekend, which breaks the whole training pipeline. This is because ADX (and some other indicators) require a continuous series of bars to calculate their values, and the gap caused by weekends leads to NaNs that propagate forward. By filtering out weekends, we ensure that the indicators can be calculated correctly and produce valid feature values for all bars.
        filter_weekends_flag=True
    )
    df_daily = csv.load_csv(
        os.path.join(dir_config.DATA_DIR, "eurusd_daily.csv"),
        start_date=timeframes.DATALOAD_START,
        end_date=timeframes.DATALOAD_END,
        ## Without filtering weekends some indicators like ADX will just produce Nan values for all bars after the first weekend, which breaks the whole training pipeline. This is because ADX (and some other indicators) require a continuous series of bars to calculate their values, and the gap caused by weekends leads to NaNs that propagate forward. By filtering out weekends, we ensure that the indicators can be calculated correctly and produce valid feature values for all bars.
        filter_weekends_flag=True
    )

    print(f"M15: {len(df_m15)} rows, 4Hours: {len(df_4hours)} rows, Daily: {len(df_daily)} rows")

    return df_m15, df_4hours, df_daily


def calculate_features(df_m15, df_4hours, df_daily):
    """Calculate technical indicators and combine timeframes."""
    print("\n" + "=" * 80)
    print("CALCULATING FEATURES IN ALL TIMEFRAMES")
    print("=" * 80 + "\n")

    ## Features are now correctly shifted. 
    ## - m15 probably also needs no shift. 
    #  - 4hours and daily don't need a shift because they are forward-filled onto M15 after shifting to bar-close time, which aligns with how Java uses the features.
    # This ensures that all features are applied at the correct times without any lookahead bias.
    features_m15 = features.add_features(df_m15, "m15", apply_shift=False).add_prefix("m15_")
    if 'm15_adx' in features_m15.columns:
        print(f"ADX print 2 max value: {features_m15['m15_adx'].max():.2f}, min value: {features_m15['m15_adx'].min():.2f}, mean value: {features_m15['m15_adx'].mean():.2f}")
    features_4hours = features.add_features(df_4hours, "4hours", apply_shift=False).add_prefix("4hours_")
    features_daily = features.add_features(df_daily, "daily", apply_shift=False).add_prefix("daily_")

    # Align 4hours and daily to M15.
    # Dukascopy bar timestamps are bar-open times. Java only uses a bar's features after it
    # closes (via getPreviousBarStart), so we shift each index to bar-close time before
    # forward-filling — otherwise Python applies new features up to N hours too early.
    # Daily bars close at 22:00 UTC (NY session); 4-hour bars close 4 h after open.
    features_4hours_shifted = features_4hours.copy()
    features_4hours_shifted.index = features_4hours_shifted.index + pd.Timedelta(hours=4)
    features_daily_shifted = features_daily.copy()
    features_daily_shifted.index = features_daily_shifted.index + pd.Timedelta(hours=24)

    hours4_aligned = features_4hours_shifted.reindex(features_m15.index, method="ffill")
    daily_aligned = features_daily_shifted.reindex(features_m15.index, method="ffill")

    # Combine
    combined = pd.concat([features_m15, hours4_aligned, daily_aligned], axis=1)
    combined = datahandling.remove_duplicates(combined, "combined")

    # Cross-timeframe features (mtf_*): need columns from more than one
    # timeframe, so they can only exist after the merge. No-op unless enabled.
    # strict: fail here with the real cause instead of an all-NaN feature that
    # surfaces later as "No rows without NaN values found" in the alignment.
    combined = features.add_cross_timeframe_features(combined, strict=True)

    return combined, features_m15


def report_nan_columns(X, top=25):
    """Diagnosis for the 'No rows without NaN values found' abort: names the
    columns whose NaN share makes a complete-case row impossible, so a
    misconfigured features config points at itself instead of at 'feature
    calculations' in general.

    Known structural causes (documented in CLAUDE.md):
      - A9 volume/spread members need the M15 'mid' column, which the training
        loaders drop (keep_mid=False) -> all-NaN.
      - A10 cross-pair members join the nine pair M15 files exactly (never
        ffilled); the pair data has a coverage hole ~2023-02..2025-12 -> all-NaN
        inside it.
    """
    share = X.isna().mean().sort_values(ascending=False)
    all_nan = share[share >= 0.999999]
    partial = share[(share > 0.5) & (share < 0.999999)]
    lines = [f"\nNaN diagnosis over {len(X):,} rows x {X.shape[1]} feature columns:"]
    if len(all_nan):
        lines.append(f"  ALL-NaN columns ({len(all_nan)}): "
                     + ", ".join(all_nan.index[:top])
                     + (" ..." if len(all_nan) > top else ""))
    if len(partial):
        lines.append(f"  >50% NaN columns ({len(partial)}):")
        for name, s in partial.head(top).items():
            lines.append(f"    {name}: {s:.1%}")
    if not len(all_nan) and not len(partial):
        lines.append("  no single column dominates - the NaN pattern is a union "
                     "of many warm-ups; inspect X.isna().mean() manually")
    lines.append("  Hint: training-only families with unmet prerequisites are the "
                 "usual cause - A9 volume/spread needs keep_mid at the load "
                 "sites; A10 cross-pair data has a ~2023-02..2025-12 coverage "
                 "hole (see CLAUDE.md).")
    return "\n".join(lines)


# The label series proper. `metadata` is also part of the contract below, but it is the
# targets frame, not a label — anything iterating labels to compute a per-key statistic
# must use this tuple instead of the dict keys.
_LABEL_SERIES_KEYS = ('long_fast', 'short_fast', 'long_slow', 'short_slow', 'reg')
_LABELS_REQUIRED_KEYS = _LABEL_SERIES_KEYS + ('metadata',)


def _validate_labels_contract(labels, index, label_mode):
    """Every label-mode branch in generate_labels() must produce this shape.

    Catches a broken/incomplete mode implementation early instead of failing
    downstream with a confusing KeyError/shape-mismatch during training.
    """
    missing = [k for k in _LABELS_REQUIRED_KEYS if k not in labels]
    if missing:
        raise ValueError(
            f"[label-mode={label_mode}] generate_labels() output is missing keys {missing}; "
            f"expected all of {_LABELS_REQUIRED_KEYS}"
        )
    for key in _LABEL_SERIES_KEYS:
        series = labels[key]
        if not isinstance(series, pd.Series):
            raise ValueError(f"[label-mode={label_mode}] labels['{key}'] must be a pd.Series, got {type(series)}")
        if not series.index.equals(index):
            raise ValueError(f"[label-mode={label_mode}] labels['{key}'] index does not match df_m15 index")


def generate_labels(df_m15, args=None, regime_labels=None, df_daily=None):
    """
    Generate training targets/labels for both the "fast" and "slow" models.

    This function is a thin orchestration layer around:
      1) `calculate_dynamic_targets(...)`  -> defines per-bar TP/SL + horizon (esp. for slow labels)
      2) `generate_dynamic_labels(...)`    -> scans forward and emits label series for:
           - long_fast / short_fast (entry-quality labels)
           - long_slow / short_slow (TP/SL outcome labels)
           - reg (aux regression target, depends on your dynamic_labels implementation)

    Inputs
    ------
    df_m15 : pd.DataFrame
        M15 OHLC data indexed by time (must contain at least open/high/low/close for label generation).

    args : argparse.Namespace | None
        Optional CLI args that influence LabelConfig. Supported arguments:
          - --label-mode: 'static' or 'atr_scaled'
              * 'static'    -> slow targets are fixed in pips (see pip_target/stop_pips below)
              * 'atr_scaled'-> slow targets are derived from ATR (see ATR parameters below)
          - --atr-multiplier: float
              ATR multiplier for the slow-model profit target (only in atr_scaled mode).
          - --atr-stop-multiplier: float
              ATR multiplier for the slow-model stop distance (only in atr_scaled mode).
          - --use-regime-labels: flag
              If set AND label-mode is atr_scaled AND regime_labels is provided, the ATR-scaled
              targets may be adjusted per regime (implementation lives in calculate_dynamic_targets).
          - --mfe-threshold: float
              Threshold (in pips) for the fast-model "MFE-before-MAE" label.
          - --mfe-horizon: int
              Lookahead (in bars) used to evaluate fast labels (e.g. 48 bars = 12h on M15).

        If args is None, defaults are used (static slow labels + mfe-threshold=15, mfe-horizon=48).

    regime_labels : pd.DataFrame | None
        Optional regime annotation aligned to time (e.g. trend/range, high/low volatility).
        It is only used when:
          - args.use_regime_labels is True, AND
          - label-mode == 'atr_scaled'
        In that case it is forwarded to `calculate_dynamic_targets(...)` which can adjust
        ATR-scaled target/stop sizing by regime.

    LabelConfig (what is defined here)
    ----------------------------------
    The function constructs a `LabelConfig` with:
      - mode:
          'static' or 'atr_scaled' (from args.label_mode)
      - Slow label base targets (static mode):
          * pip_target = x pips (take-profit distance)
          * stop_pips  = x pips  (stop-loss distance)
      - Slow label adaptive targets (atr_scaled mode):
          * atr_period = 14
          * atr_target_multiplier = args.atr_multiplier (default 2.5)
          * atr_stop_multiplier   = args.atr_stop_multiplier (default 0.875)
          * plus internal min/max bounds exposed via:
              label_config.min_target_pips, label_config.max_target_pips,
              label_config.min_stop_pips,   label_config.max_stop_pips
      - Slow label evaluation horizon:
          * horizon_min = 144 bars (36h on M15)
          * horizon_max = 288 bars (72h on M15)
        Interpretation: the slow outcome labels are determined by whether price hits the
        per-bar TP/SL within the allowed lookahead window (bounded by min/max horizon).
      - Fast label definition ("MFE-before-MAE"):
          * mfe_threshold_pips = args.mfe_threshold (default 20)
          * mfe_horizon        = args.mfe_horizon (default 96 bars)
        Interpretation: the fast labels are "entry quality" signals computed by scanning
        forward up to `mfe_horizon` bars and checking whether price achieves at least the
        required favorable excursion (MFE) before exceeding the adverse excursion (MAE),
        using the threshold in pips. (Exact mechanics are in `generate_dynamic_labels`.)

    ATR + Regime adjustment (where it happens)
    ------------------------------------------
    - ATR sizing and any regime-aware scaling is applied when building `targets_df` via
      `calculate_dynamic_targets(...)`.
    - `targets_df` is then consumed by `generate_dynamic_labels(...)` to produce the final
      label series (fast + slow) for each timestamp.

    Returns
    -------
    (labels, pip_value, targets_df, raw_labels) : tuple
      labels : dict[str, pd.Series]
          A dict of aligned label Series (index = df_m15 timestamps), expected keys:
            - 'long_fast', 'short_fast' : fast entry-quality binary labels
            - 'long_slow', 'short_slow' : slow TP/SL outcome binary labels
            - 'reg'                     : auxiliary regression target (may contain NaNs)
          If denoising is enabled, these labels have been post-processed to suppress
          isolated counter-trend signals.
      pip_value : float
          Pip value for the symbol (EURUSD) used to convert between price moves and pips.
      targets_df : pd.DataFrame
          Per-timestamp target specification produced by `calculate_dynamic_targets(...)`
          (e.g., dynamic TP/SL distances and horizons). This is what the label generator
          uses as the ground-truth target definition (especially for slow labels).
      raw_labels : dict[str, pd.Series] or None
          If denoising is enabled, contains the original un-denoised labels for comparison
          and CSV export review. None if denoising is disabled.
    """
    print("\n" + "=" * 80)
    print("GENERATING TARGET LABELS")
    print("=" * 80 + "\n")

    symbol = "EURUSD"
    pip_value = forex.pip_value_for_symbol(symbol)

    # Configure label generation based on args (or use defaults)
    label_mode = getattr(args, 'label_mode', 'static') if args else 'static'
    lookahead_horizon   = getattr(args, 'lookahead_horizon',   96)   if args else 96
    lookahead_pct       = getattr(args, 'lookahead_pct',       20.0) if args else 20.0
    lookahead_lookback  = getattr(args, 'lookahead_lookback',  2880) if args else 2880
    lookahead_stop_pips = getattr(args, 'lookahead_stop_pips', 65.0) if args else 65.0
    direction_horizon   = getattr(args, 'direction_horizon',        192)  if args else 192
    direction_dead_zone = getattr(args, 'direction_dead_zone_pips', 12.0) if args else 12.0
    label_file          = getattr(args, 'label_file',   None) if args else None
    label_column        = getattr(args, 'label_column', None) if args else None
    daily_vol_span = getattr(args, 'daily_vol_span', 100) if args else 100
    slow_hysteresis_multiplier = getattr(args, 'slow_hysteresis_multiplier', 0.0) if args else 0.0
    atr_multiplier = getattr(args, 'atr_multiplier', 1.5) if args else 1.5
    atr_stop_multiplier = getattr(args, 'atr_stop_multiplier', 0.5) if args else 0.5
    atr_smoothing_period = getattr(args, 'atr_smoothing_period', 1920) if args else 1920
    use_regime = getattr(args, 'use_regime_labels', False) if args else False
    mfe_threshold = getattr(args, 'mfe_threshold', 20.0) if args else 20.0
    mfe_horizon = getattr(args, 'mfe_horizon', 96) if args else 96
    denoise_enabled = getattr(args, 'denoise', False) if args else False
    denoise_window_fraction = getattr(args, 'denoise_window_fraction', 0.5) if args else 0.5
    denoise_threshold = getattr(args, 'denoise_threshold', 0.65) if args else 0.65
    denoise_trend_aggression = getattr(args, 'denoise_trend_aggression', 0.8) if args else 0.8

    pip_target = getattr(args, 'pip_target', None) if args else None
    stop_pips = getattr(args, 'stop_pips', None) if args else None
    horizon_min = getattr(args, 'horizon_min', None) if args else None
    horizon_max = getattr(args, 'horizon_max', None) if args else None

    # LabelConfig mode mapping:
    # - 'lookahead'          → pass 'static' (slow labels replaced afterwards)
    # - 'daily_vol_scaled'   → pass 'static' (slow labels use external daily volatility targets)
    # - 'trend_only'         → pass 'static' (static targets used for trend bars; range bars zeroed in routing)
    # - 'regime_conditional' → passed through directly
    # - 'window_cascade'     → pass 'static' (LabelConfig not used; window_labels handles everything)
    # - 'direction_horizon'  → pass 'static' (fast labels/reg/metadata still come from
    #                          generate_dynamic_labels; slow labels overridden afterwards)
    # - 'file'               → pass 'static' (no labels are computed at all; the static
    #                          targets_df only serves as `metadata` / label_targets.parquet)
    # - all others           → passed through directly
    label_config = LabelConfig(
        mode=label_mode if label_mode not in ('lookahead', 'daily_vol_scaled', 'window_cascade', 'trend_only', 'direction_horizon', 'file') else 'static',
        pip_target=pip_target if pip_target is not None else 80,
        stop_pips=stop_pips if stop_pips is not None else 35,
        atr_period=14,
        atr_smoothing_period=atr_smoothing_period,
        atr_target_multiplier=atr_multiplier,
        atr_stop_multiplier=atr_stop_multiplier,
        slow_hysteresis_multiplier=slow_hysteresis_multiplier,
        horizon_min=horizon_min if horizon_min is not None else 96,  # 24h
        horizon_max=horizon_max if horizon_max is not None else 384,  # 96h
        mfe_threshold_pips=mfe_threshold,
        mfe_horizon=mfe_horizon,
        denoise_enabled=denoise_enabled,
        denoise_window_fraction=denoise_window_fraction,
        denoise_dominance_threshold=denoise_threshold,
        denoise_trend_aggression=denoise_trend_aggression,
    )

    print(f"Label generation mode: {label_mode}")
    if label_mode != 'file':
        print(f"  Fast labels: MFE-before-MAE ({label_config.mfe_threshold_pips} pips, {label_config.mfe_horizon} bars)")
    if label_mode == 'file':
        print(f"  All labels: read from {label_file} (no label computation)")
        if label_column:
            print(f"  Direction column: {label_column}")
    elif label_config.mode == 'atr_scaled':
        print(f"  Slow labels: ATR-scaled (multiplier={label_config.atr_target_multiplier}, "
              f"stop={label_config.atr_stop_multiplier})")
        print(f"  Target bounds: {label_config.min_target_pips}-{label_config.max_target_pips} pips")
        print(f"  Stop bounds: {label_config.min_stop_pips}-{label_config.max_stop_pips} pips")
        if use_regime and regime_labels is not None:
            print(f"  Regime adjustments: ENABLED")
        else:
            print(f"  Regime adjustments: DISABLED")
    elif label_mode == 'lookahead':
        print(f"  Slow labels: Lookahead percentile ({lookahead_horizon} bars, "
              f"top/bot {lookahead_pct:.0f}%, lookback {lookahead_lookback} bars)")
    elif label_mode == 'daily_vol_scaled':
        print(f"  Slow labels: Daily-vol-scaled (daily EWM span={daily_vol_span}, "
              f"target_mult={label_config.atr_target_multiplier}, stop_mult={label_config.atr_stop_multiplier}, "
              f"hysteresis_mult={label_config.slow_hysteresis_multiplier})")
    elif label_mode == 'direction_horizon':
        print(f"  Slow labels: Direction-horizon ({direction_horizon} bars, "
              f"dead zone +/-{direction_dead_zone:.1f} pips)")
    elif label_mode == 'regime_conditional':
        print(f"  Trend bars: Static target/stop ({label_config.pip_target} pips / {label_config.stop_pips} pips stop)")
        print(f"  Range bars: BB mean-reversion (period={label_config.mean_rev_bb_period}, "
              f"std={label_config.mean_rev_bb_std}, stop={label_config.mean_rev_stop_pips} pips)")
    elif label_mode == 'window_cascade':
        timing_entry = getattr(args, 'timing_entry', 'fhl') if args else 'fhl'
        mfe_pips = float(pip_target) if pip_target is not None else 75.0
        _hmax = horizon_max if horizon_max is not None else 288
        print(f"  Setup model (slow): regime-gated windows, MFE >= {mfe_pips:.0f} pips, "
              f"horizon {_hmax} bars")
        print(f"  Timing model (fast): {timing_entry.upper()} entry strategy")
    elif label_mode == 'trend_only':
        print(f"  Trend bars: Static target/stop ({label_config.pip_target} pips / {label_config.stop_pips} pips stop)")
        print(f"  Range bars: label=0 (no entry signal)")
    else:
        print(f"  Slow labels: Static ({label_config.pip_target} pips target, "
              f"{label_config.stop_pips} pips stop)")
    if label_config.denoise_enabled:
        print(f"  Denoising: ENABLED (window_frac={label_config.denoise_window_fraction}, "
              f"threshold={label_config.denoise_dominance_threshold}, "
              f"trend_aggression={label_config.denoise_trend_aggression})")
    else:
        print(f"  Denoising: DISABLED")

    # Ensure df_m15 is sorted chronologically
    try:
        df_m15 = df_m15.sort_index()
    except Exception as e:
        print(f"[ERROR] Failed to sort df_m15: {type(e).__name__}: {e}")
        raise

    # Calculate dynamic targets (pass regime_labels if enabled)
    try:
        regime_for_targets = regime_labels if (use_regime and regime_labels is not None) else None
    except Exception as e:
        print(f"[ERROR] Failed to set regime_for_targets: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        if label_mode == 'daily_vol_scaled':
            if df_daily is None:
                raise ValueError(
                    "--label-mode daily_vol_scaled requires df_daily to be passed "
                    "to generate_labels()."
                )

            daily_sorted = df_daily.sort_index()
            if 'close' not in daily_sorted.columns:
                raise ValueError("df_daily must contain 'close' column for daily_vol_scaled mode")

            daily_returns = np.log(daily_sorted['close'] / daily_sorted['close'].shift(1))
            daily_vol = daily_returns.ewm(span=daily_vol_span).std().shift(1)
            daily_vol_to_m15 = daily_vol.reindex(df_m15.index, method='ffill')

            targets_df = calculate_volatility_scaled_targets(
                df_m15,
                label_config,
                volatility_series=daily_vol_to_m15,
                pip_value=pip_value,
                symbol=symbol,
            )
        else:
            targets_df = calculate_dynamic_targets(
                df_m15, label_config, regime_labels=regime_for_targets,
                pip_value=pip_value, symbol=symbol
            )
    except Exception as e:
        print(f"[ERROR] Failed in calculate_dynamic_targets: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Generate labels — routing depends on label mode
    if label_mode == 'window_cascade':
        timing_entry = getattr(args, 'timing_entry', 'fhl') if args else 'fhl'
        mfe_pips_wc = float(pip_target) if pip_target is not None else 75.0
        hmax_wc = horizon_max if horizon_max is not None else 288
        # Window validation accepts `mfe >= mfe_pips OR mfe >= atr_mult * ATR(14)`. At the
        # 2.5 default the ATR branch binds at ~23 pips on EURUSD M15 and a larger
        # --pip-target never takes effect; --window-atr-mult makes that choice explicit.
        atr_mult_wc = float(getattr(args, 'window_atr_mult', 2.5)) if args else 2.5

        print("\n  [window_cascade] Computing regime windows (long)...")
        windows_long = compute_regime_windows(df_m15, direction='long')
        print("  [window_cascade] Computing regime windows (short)...")
        windows_short = compute_regime_windows(df_m15, direction='short')

        print(f"  [window_cascade] Generating setup labels "
              f"(MFE >= {mfe_pips_wc:.0f} pips OR >= {atr_mult_wc:g} x ATR)...")
        setup_long = generate_setup_labels(df_m15, windows_long,
                                           mfe_pips=mfe_pips_wc, atr_mult=atr_mult_wc,
                                           horizon_max=hmax_wc,
                                           direction='long', symbol=symbol)
        setup_short = generate_setup_labels(df_m15, windows_short,
                                            mfe_pips=mfe_pips_wc, atr_mult=atr_mult_wc,
                                            horizon_max=hmax_wc,
                                            direction='short', symbol=symbol)

        print(f"  [window_cascade] Generating timing labels ({timing_entry})...")
        if timing_entry == 'fhl':
            timing_long = generate_timing_labels_fhl(df_m15, setup_long, windows_long,
                                                     mfe_pips=mfe_pips_wc, horizon_max=hmax_wc,
                                                     direction='long', symbol=symbol)
            timing_short = generate_timing_labels_fhl(df_m15, setup_short, windows_short,
                                                      mfe_pips=mfe_pips_wc, horizon_max=hmax_wc,
                                                      direction='short', symbol=symbol)
        else:  # rebound
            timing_long = generate_timing_labels_rebound(df_m15, setup_long, windows_long,
                                                         mfe_pips=mfe_pips_wc, horizon_max=hmax_wc,
                                                         direction='long', symbol=symbol)
            timing_short = generate_timing_labels_rebound(df_m15, setup_short, windows_short,
                                                          mfe_pips=mfe_pips_wc, horizon_max=hmax_wc,
                                                          direction='short', symbol=symbol)

        print_window_label_stats(setup_long, timing_long, 'long')
        print_window_label_stats(setup_short, timing_short, 'short')

        labels = {
            'long_slow':  setup_long.astype(float),
            'short_slow': setup_short.astype(float),
            'long_fast':  timing_long.astype(float),
            'short_fast': timing_short.astype(float),
            'reg':        pd.Series(np.nan, index=df_m15.index, dtype=float),
            'metadata':   targets_df,
        }

    elif label_mode == 'regime_conditional':
        if regime_labels is None:
            raise ValueError(
                "--label-mode regime_conditional requires regime_labels to be passed "
                "to generate_labels(). Generate them with generate_regime_labels() first."
            )
        try:
            labels = generate_regime_conditional_labels(
                df_m15, targets_df, label_config, regime_labels, symbol, verbose=True
            )
        except Exception as e:
            print(f"[ERROR] Failed in generate_regime_conditional_labels: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    elif label_mode == 'file':
        if not label_file:
            raise ValueError(
                "--label-mode file requires --label-file <path to parquet> "
                "(args.label_file was empty)."
            )
        file_labels = load_labels_from_file(
            label_file, df_m15.index, label_column=label_column, verbose=True
        )
        labels = {
            **file_labels,
            'reg':      pd.Series(np.nan, index=df_m15.index, dtype=float),
            'metadata': targets_df,
        }

    elif label_mode == 'trend_only':
        if regime_labels is None:
            raise ValueError(
                "--label-mode trend_only requires regime_labels to be passed "
                "to generate_labels(). Generate them with generate_regime_labels() first."
            )
        try:
            labels = generate_trend_only_labels(
                df_m15, targets_df, label_config, regime_labels, symbol, verbose=True
            )
        except Exception as e:
            print(f"[ERROR] Failed in generate_trend_only_labels: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        labels = generate_dynamic_labels(df_m15, targets_df, label_config, symbol, verbose=True)

    # Lookahead mode: replace slow labels with percentile-ranked forward return labels
    if label_mode == 'lookahead':
        print(f"\n  [lookahead] Replacing slow labels with percentile forward-return labels...")
        lookahead_result = generate_lookahead_slow_labels(
            df_m15,
            horizon=lookahead_horizon,
            top_pct=lookahead_pct,
            lookback_window=lookahead_lookback,
            stop_pips=lookahead_stop_pips,
            min_pip_target=getattr(args, 'lookahead_min_pips', 30.0),
            verbose=True,
        )
        labels['long_slow']  = lookahead_result['long_slow']
        labels['short_slow'] = lookahead_result['short_slow']

    # Direction-horizon mode: replace slow labels with fixed-horizon sign-of-return labels
    if label_mode == 'direction_horizon':
        print(f"\n  [direction_horizon] Replacing slow labels with fixed-horizon sign-of-return labels...")
        direction_result = generate_direction_horizon_labels(
            df_m15,
            horizon=direction_horizon,
            dead_zone_pips=direction_dead_zone,
            pip_value=pip_value,
            verbose=True,
        )
        labels['long_slow']  = direction_result['long_slow']
        labels['short_slow'] = direction_result['short_slow']

    # Apply label denoising if enabled
    raw_labels = None
    denoise_stats = {}
    if label_config.denoise_enabled:
        # Save raw labels before denoising (for review export)
        raw_labels = {
            'long_slow': labels['long_slow'].copy(),
            'short_slow': labels['short_slow'].copy(),
            'long_fast': labels['long_fast'].copy(),
            'short_fast': labels['short_fast'].copy(),
        }
        labels, denoise_stats = denoise_labels(
            labels, label_config,
            regime_labels=regime_labels,
            verbose=True
        )

    # Keep fast labels only where the matching slow label exists.
    labels = mask_fast_labels_by_slow(labels)

    # Print target statistics (for dynamic modes)
    if label_config.mode != 'static' or label_mode == 'daily_vol_scaled':
        print_target_statistics(targets_df)

    # Print label distribution
    print_label_distribution(labels)

    # Validate and warn about distribution issues
    validation = validate_label_distribution(labels)
    if validation['warnings']:
        print("\nLabel distribution warnings:")
        for warning in validation['warnings']:
            print(f"  {warning}")

    # Print weekly label breakdown (summary + outlier weeks only)
    weekly_df = analyze_weekly_labels(labels, df_m15.index)
    print_weekly_label_summary(weekly_df)

    _validate_labels_contract(labels, df_m15.index, label_mode)

    # The forward horizon the labels actually peeked over, in M15 bars. Carried on the
    # frame rather than added to the return tuple so no caller signature changes. The CV
    # embargo (--cv-gap) is derived from it: a label at bar t is resolved from bars up to
    # t + horizon, so without a gap of at least this size the last rows of every training
    # fold were computed from bars that lie inside the validation fold.
    targets_df.attrs['label_horizon_bars'] = int(
        max(label_config.horizon_max, label_config.mfe_horizon)
    )

    return labels, pip_value, targets_df, raw_labels


def signal_regime_mask(regime_df, model_key, regime_aware, direction_aware):
    """Select the bars in which ``model_key``'s positive labels can occur.

    This mirrors the label semantics of ``generate_trend_only_labels`` /
    ``generate_regime_conditional_labels``:

    * Direction-agnostic regimes (``regime_trend`` in {1=trend, 0=range}): every
      trend bar carries both long and short labels, so both directions share the
      trend bucket.
    * Direction-aware regimes (``regime_trend`` in {1=uptrend, -1=downtrend,
      0=range}): long labels exist **only** in uptrend bars, short labels **only**
      in downtrend bars; the opposite trend direction is forced to 0.

    Getting the second case wrong is not cosmetic.  A short model's downtrend bars
    are where all of its positives live; bucketing them together with range bars
    hands MI a downtrend-vs-range *detection* problem — exactly the artefact that
    per-regime MI exists to avoid — while its "signal" bucket (uptrend) would be
    all-zero and score 0 for every feature.

    Returns:
        None when regime-aware MI is off, otherwise a Series the MI filter buckets
        with ``> 0.5``.  NaN regimes stay NaN and fall to the non-signal side.
    """
    if not regime_aware or regime_df is None or 'regime_trend' not in regime_df.columns:
        return None

    trend = regime_df['regime_trend']
    if not direction_aware:
        # {1=trend, 0=range} — the raw Series is already the signal mask.
        return trend

    target = 1 if model_key.startswith('long') else -1
    # astype(float) + where() keeps NaN as NaN so the MI filter's NaN-safe `> 0.5`
    # still buckets unknown-regime bars as non-signal.
    return trend.eq(target).astype(float).where(trend.notna())


def fast_setup_row_index(index, y_fast, model_key, conditional):
    """The M15 bars a fast model trains on.

    ``conditional=False`` (the default pipeline behaviour) returns ``index``
    unchanged: the fast model sees every bar, and since its positives can only
    occur where the matching slow label is 1, it is really a joint
    P(direction AND timing) model — it has to re-derive the direction question
    before the timing question can pay off.

    ``conditional=True`` (--fast-conditional-on-setup) restricts the rows to that
    model's setup bars, so the label it fits is P(timing | setup) — the question
    the live entry gate asks, and the one whose precision is readable as the hit
    rate of the trades actually taken.

    Args:
        index: DatetimeIndex of the fast training frame.
        y_fast: dict of label Series (needs 'long_slow' / 'short_slow').
        model_key: 'long_fast' or 'short_fast'.
        conditional: whether to restrict to the setup bars.

    Returns:
        The index to train on (``index`` itself when ``conditional`` is False).
    """
    if not conditional:
        return index

    slow_key = 'long_slow' if model_key.startswith('long') else 'short_slow'
    if slow_key not in y_fast:
        raise ValueError(
            f"--fast-conditional-on-setup needs '{slow_key}' to condition '{model_key}' on, "
            f"but the fast label set only has {sorted(y_fast)}."
        )

    setup = y_fast[slow_key].reindex(index).fillna(0)
    keep = index[setup.values == 1]
    if len(keep) == 0:
        raise ValueError(
            f"--fast-conditional-on-setup: '{slow_key}' has no positive bar in the fast "
            f"training scope, so '{model_key}' would train on an empty frame. Check the "
            f"label mode and the training window."
        )
    return keep


def run_feature_selection(X_train, y_train, args, output_dir, model=None, regime_mask=None):
    """Run feature selection pipeline.

    Args:
        X_train: Feature matrix (DataFrame).
        y_train: Target labels (Series).
        args: Parsed CLI args with mi_threshold, pfi_threshold, skip_mi, skip_pfi, cv_splits.
        output_dir: Directory to write MI/PFI score CSVs.
        model: Pre-trained model used for PFI (optional).
        regime_mask: Optional boolean Series aligned to X_train's index where
            True = trend bar, False = range bar.  When provided, MI is computed
            independently per regime and the union of selected features is kept.
            Should only be passed when label_mode == 'regime_conditional'.
    """
    print("\n" + "=" * 80)
    print("FEATURE SELECTION PIPELINE"
          + (" [regime-aware MI]" if regime_mask is not None else ""))
    print("=" * 80 + "\n")

    results = {}
    selected_features = list(X_train.columns)

    # 1. Mutual Information Filter
    if not args.skip_mi:
        print("[Step 1] Running Mutual Information Filter...")
        mi_filter = MutualInformationFilter(
            min_mi=args.mi_threshold,
            task='classification',
            n_permutations=getattr(args, 'mi_permutations', 20),
            legacy=getattr(args, 'legacy_mi', False),
        )
        mi_filter.fit(X_train, y_train, regime_mask=regime_mask)
        mi_filter.print_summary()

        # Save MI results. `*_raw` are the undeduplicated scores kept only so old runs
        # stay comparable — they are tie-inflated for any feature carried from a
        # coarser timeframe; see MutualInformationFilter's docstring.
        mi_results = mi_filter.get_results()
        mi_path = os.path.join(output_dir, "mi_scores.csv")
        mi_df = mi_results['mi_scores'].to_frame('mi_score')
        mi_df['mi_informative'] = mi_results['mi_informative']
        if 'mi_scores_trend' in mi_results:
            mi_df['mi_score_trend'] = mi_results['mi_scores_trend']
            mi_df['mi_null_trend'] = mi_results['mi_null_trend']
            mi_df['n_eff_trend'] = mi_results['n_eff_trend']
            mi_df['mi_score_range'] = mi_results['mi_scores_range']
            mi_df['mi_null_range'] = mi_results['mi_null_range']
            mi_df['n_eff_range'] = mi_results['n_eff_range']
            mi_df['mi_score_trend_raw'] = mi_results['mi_scores_trend_raw']
            mi_df['mi_score_range_raw'] = mi_results['mi_scores_range_raw']
        else:
            mi_df['mi_null'] = mi_results['mi_null_trend']
            mi_df['n_eff'] = mi_results['n_eff_trend']
        mi_df.to_csv(mi_path)
        print(f"MI scores saved to: {mi_path}")

        selected_features = mi_filter.selected_features_
        results['mi_filter'] = mi_results
    
    # 2. Permutation Feature Importance
    if not args.skip_pfi and model is not None:
        print("\n[Step 2] Running Permutation Feature Importance...")
        pfi_filter = PermutationImportanceFilter(
            min_importance=args.pfi_threshold,
            n_cv_splits=args.cv_splits
        )
        
        X_for_pfi = X_train[selected_features]
        pfi_filter.fit(model, X_for_pfi, y_train, use_cv=True)
        pfi_filter.print_summary()
        
        # Save PFI results
        pfi_results = pfi_filter.get_results()
        pfi_path = os.path.join(output_dir, "pfi_scores.csv")
        pd.DataFrame({
            'feature': pfi_results['importance_scores'].index,
            'importance': pfi_results['importance_scores'].values,
            'std': pfi_results['importance_std'].values
        }).to_csv(pfi_path, index=False)
        print(f"PFI scores saved to: {pfi_path}")
        
        if len(pfi_filter.selected_features_) == 0:
            print(f"\n[!] PFI removed ALL {len(X_for_pfi.columns)} features -- falling back to MI-selected features.")
            print("   (All PFI scores were negative, likely due to insufficient pre-training rounds.)")
            # Keep MI-selected features unchanged
        else:
            selected_features = pfi_filter.selected_features_
        results['pfi_filter'] = pfi_results
    elif args.skip_pfi:
        print("\n[Step 2] Skipping Permutation Feature Importance (--skip-pfi)")
    
    print(f"\nFinal selected features: {len(selected_features)}")
    
    # Save selected feature list
    selected_path = os.path.join(output_dir, "selected_features.txt")
    with open(selected_path, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"Selected features saved to: {selected_path}")
    
    results['selected_features'] = selected_features
    
    return results


def _fmt_auc(val):
    """Format AUC-ROC value for display, returning 'N/A' for NaN."""
    return f"{val:.4f}" if not np.isnan(val) else "N/A"


def _none_if_nan(val):
    """JSON-safe float: NaN (e.g. AUC on a single-class fold) becomes None."""
    return None if val is None or np.isnan(val) else float(val)


def _safe_mcc(y_true, y_pred):
    """Matthews correlation coefficient at the caller's decision threshold.

    Reported next to F1/AUC because it is the one gate metric a collapsed model cannot
    pass by accident: a classifier predicting a single class everywhere scores exactly
    **0**, by construction. AUC scores 0.500 for such a model - a value a perfectly
    reasonable model also produces on a hard fold - and F1 stays visibly non-zero when
    the constant it predicts is the positive class. This project keeps meeting that
    failure (see `degenerate_folds` / `label_coverage`); MCC turns it from something
    inferred out of three numbers into one number that reads 0.

    Threshold-dependent like F1, so it is computed on the SAME `y_pred` the F1 and the
    confusion matrix use - never on a 0.5 default, which would make it incomparable with
    everything printed beside it.

    Returns 0.0 on the degenerate cases (empty input, or a zero denominator because one
    row of the confusion matrix is empty) - what the coefficient is defined to be there,
    and what sklearn returns anyway.
    """
    if len(y_true) == 0:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def _serialize_cv_metrics(cv_res, prefix):
    """Extract comprehensive CV metrics into a flat dict with given prefix.

    Aggregated metrics are computed globally on pooled CV val predictions using a
    precision-at-recall-K threshold rule (see run_time_series_cv). Per-fold lists
    are preserved for transparency. Converts NaN AUC-ROC values to None for JSON.
    """
    d = {
        f'{prefix}_global_val_f1': cv_res['global_val_f1'],
        f'{prefix}_mean_train_f1': cv_res['mean_train'],
        f'{prefix}_global_val_precision': cv_res['global_val_precision'],
        f'{prefix}_global_val_recall': cv_res['global_val_recall'],
        f'{prefix}_global_val_mcc': cv_res['global_val_mcc'],
        f'{prefix}_global_val_mcc_healthy_folds': cv_res.get('global_val_mcc_healthy_folds'),
        f'{prefix}_global_val_brier': cv_res['global_val_brier'],
        f'{prefix}_global_val_ece': cv_res['global_val_ece'],
        f'{prefix}_global_val_confusion_matrix': cv_res['global_val_confusion_matrix'],
        f'{prefix}_global_threshold': cv_res['global_threshold'],
        f'{prefix}_target_recall': cv_res['target_recall'],
        f'{prefix}_target_recall_trend': cv_res.get('target_recall_trend'),
        f'{prefix}_target_recall_range': cv_res.get('target_recall_range'),
        f'{prefix}_global_val_recall_trend': _none_if_nan(
            cv_res.get('global_val_recall_trend', float('nan'))),
        f'{prefix}_global_val_recall_range': _none_if_nan(
            cv_res.get('global_val_recall_range', float('nan'))),
        f'{prefix}_n_single_class_val_folds': cv_res.get('n_single_class_val_folds'),
        f'{prefix}_n_single_class_train_folds': cv_res.get('n_single_class_train_folds'),
        f'{prefix}_n_degenerate_folds': cv_res.get('n_degenerate_folds'),
        f'{prefix}_degenerate_folds': cv_res.get('degenerate_folds'),
        f'{prefix}_n_cv_folds': cv_res.get('n_cv_folds'),
        f'{prefix}_val_f1_per_fold': cv_res['val_scores'],
        f'{prefix}_val_precision_per_fold': cv_res['val_precisions'],
        f'{prefix}_val_recall_per_fold': cv_res['val_recalls'],
        f'{prefix}_val_mcc_per_fold': cv_res['val_mccs'],
        f'{prefix}_val_brier_per_fold': cv_res['val_brier_scores'],
        f'{prefix}_val_ece_per_fold': cv_res['val_eces'],
        f'{prefix}_val_confusion_matrices': cv_res['val_confusion_matrices'],
        f'{prefix}_val_thresholds_per_fold': cv_res['val_thresholds'],
    }
    d[f'{prefix}_global_val_auc_roc'] = _none_if_nan(cv_res['global_val_auc_roc'])
    d[f'{prefix}_global_val_auc_roc_healthy_folds'] = _none_if_nan(
        cv_res.get('global_val_auc_roc_healthy_folds', float('nan')))
    d[f'{prefix}_val_auc_roc_per_fold'] = [_none_if_nan(a) for a in cv_res['val_auc_rocs']]
    return d


# Columns carried alongside every out-of-fold score so a regime-conditioned evaluation
# needs no second training pass. `regime_combined` is what _print_per_regime_metrics
# partitions on; `regime_trend` is what tells a TREND bar from a RANGE bar.
OOF_REGIME_COLUMNS = ('regime_trend', 'regime_combined')


def build_oof_frame(cv_res, index, model_key, stage, regime_df=None):
    """Tabulate a CV run's out-of-fold validation scores.

    ``run_time_series_cv`` already forms the pooled validation predictions — it needs
    them for the global threshold, AUC and ECE — but then discards them once the
    aggregates are computed. Only the aggregates reach ``training_summary.json``, and an
    aggregate cannot be turned back into a ROC, a precision/recall curve, a reliability
    diagram or a confusion matrix at a different threshold. Persisting the scores makes
    every one of those drawable afterwards, as often as wanted, without retraining.

    The frame is deliberately long rather than wide: one row per (model, stage, bar), so
    the four models and the pre/post/final stages concatenate into a single file.

    Args:
        cv_res: the dict returned by run_time_series_cv.
        index: the frame index the CV ran over (bits['index']) — positional validation
            indices are resolved against it to recover the bar timestamps.
        model_key: one of MODEL_KEYS.
        stage: 'pre' | 'post' | 'final' — which point in the pipeline this measured.
        regime_df: optional frame aligned with ``index``; its OOF_REGIME_COLUMNS are
            attached so within-regime metrics can be computed against the raw outcome.

    Returns:
        pd.DataFrame, empty (with the right columns) when the CV produced no folds.
    """
    cols = ['timestamp', 'model', 'stage', 'fold', 'y_true', 'y_score']
    positions = np.asarray(cv_res.get('pooled_val_indices', []), dtype=int)
    y_true = np.asarray(cv_res.get('pooled_y_val', []))
    y_score = np.asarray(cv_res.get('pooled_val_prob', []))
    folds = np.asarray(cv_res.get('pooled_fold_ids', []), dtype=int)

    if len(positions) == 0 or len(y_true) != len(positions):
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame({
        'timestamp': pd.Index(index)[positions],
        'model': model_key,
        'stage': stage,
        'fold': folds if len(folds) == len(positions) else -1,
        'y_true': y_true.astype(np.int8),
        'y_score': y_score.astype(np.float32),
    })

    if regime_df is not None:
        for col in OOF_REGIME_COLUMNS:
            if col in regime_df.columns:
                out[col] = np.asarray(regime_df[col].iloc[positions].values)
    return out


def _compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.
    Lower ECE indicates better calibrated probability estimates.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    if n_total == 0:
        return 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)
    return ece


DEFAULT_SWEEP_THRESHOLDS = tuple(round(0.10 + 0.05 * i, 2) for i in range(13))  # 0.10..0.70


def _compute_threshold_sweep(y_true, y_prob, thresholds=DEFAULT_SWEEP_THRESHOLDS):
    """Return a list of dicts with precision/recall/F1/positive-rate per threshold.

    y_true / y_prob are 1-D arrays of equal length (typically pooled across CV folds).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    base_rate = float(y_true.mean()) if n > 0 else 0.0
    rows = []
    for thr in thresholds:
        y_pred = (y_prob > thr).astype(int)
        pos_count = int(y_pred.sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        prec = (tp / pos_count) if pos_count > 0 else 0.0
        actual_pos = int(y_true.sum())
        rec = (tp / actual_pos) if actual_pos > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        lift = (prec / base_rate) if base_rate > 0 else 0.0
        rows.append({
            'threshold': float(thr),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'predicted_positive_rate': pos_count / n if n > 0 else 0.0,
            'lift': lift,
        })
    return rows


def _extract_runs(arr):
    """Return list of (start, end) inclusive index pairs for contiguous 1-runs."""
    arr = np.asarray(arr)
    if len(arr) == 0:
        return []
    padded = np.concatenate(([0], arr.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _compute_event_metrics(y_true, y_pred, val_indices=None):
    """Event-level metrics for time series binary classification.

    val_indices: original positional indices of pooled samples — used to detect
    fold boundaries (non-consecutive indices = gap between folds) so events are
    never merged across folds.

    Returns dict with event_precision, event_recall, detection_lag_mean,
    mean_preds_per_event, n_actual_events, n_detected_events, n_predicted_events.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if val_indices is not None and len(val_indices) > 1:
        gaps = np.where(np.diff(val_indices) > 1)[0] + 1
        segs_y    = np.split(y_true, gaps)
        segs_pred = np.split(y_pred, gaps)
    else:
        segs_y    = [y_true]
        segs_pred = [y_pred]

    n_actual = n_detected = n_predicted = n_true_predicted = 0
    all_lags = []
    all_counts = []

    for seg_y, seg_pred in zip(segs_y, segs_pred):
        actual_runs = _extract_runs(seg_y)
        pred_runs   = _extract_runs(seg_pred)
        n_actual    += len(actual_runs)
        n_predicted += len(pred_runs)

        for s, e in actual_runs:
            preds_in = int(seg_pred[s:e + 1].sum())
            all_counts.append(preds_in)
            if preds_in > 0:
                n_detected += 1
                all_lags.append(int(np.argmax(seg_pred[s:e + 1])))

        for s, e in pred_runs:
            if any(a_s <= e and a_e >= s for a_s, a_e in actual_runs):
                n_true_predicted += 1

    ev_recall    = n_detected       / n_actual     if n_actual     > 0 else float('nan')
    ev_precision = n_true_predicted / n_predicted  if n_predicted  > 0 else float('nan')
    lag_mean     = float(np.mean(all_lags))   if all_lags   else float('nan')
    preds_mean   = float(np.mean(all_counts)) if all_counts else float('nan')

    return {
        'event_precision':      ev_precision,
        'event_recall':         ev_recall,
        'detection_lag_mean':   lag_mean,
        'mean_preds_per_event': preds_mean,
        'n_actual_events':      n_actual,
        'n_detected_events':    n_detected,
        'n_predicted_events':   n_predicted,
    }


def compute_per_regime_metrics(cv_results, regime_df):
    """Per-regime val metrics rows over the pooled CV predictions.

    Partitions the pooled val predictions by *regime_combined* (e.g. TREND_HIGH,
    RANGE_LOW) and computes F1, Precision, Recall, AUC-ROC, MCC and Brier per
    group, plus an 'Overall' row and a 'TREND (all)' aggregate — the operative
    population the strategy actually trades. Under trend_only/regime_conditional
    labels, range bars are all negatives and trivially separated, which inflates
    the Overall row; the TREND aggregate is the honest within-tradeable-population
    number. Decision threshold is the global threshold from cv_results (selected
    via precision-at-recall-K on pooled predictions).

    Returns (rows, threshold) — rows is a list of JSON-safe dicts (AUC is None on
    single-class groups), empty when there are no pooled predictions or no
    regime_combined column.
    """
    best_thr = float(cv_results.get('global_threshold', 0.5))
    indices = cv_results.get('pooled_val_indices')
    y_all   = cv_results.get('pooled_y_val')
    p_all   = cv_results.get('pooled_val_prob')
    if indices is None or len(indices) == 0 or regime_df is None or len(y_all) == 0:
        return [], best_thr
    if 'regime_combined' not in regime_df.columns:
        return [], best_thr

    regime_combined = regime_df['regime_combined'].iloc[indices].values

    def _row(name, y, p):
        if len(y) == 0:
            return None
        pred = (p >= best_thr).astype(int)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
        return {
            'regime': name,
            'n': int(len(y)),
            'pos_rate': float(np.mean(y)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'auc': _none_if_nan(auc),
            'mcc': _safe_mcc(y, pred),
            'brier': float(brier_score_loss(y, p)),
        }

    rows = []
    overall = _row('Overall', y_all, p_all)
    if overall:
        rows.append(overall)

    trend_mask = np.array([isinstance(r, str) and r.startswith('TREND') for r in regime_combined])
    if trend_mask.any():
        trend_row = _row('TREND (all)', y_all[trend_mask], p_all[trend_mask])
        if trend_row:
            rows.append(trend_row)

    for regime_name in sorted(set(r for r in regime_combined if isinstance(r, str))):
        mask = (regime_combined == regime_name)
        row = _row(regime_name, y_all[mask], p_all[mask])
        if row:
            rows.append(row)
    return rows, best_thr


def _print_per_regime_metrics(scope_label, cv_results, regime_df):
    """Print per-regime val metrics using pooled CV predictions."""
    rows, best_thr = compute_per_regime_metrics(cv_results, regime_df)
    if not rows:
        return

    def _fmt_row(row):
        auc_s = f"{row['auc']:.4f}" if row['auc'] is not None else "  N/A "
        return (f"    {row['regime']:<22} {row['n']:>6} {row['pos_rate']:>5.1%} "
                f"{row['f1']:>7.4f} {row['precision']:>7.4f} {row['recall']:>7.4f} "
                f"{auc_s:>7} {row['mcc']:>7.4f} {row['brier']:>7.4f}")

    print(f"\n  Per-regime metrics ({scope_label}, thr={best_thr:.3f}):")
    print(f"    {'Regime':<22} {'N':>6} {'Pos%':>5} {'F1':>7} {'Prec':>7} {'Recall':>7} "
          f"{'AUC':>7} {'MCC':>7} {'Brier':>7}")
    print("    " + "-" * 80)
    for row in rows:
        print(_fmt_row(row))


def _print_threshold_sweep(name, sweep, base_rate=None):
    """Pretty-print a threshold-sweep table."""
    header = f"  Thr   Prec    Recall  F1      PredPos  Lift"
    print(f"\n  [{name}]" + (f"  base_rate={base_rate:.3f}" if base_rate is not None else ""))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sweep:
        print(f"  {r['threshold']:.2f}  {r['precision']:.4f}  {r['recall']:.4f}  "
              f"{r['f1']:.4f}  {r['predicted_positive_rate']:.4f}   {r['lift']:.2f}x")


def _parse_sweep_thresholds(arg_value):
    """Parse the --sweep-thresholds CLI string. Returns the default tuple if None/empty."""
    if not arg_value:
        return DEFAULT_SWEEP_THRESHOLDS
    parts = [p.strip() for p in arg_value.split(',') if p.strip()]
    return tuple(float(p) for p in parts)


def _select_threshold_at_target_recall(y_true, y_prob, target_recall):
    """Pick the threshold that maximises precision subject to recall >= target_recall.

    Falls back to F1-optimal if no threshold on the PR curve reaches the target,
    or 0.5 if the validation set has only one class.
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns arrays of length n+1 for prec/rec and n for thresholds.
    # Pair the first n points with their thresholds.
    prec = prec_curve[:-1]
    rec = rec_curve[:-1]
    if len(thresholds) == 0:
        return 0.5
    eligible = rec >= target_recall
    if eligible.any():
        # Among thresholds meeting the recall floor, choose the one with max precision.
        # Ties: prefer the higher threshold (more conservative → higher precision in practice).
        eligible_idx = np.where(eligible)[0]
        best_local = eligible_idx[np.argmax(prec[eligible_idx])]
        return float(thresholds[best_local])
    # Fallback: F1-optimal
    f1_curve = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thresholds[np.argmax(f1_curve)]) if len(f1_curve) > 0 else 0.5


def _resolve_overall_target_recall(target_recall, target_recall_trend, target_recall_range):
    """Resolve the overall recall floor from the raw CLI values.

    ``--target-recall`` defaults to None so that "not given" is distinguishable from an
    explicit 0.5: without any floor the historical default 0.5 applies, but when only a
    regime-scoped floor is given the overall floor is dropped (0.0). Otherwise the
    silent 0.5 default dominates the regime floor — measured on regime_conditional
    (2026-09-06): ``--target-recall-trend 0.3`` alone landed at overall recall 0.53 /
    within-TREND 0.49 with the threshold dragged from 0.449 to 0.188 by the default,
    not by the trend floor the user asked for.
    """
    if target_recall is not None:
        return target_recall
    if target_recall_trend is not None or target_recall_range is not None:
        return 0.0
    return 0.5


def _select_threshold_with_recall_floors(y_true, y_prob, target_recall,
                                         regime_trend=None,
                                         target_recall_trend=None,
                                         target_recall_range=None):
    """Max precision s.t. overall recall >= target_recall AND every regime floor holds.

    ``regime_trend`` carries the label-generation regime per bar: TREND is
    ``regime_trend != 0`` (up- and downtrend pooled — under direction-aware regimes a
    long model's positives live in +1 bars and a short model's in -1 bars, but both are
    trend bars), RANGE is ``regime_trend == 0``; NaN regimes belong to neither scope.
    ``target_recall_trend`` / ``target_recall_range`` constrain the recall computed over
    the positives of that scope alone — the overall floor cannot see that a model earns
    its recall entirely on the trivially-separated range positives while catching none of
    the trend positives (measured: overall recall 0.30 at within-TREND recall 0.25).

    A scope with no positive labels cannot constrain anything and is skipped with a
    warning. If no threshold satisfies all floors simultaneously, the selection falls
    back to the overall-only rule (which itself falls back to F1-optimal). Without any
    regime floor this is exactly ``_select_threshold_at_target_recall``.
    """
    has_regime_floor = (regime_trend is not None
                        and (target_recall_trend is not None
                             or target_recall_range is not None))
    if not has_regime_floor:
        return _select_threshold_at_target_recall(y_true, y_prob, target_recall)
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_prob)
    if len(y_arr) == 0 or len(np.unique(y_arr)) < 2:
        return 0.5
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_arr, p_arr)
    prec = prec_curve[:-1]
    rec = rec_curve[:-1]
    if len(thresholds) == 0:
        return 0.5
    eligible = rec >= target_recall
    rt = np.asarray(regime_trend, dtype=float)
    for scope_name, scope_mask, floor in (
            ('TREND', (~np.isnan(rt)) & (rt != 0), target_recall_trend),
            ('RANGE', rt == 0, target_recall_range)):
        if floor is None:
            continue
        pos_scores = np.sort(p_arr[scope_mask & (y_arr == 1)])
        if len(pos_scores) == 0:
            print(f"  WARNING: no positive labels in the {scope_name} scope — the "
                  f"{scope_name} recall floor cannot constrain the threshold here.")
            continue
        # Scope recall at threshold t = share of scope positives scoring >= t
        # (precision_recall_curve pairs each threshold with `score >= thr` predictions).
        n_hit = len(pos_scores) - np.searchsorted(pos_scores, thresholds, side='left')
        eligible &= (n_hit / len(pos_scores)) >= floor
    if eligible.any():
        eligible_idx = np.where(eligible)[0]
        best_local = eligible_idx[np.argmax(prec[eligible_idx])]
        return float(thresholds[best_local])
    print("  WARNING: no threshold satisfies all recall floors simultaneously — "
          "falling back to the overall-only rule (max precision s.t. overall "
          f"recall >= {target_recall:.2f}).")
    return _select_threshold_at_target_recall(y_arr, p_arr, target_recall)


def _base_score_from_labels(y, eps=1e-3):
    """Compute base_score from label data: classifier -> mean(y) clipped; regressor -> mean(y).

    For binary targets (values in {0,1}) this returns the positive-class rate, clipped to
    (eps, 1-eps) to avoid degenerate logit at exactly 0 or 1. For continuous regression
    targets, returns the raw mean (NaNs ignored). Returns 0.5 if no valid labels.
    """
    y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
    if y_arr.dtype.kind == 'f':
        y_arr = y_arr[~np.isnan(y_arr)]
    if len(y_arr) == 0:
        return 0.5
    mean = float(np.mean(y_arr))
    uniq = np.unique(y_arr)
    is_binary = set(uniq.tolist()).issubset({0, 1, 0.0, 1.0})
    if is_binary:
        return float(np.clip(mean, eps, 1.0 - eps))
    return mean


def resolve_boost_rounds(label, fit_params, X, y, default_rounds,
                         es_rounds=None, es_val_frac=0.15, es_metric=None,
                         verbose=True):
    """Find the optimal num_boost_round via early stopping on a temporal holdout.

    The chronologically last `es_val_frac` of (X, y) is held out as the validation
    set; training runs up to `default_rounds` with `early_stopping_rounds`, and
    `best_iteration + 1` is returned so the caller can refit on the FULL dataset
    for exactly that many rounds. Refitting on full data (rather than slicing the
    booster) is what keeps the exported ONNX model — which ignores best_iteration —
    trimmed to the generalising depth.

    Early-stopping metric: for binary classifiers (`objective == binary:logistic`)
    the holdout is evaluated on AUC, NOT the training `eval_metric`. This matters
    because these models train with an aggressive `scale_pos_weight` that distorts
    predicted probabilities — logloss penalises that calibration shift and stalls at
    round 1 even while the ranking keeps improving. AUC is invariant to the spw
    distortion and matches the project's operative CV metric. `es_metric` overrides
    this; other objectives keep their own `eval_metric`.

    Patience: `es_rounds=None` (default) auto-derives it from the round cap as
    `max(20, default_rounds // 4)`, so it scales with each model's num_boost_round
    without a separate knob. A positive int overrides; `es_rounds <= 0` disables
    early stopping (returns `default_rounds` unchanged). Also returns
    `default_rounds` when the data is too small to carve a meaningful holdout, or
    when an AUC holdout would be single-class (AUC undefined).
    """
    if es_rounds is None:
        es_rounds = max(20, default_rounds // 4)
    if es_rounds <= 0:
        return default_rounds
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    n = len(y_arr)
    n_val = int(n * es_val_frac)
    if n_val < 20 or (n - n_val) < 50:
        if verbose:
            print(f"  [early-stop] {label}: skipped (only {n} bars) -> {default_rounds} rounds")
        return default_rounds
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y_arr[:-n_val], y_arr[-n_val:]

    if es_metric is None:
        es_metric = 'auc' if fit_params.get('objective') == 'binary:logistic' \
            else fit_params.get('eval_metric', 'rmse')
    if es_metric == 'auc' and len(np.unique(y_val)) < 2:
        if verbose:
            print(f"  [early-stop] {label}: skipped (single-class holdout) -> {default_rounds} rounds")
        return default_rounds

    booster = xgb.train(
        {**fit_params, 'eval_metric': es_metric},
        xgb.DMatrix(X_tr, label=y_tr),
        num_boost_round=default_rounds,
        evals=[(xgb.DMatrix(X_val, label=y_val), 'val')],
        early_stopping_rounds=es_rounds,
        verbose_eval=False)
    best = booster.best_iteration + 1
    if verbose:
        print(f"  [early-stop] {label}: best {best}/{default_rounds} rounds "
              f"(patience={es_rounds}, val {es_metric}={booster.best_score:.4f})")
    return best


def best_round_from_curve(ev, default_rounds, flat_tol=0.005, min_rounds=None):
    """Pick the boost round from ONE val-AUC curve — normally a single CV fold.

    `ev` is an evals_result-shaped dict {'val': {metric: [per-round...]}} (or None) from
    the TimeSeriesSplit folds of the UNSAMPLED training data — the regime the project's
    CV metrics are based on, and a far more stable basis than a one-shot holdout on the
    stride-sampled final-fit data. `best_round_from_folds` applies this rule per fold and
    takes the median; see there for why the folds are not averaged first.

    Only the val AUC curve is used. When the curve reports no AUC (single-class
    validation fold) this returns None; the logloss curve of an all-negative fold is
    dominated by the majority class and must not drive the round count.

    Early stopping should only fire when continuing actually HURTS validation. So this
    cuts to the peak round ONLY if val AUC degrades past it by more than `flat_tol` (peak
    minus the mean of the last 10% of the curve). On a flat curve — where extra rounds
    neither help nor hurt the ranking metric — the argmax is just noise (e.g. the fast
    M15 models saturate in a few rounds then wiggle), so `default_rounds` is kept instead
    of cutting to a noise-driven tiny value.

    A cut is floored at `min_rounds` (default `max(20, default_rounds // 10)`, never above
    `default_rounds`) so a noise-driven early peak cannot export a 1-4 tree stump.

    Returns the 1-based peak round (floored), `default_rounds` when the curve is flat,
    or None when no usable AUC curve is present.
    """
    if not ev:
        return None
    auc = ev.get('val', {}).get('auc')
    if not auc:
        return None
    a = np.asarray(auc, dtype=float)
    best_i = int(np.argmax(a))
    tail = float(np.mean(a[-max(1, len(a) // 10):]))
    if (a[best_i] - tail) <= flat_tol:
        return default_rounds
    floor = max(20, default_rounds // 10) if min_rounds is None else min_rounds
    return int(max(best_i + 1, min(floor, default_rounds)))


def best_round_from_folds(fold_curves, default_rounds, flat_tol=0.005, min_rounds=None,
                          skip_folds=None):
    """Resolve num_boost_round as the MEDIAN of the per-fold round decisions.

    Every CV fold votes independently via best_round_from_curve — same rule as always:
    argmax of that fold's val AUC, applied only on a real post-peak decline, floored at
    `min_rounds`, and `default_rounds` when the fold's curve is flat. The median vote is
    the round the final model is refit for.

    WHY THE FOLDS ARE NOT AVERAGED INTO ONE CURVE FIRST
    ---------------------------------------------------
    Averaging raw AUC per round and taking the argmax looks like the natural aggregate,
    but the argmax of a mean is driven by the fold with the largest AMPLITUDE, not by
    fold consensus — and in this project the largest amplitude reliably belongs to the
    fold with the LEAST signal, because an AUC that wanders around 0.5 swings far more
    than one sitting at 0.8.

    Measured on the long_fast model (run of 2026-08-08, 5 TimeSeriesSplit folds):

        fold          1      2      3      4      5
        own argmax  107      1     59    115    123
        peak AUC  0.524  0.585  0.785  0.778  0.960
        AUC @200  0.514  0.459  0.778  0.769  0.958

        argmax of the averaged curve .................  7
        argmax of the averaged curve without fold 2 .. 114
        median of the per-fold decisions ............. 107

    Fold 2 never leaves chance level (0.459-0.585 — its "collapse" is noise around 0.5,
    it costs nothing real), yet its 0.126 swing outvoted four folds that carry genuine
    signal and want 59-123 rounds. Cutting to 7 would have cost folds 3/4/5 between
    0.007 and 0.029 AUC — a real loss traded for an imaginary gain.

    A median is a rank statistic: it is immune to both the level and the amplitude of
    any single fold, so one degenerate regime can no longer set the capacity of a model
    that has to trade all of them. The trade-off is deliberate: the mean maximises
    *expected* AUC across folds and would be right if every fold's AUC measured
    something real, which is exactly the assumption that fails here.

    CONSENSUS GUARD
    ---------------
    A median is only meaningful if the votes describe the same thing. When the votes span
    the whole legal range — at least one fold at the floor ("this fold overfits before
    round `min_rounds`") and at least one at the cap ("this fold never degrades") — the
    folds are not disagreeing about a number, they are disagreeing about whether to cut at
    all. The median then lands on an arbitrary value with no fold actually supporting it,
    so `default_rounds` is kept instead: same principle as `flat_tol`, only act on
    evidence.

    This is not hypothetical. In the es_off/es_on A/B (2026-08-08, 18-month window)
    `long_slow` voted [59, 20, 200, 99, 20] -> median 58 and later [59, 20, 20, 200, 20]
    -> median 20, cutting the one model that gates entries from 200 rounds to 20. The
    backtest halved (448 -> 113 pips, win rate 52% -> 23%) while every CV gate stayed
    bit-identical. Straddling votes are the signature of validation folds too small to
    carry signal — here ~3.5 months of 4h bars each.

    DEGENERATE FOLDS DO NOT VOTE
    ----------------------------
    `skip_folds` is the 1-based list of folds whose model came out constant (see
    `run_time_series_cv`, key `degenerate_folds`). Such a fold's val AUC curve is a flat
    line at 0.500, which `best_round_from_curve` reads as "never degrades" and turns into
    a vote for the cap — but that is not evidence that more rounds help, it is evidence
    that the fold had nothing to learn from. Folds without a usable AUC (single-class
    validation) already cast no vote; a constant model is the same situation observed on
    the other side of the split, so it is treated the same way.

    NOTE on coverage: a *strictly* single-class training fold is already excluded
    implicitly, because `_eval_metrics_with_auc` then emits no AUC series and the fold
    produces no curve to vote with. Verified on run honest_20260829: `long_slow` votes
    [62, 20, 76, 200] both with and without `skip_folds` — the dead fold never voted.
    `skip_folds` closes the gap the implicit path leaves open: a fold with a handful of
    positives is not single-class, still trains to a constant model, and still emits a
    flat AUC curve that votes for the cap.

    Returns `(rounds, votes)` — the resolved round count and the per-fold votes for
    diagnostics — or `(None, [])` when no fold produced a usable AUC curve, in which case
    the caller falls back to the holdout resolver.
    """
    skip = set(skip_folds or ())
    votes = [best_round_from_curve(ev, default_rounds, flat_tol=flat_tol,
                                   min_rounds=min_rounds)
             for fold, ev in enumerate(fold_curves or [], 1)
             if fold not in skip]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None, []
    floor = min(max(20, default_rounds // 10) if min_rounds is None else min_rounds,
                default_rounds)
    if floor < default_rounds and floor in votes and default_rounds in votes:
        return default_rounds, votes
    return int(round(float(np.median(votes)))), votes


def round_curve(ev, nd=6):
    """Round every metric series in an evals_result dict (JSON size control)."""
    return {split: {m: [round(float(x), nd) for x in series]
                    for m, series in metrics.items()}
            for split, metrics in (ev or {}).items()}


def aggregate_fold_curves(fold_curves):
    """Average the per-round train/val metrics of every CV fold into one curve.

    `fold_curves` is the list of xgb evals_result dicts collected per TimeSeriesSplit
    fold by run_time_series_cv. This is the VISUALISATION/diagnostic view only — one
    readable curve per model for `learning_curves.png/.json` instead of five overlapping
    ones.

    It is explicitly NOT how num_boost_round is chosen: the argmax of an averaged curve
    follows whichever fold swings hardest, which is the fold with the least signal. The
    round count comes from best_round_from_folds (per-fold decision + median) — read the
    rationale there before using this curve to reason about round counts.

    Each metric is averaged only over the folds that actually reported it (a single-class
    validation fold contributes no AUC) and truncated to the shortest series. Returns
    {'train': {...}, 'val': {...}, 'n_folds': k, 'n_auc_folds': j}, or None when no fold
    produced a curve.
    """
    curves = [ev for ev in (fold_curves or []) if ev]
    if not curves:
        return None

    out = {}
    n_auc_folds = 0
    for split in ('train', 'val'):
        by_metric = {}
        for ev in curves:
            for metric, series in ev.get(split, {}).items():
                by_metric.setdefault(metric, []).append(np.asarray(series, dtype=float))
        agg = {}
        for metric, arrs in by_metric.items():
            n = min(len(a) for a in arrs)
            agg[metric] = np.mean(np.vstack([a[:n] for a in arrs]), axis=0).tolist()
            if split == 'val' and metric == 'auc':
                n_auc_folds = len(arrs)
        out[split] = agg
    out['n_folds'] = len(curves)
    out['n_auc_folds'] = n_auc_folds
    return out


def _eval_metrics_with_auc(params, y_tr, y_val):
    """Fold eval_metric list, with AUC appended when both splits carry both classes.

    XGBoost cannot compute AUC on a single-class split, so the metric is requested only
    when it is well defined. A fold without AUC simply casts no vote in
    best_round_from_folds and is skipped by aggregate_fold_curves.
    """
    eval_metric = params.get('eval_metric', 'logloss')
    metrics = [eval_metric] if isinstance(eval_metric, str) else list(eval_metric)
    if 'auc' not in metrics and len(np.unique(y_tr)) > 1 and len(np.unique(y_val)) > 1:
        metrics.append('auc')
    return metrics


def plot_learning_curves(curves, out_path, chosen_rounds=None):
    """Plot per-label train/val learning curves into a single PNG (2 cols: logloss, AUC).

    Curves are the fold-averaged output of aggregate_fold_curves(); the fold count is
    shown in each title so a curve built on fewer folds (single-class val folds carry no
    AUC) is not mistaken for the full average.

    Curves always span the full round cap — you cannot judge whether a cut was right
    without seeing past it. `chosen_rounds` (model key -> resolved num_boost_round) draws
    the round the final model was actually refit for, so a curve running to the cap is not
    misread as "early stopping did not fire".
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    items = [(name, ev) for name, ev in curves.items() if ev is not None]
    if not items:
        print("  (no learning curves to plot)")
        return
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)
    for row, (name, ev) in enumerate(items):
        train_m = ev.get('train', {})
        val_m = ev.get('val', {})
        n_folds = ev.get('n_folds')
        n_auc = ev.get('n_auc_folds')
        chosen = (chosen_rounds or {}).get(name)

        def _mark(ax):
            if chosen:
                ax.axvline(chosen - 1, color='crimson', ls='--', lw=1.2,
                           label=f'trained: {chosen} rounds')

        loss_key = next((k for k in ('logloss', 'error', 'rmse') if k in train_m), None)
        ax = axes[row][0]
        if loss_key:
            ax.plot(train_m[loss_key], label=f'train {loss_key}')
            ax.plot(val_m.get(loss_key, []), label=f'val {loss_key}')
            _mark(ax)
            ax.set_xlabel('boost round')
            ax.set_ylabel(loss_key)
            ax.set_title(f'{name} — {loss_key}' + (f' (mean of {n_folds} folds)' if n_folds else ''))
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        ax = axes[row][1]
        if 'auc' in train_m:
            ax.plot(train_m['auc'], label='train auc')
            ax.plot(val_m.get('auc', []), label='val auc')
            _mark(ax)
            ax.set_xlabel('boost round')
            ax.set_ylabel('auc')
            ax.set_title(f'{name} — AUC' + (f' (mean of {n_auc} folds)' if n_auc else ''))
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Learning curves saved: {out_path}")

def _fold_period_label(index, idx):
    """Render a fold's row range, appending the bar timestamps when they are known.

    `index` is the DatetimeIndex of the frame the CV runs on (the CV itself sees a
    plain numpy array, so the dates have to be handed in). Falls back to the bare
    row positions for array input or a non-datetime index.
    """
    span = f"{idx[0]} -> {idx[-1]} ({len(idx)} samples)"
    if not isinstance(index, pd.DatetimeIndex) or len(index) <= idx[-1]:
        return span
    fmt = '%Y-%m-%d %H:%M'
    return f"{span}  [{index[idx[0]].strftime(fmt)} -> {index[idx[-1]].strftime(fmt)}]"


def label_coverage_report(y, index, n_splits=5, gap=0):
    """Check, BEFORE training, whether every CV fold has something to learn from.

    `trend_only` (and every other regime-conditioned mode) sets whole calendar stretches
    to label 0. A `TimeSeriesSplit` fold whose training part falls entirely inside such a
    stretch trains on zero positives, produces a constant model, and contributes an AUC of
    exactly 0.500 to the pooled number — see `run_time_series_cv`. That is a property of
    the WINDOW, not of the model, and it is knowable before a single tree is grown.

    Measured on the recommended 18-month window (2024-04-01..2025-09-30, `trend_only`):
    2024-04, 2024-05, 2024-06, 2025-02, 2025-08 and 2025-09 all have a `long_slow`
    positive rate of exactly 0.000 — a third of the window carries no positive label at
    all, and the first three months are contiguous, which is what kills fold 1.

    Args:
        y: 1-D array/Series of 0/1 labels in chronological order.
        index: DatetimeIndex aligned with `y` (for the per-month table).
        n_splits/gap: the geometry `run_time_series_cv` will use.

    Returns:
        dict with `per_month` (period -> {n, positives, rate}), `folds`
        (list of per-fold train/val counts and periods), `empty_months`,
        `degenerate_fold_numbers` and `n_degenerate_folds`.
    """
    y_arr = np.asarray(y.values if hasattr(y, 'values') else y).astype(float)
    idx = pd.DatetimeIndex(index)

    per_month = {}
    if len(idx) == len(y_arr) and len(y_arr):
        s = pd.Series(y_arr, index=idx)
        grouped = s.groupby(s.index.to_period('M'))
        for period, chunk in grouped:
            per_month[str(period)] = {
                'n': int(len(chunk)),
                'positives': int(np.nansum(chunk.values)),
                'rate': float(np.nanmean(chunk.values)) if len(chunk) else 0.0,
            }
    empty_months = [m for m, v in per_month.items() if v['positives'] == 0]

    folds = []
    degenerate = []
    if len(y_arr) > n_splits:
        for fold, (tr, va) in enumerate(
                TimeSeriesSplit(n_splits=n_splits, gap=gap).split(y_arr), 1):
            tr_pos = int(np.nansum(y_arr[tr]))
            va_pos = int(np.nansum(y_arr[va]))
            folds.append({
                'fold': fold,
                'train_n': int(len(tr)), 'train_positives': tr_pos,
                'val_n': int(len(va)), 'val_positives': va_pos,
                'train_period': _fold_period_label(idx, tr),
                'val_period': _fold_period_label(idx, va),
            })
            if tr_pos == 0:
                degenerate.append(fold)

    return {
        'per_month': per_month,
        'empty_months': empty_months,
        'n_empty_months': len(empty_months),
        'n_months': len(per_month),
        'folds': folds,
        'degenerate_fold_numbers': degenerate,
        'n_degenerate_folds': len(degenerate),
    }


def print_label_coverage(report, label):
    """Print a label-coverage report; returns True when a fold would degenerate."""
    n_empty, n_months = report['n_empty_months'], report['n_months']
    dead = report['degenerate_fold_numbers']
    if not dead and not n_empty:
        return False

    print(f"\n  Label coverage '{label}': {n_empty}/{n_months} month(s) carry no "
          f"positive label" + (f" ({', '.join(report['empty_months'])})" if n_empty else ""))
    if dead:
        print(f"  ERROR: CV fold(s) {dead} would train on ZERO positives. Those folds "
              f"produce a constant model whose AUC of 0.500 is pooled into the headline "
              f"number. Move --train-start past the empty block, lower --cv-splits, or "
              f"change the label mode — the metrics are not interpretable as they stand.")
        for f in report['folds']:
            if f['fold'] in dead:
                print(f"    fold {f['fold']}: train {f['train_period']} "
                      f"-> {f['train_positives']} positives in {f['train_n']} rows")
    return bool(dead)


def run_time_series_cv(X_train, y_train, params, n_splits=5, num_boost_round=200,
                       spw_factor=1.0, target_recall=0.5, gap=0, index=None,
                       sample_weight=None, regime_trend=None,
                       target_recall_trend=None, target_recall_range=None):
    """Run time-series cross-validation with comprehensive metrics.

    Per fold: pick the threshold that maximises precision subject to recall >= target_recall
    (fallback: F1-optimal). Reports per-fold F1, Precision, Recall, AUC-ROC, Brier, ECE, CM.

    Aggregated metrics (`global_val_*`) are computed on pooled validation predictions across
    all folds, using a single global threshold derived via the same precision-at-recall rule.
    AUC and Brier are computed once on the pooled predictions (not per-fold averaged).

    Args:
        spw_factor: Multiplier applied to the per-fold auto scale_pos_weight.
                    1.0 = natural class ratio; <1.0 = favour precision over recall.
        target_recall: Recall floor for threshold selection. Default 0.5.
        gap: Embargo rows between each train fold and its validation fold
             (TimeSeriesSplit gap). Labels with a forward horizon peek past the
             fold seam; a gap >= label horizon (in this frame's cadence) removes
             that leakage. Default 0 = historical behaviour.
        index: Optional DatetimeIndex aligned with X_train, used only to print each
               fold's train/val period as dates. Taken from X_train automatically
               when it is a DataFrame; without it the log shows row positions only.
        regime_trend: Optional per-row regime_trend values aligned with X_train
               ({1=(up)trend, -1=downtrend, 0=range}, NaN=unknown). Enables the
               regime-scoped recall floors below and the within-TREND/RANGE recall
               reported alongside the global metrics.
        target_recall_trend / target_recall_range: Additional recall floors for the
               threshold selection, evaluated over the TREND (regime_trend != 0)
               resp. RANGE (== 0) bars only. All given floors must hold; see
               _select_threshold_with_recall_floors.
    """
    print("\n" + "=" * 80)
    print("TIME-SERIES CROSS-VALIDATION" + (f"  (embargo gap={gap} rows)" if gap else ""))
    print("=" * 80 + "\n")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    train_scores = []
    val_scores = []
    val_precisions = []
    val_recalls = []
    val_mccs = []
    val_auc_rocs = []
    val_brier_scores = []
    val_eces = []
    val_confusion_matrices = []
    val_thresholds = []
    fold_curves = []
    pooled_y_val = []
    pooled_val_prob = []
    pooled_val_indices = []
    n_single_class_train_folds = 0
    degenerate_folds = []   # folds whose model is constant — reported, never silent

    X_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    w_array = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if w_array is not None and len(w_array) != len(y_array):
        raise ValueError(
            f"sample_weight has {len(w_array)} entries but y has {len(y_array)}"
        )
    rt_array = None
    if regime_trend is not None:
        rt_array = np.asarray(
            regime_trend.values if hasattr(regime_trend, 'values') else regime_trend,
            dtype=float)
        if len(rt_array) != len(y_array):
            raise ValueError(
                f"regime_trend has {len(rt_array)} entries but y has {len(y_array)}"
            )
    _regime_floors_active = rt_array is not None and (
        target_recall_trend is not None or target_recall_range is not None)
    if index is None and isinstance(X_train, pd.DataFrame):
        index = X_train.index

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_array), 1):
        X_tr, X_val = X_array[train_idx], X_array[val_idx]
        y_tr, y_val = y_array[train_idx], y_array[val_idx]

        # Weights apply to the training OBJECTIVE only: every reported metric must
        # describe the bars as they actually occur, not as the objective reweighted them.
        # `deval_train` is therefore an unweighted view of the same rows — it is what goes
        # into `evals`, while `dtrain` (weighted) is what the boosting minimises.
        # XGBoost also refuses to evaluate weighted AUC on some folds
        # ("Check failed: auc <= 1.0"), which this split avoids as a side effect.
        _w = None if w_array is None else w_array[train_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=_w)
        deval_train = dtrain if _w is None else xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        # A single-class TRAINING fold produces a constant model: XGBoost has no
        # gradient to follow, every bar gets the same score, and the fold's val AUC
        # comes out at exactly 0.500 while still being pooled into the headline
        # number. Until this check existed only single-class *validation* folds were
        # counted, so the failure was invisible.
        #
        # Measured on run best_20260808 (2026-08-29): the feature-derived warm-up
        # moved the training window start back to the requested 2024-04-01, and
        # 2024-04..2024-06 carry a trend_only positive rate of exactly 0.000. Fold 1
        # then trained on 2024-04-01..2024-07-01 with zero positives -> val AUC
        # 0.5000, fold 2 -> 0.239, and the pooled slow AUC fell from 0.760 to 0.525
        # with nothing in the diagnostics reporting a problem.
        n_train_classes = len(np.unique(y_tr))
        if n_train_classes < 2:
            n_single_class_train_folds += 1
            print(f"  Fold {fold}: SINGLE-CLASS TRAINING FOLD "
                  f"({int(y_tr.sum())} positives in {len(y_tr)} rows) — the model is "
                  f"constant and this fold carries no information. "
                  f"train period: {_fold_period_label(index, train_idx)}")

        # Apply per-fold scale_pos_weight scaled by spw_factor
        fold_spw = ((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)) * spw_factor
        fold_params = {**params, 'scale_pos_weight': fold_spw, 'base_score': _base_score_from_labels(y_tr)}

        # Capture this fold's per-round learning curve. Each fold's curve casts one vote
        # for num_boost_round (see best_round_from_folds), so no separate training pass is
        # needed for early stopping. Eval metrics do not affect boosting.
        fold_ev = {}
        model = xgb.train(
            {**fold_params, 'eval_metric': _eval_metrics_with_auc(fold_params, y_tr, y_val)},
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(deval_train, 'train'), (dval, 'val')],
            evals_result=fold_ev,
            verbose_eval=False
        )
        fold_curves.append(fold_ev)

        train_prob = model.predict(deval_train)
        val_prob = model.predict(dval)

        # Constant scores are the observable symptom of a fold that learned nothing.
        # Checked on the predictions rather than only on the label counts, because a
        # fold with a handful of positives can degenerate the same way without being
        # strictly single-class.
        if np.ptp(val_prob) < 1e-12:
            degenerate_folds.append(fold)
            print(f"  Fold {fold}: CONSTANT PREDICTIONS (score = {val_prob[0]:.6f} on "
                  f"every validation bar) — its AUC of 0.500 is an artefact, not a "
                  f"measurement.")

        # Per-fold threshold: max precision subject to recall >= target_recall plus any
        # regime-scoped floors (fallback F1-optimal / overall-only rule).
        best_threshold = _select_threshold_with_recall_floors(
            y_val, val_prob, target_recall,
            regime_trend=None if rt_array is None else rt_array[val_idx],
            target_recall_trend=target_recall_trend,
            target_recall_range=target_recall_range)

        train_pred = (train_prob > best_threshold).astype(int)
        val_pred   = (val_prob   > best_threshold).astype(int)

        train_f1 = f1_score(y_tr,  train_pred, zero_division=0)
        val_f1   = f1_score(y_val, val_pred,   zero_division=0)
        val_prec = precision_score(y_val, val_pred, zero_division=0)
        val_rec  = recall_score(y_val, val_pred, zero_division=0)
        val_mcc  = _safe_mcc(y_val, val_pred)
        val_pos_rate = y_val.mean()

        # AUC-ROC: requires both classes present in y_val
        if len(np.unique(y_val)) > 1:
            val_auc = roc_auc_score(y_val, val_prob)
        else:
            val_auc = float('nan')

        # Brier score: measures calibration quality of probability estimates
        val_brier = brier_score_loss(y_val, val_prob)

        # Expected Calibration Error
        val_ece = _compute_ece(y_val, val_prob)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_val, val_pred, labels=[0, 1])

        train_scores.append(train_f1)
        val_scores.append(val_f1)
        val_precisions.append(val_prec)
        val_recalls.append(val_rec)
        val_mccs.append(val_mcc)
        val_auc_rocs.append(val_auc)
        val_brier_scores.append(val_brier)
        val_eces.append(val_ece)
        val_confusion_matrices.append(cm.tolist())
        val_thresholds.append(best_threshold)
        pooled_y_val.append(y_val)
        pooled_val_prob.append(val_prob)
        pooled_val_indices.append(val_idx)

        auc_str = f"{val_auc:.3f}" if not np.isnan(val_auc) else "N/A"
        print(f"  Fold {fold}: Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f} "
              f"(P={val_prec:.3f} R={val_rec:.3f} AUC={auc_str} MCC={val_mcc:.3f} "
              f"Brier={val_brier:.4f} ECE={val_ece:.4f} pos={val_pos_rate:.1%} thr={best_threshold:.3f})")
        print(f"           CM: TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")
        print(f"           train period: {_fold_period_label(index, train_idx)}")
        print(f"           val period:   {_fold_period_label(index, val_idx)}")

    # --- Global pooled metrics: single threshold, computed on concatenated val predictions ---
    pooled_y = np.concatenate(pooled_y_val) if pooled_y_val else np.array([])
    pooled_p = np.concatenate(pooled_val_prob) if pooled_val_prob else np.array([])

    pooled_rt = (rt_array[np.concatenate(pooled_val_indices).astype(int)]
                 if rt_array is not None and pooled_val_indices else None)

    if len(pooled_y) > 0:
        global_threshold = _select_threshold_with_recall_floors(
            pooled_y, pooled_p, target_recall,
            regime_trend=pooled_rt,
            target_recall_trend=target_recall_trend,
            target_recall_range=target_recall_range)
        global_pred = (pooled_p > global_threshold).astype(int)
        global_f1   = f1_score(pooled_y, global_pred, zero_division=0)
        global_prec = precision_score(pooled_y, global_pred, zero_division=0)
        global_rec  = recall_score(pooled_y, global_pred, zero_division=0)
        global_mcc  = _safe_mcc(pooled_y, global_pred)
        if len(np.unique(pooled_y)) > 1:
            global_auc = float(roc_auc_score(pooled_y, pooled_p))
        else:
            global_auc = float('nan')
        global_brier = float(brier_score_loss(pooled_y, pooled_p))
        global_ece   = _compute_ece(pooled_y, pooled_p)
        global_cm    = confusion_matrix(pooled_y, global_pred, labels=[0, 1]).tolist()
    else:
        global_threshold = 0.5
        global_f1 = global_prec = global_rec = global_mcc = 0.0
        global_auc = float('nan')
        global_brier = global_ece = 0.0
        global_cm = [[0, 0], [0, 0]]

    # Recall within the TREND / RANGE scopes at the SAME global threshold — the number
    # the regime-scoped floors constrain, reported whenever the regime is known so the
    # "overall recall met, trend recall not" failure mode is visible without the floors.
    global_recall_trend = global_recall_range = float('nan')
    if pooled_rt is not None and len(pooled_y) > 0:
        _trend_mask = (~np.isnan(pooled_rt)) & (pooled_rt != 0)
        _range_mask = pooled_rt == 0
        if (pooled_y[_trend_mask] == 1).any():
            global_recall_trend = float(recall_score(
                pooled_y[_trend_mask], global_pred[_trend_mask], zero_division=0))
        if (pooled_y[_range_mask] == 1).any():
            global_recall_range = float(recall_score(
                pooled_y[_range_mask], global_pred[_range_mask], zero_division=0))

    # Same pooled AUC, but over the folds where a model actually exists.
    #
    # `global_val_auc_roc` above keeps its established meaning (all folds) so older runs
    # stay comparable. A constant-prediction fold, however, does not merely contribute a
    # fold's worth of 0.5 — AUC is a rank statistic over the CONCATENATED array, so a
    # block of identical scores ties against every other fold's predictions and pulls the
    # pooled number hard toward chance. Measured on best_20260808: all folds 0.525, one
    # dead fold removed 0.700 (per-fold [0.500, 0.239, 0.784, 0.725, 0.938]).
    #
    # Read the two together, never one alone: `_healthy` is the model-quality number,
    # `n_degenerate_folds > 0` is an independent hard failure of the label coverage.
    # Excluding the fold does not fix the configuration, it only stops the defect from
    # being reported as a weak model.
    if degenerate_folds and pooled_val_indices:
        keep = [p for fold, p in enumerate(pooled_val_prob, 1) if fold not in set(degenerate_folds)]
        keep_y = [y for fold, y in enumerate(pooled_y_val, 1) if fold not in set(degenerate_folds)]
        hy = np.concatenate(keep_y) if keep_y else np.array([])
        hp = np.concatenate(keep) if keep else np.array([])
        global_auc_healthy = (float(roc_auc_score(hy, hp))
                              if len(hy) and len(np.unique(hy)) > 1 else float('nan'))
        # Same threshold as the all-folds number, so the two are read on one scale. A
        # constant fold contributes |MCC| = 0 rows here too, but unlike AUC it does not
        # also corrupt the ranking of the other folds' rows.
        global_mcc_healthy = (_safe_mcc(hy, (hp > global_threshold).astype(int))
                              if len(hy) else 0.0)
    else:
        global_auc_healthy = global_auc
        global_mcc_healthy = global_mcc

    # Folds whose validation slice holds only one class produce no AUC at all. When most
    # folds are like that the pooled AUC rests on the handful that remain, so it must not
    # be read as a 5-fold estimate — say so loudly rather than printing a clean number.
    n_single_class_folds = sum(1 for a in val_auc_rocs if np.isnan(a))
    n_folds_done = len(val_auc_rocs)

    auc_summary = f"{global_auc:.4f}" if not np.isnan(global_auc) else "N/A"
    print(f"\nCV Summary (target_recall={target_recall:.2f}, global_thr={global_threshold:.3f}):")
    if n_single_class_folds:
        print(f"  WARNING: {n_single_class_folds}/{n_folds_done} validation folds are "
              f"single-class (no positive labels) — no AUC from those folds. The global AUC "
              f"below rests on {n_folds_done - n_single_class_folds} fold(s) only; treat it "
              f"as unreliable and check the temporal distribution of the labels.")
    if degenerate_folds:
        print(f"  WARNING: fold(s) {degenerate_folds} produced CONSTANT predictions "
              f"({n_single_class_train_folds} of them had no positive TRAINING label at "
              f"all). Their AUC of exactly 0.500 is pooled into the global number below "
              f"and drags it toward chance. This is a label-coverage problem in the "
              f"training window, not a model quality result — check the per-month "
              f"positive rate before reading any metric here.")
        _h = f"{global_auc_healthy:.4f}" if not np.isnan(global_auc_healthy) else "N/A"
        print(f"           Global Val AUC over the healthy folds only: {_h}")
    print(f"  Per-fold thresholds: [{', '.join(f'{t:.3f}' for t in val_thresholds)}]")
    if _regime_floors_active:
        _fmt_sr = lambda v: f"{v:.4f}" if not np.isnan(v) else "N/A"
        _floor_bits = []
        if target_recall_trend is not None:
            _floor_bits.append(f"TREND >= {target_recall_trend:.2f} "
                               f"(achieved {_fmt_sr(global_recall_trend)})")
        if target_recall_range is not None:
            _floor_bits.append(f"RANGE >= {target_recall_range:.2f} "
                               f"(achieved {_fmt_sr(global_recall_range)})")
        print(f"  Regime recall floors: {', '.join(_floor_bits)}")
    print(f"  Global Val F1:     {global_f1:.4f}")
    print(f"  Global Val Prec:   {global_prec:.4f}")
    print(f"  Global Val Recall: {global_rec:.4f}")
    print(f"  Global Val AUC:    {auc_summary}")
    print(f"  Global Val MCC:    {global_mcc:.4f}"
          + ("   <-- 0.0000 means the model predicts one class everywhere"
             if abs(global_mcc) < 1e-12 else ""))
    print(f"  Global Val Brier:  {global_brier:.4f}")
    print(f"  Global Val ECE:    {global_ece:.4f}")
    print(f"  Global CM: TN={global_cm[0][0]} FP={global_cm[0][1]} "
          f"FN={global_cm[1][0]} TP={global_cm[1][1]}")

    # Event-level metrics (fold-boundary-aware)
    _pooled_idx = np.concatenate(pooled_val_indices).astype(int) if pooled_val_indices else None
    ev = _compute_event_metrics(pooled_y, global_pred, val_indices=_pooled_idx)
    _fmtev = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"
    print(f"  --- Event-level metrics ---")
    print(f"  Event Recall:           {_fmtev(ev['event_recall'])}"
          f"  ({ev['n_detected_events']}/{ev['n_actual_events']} events detected)")
    print(f"  Event Precision:        {_fmtev(ev['event_precision'])}"
          f"  ({ev['n_predicted_events']} predicted events)")
    print(f"  Detection Lag (mean):   {_fmtev(ev['detection_lag_mean'])} bars")
    print(f"  Preds per Event (mean): {_fmtev(ev['mean_preds_per_event'])}")

    return {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'mean_train': np.mean(train_scores) if train_scores else 0.0,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
        'val_mccs': val_mccs,
        'val_auc_rocs': val_auc_rocs,
        'val_brier_scores': val_brier_scores,
        'val_eces': val_eces,
        'val_confusion_matrices': val_confusion_matrices,
        'val_thresholds': val_thresholds,
        # Global pooled metrics (replace prior arithmetic-mean aggregates)
        'global_val_f1': global_f1,
        'global_val_precision': global_prec,
        'global_val_recall': global_rec,
        # Matthews correlation at `global_threshold`. Gate metric: 0 exactly for a
        # constant classifier, which neither AUC nor F1 reports unambiguously.
        'global_val_mcc': global_mcc,
        'global_val_auc_roc': global_auc,
        # Pooled AUC over the folds whose model is not constant. Equals
        # global_val_auc_roc when no fold degenerated. See the computation above.
        'global_val_auc_roc_healthy_folds': global_auc_healthy,
        'global_val_mcc_healthy_folds': global_mcc_healthy,
        'global_val_brier': global_brier,
        'global_val_ece': global_ece,
        'global_val_confusion_matrix': global_cm,
        'global_threshold': global_threshold,
        'target_recall': target_recall,
        'target_recall_trend': target_recall_trend,
        'target_recall_range': target_recall_range,
        # Recall within TREND / RANGE bars at global_threshold (NaN when the regime is
        # unknown or the scope has no positives) — what the regime floors constrain.
        'global_val_recall_trend': global_recall_trend,
        'global_val_recall_range': global_recall_range,
        # How many validation folds had no positives — the global AUC is only as
        # trustworthy as (n_folds - n_single_class_val_folds).
        'n_single_class_val_folds': n_single_class_folds,
        # ...and how many TRAINING folds had no positives. A single-class training
        # fold yields a constant model whose 0.500 AUC is still pooled into the
        # headline number, so this has to be visible next to it. See the fold loop.
        'n_single_class_train_folds': n_single_class_train_folds,
        'degenerate_folds': degenerate_folds,
        'n_degenerate_folds': len(degenerate_folds),
        'n_cv_folds': n_folds_done,
        # Per-fold evals_result dicts. -> best_round_from_folds() for the round choice,
        # -> aggregate_fold_curves() for the plotted/exported diagnostic curve.
        'fold_curves': fold_curves,
        'pooled_y_val': pooled_y,
        'pooled_val_prob': pooled_p,
        'pooled_val_indices': np.concatenate(pooled_val_indices).astype(int) if pooled_val_indices else np.array([], dtype=int),
        # Which fold each pooled row came from. The pooled arrays above are a
        # concatenation, so fold membership is otherwise unrecoverable — and without it
        # a per-fold ROC/calibration curve cannot be drawn from the exported scores.
        'pooled_fold_ids': (np.concatenate([np.full(len(v), f, dtype=int)
                                            for f, v in enumerate(pooled_val_indices, 1)])
                            if pooled_val_indices else np.array([], dtype=int)),
        # Event-level metrics
        'event_precision':            ev['event_precision'],
        'event_recall':               ev['event_recall'],
        'event_detection_lag_mean':   ev['detection_lag_mean'],
        'event_mean_preds_per_event': ev['mean_preds_per_event'],
        'event_n_actual':             ev['n_actual_events'],
        'event_n_detected':           ev['n_detected_events'],
        'event_n_predicted':          ev['n_predicted_events'],
    }


def _target_key(model_key):
    """Map a config model key to the trained-model dict key / ONNX file stem."""
    return f'target_{model_key}'


def train_models(model_specs, feature_map_dir, args=None, round_overrides=None):
    """Train XGBoost models — one independent feature set/scaler per model.

    ``model_specs`` maps each config model key (long_fast, short_fast, long_slow,
    short_slow) to a spec dict:
        X:               scaled ndarray (rows at the model's cadence)
        y:               label Series or ndarray
        params:          xgb params (objective already set)
        spw_factor:      scale_pos_weight multiplier (classifiers)
        features:        selected feature name list (for importance export)
        num_boost_round: default round count

    ``round_overrides`` maps a trained-model key ('target_long_fast', etc.) to a
    num_boost_round pre-resolved by best_round_from_folds (median of the per-fold CV
    votes) — more stable than the temporal-holdout fallback resolve_boost_rounds, which
    reads a single recent slice of the stride-sampled final-fit data.
    """
    print("\n" + "=" * 80)
    print("TRAINING MODELS (PER-MODEL FEATURE SETS)")
    print("=" * 80 + "\n")

    # Holdout fallback: 0 disables it, <0 (or unset) auto-derives the patience.
    _es_arg = getattr(args, 'early_stopping_rounds', None) if args else None
    es_val_frac = getattr(args, 'es_val_frac', 0.15) if args else 0.15
    es_disabled = _es_arg == 0
    es_rounds = None if (_es_arg is None or _es_arg < 0) else _es_arg
    use_cv_rounds = not getattr(args, 'no_cv_boost_rounds', False) if args else True

    # Apply per-label training sampling when enabled and y carries a DatetimeIndex.
    def _sample_xy(X_scaled, y, label_name):
        if args is None or not getattr(args, 'training_sampling', False):
            return X_scaled, (y.values if isinstance(y, pd.Series) else y)
        if not isinstance(y, pd.Series):
            return X_scaled, y
        keep_idx = compute_sampling_index(y, args, label_name=label_name)
        if len(keep_idx) == len(y):
            return X_scaled, y.values
        mask = y.index.isin(keep_idx)
        return X_scaled[mask], y.loc[mask].values

    def _resolve_rounds(model_key, label, fit_params, X, y, default_rounds):
        # The per-fold CV vote is the primary source and is independent of the holdout
        # fallback's patience knob — it must be checked FIRST, otherwise
        # --early-stopping-rounds 0 (the default) would silently discard it too.
        if use_cv_rounds and round_overrides and round_overrides.get(model_key):
            r = round_overrides[model_key]
            print(f"  [early-stop] {label}: {r}/{default_rounds} rounds (median of CV fold votes)")
            return r
        if es_disabled:
            return default_rounds
        return resolve_boost_rounds(label, fit_params, X, y, default_rounds,
                                    es_rounds=es_rounds, es_val_frac=es_val_frac)

    models = {}
    boost_rounds = {}
    for mk, spec in model_specs.items():
        tkey = _target_key(mk)
        X_scaled = spec['X']
        y = spec['y']
        params = spec['params']
        default_rounds = spec['num_boost_round']

        X_s, y_s = _sample_xy(X_scaled, y, mk)
        y_arr = y_s.values if isinstance(y_s, pd.Series) else y_s
        spw = ((y_arr == 0).sum() / max((y_arr == 1).sum(), 1)) * spec.get('spw_factor', 1.0)
        print(f"  {mk} label rate: {(y_arr == 1).mean():.1%}  ->  scale_pos_weight={spw:.2f}")
        # Uniqueness weights, when the caller supplied them. Length must match the rows
        # AFTER --training-sampling; a mismatch means the two mechanisms were combined
        # without threading the mask through, and silently mis-weighting is worse than
        # not weighting, so drop them and say so.
        w = spec.get('sample_weight')
        if w is not None and len(w) != len(y_arr):
            print(f"  WARNING: {mk} sample weights ({len(w)}) do not match the training "
                  f"rows ({len(y_arr)}) after sampling — training unweighted.")
            w = None
        dtrain = xgb.DMatrix(X_s, label=y_s, weight=w)
        p = {**params, 'scale_pos_weight': spw, 'base_score': _base_score_from_labels(y_arr)}
        nbr = _resolve_rounds(tkey, mk, p, X_s, y_arr, default_rounds)
        boost_rounds[mk] = {'rounds': nbr, 'cap': default_rounds}
        models[tkey] = xgb.train(p, dtrain, num_boost_round=nbr)
        print(f"Trained: XGBoost Classifier {mk} [{len(y_arr)} bars]")

        features.save_feature_importance(models[tkey], spec['features'], tkey, feature_map_dir)

    return {'models': models, 'boost_rounds': boost_rounds}


def calibrate_models(model_specs, models, method='platt', cal_fraction=0.2):
    """
    Calibrate ALL classifier models using Platt scaling or isotonic regression.

    Uses the last `cal_fraction` of each model's own (cadence-aligned) training
    data as a held-out calibration set.

    Args:
        model_specs: dict model_key -> spec with 'X' (scaled matrix), 'y' (label).
                     Each model is calibrated on its own feature set.
        models: dict of trained XGBoost Booster models (keyed by target key)
        method: 'platt' (sigmoid) or 'isotonic'
        cal_fraction: fraction of data to hold out for calibration

    Returns:
        calibrators: dict mapping model_name -> calibrator object
        cal_params: dict mapping model_name -> exportable parameters (for Java)
    """
    print("\n" + "=" * 80)
    print(f"PROBABILITY CALIBRATION (method={method})")
    print("=" * 80 + "\n")

    if method == 'none':
        print("Calibration disabled.")
        return {}, {}

    calibrators = {}
    cal_params = {}

    for mk, spec in model_specs.items():
        model_name = _target_key(mk)
        if model_name not in models:
            continue

        X_scaled = spec['X']
        n_samples = len(X_scaled)
        cal_start = int(n_samples * (1 - cal_fraction))
        X_cal = X_scaled[cal_start:]

        model = models[model_name]
        y_all = spec['y']
        y_cal = y_all[cal_start:] if isinstance(y_all, np.ndarray) else y_all.values[cal_start:]

        print(f"  {model_name}: calibrating on last {n_samples - cal_start} samples")

        # Both calibrators need both classes present. Sparse label modes (e.g.
        # trend_only on a short window) can leave the tail slice single-class —
        # skip calibration for that model instead of aborting the whole run.
        n_positive = int(np.sum(y_cal))
        if n_positive == 0 or n_positive == len(y_cal):
            print(f"    SKIPPED: calibration slice is single-class "
                  f"({n_positive}/{len(y_cal)} positive) — using raw probabilities")
            continue

        # Get raw probabilities on calibration set
        dcal = xgb.DMatrix(X_cal)
        raw_probs = model.predict(dcal)

        if method == 'platt':
            lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            lr.fit(raw_probs.reshape(-1, 1), y_cal)

            calibrators[model_name] = lr
            A = float(lr.coef_[0][0])
            B = float(lr.intercept_[0])
            cal_params[model_name] = {
                'method': 'platt',
                'A': A,
                'B': B,
            }

            cal_probs = lr.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            _print_calibration_stats(model_name, raw_probs, cal_probs, y_cal, A, B)

        elif method == 'isotonic':
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            ir.fit(raw_probs, y_cal)

            calibrators[model_name] = ir
            cal_params[model_name] = {
                'method': 'isotonic',
                'X_thresholds': ir.X_thresholds_.tolist() if hasattr(ir, 'X_thresholds_') else [],
                'y_thresholds': ir.y_thresholds_.tolist() if hasattr(ir, 'y_thresholds_') else [],
            }

            cal_probs = ir.predict(raw_probs)
            _print_calibration_stats(model_name, raw_probs, cal_probs, y_cal)

    return calibrators, cal_params


def _print_calibration_stats(model_name, raw_probs, cal_probs, y_true, A=None, B=None):
    """Print calibration quality statistics."""
    from sklearn.metrics import brier_score_loss, log_loss

    brier_raw = brier_score_loss(y_true, raw_probs)
    brier_cal = brier_score_loss(y_true, cal_probs)

    # Reliability: bin predictions and compare with actual frequency
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    print(f"\n  {model_name}:")
    if A is not None and B is not None:
        print(f"    Platt params: A={A:.4f}, B={B:.4f}")
        print(f"    Formula: calibrated = sigmoid({A:.4f} * raw + {B:.4f})")
    print(f"    Brier score: {brier_raw:.4f} (raw) -> {brier_cal:.4f} (calibrated)")
    print(f"    Raw  prob range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}], mean={raw_probs.mean():.3f}")
    print(f"    Cal  prob range: [{cal_probs.min():.3f}, {cal_probs.max():.3f}], mean={cal_probs.mean():.3f}")
    print(f"    Actual positive rate: {y_true.mean():.3f}")

    # Reliability diagram (text)
    print(f"    Reliability (bin_center -> actual_rate, n_samples):")
    for i in range(n_bins):
        mask = (cal_probs >= bin_edges[i]) & (cal_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            actual_rate = y_true[mask].mean()
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            print(f"      {bin_center:.2f}: actual={actual_rate:.3f} (n={mask.sum()})")


def apply_calibration(model, calibrator, X):
    """
    Get calibrated probabilities from a model + calibrator.

    Args:
        model: XGBoost Booster
        calibrator: fitted Platt (LogisticRegression) or IsotonicRegression
        X: np.ndarray or xgb.DMatrix

    Returns:
        np.ndarray of calibrated probabilities
    """
    if not isinstance(X, xgb.DMatrix):
        X = xgb.DMatrix(X)
    raw_probs = model.predict(X)

    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    elif isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(raw_probs)
    else:
        return raw_probs


def run_shap_analysis(models, shap_specs, output_dir, args):
    """Run SHAP analysis on all trained models with per-model feature sets.

    ``shap_specs`` maps each config model key to a tuple
    (X_df, feature_names, regime_labels) aligned to that model's cadence.
    """
    if args.skip_shap:
        print("\n[SHAP] Skipping SHAP analysis (--skip-shap)")
        return {}

    print("\n" + "=" * 80)
    print("SHAP ANALYSIS")
    print("=" * 80 + "\n")

    results = {}

    for mk, (X, feat_names, regime_labels) in shap_specs.items():
        model_name = _target_key(mk)
        if model_name not in models:
            print(f"Warning: Model '{model_name}' not found, skipping SHAP analysis")
            continue

        model = models[model_name]
        print(f"\nAnalyzing model: {model_name} ({len(feat_names)} features)")
        model_output_dir = os.path.join(output_dir, model_name)

        shap_results = run_full_shap_analysis(
            model=model,
            X=X,
            feature_names=feat_names,
            output_dir=model_output_dir,
            regime_labels=regime_labels,
            model_name=model_name,
            max_samples=1000
        )

        results[model_name] = shap_results

    # Cross-scope comparison: slow vs fast for long and short models
    for direction in ('long', 'short'):
        slow_key = f'target_{direction}_slow'
        fast_key = f'target_{direction}_fast'
        slow_regime = results.get(slow_key, {}).get('regime_analysis', {})
        fast_regime = results.get(fast_key, {}).get('regime_analysis', {})
        if not (slow_regime and fast_regime):
            continue

        aggregated_keys = [k for k in set(slow_regime) | set(fast_regime)
                           if not _is_combined_regime_key(k)]
        combined_keys = sorted(k for k in set(slow_regime) | set(fast_regime)
                               if _is_combined_regime_key(k))

        suffix = f' — {direction.title()} Model'
        for keys, label in [(aggregated_keys, 'aggregated'), (combined_keys, 'combined')]:
            if not keys:
                continue
            path = os.path.join(output_dir, f'shap_scope_comparison_{direction}_{label}.png')
            plot_scope_comparison_by_regime(
                slow_regime, fast_regime,
                output_path=path,
                regime_keys=keys,
                title_suffix=f'{suffix} ({label.title()})',
            )

    return results


def save_models(models, feature_counts, output_dir, calibrators=None, cal_params=None):
    """Save all models as .pkl, .json (XGBoost native), and .onnx files.

    ``feature_counts`` maps each trained-model key (e.g. 'target_long_fast')
    to its own ONNX input width. Each model has an independent input shape.
    Also saves calibrators and calibration parameters."""
    print("\n" + "=" * 80)
    print("SAVE & EXPORT MODELS")
    print("=" * 80 + "\n")

    for model_name, model in models.items():
        initial_type = [("input", FloatTensorType([None, feature_counts[model_name]]))]

        # Save as PKL (for backtest.py compatibility)
        pkl_path = os.path.join(output_dir, f"strategy_model_{model_name}.pkl")
        joblib.dump(model, pkl_path)

        # Save as JSON (XGBoost native format)
        json_path = os.path.join(output_dir, f"strategy_model_{model_name}.json")
        model.save_model(json_path)

        # Save as ONNX (for Java inference)
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type, target_opset=12)
        onnx_path = os.path.join(output_dir, f"strategy_model_{model_name}.onnx")
        onnxmltools.utils.save_model(onnx_model, onnx_path)

        print(f"Saved: {model_name} (.pkl, .json, .onnx)")

    # Save calibrators.
    # Any model WITHOUT a fresh calibrator must have its stale .pkl removed: backtest.py
    # loads calibrator_*.pkl by filename, so a leftover from a previous training run would
    # be silently applied to this run's model and corrupt every probability it produces.
    calibrators = calibrators or {}
    for model_name in models:
        cal_pkl_path = os.path.join(output_dir, f"calibrator_{model_name}.pkl")
        if model_name in calibrators:
            joblib.dump(calibrators[model_name], cal_pkl_path)
            print(f"Saved: calibrator for {model_name} (.pkl)")
        elif os.path.exists(cal_pkl_path):
            os.remove(cal_pkl_path)
            print(f"Removed stale calibrator for {model_name} (.pkl) — not calibrated in this run")

    # Save calibration parameters (JSON for Java) — only if there are actual params
    if cal_params and len(cal_params) > 0:
        cal_json_path = os.path.join(output_dir, "calibration_params.json")
        with open(cal_json_path, 'w') as f:
            json.dump(cal_params, f, indent=2)
        print(f"Saved: calibration_params.json (for Java inference)")
    else:
        # Remove stale calibration file if calibration is disabled
        cal_json_path = os.path.join(output_dir, "calibration_params.json")
        if os.path.exists(cal_json_path):
            os.remove(cal_json_path)
            print("Removed stale calibration_params.json (calibration disabled)")

    print("\nOK: All models saved (" +
          ", ".join(f"{k}: {v}" for k, v in feature_counts.items()) + " features)")


def main():
    """Main training pipeline."""
    args = parse_args()
    validate_regime_args(args)
    validate_sampling_args(args)

    if args.features_config:
        feature_config.set_feature_config(args.features_config)
        print(f"Using features config: {feature_config.get_feature_config().config_path}")

    # Run fingerprint analysis if requested (auto-selects training window)
    if args.use_fingerprint:
        if args.train_start or args.train_end:
            print("Warning: --use-fingerprint overrides --train-start and --train-end")
        fp_train_start, fp_train_end = run_fingerprint_analysis(args.fingerprint_top_n)
        args.train_start = fp_train_start
        args.train_end = fp_train_end

    # Override timeframes if provided
    if args.train_start:
        timeframes.TRAIN_START = pd.to_datetime(args.train_start)
    if args.train_end:
        timeframes.TRAIN_END = pd.to_datetime(args.train_end)
    if args.backtest_start:
        timeframes.BACKTEST_START = pd.to_datetime(args.backtest_start)
    if args.backtest_end:
        timeframes.BACKTEST_END = pd.to_datetime(args.backtest_end)

    # Sealed hold-out guard. Training on hold-out bars destroys the hold-out just as
    # surely as backtesting on them, so both ends are checked here.
    _holdout_hit = timeframes.holdout_violation(
        timeframes.TRAIN_END, timeframes.BACKTEST_END)
    if _holdout_hit is not None and not getattr(args, 'unseal_holdout', False):
        raise SystemExit(
            f"{_holdout_hit.date()} reaches into the SEALED HOLD-OUT (from "
            f"{timeframes.HOLDOUT_START.date()}). That window is the only genuinely "
            f"unseen data in this project — everything before it has been through "
            f"~1,800 backtests — and is meant to be used exactly once, on the "
            f"configuration the pre-registered protocol nominates "
            f"(docs/preregistration.md). Shorten the window, or pass --unseal-holdout "
            f"if this IS that single final evaluation."
        )
    if getattr(args, 'unseal_holdout', False):
        print("=" * 78)
        print("SEALED HOLD-OUT UNSEALED — this run consumes the one-shot final evaluation.")
        print("Any configuration choice made after reading it is no longer out-of-sample.")
        print("=" * 78)

    # Adjust data load dates.
    # The warm-up is derived from the feature config's longest rolling window
    # (get_warmup_days: currently 512 bars -> ~2.5 calendar years, because 512
    # *daily* bars is the binding case) instead of a flat 90-day buffer. 90 days
    # is ~64 daily bars against a `lookback: 500`, and a short warm-up does not
    # fail loudly: the percentile features simply mean something else over the
    # stretch where the rolling window is not yet full, so fold-to-fold variation
    # partly measures how much history each fold happened to load.
    warmup_days = feature_config.get_feature_config().get_warmup_days()
    timeframes.apply_feature_warmup(warmup_days)
    
    if args.train_end or args.backtest_end:
        latest = max(
            timeframes.TRAIN_END if args.train_end else timeframes.DATALOAD_END,
            timeframes.BACKTEST_END if args.backtest_end else timeframes.DATALOAD_END
        )
        timeframes.DATALOAD_END = latest
    
    # Setup directories
    run_dirs = setup_directories(args.run_id)
    
    print("\n" + "=" * 80)
    print("ADVANCED TRAINING PIPELINE")
    print("=" * 80)
    print(f"Run ID: {args.run_id or 'default'}")
    print(f"MI Threshold: {args.mi_threshold}")
    print(f"PFI Threshold: {args.pfi_threshold}")
    print(f"CV Splits: {args.cv_splits}")
    print(f"Training period: {timeframes.TRAIN_START} to {timeframes.TRAIN_END}")
    print(f"Data load range: {timeframes.DATALOAD_START} to {timeframes.DATALOAD_END} "
          f"({warmup_days}d warm-up for a "
          f"{feature_config.get_feature_config().get_max_lookback_bars()}-bar max lookback)")
    print("=" * 80 + "\n")

    # Experiment tracking (opt-in --wandb). Group = run_id, so this training and
    # every backtest later executed against the same run_id share one W&B group.
    wandb_config = experiment_tracking.args_config_dict(args)
    wandb_config.update({
        'train_start': str(timeframes.TRAIN_START),
        'train_end': str(timeframes.TRAIN_END),
        'backtest_start': str(timeframes.BACKTEST_START),
        'backtest_end': str(timeframes.BACKTEST_END),
        # --no-cv-boost-rounds is the only negatively-named flag here, so its raw
        # 'false' reads as "not disabled". State the feature positively as well.
        'cv_boost_rounds_enabled': not args.no_cv_boost_rounds,
    })
    experiment_tracking.init_wandb_run(
        args, job_type='train', run_id=args.run_id, config=wandb_config)

    # 1. Load data
    df_m15, df_4hours, df_daily = load_and_prepare_data(args)

    # 2. Calculate features
    combined, features_m15 = calculate_features(df_m15, df_4hours, df_daily)
    
    # 3. Generate regime labels
    print("\n" + "=" * 80)
    print("GENERATING REGIME LABELS")
    print("=" * 80 + "\n")
    
    # Generate regime labels from daily OHLC data (more stable: 500-bar lookback = ~2 years vs ~5 days on M15)
    # --regime-label-source selects the regime DEFINITION behind these labels:
    #   rule -> ADX/price-efficiency thresholds (the baseline arm)
    #   ml   -> the fitted regime model (data/regime_daily.csv), so the ML arm is
    #           labelled by its own regime definition rather than the rule's.
    _regime_label_source = getattr(args, 'regime_label_source', 'rule')
    _direction_aware = getattr(args, 'direction_aware_regime', False)
    daily_for_regime = df_daily[['high', 'low', 'close']].copy()
    if _regime_label_source == 'ml':
        from ModelTrading.source.python.features.regime_model import regime_csv_path
        # Read the same algorithm's file the rgm_* FEATURES come from (features.yaml
        # parameters.rgm_algo). Labelling from one algorithm while feeding another's
        # features would make the target unlearnable from the inputs.
        _rgm_algo = feature_config.get_feature_config().get_parameters().get('rgm_algo')
        _rgm_csv = regime_csv_path('daily', _rgm_algo)
        if not _rgm_csv.exists():
            raise FileNotFoundError(
                f"--regime-label-source ml requires {_rgm_csv}, which is missing. "
                f"Run: python data/update_regime_model_data.py --algo {_rgm_algo}"
            )
        _rgm = pd.read_csv(_rgm_csv, parse_dates=['date']).set_index('date').sort_index()
        # Align to the daily bars used for labelling; values are unshifted, matching
        # the rule-based variant which also reads current-bar indicators.
        _rgm = _rgm.reindex(daily_for_regime.index)
        regime_labels_daily = generate_regime_labels_from_scores(
            _rgm,
            trend_threshold=getattr(args, 'regime_label_trend_threshold', 0.15),
            direction_aware=_direction_aware,
        )
        print(f"Regime labels from ML regime model ({_rgm_csv.name}), "
              f"trend threshold |rgm_trend_score| > {getattr(args, 'regime_label_trend_threshold', 0.15)}")
    else:
        regime_labels_daily = generate_regime_labels(
            daily_for_regime,
            direction_aware=_direction_aware,
        )
    # Forward-fill to M15 granularity — each M15 bar inherits the regime of its trading day
    regime_labels = regime_labels_daily.reindex(combined.index, method='ffill')
    print_regime_statistics(regime_labels.dropna())
    
    # Save regime labels
    regime_path = os.path.join(run_dirs['regime_dir'], "regime_labels.parquet")
    regime_labels.to_parquet(regime_path)
    print(f"Regime labels saved to: {regime_path}")
    
    # 4. Prepare feature matrix
    config = feature_config.get_feature_config()
    usedInModel_features = config.get_usedInModel_features()
    required_features = [f for f in usedInModel_features if f in combined.columns]
    
    X = combined[required_features].copy()
    X = datahandling.remove_duplicates(X, "X")
    X = X.astype(np.float32)
    
    # 5. Generate labels (returns dict with long_fast, long_slow, short_fast, short_slow, reg)
    # Pass args for label mode configuration and regime_labels for regime-aware adjustments
    labels, pip_value, targets_df, raw_labels = generate_labels(
        df_m15,
        args=args,
        regime_labels=regime_labels,
        df_daily=df_daily,
    )

    # Label-leak test features (see m15_y_test / daily_y_test in features.yaml).
    # Injected here because labels don't exist when calculate_features() runs.
    if 'm15_y' in usedInModel_features or 'daily_y' in usedInModel_features:
        y_signed = (labels['long_slow'].astype('float32')
                    - labels['short_slow'].astype('float32'))
        if 'm15_y' in usedInModel_features:
            X['m15_y'] = y_signed.reindex(X.index).astype(np.float32)
            print("LABEL-LEAK TEST: injected m15_y = long_slow - short_slow")
        if 'daily_y' in usedInModel_features:
            X['daily_y'] = y_signed.reindex(X.index).astype(np.float32)
            print("LABEL-LEAK TEST: injected daily_y = long_slow - short_slow")

    # Plot label distribution if --plot flag is set
    if args.plot:
        ohlc_for_plot = df_m15[['open', 'high', 'low', 'close']].copy()
        plot.plot_label_distribution(
            labels['long_slow'], labels['short_slow'],
            ohlc_for_plot, pip_value, 9999999,
            long_fast=labels['long_fast'],
            short_fast=labels['short_fast'],
            last_weeks=args.plot_last_weeks,
        )
    else:
        print("Skipping label distribution plot (use --plot or -p to enable)")

    # 6. Align data
    print("\n" + "=" * 80)
    print("DATA ALIGNMENT")
    print("=" * 80 + "\n")

    # Get first valid row
    nan_per_row = X.isna().sum(axis=1)
    has_valid_rows = (nan_per_row == 0).any()

    if not has_valid_rows:
        print("ERROR: No rows without NaN values found. Check feature calculations.")
        print(report_nan_columns(X))
        sys.exit(1)

    first_valid_idx = nan_per_row[nan_per_row == 0].index[0]
    first_valid_pos = X.index.get_loc(first_valid_idx)

    X = X.iloc[first_valid_pos:].copy()
    valid_index = X.index

    # Reindex all labels
    for key in labels:
        labels[key] = labels[key].reindex(valid_index)
    regime_labels = regime_labels.reindex(valid_index)

    # Extract OHLC for backtesting (aligned to features)
    ohlc = features_m15.reindex(valid_index)[["m15_open", "m15_high", "m15_low", "m15_close"]].copy()

    # Remove NaN targets (use slow labels as reference, they have fewer NaNs)
    mask = (~labels['long_slow'].isna() & ~labels['short_slow'].isna() &
            ~labels['long_fast'].isna() & ~labels['short_fast'].isna() &
            ~ohlc.isna().any(axis=1))
    X = X.loc[mask]
    for key in labels:
        labels[key] = labels[key].loc[mask]
    regime_labels = regime_labels.loc[mask]
    ohlc = ohlc.loc[mask]

    print(f"Final dataset: {len(X)} rows, {len(X.columns)} features")

    # 7. Train/test split
    train_mask = (X.index >= timeframes.TRAIN_START) & (X.index <= timeframes.TRAIN_END)

    X_train = X.loc[train_mask].copy()
    y_train = {key: labels[key].loc[train_mask].copy() for key in labels}
    regime_train = regime_labels.loc[train_mask].copy()
    ohlc_train = ohlc.loc[train_mask].copy()

    print(f"Training set: {len(X_train)} rows")

    # 7b. Regime filtering (optional)
    if getattr(args, 'regime_filter', False):
        X_train, y_train, ohlc_train, _ = apply_regime_filter(
            X_train, y_train, ohlc_train, regime_train, args.regime_type,
        )
        regime_train = regime_train.loc[X_train.index].copy()
        print(f"Training set after regime filter: {len(X_train)} rows")

    # Label distribution nach Filter.
    #
    # Iterate the label keys, NOT the dict: `labels` also carries 'metadata', the targets
    # frame. Summing that used to print a meaningless number; since the barrier resolution
    # bars are recorded on it as t1_* TIMESTAMP columns, summing it raises
    # "unsupported operand type(s) for +: 'float' and 'Timestamp'" and kills every label
    # mode that routes through generate_dynamic_labels (static, atr_scaled,
    # daily_vol_scaled, regime_conditional, trend_only).
    for key in _LABEL_SERIES_KEYS:
        if key not in y_train:
            continue
        total = len(y_train[key])
        pos = float(y_train[key].to_numpy().sum())
        print(f"  {key}: {pos:.0f}/{total} = {pos/total:.1%} positive")

    # 7c. Per-scope fast training window override (decouples M15 start from 4h/daily start)
    _fast_train_start = getattr(args, 'fast_train_start', None)
    if _fast_train_start:
        _fast_ts = pd.to_datetime(_fast_train_start)
        fast_scope_mask = (X.index >= _fast_ts) & (X.index <= timeframes.TRAIN_END)
        X_train_fast = X.loc[fast_scope_mask].copy()
        y_train_fast = {key: labels[key].loc[fast_scope_mask].copy() for key in labels}
        regime_train_fast = regime_labels.loc[fast_scope_mask].copy()
        print(f"Fast scope override: {_fast_ts.date()} to {timeframes.TRAIN_END.date()} ({len(X_train_fast)} M15 bars)")
        # The fast scope is rebuilt from the UNFILTERED X/labels above, so the regime
        # filter applied to the slow scope must be re-applied here. Without this,
        # --regime-filter silently has NO effect on the fast models whenever
        # --fast-train-start is set (the else-branch below inherits the filtered set).
        if getattr(args, 'regime_filter', False):
            X_train_fast, y_train_fast, _, _ = apply_regime_filter(
                X_train_fast, y_train_fast, None, regime_train_fast, args.regime_type,
            )
            regime_train_fast = regime_train_fast.loc[X_train_fast.index].copy()
            print(f"Fast scope after regime filter: {len(X_train_fast)} rows")
    else:
        X_train_fast = X_train
        y_train_fast = y_train
        regime_train_fast = regime_train

    # 7d. Optionally replace fast labels with slow-style directional labels
    if getattr(args, 'fast_use_slow_label', False):
        y_train_fast['long_fast'] = y_train_fast['long_slow'].copy()
        y_train_fast['short_fast'] = y_train_fast['short_slow'].copy()
        pos = float(y_train_fast['long_fast'].sum())
        total = len(y_train_fast['long_fast'])
        print(f"Fast scope: using slow-style labels (long_slow -> long_fast)  "
              f"{pos:.0f}/{total} = {pos/total:.1%} positive")

    # 8. Per-model candidate feature sets (config-driven). Each model gets its
    #    own feature set, selection, scaler and ONNX input shape. fast/slow is
    #    now only a *cadence* grouping: *_fast models train on M15 bars,
    #    *_slow on 4h-resampled bars. The old prefix split and the
    #    --fast-4h-context CLI hack are replaced by the `models:` tags in
    #    features.yaml (see FeatureConfig.get_model_features).
    MODEL_KEYS = config.get_models()
    fast_models = [m for m in MODEL_KEYS if feature_config.model_cadence(m) == 'fast']
    slow_models = [m for m in MODEL_KEYS if feature_config.model_cadence(m) == 'slow']

    def _existing(cols):
        present = [c for c in cols if c in X_train.columns]
        missing = [c for c in cols if c not in X_train.columns]
        if missing:
            print(f"WARNING: configured features not found in computed columns, skipped: {missing}")
        return present

    cand_cols = {m: _existing(config.get_model_features(m)) for m in MODEL_KEYS}
    print("\nPer-model candidate feature sets:")
    for m in MODEL_KEYS:
        print(f"  {m} ({feature_config.model_cadence(m)} cadence): {len(cand_cols[m])} features")

    # Columns needed at slow (4h) cadence = union across all slow-cadence models.
    slow_cols_union = [c for c in X_train.columns
                       if any(c in cand_cols[m] for m in slow_models)]

    _seed = getattr(args, 'seed', None)

    # Regime-scoped recall floors (None = off): applied on top of the overall target in
    # the threshold selection, for every model. See _select_threshold_with_recall_floors.
    _target_recall_trend = getattr(args, 'target_recall_trend', None)
    _target_recall_range = getattr(args, 'target_recall_range', None)
    _global_target_recall = _resolve_overall_target_recall(
        getattr(args, 'target_recall', None), _target_recall_trend, _target_recall_range)
    if (getattr(args, 'target_recall', None) is None
            and (_target_recall_trend is not None or _target_recall_range is not None)):
        print("Overall --target-recall not given while a regime floor is set -> overall "
              "recall floor disabled (0.00); the regime floor(s) alone constrain the "
              "threshold. Pass --target-recall explicitly to add an overall floor.")
    _fast_target_recall = getattr(args, 'fast_target_recall', None)
    _fast_target_recall = _fast_target_recall if _fast_target_recall is not None else _global_target_recall
    _slow_target_recall = getattr(args, 'slow_target_recall', None)
    _slow_target_recall = _slow_target_recall if _slow_target_recall is not None else _global_target_recall

    _fast_spw_factor = getattr(args, 'fast_spw_factor', 1.0)
    _fast_nbr = getattr(args, 'fast_num_boost_round', None)
    fast_num_boost_round = _fast_nbr if _fast_nbr is not None else getattr(args, 'num_boost_round', 200)
    _fast_mcw = getattr(args, 'fast_min_child_weight', None)
    _fast_mcw_val = _fast_mcw if _fast_mcw is not None else getattr(args, 'min_child_weight', 3)
    _fast_md = getattr(args, 'fast_max_depth', None)
    _fast_md_val = _fast_md if _fast_md is not None else getattr(args, 'max_depth', 5)

    params_fast_cls = {
        "max_depth": _fast_md_val,
        "eta": getattr(args, 'eta', 0.1),
        "subsample": getattr(args, 'subsample', 0.8),
        "colsample_bytree": getattr(args, 'colsample_bytree', 0.8),
        "colsample_bynode": getattr(args, 'colsample_bynode', 1.0),
        "min_child_weight": _fast_mcw_val,
        "lambda": getattr(args, 'fast_lambda', 1.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        **({'seed': _seed} if _seed is not None else {}),
    }

    params_slow_cls = {
        "max_depth": getattr(args, 'slow_max_depth', 4),
        "eta": getattr(args, 'eta', 0.1),
        "subsample": getattr(args, 'subsample', 0.8),
        "colsample_bytree": getattr(args, 'colsample_bytree', 0.8),
        "colsample_bynode": getattr(args, 'colsample_bynode', 1.0),
        "min_child_weight": getattr(args, 'slow_min_child_weight', 5),
        "lambda": getattr(args, 'slow_lambda', 2.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        **({'seed': _seed} if _seed is not None else {}),
    }

    _slow_spw_factor = getattr(args, 'slow_spw_factor', 1.0)

    # 9. Build the 4h-resampled slow frame + labels. M15 forward-fill creates ~16
    # identical rows per 4h bar, inflating effective sample size and causing severe
    # overfitting. Deduplicate by taking the first M15 bar of each 4h period.
    X_train_slow_4h = X_train[slow_cols_union].resample('4h').first().dropna(how='all')
    # window_cascade produces sparse setup labels (a few M15 bars per window); .first()
    # would drop them whenever the 4h bar's opening M15 isn't labeled. Use .max() so any
    # positive M15 bar inside the 4h period marks the 4h bar positive.
    _slow_label_agg = 'max' if getattr(args, 'label_mode', None) == 'window_cascade' else 'first'
    y_train_slow_4h = {
        'long_slow':  getattr(y_train['long_slow'].resample('4h'), _slow_label_agg)().reindex(X_train_slow_4h.index, fill_value=0),
        'short_slow': getattr(y_train['short_slow'].resample('4h'), _slow_label_agg)().reindex(X_train_slow_4h.index, fill_value=0),
    }
    print(f"\nSlow cadence: {len(X_train_slow_4h)} 4h bars from {len(X_train)} M15 bars")

    # Regime masks for regime-aware MI + per-regime metrics.
    # Both regime_conditional and trend_only force range bars to label 0, so without
    # a regime mask the global MI would select trend-vs-range detector features rather
    # than features that discriminate winners *within* a trend. Routing both modes
    # through per-regime MI fixes that.
    _regime_aware = getattr(args, 'label_mode', 'static') in ('regime_conditional', 'trend_only')
    regime_slow_4h_df = regime_train.resample('4h').first().reindex(X_train_slow_4h.index)
    # Detect the regime encoding once, on the full training frame, so both cadences
    # agree even if a resampled frame happens to contain no downtrend bars.
    _direction_aware_labels = bool((regime_train['regime_trend'] == -1).any())
    # The signal bucket is resolved per MODEL, not just per cadence — with
    # direction-aware regimes a short model's positives live in downtrend bars and a
    # long model's in uptrend bars. Built inside _cadence_bits(); see signal_regime_mask().

    # Per-cadence feature-selection arg variants (fast/slow MI + skip-PFI overrides).
    if getattr(args, 'fast_mi_threshold', None) is not None:
        args_fast = copy.copy(args)
        args_fast.mi_threshold = args.fast_mi_threshold
    else:
        args_fast = args
    if getattr(args, 'skip_pfi_slow', False) or getattr(args, 'slow_mi_threshold', None) is not None:
        args_slow = copy.copy(args)
        if getattr(args, 'skip_pfi_slow', False):
            args_slow.skip_pfi = True
        if getattr(args, 'slow_mi_threshold', None) is not None:
            args_slow.mi_threshold = args.slow_mi_threshold
    else:
        args_slow = args

    # --cv-gap is given in M15 bars; the slow cadence trains on 4h rows
    # (1 row = 16 M15 bars), so its embargo is the same wall-clock span in 4h rows.
    _label_horizon = int(targets_df.attrs.get('label_horizon_bars', 0) or 0)
    _cv_gap_m15 = int(getattr(args, 'cv_gap', 0) or 0)
    if _cv_gap_m15 < 0:
        # --cv-gap -1 = auto: embargo exactly the span the labels peek over.
        _cv_gap_m15 = _label_horizon
        print(f"CV embargo auto-derived from the label horizon: {_cv_gap_m15} M15 bars")
    _cv_gap_4h = int(np.ceil(_cv_gap_m15 / 16)) if _cv_gap_m15 > 0 else 0
    # The W&B config was written at init from the raw args, where --cv-gap -1 only
    # says "auto" — record the embargo the folds actually used.
    experiment_tracking.update_config({'cv_gap_m15_effective': _cv_gap_m15,
                                       'cv_gap_4h_effective': _cv_gap_4h,
                                       'label_horizon_bars': _label_horizon})
    if _cv_gap_m15 > 0:
        print(f"CV embargo enabled: gap={_cv_gap_m15} M15 rows (fast), {_cv_gap_4h} 4h rows (slow)")
    elif _label_horizon > 0:
        print(f"WARNING: --cv-gap is 0 but the labels peek {_label_horizon} M15 bars "
              f"({_label_horizon / 96:.1f} days) ahead. The last rows of every training "
              f"fold are computed from bars inside the following validation fold, so the "
              f"CV metrics are optimistic at the fold seam. Pass --cv-gap -1 to embargo "
              f"exactly the label horizon.")

    _fast_cond_reported = set()

    def _fast_row_index(mk):
        """`fast_setup_row_index` for this run, reporting the effect once per model."""
        idx = X_train_fast.index
        keep = fast_setup_row_index(
            idx, y_train_fast, mk,
            conditional=getattr(args, 'fast_conditional_on_setup', False))
        if len(keep) != len(idx) and mk not in _fast_cond_reported:
            _fast_cond_reported.add(mk)
            y_mk = y_train_fast[mk].reindex(idx).fillna(0)
            print(f"Fast scope conditioned on setup bars for '{mk}': "
                  f"{len(keep)}/{len(idx)} bars, positive rate "
                  f"{float(y_mk.mean()):.2%} -> {float(y_mk.loc[keep].mean()):.2%}")
        return keep

    def _regime_trend_values(rdf, index):
        """regime_trend aligned to a training index, for the regime-scoped recall floors."""
        if rdf is None or 'regime_trend' not in rdf.columns:
            return None
        return rdf['regime_trend'].reindex(index).values

    def _cadence_bits(mk):
        """Resolve the cadence-specific training inputs for a model key."""
        if feature_config.model_cadence(mk) == 'fast':
            idx = _fast_row_index(mk)
            regime_mask = signal_regime_mask(
                regime_train_fast, mk, _regime_aware, _direction_aware_labels)
            if len(idx) != len(X_train_fast.index):
                return dict(Xframe=X_train_fast.loc[idx], index=idx,
                            y=y_train_fast[mk].loc[idx],
                            params=params_fast_cls, spw=_fast_spw_factor,
                            recall=_fast_target_recall,
                            nbr=fast_num_boost_round, fs_args=args_fast, fs_dir='fast',
                            regime_mask=(regime_mask.loc[idx]
                                         if regime_mask is not None else None),
                            regime_trend=_regime_trend_values(regime_train_fast, idx),
                            gap=_cv_gap_m15)
            return dict(Xframe=X_train_fast, index=X_train_fast.index, y=y_train_fast[mk],
                        params=params_fast_cls, spw=_fast_spw_factor, recall=_fast_target_recall,
                        nbr=fast_num_boost_round, fs_args=args_fast, fs_dir='fast',
                        regime_mask=regime_mask,
                        regime_trend=_regime_trend_values(regime_train_fast, X_train_fast.index),
                        gap=_cv_gap_m15)
        return dict(Xframe=X_train_slow_4h, index=X_train_slow_4h.index, y=y_train_slow_4h[mk],
                    params=params_slow_cls, spw=_slow_spw_factor, recall=_slow_target_recall,
                    nbr=args.slow_num_boost_round, fs_args=args_slow, fs_dir='slow',
                    regime_mask=signal_regime_mask(
                        regime_slow_4h_df, mk, _regime_aware, _direction_aware_labels),
                    regime_trend=_regime_trend_values(regime_slow_4h_df, X_train_slow_4h.index),
                    gap=_cv_gap_4h)

    for fs in ('fast', 'slow'):
        os.makedirs(os.path.join(run_dirs['feature_selection_dir'], fs), exist_ok=True)

    # 10-12. Per-model: initial scale -> CV -> feature selection -> refit scaler ->
    # post-selection CV -> learning curve. Each model has its OWN selected set/scaler.
    selected, scaler, X_scaled, X_df = {}, {}, {}, {}
    cv_pre, cv_post, lc = {}, {}, {}
    label_coverage = {}   # per model: which CV folds would train on zero positives
    sample_weight_by_model = {}
    _sample_weight_mode = getattr(args, 'sample_weight', 'none')
    for mk in MODEL_KEYS:
        bits = _cadence_bits(mk)
        cols = cand_cols[mk]
        y = bits['y']
        y_arr = y.values if isinstance(y, pd.Series) else y
        # Name the MI signal bucket: under direction-aware regimes the column the MI
        # summary calls "trend" holds uptrend MI for long models and downtrend MI for
        # short models, so state which one this run actually measured.
        _sig = ('uptrend' if mk.startswith('long') else 'downtrend') if _direction_aware_labels else 'trend'
        _sig_note = f", MI signal regime: {_sig}" if bits['regime_mask'] is not None else ""
        print(f"\n=== Model '{mk}' ({len(cols)} candidate features, {len(bits['Xframe'])} bars, "
              f"{feature_config.model_cadence(mk)} cadence{_sig_note}) ===")

        # Pre-flight: a fold that will train on zero positives is knowable from the label
        # series alone, before any tree is grown. Reported for every model and recorded in
        # training_summary.json so a window with a label-empty head can never again be
        # read as a weak model instead of an uninterpretable measurement.
        label_coverage[mk] = label_coverage_report(
            y_arr, bits['index'], n_splits=args.cv_splits, gap=bits['gap'])

        # Label uniqueness. Always MEASURED (n_eff is the sample size every standard
        # error in this project should have been using); only APPLIED as training
        # weights under --sample-weight uniqueness.
        _t1 = sample_weights_mod.t1_from_metadata(
            targets_df, mk, index=bits['index'], fallback_horizon=None)
        if _t1 is not None:
            _u = sample_weights_mod.average_uniqueness(_t1)
            _n_eff = float(_u.sum())
            label_coverage[mk]['n_rows'] = int(len(_t1))
            label_coverage[mk]['n_eff'] = _n_eff
            label_coverage[mk]['mean_uniqueness'] = float(_u.mean())
            print(f"  Sample uniqueness '{mk}': n={len(_t1)}, n_eff={_n_eff:.0f} "
                  f"(mean uniqueness {_u.mean():.3f}) — standard errors computed with n "
                  f"instead of n_eff are understated by ~{np.sqrt(len(_t1) / max(_n_eff, 1)):.1f}x")
            if _sample_weight_mode == 'uniqueness':
                sample_weight_by_model[mk] = sample_weights_mod.sample_weights(_t1)
        elif _sample_weight_mode == 'uniqueness':
            print(f"  WARNING: '{mk}' has no t1_{mk} column in the label metadata "
                  f"(label mode does not run the barrier race) — training unweighted.")
        if print_label_coverage(label_coverage[mk], mk) and getattr(
                args, 'fail_on_degenerate_folds', False):
            raise SystemExit(
                f"Model '{mk}': CV fold(s) "
                f"{label_coverage[mk]['degenerate_fold_numbers']} have no positive "
                f"training label (--fail-on-degenerate-folds)."
            )

        init_scaler = StandardScaler()
        Xc_scaled = init_scaler.fit_transform(bits['Xframe'][cols])

        cv_pre[mk] = run_time_series_cv(
            Xc_scaled, y_arr, bits['params'],
            n_splits=args.cv_splits, num_boost_round=bits['nbr'],
            spw_factor=bits['spw'], target_recall=bits['recall'], gap=bits['gap'],
            index=bits['index'], sample_weight=sample_weight_by_model.get(mk),
            regime_trend=bits['regime_trend'],
            target_recall_trend=_target_recall_trend,
            target_recall_range=_target_recall_range,
        )
        # PFI is the ONLY consumer of this model (run_feature_selection guards its use
        # with `not args.skip_pfi and model is not None`), so training it whenever PFI
        # is off costs 100 rounds on the full training set for nothing. skip_pfi can
        # differ per cadence via --skip-pfi-slow, hence the per-model fs_args.
        init_model = None
        if not getattr(bits['fs_args'], 'skip_pfi', False):
            dtrain_init = xgb.DMatrix(Xc_scaled, label=y_arr)
            init_model = xgb.train(
                {**bits['params'], 'base_score': _base_score_from_labels(y_arr)},
                dtrain_init, num_boost_round=100)
        Xc_df = pd.DataFrame(Xc_scaled, columns=cols, index=bits['index'])
        fs_path = os.path.join(run_dirs['feature_selection_dir'], bits['fs_dir'], mk)
        os.makedirs(fs_path, exist_ok=True)
        fs_res = run_feature_selection(
            Xc_df, y, bits['fs_args'], fs_path,
            model=init_model, regime_mask=bits['regime_mask'])
        selected[mk] = fs_res['selected_features']

        # A selection that keeps every candidate feature IN THE SAME ORDER cannot
        # change anything downstream: StandardScaler sees the identical columns, so
        # the scaled matrix is identical, and XGBoost is deterministic. Refitting the
        # scaler and re-running the CV would spend cv_splits fits per model to
        # reproduce cv_pre bit for bit -- which is exactly what the "Before/After"
        # impact table shows when it prints `Delta F1: +0.0000`.
        # Order matters as much as membership: colsample_bytree picks columns by
        # index, so a permuted list trains different trees (see MutualInformationFilter).
        selection_is_noop = list(selected[mk]) == list(cols)

        if selection_is_noop:
            scaler[mk], Xs = init_scaler, Xc_scaled
        else:
            sc = StandardScaler()
            Xs = sc.fit_transform(bits['Xframe'][selected[mk]])
            scaler[mk] = sc
        X_scaled[mk] = Xs
        X_df[mk] = pd.DataFrame(Xs, columns=selected[mk], index=bits['index'])

        if selection_is_noop:
            print(f"  Selection kept all {len(cols)} features "
                  f"-> reusing the pre-selection CV (identical by construction)")
            cv_post[mk] = dict(cv_pre[mk])
        else:
            cv_post[mk] = run_time_series_cv(
                Xs, y_arr, bits['params'],
                n_splits=args.cv_splits, num_boost_round=bits['nbr'],
                spw_factor=bits['spw'], target_recall=bits['recall'], gap=bits['gap'],
                index=bits['index'], sample_weight=sample_weight_by_model.get(mk),
                regime_trend=bits['regime_trend'],
                target_recall_trend=_target_recall_trend,
                target_recall_range=_target_recall_range,
            )
        lc[mk] = aggregate_fold_curves(cv_post[mk].get('fold_curves'))
        print(f"  Selected {len(selected[mk])}/{len(cols)} features")

    # Representative aliases (long_* per cadence) keep the downstream impact print,
    # summary and CSV export unchanged. Per-model artefacts use the dicts above.
    selected_fast, selected_slow = selected['long_fast'], selected['long_slow']
    scaler_slow = scaler['long_slow']  # used for the CSV review export
    cv_results_fast, cv_results_fast_post = cv_pre['long_fast'], cv_post['long_fast']
    cv_results_slow, cv_results_slow_post = cv_pre['long_slow'], cv_post['long_slow']
    m15_cols, daily_cols = cand_cols['long_fast'], cand_cols['long_slow']

    # --- Learning curves (averaged over all CV folds, post-selection features) ---
    print(f"\n--- Learning curves (mean over {args.cv_splits} CV folds) ---")
    learning_curves = {mk: lc.get(mk) for mk in ('long_fast', 'short_fast', 'long_slow', 'short_slow')}

    # Derive each classifier's num_boost_round from its CV folds: every fold votes with
    # the same rule, the MEDIAN vote wins. Never average the fold curves first — the
    # argmax of a mean follows the loudest fold, which is the one with no signal. See
    # best_round_from_folds for the measured case that motivates this.
    round_votes = {}
    if getattr(args, 'no_cv_boost_rounds', False):
        round_overrides = {}
        print("  CV-derived boost rounds: disabled (--no-cv-boost-rounds)")
    else:
        round_overrides = {}
        for mk in MODEL_KEYS:
            # A fold whose model came out constant votes for the cap because its flat
            # 0.500 curve looks like "never degrades". That is an artefact of having no
            # positives to train on, not a capacity signal — it must not vote.
            dead = cv_post[mk].get('degenerate_folds') or []
            rounds, votes = best_round_from_folds(
                cv_post[mk].get('fold_curves'), _cadence_bits(mk)['nbr'],
                skip_folds=dead)
            round_votes[mk] = votes
            if rounds is not None:
                round_overrides[f'target_{mk}'] = rounds
        for mk in MODEL_KEYS:
            cap = _cadence_bits(mk)['nbr']
            dead = cv_post[mk].get('degenerate_folds') or []
            dead_note = f", {len(dead)} degenerate fold(s) {dead} excluded" if dead else ""
            if f'target_{mk}' in round_overrides:
                r = round_overrides[f'target_{mk}']
                why = ('no fold consensus (votes at floor AND cap) -> cap kept'
                       if r == cap and len(set(round_votes[mk])) > 1 else 'median of votes')
                print(f"  CV boost rounds {mk}: {r} (cap {cap}) — {why}, "
                      f"votes {round_votes[mk]}{dead_note}")
            else:
                print(f"  CV boost rounds {mk}: no usable AUC curve "
                      f"(single-class val folds{dead_note}) -> holdout/default fallback")

    # Plotted after the round choice so the PNG can mark the round each model is refit
    # for — the curves themselves always run to the full cap.
    lc_png = os.path.join(run_dirs['visualization_dir'], "learning_curves.png")
    lc_json = os.path.join(run_dirs['visualization_dir'], "learning_curves.json")
    plot_learning_curves(learning_curves, lc_png,
                         chosen_rounds={mk: round_overrides.get(f'target_{mk}')
                                        for mk in learning_curves})
    # Persist the PER-FOLD curves alongside the averaged one. The round choice is a
    # statement about fold disagreement, so without them no claim about it — and no
    # comparison of aggregation rules — can be checked without retraining. See
    # analytics/boost_round_study.py, which reads exactly this file.
    lc_payload = {}
    for mk, ev in learning_curves.items():
        if ev is None:
            continue
        lc_payload[mk] = {
            **ev,
            'cap': _cadence_bits(mk)['nbr'],
            'resolved_rounds': round_overrides.get(f'target_{mk}'),
            'votes': round_votes.get(mk, []),
            'folds': [round_curve(f) for f in (cv_post[mk].get('fold_curves') or [])],
        }
    with open(lc_json, 'w') as f:
        json.dump(lc_payload, f, indent=2)
    print(f"  Learning curve data (mean + per-fold): {lc_json}")

    # --- Gate metrics measured on the model that actually ships ------------------------
    # The CV above trains every fold at the CAP, but the exported model is refit for the
    # RESOLVED round count. Without this second pass the acceptance gates describe a model
    # that is never deployed: in the es_off/es_on A/B (2026-08-08) both runs reported
    # bit-identical CV metrics (AUC 0.7540, F1 0.3858, identical confusion matrix) while
    # shipping a 200-round and a 20-round long_slow whose backtests differed by a factor
    # of four. Re-running the CV at the resolved count makes the round choice visible to
    # the gates. Models left at the cap reuse the existing result — no extra fits.
    #
    # Caveat, deliberately not hidden: the rounds were chosen on these same folds, so
    # `*_final` removes the measured-vs-shipped MISMATCH, not the selection bias. Fully
    # unbiased numbers would need nested CV (resolve the rounds inside each fold's own
    # training part); these figures remain mildly optimistic about the round choice.
    print("\n--- CV at the resolved boost rounds (gate metrics) ---")
    cv_final = {}
    for mk in MODEL_KEYS:
        bits = _cadence_bits(mk)
        resolved = round_overrides.get(f'target_{mk}')
        if resolved is None or resolved == bits['nbr']:
            cv_final[mk] = cv_post[mk]
            note = ('cap' if resolved == bits['nbr']
                    else 'resolved later by the holdout fallback — gates measure the cap')
            print(f"  {mk}: {bits['nbr']} rounds ({note}) -> reusing post-selection CV")
            continue
        y_arr = bits['y'].values if isinstance(bits['y'], pd.Series) else bits['y']
        cv_final[mk] = run_time_series_cv(
            X_scaled[mk], y_arr, bits['params'],
            n_splits=args.cv_splits, num_boost_round=resolved,
            spw_factor=bits['spw'], target_recall=bits['recall'],
            index=bits['index'], sample_weight=sample_weight_by_model.get(mk),
            regime_trend=bits['regime_trend'],
            target_recall_trend=_target_recall_trend,
            target_recall_range=_target_recall_range)
        a0, a1 = cv_post[mk]['global_val_auc_roc'], cv_final[mk]['global_val_auc_roc']
        f0, f1v = cv_post[mk]['global_val_f1'], cv_final[mk]['global_val_f1']
        print(f"  {mk}: {bits['nbr']} -> {resolved} rounds | AUC {_fmt_auc(a0)} -> {_fmt_auc(a1)} "
              f"({a1 - a0:+.4f}) | F1 {f0:.4f} -> {f1v:.4f} ({f1v - f0:+.4f})")

    # --- Out-of-fold scores: the raw material for every model-quality figure -----------
    # `training_summary.json` carries the aggregates, and an aggregate cannot be turned
    # back into a ROC, a precision/recall curve, a reliability diagram or a confusion
    # matrix at a different threshold. The scores are exported for all four models at
    # both the cap ('post') and the resolved rounds ('final') so those diagnostics —
    # mandatory per docs/preregistration.md §5 — can be drawn without retraining.
    _oof_regime = {'fast': regime_train_fast, 'slow': regime_slow_4h_df}
    oof_parts = []
    # Per-model, per-regime gate metrics at the RESOLVED rounds — recorded in the
    # summary (and W&B) so the regime breakdown is not console-only. Computed here
    # because the regime frame is already reindexed to each model's own index.
    per_regime_final = {}
    for mk in MODEL_KEYS:
        bits = _cadence_bits(mk)
        rdf = _oof_regime.get(feature_config.model_cadence(mk))
        if rdf is not None and not rdf.index.equals(pd.Index(bits['index'])):
            rdf = rdf.reindex(bits['index'])
        regime_rows, regime_thr = compute_per_regime_metrics(cv_final[mk], rdf)
        per_regime_final[mk] = {'threshold': regime_thr, 'rows': regime_rows}
        for stage, res in (('post', cv_post[mk]), ('final', cv_final[mk])):
            oof_parts.append(build_oof_frame(res, bits['index'], mk, stage, regime_df=rdf))
    oof_parts = [p for p in oof_parts if len(p)]
    if oof_parts:
        oof_df = pd.concat(oof_parts, ignore_index=True)
        oof_path = os.path.join(run_dirs['generated_dir'], 'oof_predictions.parquet')
        oof_df.to_parquet(oof_path, index=False)
        print(f"\nOut-of-fold predictions saved: {oof_path} ({len(oof_df)} rows, "
              f"{oof_df['model'].nunique()} models x {oof_df['stage'].nunique()} stages)")

    print("\n" + "=" * 80)
    print("FEATURE SELECTION IMPACT")
    print("=" * 80)

    _floor_note = ''
    if _target_recall_trend is not None:
        _floor_note += f", TREND recall >= {_target_recall_trend:.2f}"
    if _target_recall_range is not None:
        _floor_note += f", RANGE recall >= {_target_recall_range:.2f}"
    print(f"Threshold rule: max precision s.t. recall >= target "
          f"(fast target={_fast_target_recall:.2f}, slow target={_slow_target_recall:.2f}"
          f"{_floor_note}); "
          f"global threshold computed on pooled CV val predictions.")
    print(f"Fast scope:")
    print(f"  Before: {len(m15_cols)} features, Val F1 = {cv_results_fast['global_val_f1']:.4f}, "
          f"Prec = {cv_results_fast['global_val_precision']:.4f}, "
          f"Recall = {cv_results_fast['global_val_recall']:.4f}, "
          f"AUC = {_fmt_auc(cv_results_fast['global_val_auc_roc'])}, "
          f"Brier = {cv_results_fast['global_val_brier']:.4f}, "
          f"thr = {cv_results_fast['global_threshold']:.3f}")
    print(f"  After:  {len(selected_fast)} features, Val F1 = {cv_results_fast_post['global_val_f1']:.4f}, "
          f"Prec = {cv_results_fast_post['global_val_precision']:.4f}, "
          f"Recall = {cv_results_fast_post['global_val_recall']:.4f}, "
          f"AUC = {_fmt_auc(cv_results_fast_post['global_val_auc_roc'])}, "
          f"Brier = {cv_results_fast_post['global_val_brier']:.4f}, "
          f"thr = {cv_results_fast_post['global_threshold']:.3f}")
    print(f"  Delta F1: {cv_results_fast_post['global_val_f1'] - cv_results_fast['global_val_f1']:+.4f}")
    print(f"Slow scope:")
    print(f"  Before: {len(daily_cols)} features, Val F1 = {cv_results_slow['global_val_f1']:.4f}, "
          f"Prec = {cv_results_slow['global_val_precision']:.4f}, "
          f"Recall = {cv_results_slow['global_val_recall']:.4f}, "
          f"AUC = {_fmt_auc(cv_results_slow['global_val_auc_roc'])}, "
          f"Brier = {cv_results_slow['global_val_brier']:.4f}, "
          f"thr = {cv_results_slow['global_threshold']:.3f}")
    print(f"  After:  {len(selected_slow)} features, Val F1 = {cv_results_slow_post['global_val_f1']:.4f}, "
          f"Prec = {cv_results_slow_post['global_val_precision']:.4f}, "
          f"Recall = {cv_results_slow_post['global_val_recall']:.4f}, "
          f"AUC = {_fmt_auc(cv_results_slow_post['global_val_auc_roc'])}, "
          f"Brier = {cv_results_slow_post['global_val_brier']:.4f}, "
          f"thr = {cv_results_slow_post['global_threshold']:.3f}")
    print(f"  Delta F1: {cv_results_slow_post['global_val_f1'] - cv_results_slow['global_val_f1']:+.4f}")

    _print_per_regime_metrics('fast scope post-selection', cv_results_fast_post, regime_train_fast)
    _print_per_regime_metrics('slow scope post-selection', cv_results_slow_post, regime_slow_4h_df)

    # Optional threshold sweep (PR-curve table) on post-selection pooled CV predictions
    sweep_fast = sweep_slow = None
    if getattr(args, 'threshold_sweep', False):
        sweep_thresholds = _parse_sweep_thresholds(getattr(args, 'sweep_thresholds', None))
        print("\n" + "=" * 80)
        print(f"THRESHOLD SWEEP (post-selection, pooled CV val predictions)")
        print(f"Thresholds: {', '.join(f'{t:.2f}' for t in sweep_thresholds)}")
        print("=" * 80)
        y_f = cv_results_fast_post['pooled_y_val']
        p_f = cv_results_fast_post['pooled_val_prob']
        sweep_fast = _compute_threshold_sweep(y_f, p_f, sweep_thresholds)
        _print_threshold_sweep('Fast scope', sweep_fast, base_rate=float(y_f.mean()) if len(y_f) else None)

        y_s = cv_results_slow_post['pooled_y_val']
        p_s = cv_results_slow_post['pooled_val_prob']
        sweep_slow = _compute_threshold_sweep(y_s, p_s, sweep_thresholds)
        _print_threshold_sweep('Slow scope', sweep_slow, base_rate=float(y_s.mean()) if len(y_s) else None)
        print("=" * 80)
    print("=" * 80 + "\n")

    # Save per-model scalers
    for mk in MODEL_KEYS:
        joblib.dump(scaler[mk], os.path.join(run_dirs['generated_dir'], f"scaler_{mk}.save"))
    print("Scalers saved: " + ", ".join(f"{mk} ({len(selected[mk])})" for mk in MODEL_KEYS))

    # 12c. Save features.json for Java (per-model scopes)
    feature_list_path = os.path.join(run_dirs['java_config_dir'], "features.json")
    helper_features = config.get_helper_features()

    selected_features = sorted(set().union(*[set(selected[mk]) for mk in MODEL_KEYS]))

    scopes_json = {}
    features_removed = {}
    for mk in MODEL_KEYS:
        scopes_json[mk] = {
            'usedInModel_features': list(selected[mk]),
            'usedInModel_count': len(selected[mk]),
        }
        features_removed[mk] = [f for f in cand_cols[mk] if f not in selected[mk]]

    with open(feature_list_path, 'w') as f:
        json.dump({
            'helper_features': helper_features,
            'selected_features': selected_features,
            'scopes': scopes_json,
            'original_features': usedInModel_features,
            'original_count': len(usedInModel_features),
            'features_removed': features_removed,
            'parameters': config.get_parameters()
        }, f, indent=2)
    print(f"Features saved for Java to: {feature_list_path}")
    for mk in MODEL_KEYS:
        print(f"  {mk}: {len(selected[mk])} features (removed {len(features_removed[mk])})")

    # 13. Train final models — one independent feature set/scaler/ONNX shape per
    # model. model_specs carries each model's scaled training matrix, label,
    # params and metadata. Labels: fast models at M15 cadence, slow at 4h.
    model_specs = {}
    for mk in MODEL_KEYS:
        bits = _cadence_bits(mk)
        model_specs[mk] = {
            'X': X_scaled[mk],
            'y': bits['y'],
            'params': bits['params'],
            'spw_factor': bits['spw'] if bits['spw'] is not None else 1.0,
            'features': selected[mk],
            'num_boost_round': bits['nbr'],
            'sample_weight': sample_weight_by_model.get(mk),
        }

    train_result = train_models(
        model_specs, run_dirs['feature_map_dir'], args=args, round_overrides=round_overrides)
    models = train_result['models']

    # 13b. Calibrate ALL classifier models
    calibrators, cal_params = calibrate_models(
        model_specs, models,
        method=args.calibration_method,
        cal_fraction=args.calibration_fraction,
    )

    # 14. Save models (per-model ONNX input shapes) + calibrators
    feature_counts = {_target_key(mk): len(selected[mk]) for mk in MODEL_KEYS}
    save_models(models, feature_counts, run_dirs['generated_dir'],
                calibrators=calibrators, cal_params=cal_params)

    # 14b. Export per-model scalers for Java (ScalerParams.java)
    java_source_dir = os.path.join(dir_config.SOURCE_DIR, "strategy", "src", "main", "java", "jforex")
    export_scaler.export_scalers_for_java(
        {mk: (scaler[mk], selected[mk]) for mk in MODEL_KEYS},
        run_dirs['generated_dir'], java_source_dir
    )

    # 15. SHAP Analysis (per-model)
    shap_specs = {
        mk: (X_df[mk], selected[mk],
             regime_train_fast if feature_config.model_cadence(mk) == 'fast' else regime_slow_4h_df)
        for mk in MODEL_KEYS
    }
    shap_results = run_shap_analysis(models, shap_specs, run_dirs['shap_dir'], args)

    # 16. Save Features + Targets + OHLC (ALL ALIGNED) - for backtesting
    print("\n" + "=" * 80)
    print("SAVE FEATURES + TARGETS + OHLC (ALL ALIGNED)")
    print("=" * 80 + "\n")

    # Per-model feature matrices at M15 frequency (backtest/live evaluate on M15 bars).
    for mk in MODEL_KEYS:
        X[selected[mk]].to_parquet(os.path.join(run_dirs['generated_dir'], f"X_{mk}.parquet"))

    # Regime-score sidecar for the backtest's execution layer (--regime-risk /
    # --regime-source ml / --direction-source trend): persisted from the combined
    # frame so daily_rgm_* scores tagged `role: helper` (kept out of every model)
    # still reach backtest.py. X_{model}.parquet carries only selected features.
    _rgm_saved = regime_model_mod.save_regime_scores(
        combined, X.index, run_dirs['generated_dir'])
    if _rgm_saved:
        print(f"Regime scores saved: {regime_model_mod.REGIME_SCORES_FILENAME} "
              f"({', '.join(_rgm_saved)})")
    labels['long_slow'].to_frame(name="target_long_slow").to_parquet(
        os.path.join(run_dirs['generated_dir'], "y_target_long_long_slow.parquet"))
    labels['short_slow'].to_frame(name="target_short_slow").to_parquet(
        os.path.join(run_dirs['generated_dir'], "y_target_short_short_slow.parquet"))
    labels['long_fast'].to_frame(name="target_long_fast").to_parquet(
        os.path.join(run_dirs['generated_dir'], "y_target_long_fast.parquet"))
    labels['short_fast'].to_frame(name="target_short_fast").to_parquet(
        os.path.join(run_dirs['generated_dir'], "y_target_short_fast.parquet"))
    ohlc.to_parquet(os.path.join(run_dirs['generated_dir'], "ohlc.parquet"))

    targets_aligned = targets_df.reindex(X.index)
    targets_aligned.to_parquet(os.path.join(run_dirs['generated_dir'], "label_targets.parquet"))

    print(f"OK: All data saved ({len(X)} rows)")

    # 17. Export training data to CSV for review
    # Pass raw (pre-denoising) labels if available for comparison in review plots
    raw_long_slow = raw_labels['long_slow'].reindex(X_train.index) if raw_labels else None
    raw_short_slow = raw_labels['short_slow'].reindex(X_train.index) if raw_labels else None
    csv.export_training_data_to_csv(
        X_train,
        ohlc_train,
        y_train['long_slow'],
        y_train['short_slow'],
        y_train['long_fast'],
        y_train['short_fast'],
        scaler_slow,  # use slow scaler for CSV export (main features)
        run_dirs['label_review_dir'],
        y_train_cls_long_raw=raw_long_slow,
        y_train_cls_short_raw=raw_short_slow,
    )

    # 18. Save results summary
    # Build a copy-pasteable command line. Use shlex (POSIX) since the project runs in bash.
    command_line = shlex.join([sys.executable] + sys.argv)

    summary = {
        'run_id': args.run_id,
        'command_line': command_line,
        'features_config': os.path.basename(config.config_path),
        'args': {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v)
                 for k, v in vars(args).items()},
        'n_features_original': len(X_train.columns),
        'n_features_fast': len(selected_fast),
        'n_features_slow': len(selected_slow),
        'n_training_samples': len(X_train),
        'cv_global_val_f1_fast_pre': cv_results_fast['global_val_f1'],
        'cv_global_val_f1_fast_post': cv_results_fast_post['global_val_f1'],
        'cv_global_val_f1_slow_pre': cv_results_slow['global_val_f1'],
        'cv_global_val_f1_slow_post': cv_results_slow_post['global_val_f1'],
        # Per-label positive rate/count — the class-distribution diagnostic. Required to
        # compare label modes, whose label definitions (and therefore base rates) differ.
        'label_stats': validate_label_distribution(labels)['stats'],
        'mi_threshold': args.mi_threshold,
        'pfi_threshold': args.pfi_threshold,
        'train_start': str(timeframes.TRAIN_START),
        'train_end': str(timeframes.TRAIN_END),
        'calibration_method': args.calibration_method,
        'calibration_fraction': args.calibration_fraction,
        'calibration_params': cal_params,
        # Resolved num_boost_round per model + the configured cap: makes an early stop
        # (rounds < cap) visible in the run diagnostics instead of being print-only.
        'boost_rounds': train_result['boost_rounds'],
        'boost_round_source': {
            mk: ('cv_folds' if f'target_{mk}' in round_overrides else
                 ('fixed' if getattr(args, 'early_stopping_rounds', 0) == 0 else 'holdout'))
            for mk in MODEL_KEYS
        },
        # Per-fold votes behind each median. Wide spread = the folds (market regimes)
        # disagree about how much capacity generalises; a lone outlier vote is exactly
        # what the median is there to absorb.
        'boost_round_votes': round_votes,
        'learning_curve_folds': {mk: (lc[mk] or {}).get('n_auc_folds', 0) for mk in MODEL_KEYS},
        # Pre-flight label coverage per model: which months carry no positive label at all
        # and which CV folds would therefore train on nothing. A non-empty
        # degenerate_fold_numbers invalidates every CV metric of that model regardless of
        # how the AUC reads — see label_coverage_report.
        'label_coverage': {
            mk: {
                'n_months': cov['n_months'],
                'n_empty_months': cov['n_empty_months'],
                'empty_months': cov['empty_months'],
                'degenerate_fold_numbers': cov['degenerate_fold_numbers'],
                'folds': cov['folds'],
                # Effective sample size after label overlap. THIS is the n every standard
                # error should use; the row count overstates it by ~sqrt(n/n_eff).
                'n_rows': cov.get('n_rows'),
                'n_eff': cov.get('n_eff'),
                'mean_uniqueness': cov.get('mean_uniqueness'),
            }
            for mk, cov in label_coverage.items()
        },
        'sample_weight_mode': _sample_weight_mode,
        'holdout_unsealed': bool(getattr(args, 'unseal_holdout', False)),
        'holdout_start': timeframes.HOLDOUT_START.strftime('%Y-%m-%d'),
    }
    # Add comprehensive per-scope CV metrics.
    #   *_pre   = pre feature selection, at the cap
    #   *_post  = post feature selection, at the cap
    #   *_final = post feature selection, at the RESOLVED boost rounds  <- the gate metrics
    # Compare runs on *_final: *_post cannot see the round choice at all (see the
    # "CV at the resolved boost rounds" step).
    summary['cv_metrics'] = {
        **_serialize_cv_metrics(cv_results_fast, 'fast_pre'),
        **_serialize_cv_metrics(cv_results_fast_post, 'fast_post'),
        **_serialize_cv_metrics(cv_results_slow, 'slow_pre'),
        **_serialize_cv_metrics(cv_results_slow_post, 'slow_post'),
        **_serialize_cv_metrics(cv_final['long_fast'], 'fast_final'),
        **_serialize_cv_metrics(cv_final['long_slow'], 'slow_final'),
    }
    # Same numbers for all four models (the two blocks above only cover the long_*
    # representatives), keyed by model so a per-model gate check is possible. `at_cap`
    # carries the same metrics measured at the round cap, so the cost/benefit of a cut is
    # readable without a second run — the cut model is typically NOT one of the long_*
    # representatives, i.e. exactly the one the prefixed blocks cannot show.
    def _gate_metrics(cv_res):
        return {
            'auc': _none_if_nan(cv_res['global_val_auc_roc']),
            'f1': cv_res['global_val_f1'],
            'precision': cv_res['global_val_precision'],
            'recall': cv_res['global_val_recall'],
            'recall_trend': _none_if_nan(cv_res.get('global_val_recall_trend', float('nan'))),
            'recall_range': _none_if_nan(cv_res.get('global_val_recall_range', float('nan'))),
            'mcc': cv_res['global_val_mcc'],
            'brier': cv_res['global_val_brier'],
            'ece': cv_res['global_val_ece'],
            'confusion_matrix': cv_res['global_val_confusion_matrix'],
            'threshold': cv_res['global_threshold'],
        }

    summary['cv_final_metrics'] = {
        mk: {
            'rounds': train_result['boost_rounds'][mk]['rounds'],
            'cap': train_result['boost_rounds'][mk]['cap'],
            'measured_at_cap': cv_final[mk] is cv_post[mk],
            **_gate_metrics(cv_final[mk]),
            'at_cap': _gate_metrics(cv_post[mk]),
        }
        for mk in MODEL_KEYS
    }
    # Per-model regime breakdown at the resolved rounds — same numbers the console
    # regime table prints, persisted so they can be compared across runs.
    summary['per_regime_metrics'] = per_regime_final
    if sweep_fast is not None or sweep_slow is not None:
        summary['threshold_sweep'] = {
            'fast_post': sweep_fast,
            'slow_post': sweep_slow,
        }

    summary_path = os.path.join(run_dirs['generated_dir'], "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # W&B: gate metrics per model, per-regime breakdown, learning curves.
    experiment_tracking.log_training_summary(summary, learning_curves=learning_curves)

    print("\n" + "=" * 80)
    print("ADVANCED TRAINING COMPLETE")
    print("=" * 80)
    print(f"Output directory: {run_dirs['generated_dir']}")
    print(f"Fast features: {len(selected_fast)} M15 (CV global F1: {cv_results_fast['global_val_f1']:.4f})")
    print(f"Slow features: {len(selected_slow)} slow/4hours+daily (CV global F1: {cv_results_slow['global_val_f1']:.4f})")
    print("=" * 80 + "\n")

    # Run backtest if requested (must run before report so trade_list.csv is available)
    if args.run_backtest:
        backtest_start = args.backtest_start or str(timeframes.BACKTEST_START.date()) if hasattr(timeframes, 'BACKTEST_START') else None
        backtest_end = args.backtest_end or str(timeframes.BACKTEST_END.date()) if hasattr(timeframes, 'BACKTEST_END') else None
        backtest_ok = run_backtest(
            run_id=args.run_id,
                       backtest_start=backtest_start,
            backtest_end=backtest_end,
            extra_args=experiment_tracking.wandb_cli_flags(args)
        )
        if not backtest_ok:
            print("Warning: Backtest failed. Report will not include backtest metrics.")

    # Generate model health report if requested
    if args.generate_report:
        print("\n" + "=" * 80)
        print("GENERATING MODEL HEALTH REPORT")
        print("=" * 80 + "\n")
        try:
            md_path, pdf_path, claude_analysis = generate_health_report(
                run_dir=run_dirs['generated_dir'],
                skip_claude=args.skip_claude_analysis
            )
            print(f"\nReport files generated:")
            print(f"  Markdown: {md_path}")
            print(f"  PDF: {pdf_path}")
        except Exception as e:
            print(f"Warning: Failed to generate health report: {e}")

    experiment_tracking.finish_wandb_run()


if __name__ == "__main__":
    main()
