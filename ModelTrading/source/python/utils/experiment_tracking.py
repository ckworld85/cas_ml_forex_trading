"""
Weights & Biases experiment tracking (opt-in via --wandb).

Design contract:

- OPT-IN. Nothing happens unless the CLI was given ``--wandb``. A run without the
  flag is bit-identical to a run made before this module existed.
- NEVER FATAL. A missing ``wandb`` package, a failed login or a network error must
  not kill a multi-hour training or a campaign cell. Every wandb call is wrapped;
  failures print a ``[wandb]`` warning and the run continues untracked.
- GROUPING. The W&B ``group`` is the model's run id, so ONE trained model and EVERY
  backtest executed against its artefacts land in the same group; ``job_type``
  ('train' vs 'backtest') separates them inside it. Walk-forward children carry
  ``run_base`` / ``wf_fold`` / ``run_seed`` config keys parsed from their display
  name, so a campaign can be regrouped by model family in the UI (the on-disk
  run_id of a campaign child is a timestamp+uuid and carries no meaning).

The module is import-safe everywhere: ``wandb`` itself is imported lazily inside
the guarded functions, so nothing staged to TEST/PRODUCTION gains a dependency.
"""

import json
import os
import re
import shlex
import sys

import numpy as np
import pandas as pd

DEFAULT_PROJECT = 'forex-trading'

# The active wandb run for this process. One process = at most one run: the
# training script and every backtest are separate processes already.
_active_run = None


def _warn(msg):
    print(f"[wandb] WARNING: {msg}")


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def add_wandb_args(parser):
    """Attach the shared --wandb* flags to an argparse parser."""
    group = parser.add_argument_group('experiment tracking (Weights & Biases)')
    group.add_argument('--wandb', action='store_true', default=False,
                       help='Log this run to Weights & Biases. Off by default; a run '
                            'without the flag behaves exactly as before. Requires the '
                            'wandb package and a login (wandb login / WANDB_API_KEY).')
    group.add_argument('--wandb-project', type=str, default=None,
                       help=f'W&B project (default: $WANDB_PROJECT or "{DEFAULT_PROJECT}")')
    group.add_argument('--wandb-entity', type=str, default=None,
                       help='W&B entity (team/user). Default: the logged-in default entity.')
    group.add_argument('--wandb-group', type=str, default=None,
                       help='Override the W&B group. Default: the run id, so a training '
                            'and every backtest of the same model share one group.')
    group.add_argument('--wandb-run-name', type=str, default=None,
                       help='Override the W&B run display name. Default: the run id '
                            '(training) or a name derived from the backtest parameters.')
    group.add_argument('--wandb-tags', action='append', default=None, metavar='TAG[,TAG...]',
                       help='Extra W&B tags. Repeatable; each occurrence may hold a '
                            'comma-separated list.')
    group.add_argument('--wandb-mode', type=str, default=None,
                       choices=['online', 'offline', 'disabled'],
                       help='W&B mode override (default: wandb default / $WANDB_MODE).')
    return parser


def wandb_enabled(args):
    """True when the CLI opted into tracking."""
    return bool(getattr(args, 'wandb', False))


def merged_tags(args, extra_tags=()):
    """Flatten --wandb-tags occurrences + extra tags into a deduplicated list."""
    tags = []
    for chunk in (getattr(args, 'wandb_tags', None) or []):
        tags.extend(t.strip() for t in str(chunk).split(',') if t.strip())
    tags.extend(t for t in extra_tags if t)
    return list(dict.fromkeys(tags))


def wandb_cli_flags(args, extra_tags=()):
    """
    The --wandb* flags to forward to a child process (advanced_train.py or
    backtest.py) so it joins the same project with the same tags. The child derives
    its own group and run name, so --wandb-run-name is deliberately NOT forwarded.
    Returns [] when tracking is off — appending the result is always safe.
    """
    if not wandb_enabled(args):
        return []
    flags = ['--wandb']
    for flag, attr in (('--wandb-project', 'wandb_project'),
                       ('--wandb-entity', 'wandb_entity'),
                       ('--wandb-group', 'wandb_group'),
                       ('--wandb-mode', 'wandb_mode')):
        value = getattr(args, attr, None)
        if value:
            flags.extend([flag, str(value)])
    tags = merged_tags(args, extra_tags)
    if tags:
        flags.extend(['--wandb-tags', ','.join(tags)])
    return flags


# ---------------------------------------------------------------------------
# Naming / grouping helpers (pure, unit-tested)
# ---------------------------------------------------------------------------

def wandb_group_for_run(run_id):
    """Sanitize a run id into a W&B group name ('default' when there is none)."""
    if not run_id:
        return 'default'
    return str(run_id).replace('\\', '/')


def parse_run_name(name):
    """
    Extract campaign coordinates from a run display name.

    Walk-forward / multi-seed children are named '{base}[_s{seed}][_wf{NN}]'
    (iterative_training appends the suffixes in that order); backtest runs add a
    '__bt...' suffix, which is stripped first so a backtest resolves to the SAME
    run_base as its training run. Returns {'run_base': ..., 'wf_fold': int?,
    'run_seed': int?} — the base always, the coordinates only when present.
    Grouping a campaign by config.run_base in the W&B UI collects every fold and
    seed of one configuration family.
    """
    base = re.sub(r'__bt.*$', '', wandb_group_for_run(name))
    out = {}
    while True:
        m = re.match(r'^(.+)_wf(\d+)$', base)
        if m and 'wf_fold' not in out:
            out['wf_fold'] = int(m.group(2))
            base = m.group(1)
            continue
        m = re.match(r'^(.+)_s(\d+)$', base)
        if m and 'run_seed' not in out:
            out['run_seed'] = int(m.group(2))
            base = m.group(1)
            continue
        break
    out['run_base'] = base
    return out


def derive_backtest_run_name(run_id, p_open_slow=None, stop_pips=None,
                             cost_model=None, regime_gate=None):
    """Compact display name for a backtest run: '{run-id-tail}__bt' + key knobs."""
    tail = wandb_group_for_run(run_id).rsplit('/', 1)[-1]
    parts = [f"{tail}__bt"]
    if p_open_slow is not None:
        parts.append(f"slow{float(p_open_slow):g}")
    if stop_pips is not None:
        parts.append(f"stop{float(stop_pips):g}")
    if cost_model and cost_model != 'none':
        parts.append(f"cost-{cost_model}")
    if regime_gate and regime_gate != 'trending':
        parts.append(f"gate-{regime_gate}")
    return '_'.join(parts)


def args_config_dict(args):
    """
    vars(args) made W&B/JSON-safe: --wandb* bookkeeping keys dropped, non-primitive
    values stringified (same rule training_summary.json applies to its args block).
    """
    out = {}
    for key, value in vars(args).items():
        if key.startswith('wandb'):
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, dict)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def effective_config(args, entries):
    """
    The run configuration with every RESOLVED value in place of the raw argparse one.

    ``vars(args)`` is not the configuration. A ``store_true`` flag whose default
    lives in a module constant reads ``False`` on a run that used ``True``:
    backtest.py's ``OPENING_REQUIRES_SLOW_SIGNAL`` and
    ``CLOSING_AFTER_SIGNAL_REVERSAL_SLOW`` default to on, and their ``--no-*``
    partners write a *different* dest, so a plain args dump states the opposite of
    what the backtest did. The same holds for every ``default=None`` knob whose
    fallback is a constant (``--stop-pips`` -> ``STOP_PIPS = 35``) and for a mode
    that can turn itself off at runtime (the regime gate with no regime feature).

    *entries* is an iterable of ``(key, value, superseded_dests)``: the config key,
    the value the run actually used, and the argparse dests that value replaces —
    those are removed, so no key can survive holding a value the run contradicted.
    Every superseded dest is dropped before any key is written, so the entries are
    order-independent.
    """
    out = args_config_dict(args)
    entries = list(entries)
    for _, _, superseded in entries:
        for dest in superseded or ():
            out.pop(dest, None)
    for key, value, _ in entries:
        out[key] = value
    return out


# Keys of the TRAINING run's args block that a backtest cannot see from its own CLI
# but that decide what the models were taught. 'label_mode' deliberately keeps its
# name, so a training run and every backtest of it carry the SAME config key and can
# be filtered/grouped together in the W&B UI. The barrier geometry is prefixed
# 'label_' because backtest.py has its own --stop-pips, which means the EXIT stop and
# not the label's stop barrier — one key with two meanings would silently mix them.
TRAINING_PROVENANCE_ARGS = (
    ('label_mode', 'label_mode'),
    ('label_file', 'label_file'),
    ('pip_target', 'label_pip_target'),
    ('stop_pips', 'label_stop_pips'),
)

# Top-level training_summary.json keys copied verbatim (the training window; the
# backtest's own window is already in its config as backtest_start/backtest_end).
TRAINING_PROVENANCE_SUMMARY_KEYS = ('train_start', 'train_end')


def training_provenance_config(generated_dir):
    """
    Training-side provenance for a BACKTEST run's W&B config, read from that run's
    ``training_summary.json`` in *generated_dir*.

    Without this a backtest run carries only its own CLI, so ``label_mode`` — the
    single most important thing about the model being backtested — exists on the
    training run and nowhere else, and a W&B view mixing both job types cannot group
    on it.

    Never raises: a missing file (a run backtested with ``--proba-file``, or one
    predating the summary export), unreadable JSON or a summary without an ``args``
    block all yield ``{}`` and the backtest is logged without the provenance.
    """
    path = os.path.join(generated_dir or '', 'training_summary.json')
    try:
        with open(path, encoding='utf-8') as f:
            summary = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(summary, dict):
        return {}

    out = {}
    args_block = summary.get('args')
    if isinstance(args_block, dict):
        for src, dst in TRAINING_PROVENANCE_ARGS:
            value = args_block.get(src)
            if value is not None:
                out[dst] = value
    for key in TRAINING_PROVENANCE_SUMMARY_KEYS:
        value = summary.get(key)
        if value is not None:
            out[key] = value
    return out


def flatten_metrics(obj, prefix='', sep='/'):
    """
    Flatten nested dicts into {'a/b/c': number}. Only numeric leaves (bool counts as
    numeric) survive — strings, lists and None are dropped, NaN becomes nothing.
    """
    out = {}
    if not isinstance(obj, dict):
        return out
    for key, value in obj.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_metrics(value, prefix=name, sep=sep))
        elif isinstance(value, bool):
            out[name] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            if not (isinstance(value, (float, np.floating)) and np.isnan(value)):
                out[name] = float(value) if isinstance(value, (np.integer, np.floating)) else value
    return out


# ---------------------------------------------------------------------------
# Payload builders (pure, unit-tested)
# ---------------------------------------------------------------------------

def _regime_slug(name):
    """'TREND (all)' -> 'TREND_all' — a W&B metric-key-safe regime name."""
    return re.sub(r'_+', '_', re.sub(r'[^0-9A-Za-z]+', '_', str(name))).strip('_')


def training_payload(summary):
    """
    Flat metrics for a training run, from the training_summary.json dict.

    Per model: the *final* gate metrics (post feature selection, at the resolved
    boost rounds — the numbers the acceptance gates use), the positive prediction
    rate derived from the confusion matrix, label coverage / n_eff, and the
    per-regime breakdown recorded under summary['per_regime_metrics'].
    """
    out = {}
    for mk, fm in (summary.get('cv_final_metrics') or {}).items():
        for key in ('auc', 'f1', 'precision', 'recall', 'mcc', 'brier', 'ece',
                    'threshold', 'rounds', 'cap'):
            value = fm.get(key)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
                out[f'{mk}/{key}'] = value
        cm = fm.get('confusion_matrix')
        if cm and len(cm) == 2 and len(cm[0]) == 2:
            tn, fp = cm[0]
            fn, tp = cm[1]
            total = tn + fp + fn + tp
            out[f'{mk}/cm/tn'] = tn
            out[f'{mk}/cm/fp'] = fp
            out[f'{mk}/cm/fn'] = fn
            out[f'{mk}/cm/tp'] = tp
            if total > 0:
                out[f'{mk}/positive_prediction_rate'] = (fp + tp) / total

    for mk, cov in (summary.get('label_coverage') or {}).items():
        for key in ('n_rows', 'n_eff', 'mean_uniqueness', 'n_empty_months'):
            value = cov.get(key)
            if isinstance(value, (int, float)):
                out[f'{mk}/label/{key}'] = value
        out[f'{mk}/label/n_degenerate_folds'] = len(cov.get('degenerate_fold_numbers') or [])

    # Healthy-fold AUC/MCC exist only for the long_* cadence representatives.
    cv = summary.get('cv_metrics') or {}
    for prefix, mk in (('fast_final', 'long_fast'), ('slow_final', 'long_slow')):
        for metric, name in (('global_val_auc_roc_healthy_folds', 'auc_healthy_folds'),
                             ('global_val_mcc_healthy_folds', 'mcc_healthy_folds')):
            value = cv.get(f'{prefix}_{metric}')
            if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
                out[f'{mk}/{name}'] = value

    for mk, block in (summary.get('per_regime_metrics') or {}).items():
        for row in block.get('rows') or []:
            slug = _regime_slug(row.get('regime'))
            for key in ('auc', 'f1', 'precision', 'recall', 'mcc', 'brier', 'n', 'pos_rate'):
                value = row.get(key)
                if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
                    out[f'{mk}/regime/{slug}/{key}'] = value

    for key in ('n_training_samples', 'n_features_fast', 'n_features_slow'):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            out[key] = value
    out.update(flatten_metrics(summary.get('label_stats') or {}, prefix='label_stats'))
    return out


def regime_table_rows(summary):
    """Long-form rows (one per model x regime) for the regime_metrics W&B table."""
    rows = []
    for mk, block in (summary.get('per_regime_metrics') or {}).items():
        for row in block.get('rows') or []:
            rows.append({'model': mk, 'threshold': block.get('threshold'), **row})
    return rows


def backtest_payload(summary):
    """Flat numeric metrics from a backtest_summary.json dict."""
    return flatten_metrics(summary)


def regime_breakdown_rows(df_trades, regime_labels):
    """
    Per-regime trade statistics (same partition as the console regime breakdown:
    regime_combined at the trade's OPEN time, unknown bars as 'UNKNOWN').
    """
    if df_trades is None or len(df_trades) == 0:
        return []
    if regime_labels is None or 'regime_combined' not in getattr(regime_labels, 'columns', []):
        return []
    open_times = pd.to_datetime(df_trades['open_time'])
    regimes = regime_labels['regime_combined'].reindex(open_times).fillna('UNKNOWN').values
    return _grouped_trade_rows(df_trades, regimes)


def exit_reason_rows(df_trades):
    """Per-exit-reason trade statistics."""
    if df_trades is None or len(df_trades) == 0 or 'exit_reason' not in df_trades:
        return []
    reasons = df_trades['exit_reason'].fillna('unknown').astype(str).values
    return _grouped_trade_rows(df_trades, reasons)


def _grouped_trade_rows(df_trades, group_values):
    grouped = df_trades.assign(_grp=group_values)
    rows = []
    for name in sorted(grouped['_grp'].astype(str).unique()):
        sub = grouped[grouped['_grp'].astype(str) == name]
        rows.append({
            'name': str(name),
            'n_trades': int(len(sub)),
            'win_rate_pct': float((sub['pnl'] > 0).mean() * 100),
            'avg_pnl_eur': float(sub['pnl'].mean()),
            'total_pnl_eur': float(sub['pnl'].sum()),
            'total_pnl_pips': float(sub['pnl_pips'].sum()),
        })
    return rows


def breakdown_payload(rows, prefix):
    """Rows from *_rows() as flat metrics: '{prefix}/{name}/{field}'."""
    out = {}
    for row in rows:
        slug = _regime_slug(row.get('name'))
        for key, value in row.items():
            if key != 'name' and isinstance(value, (int, float)):
                out[f'{prefix}/{slug}/{key}'] = value
    return out


# ---------------------------------------------------------------------------
# Guarded wandb transport
# ---------------------------------------------------------------------------

def _wandb_module():
    try:
        import wandb
        return wandb
    except ImportError:
        _warn("--wandb requested but the 'wandb' package is not installed "
              "(pip install wandb). Continuing WITHOUT experiment tracking.")
        return None


def init_wandb_run(args, *, job_type, run_id=None, name=None, config=None, tags=()):
    """
    Start the W&B run for this process. Returns the run, or None when tracking is
    off or unavailable (every later log_* call then no-ops).
    """
    global _active_run
    if not wandb_enabled(args):
        return None
    wandb = _wandb_module()
    if wandb is None:
        return None

    resolved_name = name or getattr(args, 'wandb_run_name', None) or wandb_group_for_run(run_id)
    run_config = dict(config or {})
    run_config.setdefault('run_id', wandb_group_for_run(run_id))
    for key, value in parse_run_name(resolved_name).items():
        run_config.setdefault(key, value)
    run_config.setdefault('command_line', shlex.join([sys.executable] + sys.argv))

    try:
        _active_run = wandb.init(
            project=getattr(args, 'wandb_project', None) or os.environ.get('WANDB_PROJECT') or DEFAULT_PROJECT,
            entity=getattr(args, 'wandb_entity', None) or None,
            group=getattr(args, 'wandb_group', None) or wandb_group_for_run(run_id),
            job_type=job_type,
            name=resolved_name,
            config=run_config,
            tags=merged_tags(args) or None,
            mode=getattr(args, 'wandb_mode', None) or None,
        )
        print(f"[wandb] tracking as '{resolved_name}' "
              f"(group={getattr(args, 'wandb_group', None) or wandb_group_for_run(run_id)}, "
              f"job_type={job_type})")
    except Exception as e:
        _warn(f"wandb.init failed ({e}). Continuing WITHOUT experiment tracking.")
        _active_run = None
    return _active_run


def is_active():
    return _active_run is not None


def update_config(values):
    """
    Add/overwrite config keys on the ACTIVE run. For values that are only resolved
    after wandb.init — a sentinel like --cv-gap -1, whose effective embargo is
    derived from the label horizon mid-run. No-ops without an active run.
    """
    if _active_run is None or not values:
        return
    try:
        _active_run.config.update(values, allow_val_change=True)
    except Exception as e:
        _warn(f"failed to update config: {e}")


def log_metrics(metrics, step=None):
    if _active_run is None or not metrics:
        return
    try:
        if step is None:
            _active_run.log(metrics)
        else:
            _active_run.log(metrics, step=step)
    except Exception as e:
        _warn(f"failed to log metrics: {e}")


def log_table(key, rows, max_rows=10000):
    """Log a list-of-dicts (or DataFrame) as a wandb.Table."""
    if _active_run is None or rows is None or len(rows) == 0:
        return
    wandb = _wandb_module()
    if wandb is None:
        return
    try:
        if isinstance(rows, pd.DataFrame):
            df = rows.head(max_rows)
            columns = list(df.columns)
            data = df.values.tolist()
        else:
            rows = rows[:max_rows]
            columns = list(rows[0].keys())
            data = [[row.get(col) for col in columns] for row in rows]
        data = [[_table_cell(v) for v in row] for row in data]
        _active_run.log({key: wandb.Table(columns=columns, data=data)})
    except Exception as e:
        _warn(f"failed to log table '{key}': {e}")


def _table_cell(value):
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def log_learning_curves(curves_by_model):
    """
    Per-round train/val curves (aggregate_fold_curves output per model) as W&B
    line charts, each model on its own '{model}/curve/round' step axis.
    """
    if _active_run is None:
        return
    try:
        for mk, curve in (curves_by_model or {}).items():
            if not curve:
                continue
            train = curve.get('train') or {}
            val = curve.get('val') or {}
            metrics = sorted(set(train) | set(val))
            n_rounds = max((len(s) for s in list(train.values()) + list(val.values())), default=0)
            if n_rounds == 0:
                continue
            step_key = f'{mk}/curve/round'
            _active_run.define_metric(step_key)
            _active_run.define_metric(f'{mk}/curve/*', step_metric=step_key)
            for i in range(n_rounds):
                row = {step_key: i}
                for metric in metrics:
                    series = train.get(metric)
                    if series is not None and i < len(series):
                        row[f'{mk}/curve/train_{metric}'] = series[i]
                    series = val.get(metric)
                    if series is not None and i < len(series):
                        row[f'{mk}/curve/val_{metric}'] = series[i]
                _active_run.log(row)
    except Exception as e:
        _warn(f"failed to log learning curves: {e}")


def log_training_summary(summary, learning_curves=None):
    """One-call training logging: gate metrics + regime breakdown + curves."""
    if _active_run is None:
        return
    try:
        log_metrics(training_payload(summary))
        log_table('regime_metrics', regime_table_rows(summary))
        if learning_curves:
            log_learning_curves(learning_curves)
    except Exception as e:
        _warn(f"failed to log training summary: {e}")


def _as_timestamp(value):
    """pd.Timestamp for a timestamp-like value, or None when absent/unparseable."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts
    except (TypeError, ValueError):
        return None


def _epoch_seconds(value):
    """Unix seconds for a timestamp-like value, or None when absent/unparseable."""
    ts = _as_timestamp(value)
    return None if ts is None else float(ts.timestamp())


def log_trades(df_trades):
    """Per-trade PnL/equity series on a 'trade/idx' step axis.

    Each row also carries the trade's open/close time as unix epoch seconds
    ('trade/open_time' / 'trade/close_time') — W&B charts only numbers, but a
    panel x-axis auto-detects epoch-second values and renders them as calendar
    datetimes; pick trade/close_time as the panel's x-axis to plot the series
    over calendar time instead of trade count. 'trade/open_date' /
    'trade/close_date' are the same instants truncated to midnight (still epoch
    seconds — a YYYYMMDD int would be misread by that auto-detection as an epoch
    value in 1970), so their axis labels show pure dates.
    The cumulative view lives in log_equity_curve, not on the trade rows.
    """
    if _active_run is None or df_trades is None or len(df_trades) == 0:
        return
    try:
        _active_run.define_metric('trade/idx')
        _active_run.define_metric('trade/*', step_metric='trade/idx')
        for i, row in enumerate(df_trades.itertuples(index=False), 1):
            entry = {
                'trade/idx': i,
                'trade/pnl_eur': float(row.pnl),
                'trade/pnl_pips': float(row.pnl_pips),
                'trade/equity_eur': float(row.equity_after),
            }
            for key in ('open_time', 'close_time'):
                ts = _as_timestamp(getattr(row, key, None))
                if ts is not None:
                    entry[f'trade/{key}'] = float(ts.timestamp())
                    entry[f'trade/{key.replace("_time", "_date")}'] = float(ts.normalize().timestamp())
            _active_run.log(entry)
    except Exception as e:
        _warn(f"failed to log trades: {e}")


def log_equity_curve(df_trades, period_start, period_end):
    """Daily cumulative-PnL curve over the whole backtest window.

    One row per calendar day from period_start to period_end — including days
    without any trade — keyed 'eq_curve/date' (the step axis: epoch seconds at
    that day's midnight, which W&B's x-axis auto-detection renders as a
    calendar date) and 'eq_curve/cum_pnl' (EUR). A trade's PnL enters the
    curve on the day the trade was CLOSED; days before the first close read 0.0.
    """
    if _active_run is None:
        return
    try:
        start = _as_timestamp(period_start)
        end = _as_timestamp(period_end)
        if start is None or end is None:
            return
        daily_index = pd.date_range(start=start.normalize(), end=end.normalize(), freq='D')
        if df_trades is not None and len(df_trades) > 0:
            daily_pnl = (
                pd.Series(df_trades['pnl'].astype(float).values,
                          index=pd.to_datetime(df_trades['close_time']).dt.normalize())
                .groupby(level=0).sum()
            )
        else:
            daily_pnl = pd.Series(dtype=float)
        cum_pnl = daily_pnl.reindex(daily_index, fill_value=0.0).cumsum()
        _active_run.define_metric('eq_curve/date')
        _active_run.define_metric('eq_curve/*', step_metric='eq_curve/date')
        for day, value in cum_pnl.items():
            _active_run.log({
                'eq_curve/date': float(day.timestamp()),
                'eq_curve/cum_pnl': float(value),
            })
    except Exception as e:
        _warn(f"failed to log equity curve: {e}")


def finish_wandb_run():
    global _active_run
    if _active_run is None:
        return
    try:
        _active_run.finish()
    except Exception as e:
        _warn(f"failed to finish run: {e}")
    finally:
        _active_run = None
