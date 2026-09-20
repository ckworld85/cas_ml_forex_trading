"""
Feature pruning — which of the features already in use are worth keeping?

WHAT THIS IS FOR, AND WHAT IT CANNOT DO
---------------------------------------
No selection procedure can make a model better in the sense of creating an edge. Selection
rearranges the information that is present; measured across three model families, 84
candidate features and every available information family, the within-regime directional
AUC in this project is ~0.50. What selection *can* do is remove features that carry nothing
of their own, which reduces variance and improves calibration — and matters more here than
the row count suggests, because the effective sample is small (see the capacity budget).

The acceptance rule is therefore **non-inferiority, not improvement**. At 39 month-clusters
against the ~680 a significant P&L difference would need, "this change makes more money" is
undecidable. "The same AUC with 19 features instead of 30" is decidable, and it is a real
gain: walk-forward validated 2026-08-29 at 30 % fewer features, −22 % variance per run and
no measurable loss.

THREE STEPS
-----------
1. **Redundancy clustering** — hierarchical on 1-|corr|, one representative per cluster.
   Deliberately **label-free**: a property of the features, not of the target, so it costs
   no statistical power, carries no multiple-testing burden and cannot overfit. It is also
   why the cluster structure barely moves between label modes.

   Measured on the active set: 30 slow features collapse to **19 clusters at |corr| >= 0.80**,
   21 fast features to 15. Among the slow clusters `daily_regime_trend` / `daily_adx` /
   `daily_adx_percentile` at 0.91 — the regime detector measured three times over, and the
   mechanism behind the 0.760 -> 0.506 AUC collapse between `baseline_0` and `baseline_rgm`.

   The representative is chosen by **stationarity, never by importance**: importance is
   exactly what spreads itself across a correlated cluster, so ranking on it picks an
   arbitrary member and hides the redundancy that caused the spread.

2. **Capacity budget** — features against the EFFECTIVE sample, not the row count, and
   counted on POSITIVES: on an imbalanced problem the minority class is what limits what a
   tree can learn. `long_slow` reads a comfortable 45 observations per feature and passes,
   while having only **10.9 effective positives per feature**, which is what actually binds.
   Refuses a verdict when n_eff had to fall back to the row count (an upper bound).

3. **Paired non-inferiority test** — the reduced set against the full one on identical
   seeds and folds, scored on the AUC **within the operative regime** against the
   non-zeroed outcome. Delegated to `analytics.feature_ab.compare`.

Usage
-----
    # steps 1+2 only (label-free, instant)
    python -m ModelTrading.source.python.analytics.feature_pruning --run-id <run>

    # all three, and write the pruned config
    python -m ModelTrading.source.python.analytics.feature_pruning --run-id <run> \\
        --model all --verify --seeds 3 --label-mode trend_only \\
        --emit-config ModelTrading/config/features-trend_only-proposal.yaml

Read-only with respect to training state: it reads the artefacts a run wrote. The only
thing it writes is the file you point `--emit-config` at. Emit to a *-proposal.yaml,
never straight onto a canonical `features-<mode>.yaml` — those can carry hand-curated
selections (trend_only's was re-selected 2026-09-06 via the within-TREND screen) that a
mechanical emission would silently overwrite.
"""

import argparse
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ModelTrading.config.directories as dir_config  # noqa: E402
from ModelTrading.source.python.analytics import feature_ab  # noqa: E402
from ModelTrading.source.python.training import sample_weights as sw  # noqa: E402

MODELS = ('long_fast', 'short_fast', 'long_slow', 'short_slow')

# Non-inferiority margin in AUC: the movement reseeding alone produces (measured
# 2026-08-29, +/-0.017). A reduced set landing inside that band has not been shown to be
# worse by anything the harness can distinguish from a different random draw.
NON_INFERIORITY_MARGIN = 0.017

# Effective MINORITY-CLASS observations per feature below which the count is flagged.
# Counted on positives, not rows — see the module docstring.
MIN_EFF_POSITIVES_PER_FEATURE = 20.0

# Preference order for a cluster representative. Deliberately NOT importance.
_STATIONARY_HINTS = ('_percentile', '_efficiency', '_pband', '_percent', 'hour_sin',
                     'hour_cos', 'is_', 'rsi', 'stoch', 'adx', 'bb_position')
_NORMALISED_HINTS = ('_atr', 'deviation', '_z250', '_ratio', 'slope_efficiency')


def _stationarity_rank(name):
    """Lower is better: 0 = bounded by construction, 1 = normalised, 2 = everything else."""
    n = name.lower()
    if any(h in n for h in _STATIONARY_HINTS):
        return 0
    if any(h in n for h in _NORMALISED_HINTS):
        return 1
    return 2


def _representative(members, corr):
    """Pick one feature to stand for a correlated cluster.

    Ranked by (1) stationarity class, (2) centrality inside the cluster — the member with
    the highest mean |corr| to the others best summarises them — and (3) name, so the
    choice is deterministic and reviewable.
    """
    if len(members) == 1:
        return members[0]
    centrality = {m: float(corr.loc[m, [x for x in members if x != m]].mean())
                  for m in members}
    return sorted(members, key=lambda m: (_stationarity_rank(m), -centrality[m], m))[0]


def cluster_features(X, threshold=0.80):
    """Group features whose |corr| reaches `threshold`.

    Returns (clusters, keep) where clusters is a list of member lists ordered largest
    first, and keep is the chosen representative list in the original column order.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    cols = list(X.columns)
    if len(cols) < 2:
        return [[c] for c in cols], cols

    corr = X.corr().abs().fillna(0.0)
    # pandas 3 (copy-on-write) returns a read-only array from .values —
    # fill the diagonal on an owned copy and rebuild the frame.
    corr_arr = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 1.0)
    corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

    dist = 1.0 - corr_arr
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)

    link = linkage(squareform(dist, checks=False), method='average')
    labels = fcluster(link, t=1.0 - threshold, criterion='distance')

    groups = {}
    for col, lab in zip(cols, labels):
        groups.setdefault(lab, []).append(col)

    clusters = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    keep = {_representative(g, corr) for g in clusters}
    return clusters, [c for c in cols if c in keep]


def capacity_budget(X, y, run_dir, model):
    """Effective sample size and the per-feature ratios.

    n_eff comes from the label overlap, not the row count: barrier labels are emitted on
    every bar but resolved over the following horizon, so neighbouring rows share an
    outcome. Uses the `t1_{model}` column the labeller records; falls back to the row count
    with `n_eff_measured=False` when it is absent, because the row count is only an upper
    bound and a budget verdict computed from it would be unearned.
    """
    n_rows = int(len(X))
    n_eff, source, measured = float(n_rows), 'row count', False

    meta_path = os.path.join(run_dir, 'label_targets.parquet')
    if os.path.exists(meta_path):
        try:
            meta = pd.read_parquet(meta_path)
            t1 = sw.t1_from_metadata(meta, model, index=X.index)
            if t1 is not None:
                n_eff = sw.effective_sample_size(t1)
                source = 'label uniqueness (t1)'
                measured = True
        except Exception as exc:                                  # pragma: no cover
            source = f'row count (t1 unreadable: {exc})'

    pos_rate = float(y.mean()) if len(y) else 0.0
    n_features = int(X.shape[1])
    return {
        'n_rows': n_rows,
        'n_eff': float(n_eff),
        'n_eff_source': source,
        'n_eff_measured': bool(measured),
        'positive_rate': pos_rate,
        'n_eff_positives': float(n_eff * pos_rate),
        'n_features': n_features,
        'eff_obs_per_feature': float(n_eff / max(n_features, 1)),
        'eff_positives_per_feature': float(n_eff * pos_rate / max(n_features, 1)),
    }


def affordable_features(n_eff, positive_rate,
                        min_eff_positives=MIN_EFF_POSITIVES_PER_FEATURE):
    """How many features the effective MINORITY class supports at the stated budget."""
    n_eff_pos = float(n_eff) * float(positive_rate)
    return int(max(1, n_eff_pos // max(min_eff_positives, 1e-9)))


def verdict(res, margin=NON_INFERIORITY_MARGIN):
    """Apply the non-inferiority rule to a feature_ab.compare result.

    Accepted when the AUC delta's lower confidence bound stays above -margin: not shown to
    be worse by more than a reseed would move it. Deliberately not a test for improvement,
    which at this sample size is undecidable.

    The scope is the **operative regime** against the non-zeroed outcome: under a
    regime-conditioned label mode the global AUC measures regime recovery rather than trade
    selection. When the breakdown is unavailable this falls back to the global AUC and says
    so — `scope_is_operative` is False and the report warns.

    Note the key names: feature_ab reports per-regime statistics as `mean` / `ci95` while
    the top-level result uses `delta_mean` / `delta_ci95`. Reading the wrong pair falls
    through to the global scope with no error.
    """
    operative = res.get('operative_regime')
    stats = (res.get('regimes') or {}).get(operative)
    ci = np.asarray(stats.get('ci95', (np.nan, np.nan)), dtype=float) if stats else None

    if stats and stats.get('evaluable', True) and ci is not None and np.isfinite(ci[0]):
        delta = float(stats['mean'])
        lo, hi = float(ci[0]), float(ci[1])
        scope = f"operative regime {operative} ({stats.get('n_bars', '?')} bars)"
        operative_scope = True
    else:
        delta = float(res['delta_mean'])
        lo, hi = (float(v) for v in res['delta_ci95'])
        scope = 'GLOBAL AUC - operative-regime breakdown unavailable'
        operative_scope = False

    base = {'scope': scope, 'scope_is_operative': operative_scope,
            'delta': delta, 'ci': (lo, hi)}
    if not np.isfinite(lo):
        return {**base, 'accept': False, 'reason': 'no confidence interval - add seeds'}
    if lo > -margin:
        return {**base, 'accept': True,
                'reason': f'lower bound {lo:+.4f} stays above the -{margin:.3f} margin'}
    return {**base, 'accept': False,
            'reason': f'lower bound {lo:+.4f} falls below the -{margin:.3f} margin'}


_ROW_RE = re.compile(r'^(\s*- \{name: )([A-Za-z0-9_]+)(,.*)\}(\s*#.*)?$')


def emit_config(results, source_path, out_path, label_mode=None,
                generated_by='analytics/feature_pruning.py --emit-config'):
    """Write a features YAML in which only the kept features remain enabled.

    Mechanical, not hand-edited: for every `role: model` row that appears in a model's
    feature matrix, the `models:` tag is narrowed to exactly the models that kept it, and a
    feature kept by none is set `enabled: false`. Rows the pruning never saw (helpers,
    already-disabled features) are copied through untouched.

    Narrowing rather than disabling matters: a feature can be a cluster representative for
    the fast models and redundant for the slow ones. Measured 2026-08-29, `daily_atr_pips`
    and `daily_sma_slope_80` are exactly that, and disabling them outright would throw away
    information the fast models still use.

    A model whose step-3 verdict rejected the pruning keeps its FULL set: emitting a
    config that ignores a verification that was actually run would be exactly the kind of
    silent fallback this module exists to remove.

    Returns (disabled, narrowed, restored) for the caller to report.
    """
    # A model whose step-3 verdict REJECTED the pruning keeps its full set. Emitting a
    # config that ignores a verification you just ran is the same silent-fallback pattern
    # this module exists to remove: measured 2026-08-29, `window_cascade` / `short_slow`
    # loses 0.0266 AUC [-0.0308, -0.0223] over 8 paired seeds — well outside the 0.017
    # margin and stable across seeds, so it is a real loss, not a draw.
    keep, restored = {}, []
    for m, r in results.items():
        v = r.get('verdict')
        if v is not None and not v.get('accept', True):
            keep[m] = set(r['keep']) | set(r['drop'])
            restored.append(m)
        else:
            keep[m] = set(r['keep'])

    seen = set().union(*(set(r['keep']) | set(r['drop']) for r in results.values()))
    order = [m for m in MODELS if m in results]
    nl = chr(10)

    with io.open(source_path, encoding='utf-8') as fh:
        lines = fh.read().split(nl)

    disabled, narrowed, out = [], [], []
    for line in lines:
        m = _ROW_RE.match(line)
        if not m or 'role: model' not in line or m.group(2) not in seen:
            out.append(line)
            continue
        name, body = m.group(2), m.group(3)
        kept_for = [mm for mm in order if name in keep[mm]]
        if not kept_for:
            new = re.sub(r'enabled: (true|false)', 'enabled: false', body)
            if 'enabled:' not in body:
                new = ', enabled: false' + body
            disabled.append(name)
        else:
            new = re.sub(r'models: \[[^\]]*\]', 'models: [' + ', '.join(kept_for) + ']', body)
            if new != body:
                narrowed.append((name, kept_for))
        out.append(m.group(1) + name + new + '}' + (m.group(4) or ''))

    counts = '  '.join(f'{mm} {len(keep[mm])}' for mm in order)
    header = [
        *([f'# NOTE: {", ".join(restored)} kept the FULL set - step 3 rejected the pruning',
           '# for them (loss beyond the non-inferiority margin, stable across seeds).']
          if restored else []),
        '# Feature Configuration - PRUNED SET'
        + (f' for label mode `{label_mode}`' if label_mode else ''),
        f'# Generated mechanically by {generated_by}. Do not',
        '# hand-edit: regenerate it from the run, so that the config and the evidence',
        '# justifying it stay attached to each other.',
        f'# Kept per model: {counts}',
        '# Redundancy clustering is LABEL-FREE (a correlation threshold on the feature',
        '# matrix), so the cluster structure barely moves between label modes; what moves',
        '# is which rows enter the matrix, and therefore the verification in step 3.',
        '#',
    ]
    text = nl.join(out)
    anchors = [l for l in out if l.startswith('# Feature Configuration')]
    if anchors:
        text = text.replace(anchors[0], nl.join(header) + nl + anchors[0], 1)
    else:                                                          # pragma: no cover
        text = nl.join(header) + nl + text

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with io.open(out_path, 'w', encoding='utf-8', newline=nl) as fh:
        fh.write(text)
    return disabled, narrowed, restored


def prune(run_id, model, threshold=0.80, generated_dir=None, verify=False, seeds=8,
          margin=NON_INFERIORITY_MARGIN,
          min_eff_positives=MIN_EFF_POSITIVES_PER_FEATURE):
    """Run all three steps for one model. Returns a result dict."""
    generated_dir = generated_dir or dir_config.GENERATED_DIR
    run_dir = os.path.join(generated_dir, run_id)
    X, y, args = feature_ab.load_run(run_id, model, generated_dir=generated_dir)

    clusters, keep = cluster_features(X, threshold=threshold)
    budget_before = capacity_budget(X, y, run_dir, model)
    budget_after = capacity_budget(X[keep], y, run_dir, model)

    out = {
        'run_id': run_id, 'model': model, 'threshold': threshold,
        'label_mode': args.get('label_mode'),
        'clusters': clusters,
        'redundant_clusters': [c for c in clusters if len(c) > 1],
        'keep': keep,
        'drop': [c for c in X.columns if c not in set(keep)],
        'budget_before': budget_before,
        'budget_after': budget_after,
        'affordable_at_budget': affordable_features(
            budget_before['n_eff'], budget_before['positive_rate'], min_eff_positives),
        'min_eff_positives_per_feature': min_eff_positives,
        'verification': None,
    }

    if verify and out['drop']:
        regimes = feature_ab.load_regimes(run_id, X.index, generated_dir=generated_dir)
        res = feature_ab.compare(X, y, args, model, keep_only=keep, seeds=seeds,
                                 regimes=regimes)
        out['verification'] = res
        out['verdict'] = verdict(res, margin=margin)
    return out


def print_report(out):
    b, a = out['budget_before'], out['budget_after']
    print()
    print('=' * 88)
    print(f"FEATURE PRUNING - {out['run_id']} / {out['model']}"
          + (f" (label mode {out['label_mode']})" if out.get('label_mode') else ''))
    print('=' * 88)

    print(f"\n1. REDUNDANCY (label-free, |corr| >= {out['threshold']:.2f})")
    print(f"   {b['n_features']} features -> {len(out['keep'])} clusters "
          f"({len(out['drop'])} redundant)")
    for members in out['redundant_clusters']:
        rep = next(m for m in out['keep'] if m in members)
        print(f"     KEEP {rep}")
        for other in (m for m in members if m != rep):
            print(f"       drop {other}")

    print(f"\n2. CAPACITY BUDGET (n_eff from {b['n_eff_source']})")
    print(f"   rows {b['n_rows']:,}  ->  n_eff {b['n_eff']:,.0f}  "
          f"(positive rate {b['positive_rate']:.1%}, "
          f"n_eff positives {b['n_eff_positives']:,.0f})")
    print(f"   before: {b['n_features']:3d} features -> "
          f"{b['eff_positives_per_feature']:6.1f} eff. positives/feature")
    print(f"   after:  {a['n_features']:3d} features -> "
          f"{a['eff_positives_per_feature']:6.1f} eff. positives/feature")
    if not b['n_eff_measured']:
        print("   WARNING: n_eff fell back to the ROW COUNT, an upper bound - this label")
        print("            mode records no t1. Read the ratio only as a relative")
        print("            before/after comparison; no budget verdict is given.")
    else:
        limit = out['min_eff_positives_per_feature']
        flag = 'OK' if a['eff_positives_per_feature'] >= limit else 'OVER BUDGET'
        print(f"   at {limit:.0f} eff. positives/feature the sample supports "
              f"~{out['affordable_at_budget']} features  [{flag}]")

    print('\n3. NON-INFERIORITY VERIFICATION')
    v = out.get('verification')
    if v is None:
        print('   not run (pass --verify). Steps 1 and 2 are label-free and already valid.')
    else:
        d = out['verdict']
        print(f"   scope: {d['scope']}, {v['n_seeds']} paired seeds")
        if not d['scope_is_operative']:
            print('   WARNING: this verdict is on the GLOBAL AUC. Under a regime-conditioned')
            print('            label mode that measures regime recovery, not trade selection.')
        print(f"   full set    AUC {v['baseline_test_auc']:.4f}  "
              f"({v['baseline_n_features']} features)")
        print(f"   pruned set  AUC {v['variant_test_auc']:.4f}  "
              f"({v['variant_n_features']} features)")
        print(f"   delta {d['delta']:+.4f}  95% CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
        print(f"   -> {'ACCEPT' if d['accept'] else 'REJECT'}: {d['reason']}")

    print(f"\nPRUNED SET ({len(out['keep'])} features):")
    for col in out['keep']:
        print(f'  {col}')
    print('=' * 88)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id', required=True, help='Training run whose artefacts to read.')
    ap.add_argument('--model', default='long_slow', choices=list(MODELS) + ['all'])
    ap.add_argument('--threshold', type=float, default=0.80,
                    help='|corr| at which two features count as interchangeable (0.80).')
    ap.add_argument('--verify', action='store_true',
                    help='Run step 3. Slow; steps 1-2 are label-free and valid without it.')
    ap.add_argument('--seeds', type=int, default=8, help='Paired seeds for --verify.')
    ap.add_argument('--margin', type=float, default=NON_INFERIORITY_MARGIN,
                    help='Non-inferiority margin in AUC (default 0.017, the measured '
                         'movement from reseeding alone).')
    ap.add_argument('--min-eff-positives', type=float,
                    default=MIN_EFF_POSITIVES_PER_FEATURE,
                    help='Effective MINORITY-CLASS observations per feature below which '
                         'the count is flagged (default 20). Counted on positives, not '
                         'rows: the minority class is what limits what a tree can learn.')
    ap.add_argument('--out', default=None, help='Write the full result to this JSON.')
    ap.add_argument('--emit-config', default=None,
                    help='Write a pruned features YAML here. Requires --model all, since '
                         'the config carries a per-model tag for every feature.')
    ap.add_argument('--source-config', default=None,
                    help='Base YAML to prune (default config/features.yaml).')
    ap.add_argument('--label-mode', default=None,
                    help='Recorded in the emitted config header.')
    args = ap.parse_args()

    models = list(MODELS) if args.model == 'all' else [args.model]
    results = {}
    for model in models:
        out = prune(args.run_id, model, threshold=args.threshold, verify=args.verify,
                    seeds=args.seeds, margin=args.margin,
                    min_eff_positives=args.min_eff_positives)
        print_report(out)
        results[model] = out

    if args.emit_config:
        if args.model != 'all':
            raise SystemExit('--emit-config requires --model all: the config assigns each '
                             'feature to the models that kept it, so all four must be known.')
        src = args.source_config or os.path.join(dir_config.CONFIG_DIR, 'features.yaml')
        label_mode = args.label_mode or next(
            (r.get('label_mode') for r in results.values() if r.get('label_mode')), None)
        disabled, narrowed, restored = emit_config(results, src, args.emit_config,
                                                   label_mode=label_mode)
        print(f'\nWrote {args.emit_config}')
        print(f"  disabled ({len(disabled)}): {', '.join(disabled) or '-'}")
        for name, kept in narrowed:
            print(f"  narrowed {name} -> [{', '.join(kept)}]")
        if restored:
            print(f"  KEPT FULL SET for {', '.join(restored)} - step 3 rejected the "
                  f"pruning for them")

    if args.out:
        keep_keys = ('baseline_test_auc', 'variant_test_auc', 'delta_mean', 'delta_ci95',
                     'n_seeds', 'operative_regime', 'baseline_n_features',
                     'variant_n_features')
        payload = {}
        for m, o in results.items():
            row = {k: v for k, v in o.items() if k != 'verification'}
            if o.get('verification'):
                row['verification'] = {k: o['verification'][k] for k in keep_keys}
                row['verdict'] = o['verdict']
            payload[m] = row
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f'\nResult written to {args.out}')


if __name__ == '__main__':
    main()
