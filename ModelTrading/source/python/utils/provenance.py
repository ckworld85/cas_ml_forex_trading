"""JSON-safe provenance helpers for run summaries.

Motivation (2026-09-12): a backtest run with --opening-requires-fast-signal
produced 16 trades against 19 for the default invocation, and nothing in
backtest_summary.json recorded the difference — the summary carried thresholds,
costs and risk model, but not the entry gates or any other CLI flag. Two
summaries that disagreed on trades looked identically configured. Every
decision-relevant setting now goes into the summary, and this module keeps the
serialization from crashing on non-JSON argparse values (Paths, Timestamps,
numpy scalars) at the very end of a long run.
"""


_JSON_BASIC = (str, int, float, bool, type(None))


def sanitize_for_json(mapping):
    """Return a dict whose values survive ``json.dump`` unchanged in meaning.

    Basic JSON types pass through; numpy scalars unwrap via ``.item()``;
    lists/tuples are sanitized element-wise; nested dicts recurse; everything
    else is stringified rather than raising at dump time. Keys are sorted so
    two summaries diff cleanly.
    """
    out = {}
    for key in sorted(mapping, key=str):
        out[str(key)] = _sanitize_value(mapping[key])
    return out


def _sanitize_value(value):
    if isinstance(value, _JSON_BASIC):
        return value
    if isinstance(value, dict):
        return sanitize_for_json(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _sanitize_value(item())
        except (TypeError, ValueError):
            pass
    return str(value)
