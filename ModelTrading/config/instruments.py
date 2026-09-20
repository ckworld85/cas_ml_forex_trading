"""
Instrument registry — the single place that knows which symbols exist and how they decompose.

WHY THIS EXISTS
---------------
Until now every module carried its own `CSV_FOR = {'m15': 'eurusd_m15.csv', ...}` dict and its
own `forex.pip_value_for_symbol('EURUSD')`. That was fine while there was one instrument. It
stops being fine the moment a feature is a statement about EUR *relative to* the other
currencies, because such a feature cannot be computed from one CSV at all.

WHAT A PAIR ACTUALLY IS
-----------------------
A quoted pair is a *difference* between two currency strengths:

    log-return(EURUSD)  =  strength(EUR) - strength(USD)

so ten pairs over eight currencies are ten equations in eight unknowns. `base`/`quote` below
are what makes that system writable; everything in `features/cross_asset.py` follows from it.
Get one of them backwards and the whole decomposition silently flips a currency's sign, which
is why `test_instruments.py` checks every entry against its own filename.

PIP SIZE IS NOT COSMETIC
------------------------
`USDJPY` has a pip of 0.01, a hundred times `EURUSD`'s. A "35 pip stop" is therefore a
completely different distance on the two, and the break-even formula `S / (T + S)` is
unit-free — it will not notice the mistake. `pip_for` delegates to the existing
`utils.forex.pip_value_for_symbol` rather than re-deriving the rule.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# The base currency is the one you are long when you buy the pair.
#   EURUSD -> buying it is long EUR, short USD.
#   USDJPY -> buying it is long USD, short JPY.
PAIRS = {
    'EURUSD': ('EUR', 'USD'),
    'GBPUSD': ('GBP', 'USD'),
    'AUDUSD': ('AUD', 'USD'),
    'NZDUSD': ('NZD', 'USD'),
    'USDJPY': ('USD', 'JPY'),
    'USDCHF': ('USD', 'CHF'),
    'USDCAD': ('USD', 'CAD'),
    'EURJPY': ('EUR', 'JPY'),
    'EURGBP': ('EUR', 'GBP'),
    'AUDJPY': ('AUD', 'JPY'),
}

SYMBOLS = tuple(PAIRS)

# The instrument the models trade. Everything else exists to describe its context.
PRIMARY = 'EURUSD'

TIMEFRAMES = ('m15', '4hours', 'daily')

# The eight currencies the ten pairs span. Order is fixed so that a persisted design matrix
# and a persisted feature CSV keep meaning the same thing across runs.
CURRENCIES = ('AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD')


def csv_name(symbol, timeframe):
    """Provider CSV filename for a symbol/timeframe, e.g. ('GBPUSD', 'daily') -> gbpusd_daily.csv."""
    _check(symbol, timeframe)
    return f"{symbol.lower()}_{timeframe}.csv"


def csv_path(symbol, timeframe, data_dir=None):
    """Absolute path to the provider CSV. `data_dir` defaults to the configured DATA_DIR."""
    if data_dir is None:
        import ModelTrading.config.directories as dir_config
        data_dir = dir_config.DATA_DIR
    return os.path.join(data_dir, csv_name(symbol, timeframe))


def available(symbol, timeframe, data_dir=None):
    """Whether the CSV for this symbol/timeframe is actually on disk."""
    try:
        return os.path.exists(csv_path(symbol, timeframe, data_dir))
    except (KeyError, ValueError):
        return False


def available_symbols(timeframe, data_dir=None, symbols=None):
    """The subset of `symbols` whose CSV exists, in registry order.

    The cross-asset work has to run on whatever is present: the provider export arrives in
    waves, and a decomposition over seven pairs is still a decomposition. What must never
    happen is a *silent* change of universe, so every caller that uses this records the list
    it actually got.
    """
    return tuple(s for s in (symbols or SYMBOLS) if available(s, timeframe, data_dir))


def base_quote(symbol):
    """(base, quote) for a symbol — the two currencies its return is a difference of."""
    _check(symbol)
    return PAIRS[symbol]


def currencies_of(symbols):
    """The currencies spanned by `symbols`, in CURRENCIES order."""
    seen = set()
    for s in symbols:
        seen.update(PAIRS[s.upper()])
    return tuple(c for c in CURRENCIES if c in seen)


def pip_for(symbol):
    """Pip size. 0.01 for JPY quotes, 0.0001 otherwise — delegated, not re-derived."""
    _check(symbol)
    from ModelTrading.source.python.utils import forex
    return forex.pip_value_for_symbol(symbol)


def _check(symbol, timeframe=None):
    if symbol.upper() not in PAIRS:
        raise KeyError(f"unknown symbol {symbol!r}; known: {', '.join(SYMBOLS)}")
    if timeframe is not None and timeframe not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}; known: {', '.join(TIMEFRAMES)}")
