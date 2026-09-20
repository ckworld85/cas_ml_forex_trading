"""Transaction-cost model for the backtest.

Why this exists
---------------
Until this module was added, ``backtest.py`` read ASK closes for **both** legs of
every trade: a long bought at the ask and sold at the ask, a short sold at the ask
and bought back at the ask. Real execution crosses the book once per round trip, so
every configuration was flattered by exactly one full spread — and configurations
that trade more were flattered in proportion, which biases any comparison between
configurations with different trade volumes.

Measured on ``ModelTrading/data/eurusd_m15.csv`` (2026 sample, 20k bars): the spread
implied by ``2 * (ask_close - mid)`` has median **0.4 pips**, mean **1.8**, p95
**7.0**. Against a gross expectancy of ~6 pips per trade that is 7-30% of the entire
edge, and far more than that for high-frequency configurations.

Price convention
----------------
Every price in the pipeline is an **ASK** price (see CLAUDE.md, *Timeframes*). With
``s`` = full spread:

    ask = P                 bid = P - s

so a fill is

    buy   ->  P + slippage
    sell  ->  P - s - slippage

Applied to a round trip this costs exactly one spread plus two slippages in **both**
directions — the long pays the spread on its exit (a sell), the short pays it on its
entry (a sell). That symmetry is asserted in the tests: it is the invariant that
distinguishes a correct cost model from one that silently penalises one side.

Decision prices are **not** touched. The cost model converts a decision price into a
fill price and nothing else, so stop/take-profit triggers fire on exactly the bars
they fired on before. ``mode='none'`` is therefore bit-identical to the pre-cost
behaviour, which is what makes the regression check in the test suite meaningful.

The ``mid`` column
------------------
The provider CSVs carry ``time,open,high,low,close,volume,mid`` where OHLC is the ask
bar and ``mid`` is the mid close. Verified over 60k bars: ``mid <= close`` on every
single row (0 violations), with 2.1% of rows at exactly ``mid == close`` — a stalled
mid feed rather than a genuinely zero spread. Those rows fall back to
``min_spread_pips`` instead of trading for free.

``utils/csv.load_csv`` drops ``mid`` by default so the feature pipeline sees exactly
the columns it always saw; pass ``keep_mid=True`` to retain it.

Commission and overnight financing (Dukascopy fee schedule, 2026-09)
--------------------------------------------------------------------
The broker quotes both fees in **USD per 1M USD traded**: 18 USD trade commission per
side (the net-deposit > 50k tier) and overnight financing of 63.65 USD (long) /
28.65 USD (short) per settlement night. For EUR/USD with a EUR account the currency
conversion cancels exactly — USD traded = EUR notional x price, and the USD charge
converts back to EUR at that same price — so the quoted USD rates are charged here as
EUR per 1M EUR notional, with no FX conversion and no price plumbing.

Overnight financing is billed once per settlement the position is held through.
Settlements happen Monday-Friday at 21:00 UTC (the DST hour shift of the 17:00
New York roll is ignored; it can only matter for a position opened or closed within
that hour of the boundary). Spot FX settles T+2, so Wednesday's rollover finances the
weekend and is billed at 3x — the market-wide triple-swap convention. Saturday and
Sunday have no settlement: a position held Friday->Monday pays exactly Friday's
single night. ``count_rollover_nights`` implements this; the per-trade night count is
recorded in the trade CSV so the charge stays auditable.
"""

import os

import numpy as np
import pandas as pd

VALID_MODES = ('none', 'fixed', 'data')

# Dukascopy fee schedule for the account's tier (net deposit > 50k USD), quoted in
# USD per 1M USD traded — charged as EUR per 1M EUR notional (see module docstring
# for why the conversion cancels). Single source of truth for the CLI defaults in
# backtest.py and iterative_training.py; the class itself defaults to 0 so existing
# direct constructions (e.g. london_window_strategy) stay unchanged.
DEFAULT_COMMISSION_PER_MILLION = 18.0
DEFAULT_OVERNIGHT_LONG_PER_MILLION = 63.65
DEFAULT_OVERNIGHT_SHORT_PER_MILLION = 28.65

# FX settlement: Mon-Fri at 21:00 UTC, Wednesday billed 3x (T+2 weekend financing).
ROLLOVER_HOUR_UTC = 21
TRIPLE_ROLLOVER_WEEKDAY = 2  # Wednesday


def count_rollover_nights(open_time, close_time,
                          rollover_hour_utc=ROLLOVER_HOUR_UTC):
    """Financing nights billed for a position held over ``(open_time, close_time]``.

    Counts the Mon-Fri settlements at ``rollover_hour_utc`` strictly after
    ``open_time`` and up to ``close_time``, with Wednesday's settlement counted 3x
    (see module docstring). A position opened exactly at the settlement instant is
    not charged for it; one closed exactly on it is.
    """
    open_time = pd.Timestamp(open_time)
    close_time = pd.Timestamp(close_time)
    if pd.isna(open_time) or pd.isna(close_time) or close_time <= open_time:
        return 0

    settlement = open_time.normalize() + pd.Timedelta(hours=rollover_hour_utc)
    if settlement <= open_time:
        settlement += pd.Timedelta(days=1)

    nights = 0
    one_day = pd.Timedelta(days=1)
    while settlement <= close_time:
        weekday = settlement.weekday()
        if weekday < 5:  # Sat/Sun have no settlement
            nights += 3 if weekday == TRIPLE_ROLLOVER_WEEKDAY else 1
        settlement += one_day
    return nights


def half_spread_from_bars(df, pip_size, min_spread_pips=0.2):
    """Derive the per-bar half spread (in price units) from ask OHLC + mid close.

    Args:
        df: DataFrame carrying an ask ``close`` column and a ``mid`` column.
        pip_size: 0.0001 for non-JPY pairs (see utils.forex.pip_value_for_symbol).
        min_spread_pips: floor for the FULL spread. Rows where the mid feed stalled
            (``mid == close``) would otherwise trade at zero cost.

    Returns:
        pd.Series of half spreads in price units, indexed like ``df``.
    """
    missing = [c for c in ('close', 'mid') if c not in df.columns]
    if missing:
        raise ValueError(
            f"half_spread_from_bars needs columns {missing}; got {list(df.columns)}. "
            f"Load the CSV with load_csv(..., keep_mid=True)."
        )

    full_spread = (df['close'].astype('float64') - df['mid'].astype('float64')) * 2.0
    floor = min_spread_pips * pip_size
    # A mid above the ask is not a spread, it is bad data — floor it like a stall.
    full_spread = full_spread.clip(lower=floor)
    return full_spread / 2.0


def load_half_spread(csv_path, index, pip_size, min_spread_pips=0.2,
                     filter_weekends_flag=True, verbose=True):
    """Load the per-bar half spread for ``index`` from the raw provider CSV.

    Reuses ``utils.csv.load_csv`` rather than re-implementing the weekend filter, so
    bar selection stays identical to training (CLAUDE.md: *Do not change bar
    selection*). Bars of ``index`` that the CSV does not cover fall back to the
    median of the bars it does cover.

    Returns:
        (pd.Series aligned to ``index``, dict of diagnostics)
    """
    # Imported here so this module stays importable without the package root on the
    # path (the tests import it directly).
    import ModelTrading.source.python.utils.csv as csv_utils

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"--cost-model data needs the raw provider CSV at {csv_path} (it carries "
            f"the 'mid' column the spread is derived from). Use --cost-model fixed "
            f"with --spread-pips instead if the CSV is unavailable."
        )

    raw = csv_utils.load_csv(csv_path, filter_weekends_flag=filter_weekends_flag,
                             keep_mid=True)
    if 'mid' not in raw.columns:
        raise ValueError(
            f"{os.path.basename(csv_path)} has no 'mid' column, so the spread cannot "
            f"be derived from it. Use --cost-model fixed --spread-pips <x>."
        )

    half = half_spread_from_bars(raw, pip_size, min_spread_pips=min_spread_pips)
    aligned = half.reindex(index)

    n_missing = int(aligned.isna().sum())
    fallback = float(half.median()) if len(half) else min_spread_pips * pip_size / 2.0
    aligned = aligned.fillna(fallback)

    diagnostics = {
        'n_bars': int(len(aligned)),
        'n_missing_filled': n_missing,
        'median_spread_pips': float(aligned.median() * 2.0 / pip_size),
        'mean_spread_pips': float(aligned.mean() * 2.0 / pip_size),
        'p95_spread_pips': float(aligned.quantile(0.95) * 2.0 / pip_size),
    }
    if verbose:
        print(f"  Spread from {os.path.basename(csv_path)}: "
              f"median {diagnostics['median_spread_pips']:.2f} pips, "
              f"mean {diagnostics['mean_spread_pips']:.2f}, "
              f"p95 {diagnostics['p95_spread_pips']:.2f}"
              + (f" ({n_missing} bars filled with the median)" if n_missing else ""))

    return aligned, diagnostics


class TransactionCostModel:
    """Converts decision prices into fill prices and accounts commission.

    Args:
        mode: 'none'   -> no cost at all; fills equal decision prices (the historical
                          behaviour, kept so old results stay reproducible).
              'fixed'  -> constant ``spread_pips`` on every bar.
              'data'   -> per-bar spread from ``half_spread`` (ask - mid).
        pip_size: 0.0001 for non-JPY pairs.
        half_spread: array-like of half spreads in price units, positionally aligned
            to the bar index. Required for mode='data'.
        spread_pips: full spread in pips for mode='fixed'.
        slippage_pips: applied against the trader on EVERY leg (so a round trip pays
            2x). Models the queue position a market order actually gets.
        commission_per_million: account-currency commission per 1M notional per side.
            A round trip therefore pays 2x this on the traded notional. The broker
            quotes it in USD per 1M USD traded; for EUR/USD with a EUR account the
            conversion cancels, so the quoted rate applies to the EUR notional as-is
            (see module docstring).
        overnight_long_per_million / overnight_short_per_million: overnight financing
            per 1M notional per settlement night held, same currency cancellation as
            the commission. Charged ``nights x rate`` via :meth:`overnight_eur`, with
            the nights counted by :func:`count_rollover_nights`.
    """

    def __init__(self, mode='none', pip_size=0.0001, half_spread=None,
                 spread_pips=0.0, slippage_pips=0.0, commission_per_million=0.0,
                 overnight_long_per_million=0.0, overnight_short_per_million=0.0):
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown cost mode {mode!r}; expected one of {VALID_MODES}")

        self.mode = mode
        self.pip_size = float(pip_size)
        self.spread_pips = float(spread_pips)
        self.slippage_pips = float(slippage_pips)
        self.commission_per_million = float(commission_per_million)
        self.overnight_long_per_million = float(overnight_long_per_million)
        self.overnight_short_per_million = float(overnight_short_per_million)

        if mode == 'data':
            if half_spread is None:
                raise ValueError("mode='data' requires a half_spread series")
            self._half_spread = np.asarray(half_spread, dtype='float64')
        else:
            self._half_spread = None

        self._slippage = self.slippage_pips * self.pip_size

    @property
    def enabled(self):
        """True when this model can move a fill away from its decision price."""
        return self.mode != 'none' and (
            self.spread_pips > 0 or self._half_spread is not None
            or self.slippage_pips > 0
        )

    def full_spread(self, i):
        """Full spread in price units at bar ``i``."""
        if self.mode == 'none':
            return 0.0
        if self.mode == 'fixed':
            return self.spread_pips * self.pip_size
        return float(self._half_spread[i]) * 2.0

    def round_trip_pips(self, i_entry, i_exit):
        """Spread + slippage cost of a full round trip, in pips.

        The spread is charged once, on whichever leg is the sell: the exit for a long,
        the entry for a short. Both therefore pay one spread — this helper takes it
        from the leg it is actually paid on for the long case and is used for
        reporting only; ``fill_price`` is what the P&L is computed from.
        """
        if self.mode == 'none':
            return 0.0
        spread = self.full_spread(i_exit)
        return spread / self.pip_size + 2.0 * self.slippage_pips

    def fill_price(self, i, price, side):
        """Fill price for a decision price ``price`` at bar ``i``.

        Args:
            side: 'buy' or 'sell'. Prices are ASK, so a buy fills at the quoted price
                and a sell fills one spread below it.
        """
        if self.mode == 'none':
            return float(price)
        if side == 'buy':
            return float(price) + self._slippage
        if side == 'sell':
            return float(price) - self.full_spread(i) - self._slippage
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def entry_fill(self, i, price, is_long):
        """Fill price for opening a position (buy when long, sell when short)."""
        return self.fill_price(i, price, 'buy' if is_long else 'sell')

    def exit_fill(self, i, price, is_long):
        """Fill price for closing a position (sell when long, buy when short)."""
        return self.fill_price(i, price, 'sell' if is_long else 'buy')

    def commission_eur(self, notional):
        """Round-trip commission in account currency for ``notional``."""
        if self.mode == 'none' or self.commission_per_million == 0.0:
            return 0.0
        return 2.0 * self.commission_per_million * (float(notional) / 1_000_000.0)

    def overnight_eur(self, notional, nights, is_long):
        """Overnight financing in account currency for ``nights`` settlement nights.

        ``nights`` comes from :func:`count_rollover_nights` (Wednesday already counts
        3x there). The long and short sides carry different rates — the financing is
        an interest-rate differential plus a broker markup, not a symmetric fee.
        """
        if self.mode == 'none' or nights <= 0:
            return 0.0
        rate = (self.overnight_long_per_million if is_long
                else self.overnight_short_per_million)
        return rate * float(nights) * (float(notional) / 1_000_000.0)

    def describe(self):
        """One-line summary for the backtest header."""
        if self.mode == 'none':
            return "Transaction costs: NONE (gross P&L — not tradeable)"
        if self.mode == 'fixed':
            base = f"fixed spread {self.spread_pips:.2f} pips"
        else:
            med = float(np.median(self._half_spread)) * 2.0 / self.pip_size
            base = f"per-bar spread from ask-mid (median {med:.2f} pips)"
        return (f"Transaction costs: {base}, slippage {self.slippage_pips:.2f} pips/leg, "
                f"commission {self.commission_per_million:.2f}/M/side, "
                f"overnight {self.overnight_long_per_million:.2f}/"
                f"{self.overnight_short_per_million:.2f} per M/night (long/short)")


def build_cost_model(mode, index, pip_size, csv_path=None, spread_pips=0.0,
                     slippage_pips=0.0, commission_per_million=0.0,
                     overnight_long_per_million=0.0, overnight_short_per_million=0.0,
                     min_spread_pips=0.2, verbose=True):
    """Construct a TransactionCostModel, loading the spread series when needed."""
    diagnostics = {}
    half_spread = None
    if mode == 'data':
        half_spread, diagnostics = load_half_spread(
            csv_path, index, pip_size, min_spread_pips=min_spread_pips, verbose=verbose
        )
        half_spread = half_spread.values

    model = TransactionCostModel(
        mode=mode, pip_size=pip_size, half_spread=half_spread,
        spread_pips=spread_pips, slippage_pips=slippage_pips,
        commission_per_million=commission_per_million,
        overnight_long_per_million=overnight_long_per_million,
        overnight_short_per_million=overnight_short_per_million,
    )
    return model, diagnostics
