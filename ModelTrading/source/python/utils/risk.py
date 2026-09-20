"""Position sizing for the backtest.

Why this exists
---------------
`backtest.py` traded a fixed 1,000,000 EUR notional on every entry, independent of the
stop distance and of account equity. That is not a risk model. Against the configured
50,000 EUR of capital a 35-pip stop already risks 6.4% of the account per trade, and the
100-150 pip stop a multi-day swing design needs would risk 20-30%. Measured on run
honest_20260829 the fixed rule produced a **-38.5% maximum drawdown** where the same
trades under 1%-of-equity sizing produce **-8.4%** — the trades are identical, only the
sizing differs.

The consequence is that every drawdown-, Sharpe-, Sortino- and Calmar-derived number in
every report produced before this module describes the *sizing rule*, not the strategy.
Pips are unaffected: sizing changes no entry, no exit and no trigger.

Fixed-fractional sizing
-----------------------
Size each trade so that being stopped out always costs the same fraction of current
equity. For a EUR-base pair quoted in USD, the loss on `notional` EUR at `stop_pips` is

    loss_usd = notional * stop_pips * pip_size
    loss_eur = loss_usd / price

so solving `loss_eur = risk_eur` gives

    notional = risk_eur * price / (stop_pips * pip_size)

A wider stop therefore buys a smaller position and the risk per trade is invariant to
stop distance, regime and volatility level — which is exactly what makes drawdown
comparable across configurations that use different stops.

`max_leverage` caps the result: as `stop_pips -> 0` the formula diverges.
"""


def position_notional(equity, stop_pips, price, risk_pct=1.0, max_leverage=30.0,
                      pip_size=0.0001):
    """Notional to trade so a stop-out costs ``risk_pct`` percent of ``equity``.

    Args:
        equity: current account equity in the account currency.
        stop_pips: stop distance in pips (the *effective* one, after any regime widening).
        price: current price, used to convert the quote-currency loss into base currency.
        risk_pct: percent of equity to risk on this trade.
        max_leverage: cap on notional / equity.
        pip_size: 0.0001 for non-JPY pairs.

    Returns:
        float notional, 0.0 when the inputs cannot support a position.
    """
    equity = max(float(equity), 0.0)
    if equity <= 0 or stop_pips <= 0 or price <= 0 or risk_pct <= 0:
        return 0.0
    risk = equity * (float(risk_pct) / 100.0)
    notional = risk * float(price) / (float(stop_pips) * float(pip_size))
    return float(min(notional, equity * float(max_leverage)))


def realised_risk_pct(notional, stop_pips, price, equity, pip_size=0.0001):
    """Percent of ``equity`` a stop-out on this position would cost.

    The inverse of `position_notional`; used in tests and diagnostics to confirm that the
    risk actually held constant across stop distances.
    """
    if equity <= 0 or price <= 0:
        return 0.0
    loss = float(notional) * float(stop_pips) * float(pip_size) / float(price)
    return loss / float(equity) * 100.0
