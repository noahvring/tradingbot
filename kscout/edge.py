"""Turning a model probability + market prices into a sized recommendation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .kalshi import Market, depth_to_buy


def taker_fee_per_contract(price: float, contracts: float = 1, rate: float = 0.07) -> float:
    """Kalshi's general taker fee: ceil-to-cent(rate * C * P * (1-P)), per contract.

    Some series use a different rate (check Kalshi's fee schedule) - it's in config.
    """
    if contracts <= 0:
        return 0.0
    total = math.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100.0
    return total / contracts


@dataclass
class Rec:
    category: str
    ticker: str
    event_ticker: str
    title: str
    side: str                 # "YES" or "NO"
    price: float              # current best ask for that side ($)
    limit_price: float        # don't pay more than this
    model_prob: float         # model P(side wins)
    market_prob: float        # implied by ask
    edge: float               # model_prob - price - fee (in probability points)
    ev_per_contract: float    # $ expected profit per contract after fees
    contracts: int
    cost: float               # $ at risk
    expected_profit: float
    confidence: str           # high / medium / low
    reasoning: str
    model_yes_prob: float     # model P(YES) - stored for calibration tracking
    close_time: str = ""
    warnings: list[str] = field(default_factory=list)


def kelly_fraction(p: float, cost: float) -> float:
    """Full-Kelly fraction of bankroll for a $1 binary bought at `cost`."""
    if cost <= 0 or cost >= 1:
        return 0.0
    return max(0.0, (p - cost) / (1 - cost))


def evaluate(market: Market, p_yes: float, cfg: dict, category: str, reasoning: str,
             confidence: str = "medium", orderbook: dict | None = None,
             warnings: list[str] | None = None) -> Rec | None:
    """Return the better of YES/NO if it clears the edge bar, else None."""
    s = cfg["sizing"]
    f = cfg["filters"]
    fee_rate = cfg.get("fees", {}).get(category, cfg.get("fees", {}).get("default", 0.07))
    min_edge = f.get("min_edge_by_category", {}).get(category, f["min_edge"])
    p_yes = min(max(p_yes, 0.001), 0.999)

    best: Rec | None = None
    for side, ask, p in (("YES", market.yes_ask, p_yes), ("NO", market.no_ask, 1 - p_yes)):
        if ask is None or ask <= 0 or ask >= 1:
            continue
        if ask < f["min_price"] or ask > f["max_price"]:
            continue
        fee = taker_fee_per_contract(ask, 10, fee_rate)
        edge = p - ask - fee
        if edge < min_edge:
            continue

        # Highest price at which the trade still clears min_edge (ignoring fee drift)
        limit = round(math.floor((p - min_edge - fee) * 100) / 100, 2)
        limit = max(limit, ask)

        all_in = ask + fee
        k = kelly_fraction(p, all_in) * s["kelly_fraction"]
        dollars = min(k * s["bankroll"], s["max_per_trade"])
        contracts = int(dollars // all_in)

        w = list(warnings or [])
        if orderbook is not None:
            avail, _ = depth_to_buy(orderbook, side.lower(), limit)
            if avail < contracts:
                w.append(f"only {int(avail)} contracts offered at <= ${limit:.2f}")
                contracts = int(avail)
        if contracts < 1:
            continue

        if edge > 0.20:
            w.append(f"edge {edge * 100:.0f} pts is implausibly large for a liquid market - "
                     "assume the model is missing something until stats prove otherwise")
        ev = p - ask - fee
        rec = Rec(
            category=category, ticker=market.ticker, event_ticker=market.event_ticker,
            title=f"{market.title} — {market.subtitle}".strip(" —"),
            side=side, price=ask, limit_price=limit, model_prob=p, market_prob=ask,
            edge=edge, ev_per_contract=ev, contracts=contracts,
            cost=round(contracts * all_in, 2), expected_profit=round(contracts * ev, 2),
            confidence=confidence, reasoning=reasoning, model_yes_prob=p_yes,
            close_time=market.close_time, warnings=w,
        )
        if best is None or rec.expected_profit > best.expected_profit:
            best = rec
    return best


def passes_liquidity(market: Market, cfg: dict) -> tuple[bool, str]:
    f = cfg["filters"]
    if market.spread is not None and market.spread > f["max_spread"]:
        return False, f"spread {market.spread:.2f} too wide"
    if market.volume < f["min_volume"]:
        return False, f"volume {market.volume:.0f} too low"
    return True, ""
