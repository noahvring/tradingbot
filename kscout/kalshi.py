"""Kalshi public market data (no API key, no account, GET only)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .http import get_json

BASE = "https://api.elections.kalshi.com/trade-api/v2"


# ---------- field helpers (Kalshi has both cent and dollar fields) ----------

def _price(m: dict, name: str) -> float | None:
    """Return a price in dollars (0-1) for e.g. name='yes_ask'."""
    d = m.get(f"{name}_dollars")
    if d not in (None, ""):
        try:
            return float(d)
        except (TypeError, ValueError):
            pass
    c = m.get(name)
    if c in (None, ""):
        return None
    try:
        return float(c) / 100.0
    except (TypeError, ValueError):
        return None


def _num(m: dict, name: str) -> float:
    for key in (name, f"{name}_fp"):
        v = m.get(key)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


@dataclass
class Market:
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    last_price: float | None
    volume: float
    volume_24h: float
    open_interest: float
    close_time: str
    strike_type: str | None
    floor_strike: float | None
    cap_strike: float | None
    status: str
    result: str
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_api(cls, m: dict) -> "Market":
        yes_bid, yes_ask = _price(m, "yes_bid"), _price(m, "yes_ask")
        no_bid, no_ask = _price(m, "no_bid"), _price(m, "no_ask")
        # Derive missing sides: buying NO at x == selling YES at 1-x.
        if no_ask is None and yes_bid is not None:
            no_ask = round(1 - yes_bid, 4)
        if no_bid is None and yes_ask is not None:
            no_bid = round(1 - yes_ask, 4)
        # Kalshi reports 0 bid / 100 ask for an empty side: treat as missing
        if yes_ask is not None and yes_ask >= 1.0:
            yes_ask = None
        if no_ask is not None and no_ask >= 1.0:
            no_ask = None
        fs, cs = m.get("floor_strike"), m.get("cap_strike")
        return cls(
            ticker=m.get("ticker", ""),
            event_ticker=m.get("event_ticker", ""),
            title=m.get("title", ""),
            subtitle=m.get("yes_sub_title") or m.get("subtitle") or "",
            yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
            last_price=_price(m, "last_price"),
            volume=_num(m, "volume"),
            volume_24h=_num(m, "volume_24h"),
            open_interest=_num(m, "open_interest"),
            close_time=m.get("close_time", ""),
            strike_type=m.get("strike_type"),
            floor_strike=float(fs) if fs is not None else None,
            cap_strike=float(cs) if cs is not None else None,
            status=m.get("status", ""),
            result=(m.get("result") or "").lower(),
            raw=m,
        )

    @property
    def mid(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def spread(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None


# ------------------------------- endpoints -------------------------------

def get_markets(series_ticker: str | None = None, event_ticker: str | None = None,
                status: str = "open", max_pages: int = 10) -> list[Market]:
    params: dict = {"limit": 1000}
    if status:
        params["status"] = status
    if series_ticker:
        params["series_ticker"] = series_ticker
    if event_ticker:
        params["event_ticker"] = event_ticker
    out: list[Market] = []
    cursor = None
    for _ in range(max_pages):
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{BASE}/markets", params=params)
        out.extend(Market.from_api(m) for m in data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def get_events(status: str = "open", max_pages: int = 15) -> list[dict]:
    """Open events with their markets nested (used for category scans)."""
    params: dict = {"limit": 200, "status": status, "with_nested_markets": "true"}
    out: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{BASE}/events", params=params)
        out.extend(data.get("events", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def get_market(ticker: str) -> Market:
    data = get_json(f"{BASE}/markets/{ticker}")
    return Market.from_api(data.get("market", data))


def get_orderbook(ticker: str) -> dict:
    data = get_json(f"{BASE}/markets/{ticker}/orderbook")
    return parse_orderbook(data)


def parse_orderbook(data: dict) -> dict:
    """Normalize to {'yes': [(price$, qty)], 'no': [(price$, qty)]} bids, best first.

    Kalshi only lists BIDS. The ask for YES is (1 - best NO bid) and vice versa.
    """
    ob = data.get("orderbook") or data.get("orderbook_fp") or data
    result = {}
    for side in ("yes", "no"):
        levels = ob.get(f"{side}_dollars")
        if levels:
            parsed = [(float(p), float(q)) for p, q in levels]
        else:
            parsed = [(float(p) / 100.0, float(q)) for p, q in (ob.get(side) or [])]
        result[side] = sorted(parsed, key=lambda x: -x[0])
    return result


def depth_to_buy(orderbook: dict, side: str, limit_price: float) -> tuple[float, float]:
    """Contracts available to BUY `side` at or below limit_price, and avg price.

    Buying YES lifts NO bids: a NO bid at q is a YES offer at 1-q.
    """
    opposite = "no" if side == "yes" else "yes"
    qty_total, cost_total = 0.0, 0.0
    for bid_price, qty in orderbook.get(opposite, []):
        ask = round(1 - bid_price, 4)
        if ask > limit_price + 1e-9:
            break
        qty_total += qty
        cost_total += qty * ask
    avg = cost_total / qty_total if qty_total else 0.0
    return qty_total, avg


_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def event_date(event_ticker: str) -> date | None:
    """KXHIGHNY-26NOV05 -> date(2026, 11, 5)."""
    m = _DATE_RE.search(event_ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    if mon not in _MONTHS:
        return None
    try:
        return date(2000 + int(yy), _MONTHS[mon], int(dd))
    except ValueError:
        return None
