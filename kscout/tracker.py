"""SQLite log of every recommendation, what you actually took, and outcomes."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone

from .edge import Rec, taker_fee_per_contract
from .kalshi import get_market

SCHEMA = """
CREATE TABLE IF NOT EXISTS recs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT, category TEXT, ticker TEXT, event_ticker TEXT, title TEXT,
    side TEXT, price REAL, limit_price REAL, model_prob REAL, model_yes_prob REAL,
    edge REAL, contracts INTEGER, cost REAL, expected_profit REAL, confidence TEXT,
    reasoning TEXT, close_time TEXT,
    result TEXT, settled_at TEXT
);
CREATE TABLE IF NOT EXISTS taken (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_id INTEGER, ticker TEXT, side TEXT, price REAL, contracts INTEGER,
    taken_at TEXT, note TEXT
);
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def log_recs(con: sqlite3.Connection, recs: list[Rec], dedupe_hours: float = 6) -> list[int]:
    """Insert recs; skip a ticker/side already logged in the last few hours."""
    ids = []
    now = datetime.now(timezone.utc)
    for r in recs:
        prev = con.execute(
            "SELECT id, created_at FROM recs WHERE ticker=? AND side=? ORDER BY id DESC LIMIT 1",
            (r.ticker, r.side)).fetchone()
        if prev:
            age = (now - datetime.fromisoformat(prev["created_at"])).total_seconds() / 3600
            if age < dedupe_hours:
                ids.append(prev["id"])
                continue
        d = asdict(r)
        cur = con.execute(
            """INSERT INTO recs (created_at, category, ticker, event_ticker, title, side, price,
               limit_price, model_prob, model_yes_prob, edge, contracts, cost, expected_profit,
               confidence, reasoning, close_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now.isoformat(), d["category"], d["ticker"], d["event_ticker"], d["title"], d["side"],
             d["price"], d["limit_price"], d["model_prob"], d["model_yes_prob"], d["edge"],
             d["contracts"], d["cost"], d["expected_profit"], d["confidence"], d["reasoning"],
             d["close_time"]))
        ids.append(cur.lastrowid)
    con.commit()
    return ids


def mark_taken(con, rec_id: int, price: float, contracts: int, note: str = "") -> None:
    row = con.execute("SELECT ticker, side FROM recs WHERE id=?", (rec_id,)).fetchone()
    if not row:
        raise SystemExit(f"No recommendation with id {rec_id}")
    con.execute("INSERT INTO taken (rec_id, ticker, side, price, contracts, taken_at, note) "
                "VALUES (?,?,?,?,?,?,?)",
                (rec_id, row["ticker"], row["side"], price, contracts,
                 datetime.now(timezone.utc).isoformat(), note))
    con.commit()


def settle(con) -> list[str]:
    msgs = []
    tickers = [r["ticker"] for r in con.execute(
        "SELECT DISTINCT ticker FROM recs WHERE result IS NULL")]
    for t in tickers:
        try:
            m = get_market(t)
        except Exception as e:
            msgs.append(f"{t}: lookup failed ({e})")
            continue
        if m.result in ("yes", "no"):
            con.execute("UPDATE recs SET result=?, settled_at=? WHERE ticker=? AND result IS NULL",
                        (m.result, datetime.now(timezone.utc).isoformat(), t))
            msgs.append(f"{t}: settled {m.result.upper()}")
    con.commit()
    return msgs


def _pnl(side: str, price: float, contracts: int, result: str, fee_rate: float = 0.07) -> float:
    won = (side == "YES" and result == "yes") or (side == "NO" and result == "no")
    fee = taker_fee_per_contract(price, contracts, fee_rate) * contracts
    return contracts * ((1 - price) if won else -price) - fee


def stats(con) -> str:
    rows = con.execute("SELECT * FROM recs WHERE result IN ('yes','no')").fetchall()
    lines = []
    if not rows:
        return "No settled recommendations yet. Run `python scout.py settle` after markets resolve."

    def block(name, rs):
        if not rs:
            return
        n = len(rs)
        brier = sum((r["model_yes_prob"] - (r["result"] == "yes")) ** 2 for r in rs) / n
        mkt_brier = sum(((r["price"] if r["side"] == "YES" else 1 - r["price"]) -
                         (r["result"] == "yes")) ** 2 for r in rs) / n
        ll = -sum(math.log(max(1e-6, r["model_yes_prob"] if r["result"] == "yes"
                               else 1 - r["model_yes_prob"])) for r in rs) / n
        wins = sum(1 for r in rs if (r["side"] == "YES") == (r["result"] == "yes"))
        pnl = sum(_pnl(r["side"], r["price"], r["contracts"], r["result"]) for r in rs)
        risk = sum(r["cost"] for r in rs)
        exp = sum(r["expected_profit"] for r in rs)
        lines.append(f"\n== {name} ({n} settled) ==")
        lines.append(f"  Hit rate:            {wins}/{n} = {wins / n:.0%}")
        lines.append(f"  Brier (model):       {brier:.4f}   (lower is better)")
        lines.append(f"  Brier (market price): {mkt_brier:.4f}  <- model must beat this")
        lines.append(f"  Log loss:            {ll:.4f}")
        lines.append(f"  Paper P&L:           ${pnl:+.2f} on ${risk:.2f} risked "
                     f"(ROI {pnl / risk:+.1%})" if risk else "")
        lines.append(f"  Expected P&L:        ${exp:+.2f}  (big gap vs actual = model overconfident)")

    block("ALL", rows)
    for cat in sorted({r["category"] for r in rows}):
        block(cat.upper(), [r for r in rows if r["category"] == cat])

    # Calibration buckets on P(side wins)
    lines.append("\n== Calibration (P(our side wins) vs how often it did) ==")
    buckets: dict[int, list] = {}
    for r in rows:
        won = (r["side"] == "YES") == (r["result"] == "yes")
        b = min(int(r["model_prob"] * 10), 9)
        buckets.setdefault(b, []).append(won)
    for b in sorted(buckets):
        ws = buckets[b]
        lines.append(f"  {b * 10:>3}-{b * 10 + 10:<3}%: predicted ~{b * 10 + 5}%, "
                     f"actual {sum(ws) / len(ws):.0%}  (n={len(ws)})")

    taken = con.execute("""SELECT t.*, r.result FROM taken t JOIN recs r ON r.id = t.rec_id
                           WHERE r.result IN ('yes','no')""").fetchall()
    if taken:
        real = sum(_pnl(t["side"], t["price"], t["contracts"], t["result"]) for t in taken)
        lines.append(f"\n== Trades you actually took: {len(taken)}, real P&L ${real:+.2f} ==")
    return "\n".join(lines)
