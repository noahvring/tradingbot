"""Printing and saving recommendations."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .edge import Rec


def _fmt_close(s: str) -> str:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone().strftime("%a %b %d %I:%M%p")
    except (ValueError, AttributeError):
        return s or "?"


def print_recs(recs: list[Rec], ids: list[int], leads: list[dict], log: list[str], verbose: bool) -> None:
    if verbose:
        print("\n".join(log))
        print()
    if not recs:
        print("No trades clear your edge threshold right now. That's normal - most of the time "
              "the market is priced about right.\n")
    else:
        total_cost = sum(r.cost for r in recs)
        total_ev = sum(r.expected_profit for r in recs)
        print(f"\n{len(recs)} RECOMMENDATIONS  (total risk ${total_cost:.2f}, "
              f"expected profit ${total_ev:+.2f})\n" + "=" * 78)
        for i, (r, rid) in enumerate(zip(recs, ids), 1):
            print(f"\n#{i}  [id {rid}]  {r.category.upper()}  |  confidence: {r.confidence}")
            print(f"    {r.title}")
            print(f"    Ticker: {r.ticker}")
            print(f"    >>> BUY {r.side}  x{r.contracts}  at ${r.price:.2f}  "
                  f"(limit order, max ${r.limit_price:.2f})  cost ${r.cost:.2f}")
            print(f"    P(win) {r.model_prob:.1%} (model/market blend) vs price {r.market_prob:.1%}  ->  "
                  f"edge {r.edge * 100:+.1f} pts after fees, EV ${r.expected_profit:+.2f}")
            print(f"    Why: {r.reasoning}")
            print(f"    Closes: {_fmt_close(r.close_time)}")
            for w in r.warnings:
                print(f"    ! {w}")
    if leads:
        print("\n" + "=" * 78 + "\nPOLITICS LEADS TO RESEARCH (not trade recs — read both "
              "markets' resolution rules first)\n")
        for L in leads:
            print(f"  {L['ticker']}: Kalshi {L['kalshi_mid']:.0%} vs Polymarket {L['poly_yes']:.0%} "
                  f"(gap {L['gap'] * 100:+.0f} pts, match {L['similarity']:.2f})")
            print(f"     K: {L['kalshi_q'][:90]}")
            print(f"     P: {L['poly_q'][:90]}")
    print()


def save(recs: list[Rec], ids: list[int], leads: list[dict], out_dir: str) -> tuple[Path, Path]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = Path(out_dir) / f"recs_{stamp}.csv"
    md_path = Path(out_dir) / f"recs_{stamp}.md"

    fields = ["id"] + list(asdict(recs[0]).keys()) if recs else ["id"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, rid in zip(recs, ids):
            row = asdict(r)
            row["warnings"] = "; ".join(row["warnings"])
            w.writerow({"id": rid, **row})

    lines = [f"# Kalshi Scout — {datetime.now():%a %b %d, %Y %I:%M %p}", ""]
    if recs:
        lines += ["| id | Cat | Market | Buy | Qty | Price | Max | P(win) | Edge | EV | Conf |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r, rid in zip(recs, ids):
            lines.append(f"| {rid} | {r.category} | {r.title[:60]} (`{r.ticker}`) | {r.side} | "
                         f"{r.contracts} | ${r.price:.2f} | ${r.limit_price:.2f} | {r.model_prob:.0%} | "
                         f"{r.edge * 100:+.1f} | ${r.expected_profit:+.2f} | {r.confidence} |")
        lines += ["", "## Reasoning", ""]
        for r, rid in zip(recs, ids):
            lines.append(f"- **{rid} · {r.ticker}** — {r.reasoning}"
                         + (f" ⚠️ {'; '.join(r.warnings)}" if r.warnings else ""))
    else:
        lines.append("No trades clear the edge threshold right now.")
    if leads:
        lines += ["", "## Politics leads (research, not trades)", ""]
        for L in leads:
            lines.append(f"- `{L['ticker']}` Kalshi {L['kalshi_mid']:.0%} vs Polymarket "
                         f"{L['poly_yes']:.0%} — {L['kalshi_q'][:80]} / [{L['poly_q'][:60]}]({L['poly_url']})")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path
