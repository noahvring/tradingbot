"""Politics markets.

There's no free, reliable, machine-readable "fair probability" feed for
politics the way sportsbooks provide one for games. So this module does two
honest things instead of pretending:

1. YOUR PRIORS (makes real recommendations)
   You write your own probability for specific tickers in politics_priors.yaml
   (from polling averages, forecast models, your own research). The tool
   compares them to live prices and sizes trades. Your research, its math.

2. CROSS-MARKET CHECK (review list only, never a "trade")
   It looks for similar questions on Polymarket and lists big price gaps for
   you to investigate. Resolution rules often differ between the two sites,
   so a gap is a lead to read the rules - not a free arbitrage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .edge import Rec, evaluate, passes_liquidity
from .http import get_json
from .kalshi import Market, get_events, get_market, get_orderbook

POLITICS_CATEGORIES = {"politics", "elections", "world", "policy"}


def load_priors(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or []
    return [d for d in data if isinstance(d, dict) and d.get("ticker") and d.get("prob") is not None]


def from_priors(cfg: dict) -> tuple[list[Rec], list[str]]:
    recs, log = [], []
    priors = load_priors(cfg["politics"]["priors_file"])
    if not priors:
        return [], ["[politics] no priors in politics_priors.yaml - add some to get trade recs"]
    for pr in priors:
        try:
            m = get_market(pr["ticker"])
        except Exception as e:
            log.append(f"[politics] {pr['ticker']}: not found ({e})")
            continue
        if m.status not in ("open", "active"):
            log.append(f"[politics] {pr['ticker']}: market is {m.status}")
            continue
        ok, why = passes_liquidity(m, cfg)
        if not ok:
            log.append(f"[politics] {pr['ticker']}: skipped, {why}")
            continue
        p = float(pr["prob"])
        reason = f"Your prior {p:.0%}" + (f" — {pr['note']}" if pr.get("note") else "")
        rec = evaluate(m, p, cfg, "politics", reason, pr.get("confidence", "medium"))
        if rec:
            try:
                rec = evaluate(m, p, cfg, "politics", reason, pr.get("confidence", "medium"),
                               orderbook=get_orderbook(m.ticker))
            except Exception:
                rec.warnings.append("order book not checked")
            if rec:
                recs.append(rec)
        else:
            mp = m.mid
            log.append(f"[politics] {pr['ticker']}: no edge (yours {p:.0%}, market "
                       f"{mp:.0%})" if mp is not None else f"[politics] {pr['ticker']}: no edge")
    return recs, log


# ----------------------------- cross-market -----------------------------

_STOP = set("will the a an of in on to be by for and or who which what is 2026 2027 2028 before after "
            "win wins next than more less".split())


def _tok(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 2}


def similarity(a: str, b: str) -> float:
    ta, tb = _tok(a), _tok(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def polymarket_markets(limit_pages: int = 5) -> list[dict]:
    out = []
    for page in range(limit_pages):
        data = get_json("https://gamma-api.polymarket.com/markets", params={
            "active": "true", "closed": "false", "limit": 500, "offset": page * 500,
            "order": "volume", "ascending": "false",
        })
        if not data:
            break
        for d in data:
            try:
                prices = json.loads(d.get("outcomePrices") or "[]")
                outcomes = json.loads(d.get("outcomes") or "[]")
            except (TypeError, ValueError):
                continue
            if outcomes[:2] == ["Yes", "No"] and prices:
                out.append({"question": d.get("question", ""), "yes": float(prices[0]),
                            "volume": float(d.get("volume") or 0), "slug": d.get("slug", "")})
    return out


def kalshi_politics_markets() -> list[Market]:
    ms = []
    for ev in get_events():
        if (ev.get("category") or "").lower() not in POLITICS_CATEGORIES:
            continue
        for raw in ev.get("markets", []):
            m = Market.from_api(raw)
            if not m.title:
                m.title = ev.get("title", "")
            ms.append(m)
    return ms


def cross_check(cfg: dict) -> tuple[list[dict], list[str]]:
    pc = cfg["politics"]
    try:
        k_markets = kalshi_politics_markets()
        p_markets = polymarket_markets()
    except Exception as e:
        return [], [f"[politics] cross-check unavailable ({e})"]
    leads = []
    for m in k_markets:
        if m.mid is None or m.volume < cfg["filters"]["min_volume"]:
            continue
        q = f"{m.title} {m.subtitle}"
        best, score = None, 0.0
        for pm in p_markets:
            s = similarity(q, pm["question"])
            if s > score:
                best, score = pm, s
        if best and score >= pc["min_similarity"] and abs(best["yes"] - m.mid) >= pc["min_gap"]:
            leads.append({"ticker": m.ticker, "kalshi_q": q, "kalshi_mid": m.mid,
                          "poly_q": best["question"], "poly_yes": best["yes"],
                          "gap": best["yes"] - m.mid, "similarity": score,
                          "poly_url": f"https://polymarket.com/event/{best['slug']}"})
    leads.sort(key=lambda x: -abs(x["gap"]))
    return leads[: pc["max_leads"]], [f"[politics] cross-check: {len(k_markets)} Kalshi vs "
                                      f"{len(p_markets)} Polymarket markets, {len(leads)} leads"]


def recommend(cfg: dict) -> tuple[list[Rec], list[str], list[dict]]:
    recs, log = from_priors(cfg)
    leads: list[dict] = []
    if cfg["politics"].get("cross_check", True):
        leads, l2 = cross_check(cfg)
        log += l2
    return recs, log, leads
