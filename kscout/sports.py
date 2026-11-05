"""Game-winner markets (NFL, NBA, NHL, MLB, college).

Model
-----
Sportsbooks price games very efficiently, so we don't try to out-model them.
Instead we take a "fair" probability from sharp books (Pinnacle first; else the
median of all books), remove the bookmaker's margin ("devig"), and flag Kalshi
prices that disagree by more than fees + your minimum edge.

Needs a free API key from https://the-odds-api.com (500 requests/month free).
Put it in config.yaml or the ODDS_API_KEY environment variable.
"""
from __future__ import annotations

import os
import re
import statistics
from datetime import datetime, timezone
from difflib import SequenceMatcher

from .edge import Rec, evaluate, passes_liquidity
from .http import get_json
from .kalshi import Market, get_markets, get_orderbook

ODDS_BASE = "https://api.the-odds-api.com/v4/sports"


def devig(decimal_odds: list[float]) -> list[float]:
    """Proportional (multiplicative) devig: implied probs scaled to sum to 1."""
    implied = [1 / o for o in decimal_odds]
    total = sum(implied)
    return [p / total for p in implied]


def fair_probs(event: dict, sharp_books: list[str], min_books: int) -> tuple[dict[str, float], str]:
    """Return {team_name: fair_prob}, plus a description of the source."""
    per_book: dict[str, dict[str, float]] = {}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            outs = mk.get("outcomes", [])
            if len(outs) != 2:
                continue  # skip 3-way (draw) markets
            probs = devig([float(o["price"]) for o in outs])
            per_book[bk["key"]] = {o["name"]: p for o, p in zip(outs, probs)}

    for sharp in sharp_books:
        if sharp in per_book:
            return per_book[sharp], f"{sharp} (devigged)"
    if len(per_book) >= min_books:
        teams = list(next(iter(per_book.values())).keys())
        med = {t: statistics.median(b[t] for b in per_book.values() if t in b) for t in teams}
        s = sum(med.values())
        return {t: v / s for t, v in med.items()}, f"median of {len(per_book)} books (devigged)"
    return {}, ""


_STOP = {"the", "fc", "at", "vs", "winner", "will", "win", "game", "to"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


def team_score(a: str, b: str) -> float:
    """Similarity between two team strings (e.g. 'Washington' vs 'Washington Commanders')."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / min(len(ta), len(tb))
    return max(overlap, SequenceMatcher(None, a.lower(), b.lower()).ratio())


def match_market(m: Market, events: list[dict], max_hours: float = 36) -> tuple[dict | None, str | None, float]:
    """Find the odds event + team that this Kalshi market's YES side refers to."""
    yes_team = m.subtitle or m.title
    best = (None, None, 0.0)
    try:
        close = datetime.fromisoformat(m.close_time.replace("Z", "+00:00"))
    except ValueError:
        close = None
    for ev in events:
        if close:
            start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            # Kalshi game markets close at/after the game (sometimes days after,
            # to allow for postponements). Same game => close within that window.
            hours = (close - start).total_seconds() / 3600
            if hours < -max_hours or hours > 24 * 15:
                continue
        teams = [ev["home_team"], ev["away_team"]]
        # Both teams should appear in the Kalshi title
        title_fit = min(max(team_score(t, m.title), team_score(t.split()[-1], m.title)) for t in teams)
        for t in teams:
            s = max(team_score(yes_team, t), team_score(yes_team, t.split()[-1]))
            score = 0.6 * s + 0.4 * title_fit
            if score > best[2]:
                best = (ev, t, score)
    return best


def recommend(cfg: dict) -> tuple[list[Rec], list[str]]:
    sp = cfg["sports"]
    key = os.environ.get("ODDS_API_KEY") or sp.get("odds_api_key")
    log: list[str] = []
    if not key:
        return [], ["[sports] skipped: no Odds API key (free at the-odds-api.com)"]

    recs: list[Rec] = []
    for league in sp["leagues"]:
        try:
            events = get_json(f"{ODDS_BASE}/{league['odds_key']}/odds", params={
                "apiKey": key, "regions": "us,eu", "markets": "h2h", "oddsFormat": "decimal",
            })
        except Exception as e:
            log.append(f"[sports] {league['name']}: odds unavailable ({e})")
            continue
        try:
            markets = get_markets(series_ticker=league["kalshi_series"])
        except Exception as e:
            log.append(f"[sports] {league['name']}: Kalshi markets unavailable ({e})")
            continue
        log.append(f"[sports] {league['name']}: {len(events)} games with odds, "
                   f"{len(markets)} Kalshi markets")

        now = datetime.now(timezone.utc)
        unmatched = 0
        for m in markets:
            ev, team, score = match_market(m, events)
            if ev is None or score < sp["min_match_score"]:
                unmatched += 1
                continue
            start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if start <= now and not sp.get("include_live", False):
                continue  # in-game prices move faster than this tool refreshes
            fair, src = fair_probs(ev, sp["sharp_books"], sp["min_books"])
            if team not in fair:
                continue
            ok, _ = passes_liquidity(m, cfg)
            if not ok:
                continue
            p = fair[team]
            reason = (f"{ev['away_team']} @ {ev['home_team']}, {start:%a %b %d %H:%M} UTC. "
                      f"Fair P({team} wins) = {p:.1%} from {src}. Match score {score:.2f}.")
            conf = "high" if "pinnacle" in src else "medium"
            warn = [] if score >= 0.85 else [f"check team match (score {score:.2f})"]
            rec = evaluate(m, p, cfg, "sports", reason, conf, warnings=warn)
            if rec:
                try:
                    rec = evaluate(m, p, cfg, "sports", reason, conf,
                                   orderbook=get_orderbook(m.ticker), warnings=warn)
                except Exception:
                    rec.warnings.append("order book not checked")
                if rec:
                    recs.append(rec)
        if unmatched:
            log.append(f"[sports] {league['name']}: {unmatched} Kalshi markets had no odds match")
    return recs, log
