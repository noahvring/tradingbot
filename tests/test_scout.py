"""Offline tests: no network needed. Run with `python -m pytest -q`."""
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from kscout import edge, kalshi, politics, sports, tracker, weather
from kscout.http import ReadOnlyViolation, get_json

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


# ----------------------------- safety ---------------------------------

def test_no_trading_code_anywhere():
    """The project must not contain any way to place/cancel orders or authenticate."""
    banned = [r"\.post\(", r"\.put\(", r"\.delete\(", r"portfolio/orders", r"KALSHI-ACCESS",
              r"private_key", r"/login"]
    for f in (ROOT / "kscout").glob("*.py"):
        text = f.read_text()
        for pat in banned:
            assert not re.search(pat, text), f"{pat} found in {f.name}"


def test_disallowed_host_blocked():
    with pytest.raises(ReadOnlyViolation):
        get_json("https://trading-api.kalshi.com/trade-api/v2/portfolio/orders")


# ----------------------------- kalshi parsing -------------------------

RAW_BRACKET = {
    "ticker": "KXHIGHNY-26NOV05-B55.5", "event_ticker": "KXHIGHNY-26NOV05",
    "title": "Highest temperature in NYC on Nov 5, 2026?", "yes_sub_title": "55° to 56°",
    "yes_bid": 30, "yes_ask": 34, "no_bid": 66, "no_ask": 70, "last_price": 32,
    "volume": 4200, "open_interest": 900, "close_time": "2026-11-06T04:59:00Z",
    "strike_type": "between", "floor_strike": 55, "cap_strike": 56, "status": "active",
}


def test_market_parse_cents_and_dollars():
    m = kalshi.Market.from_api(RAW_BRACKET)
    assert m.yes_ask == pytest.approx(0.34) and m.no_ask == pytest.approx(0.70)
    d = dict(RAW_BRACKET, yes_ask=None, yes_ask_dollars="0.3500")
    assert kalshi.Market.from_api(d).yes_ask == pytest.approx(0.35)


def test_missing_side_derived_and_empty_side_ignored():
    d = {k: v for k, v in RAW_BRACKET.items() if k not in ("no_ask", "no_bid")}
    m = kalshi.Market.from_api(d)
    assert m.no_ask == pytest.approx(0.70)
    m2 = kalshi.Market.from_api(dict(RAW_BRACKET, yes_ask=100))
    assert m2.yes_ask is None


def test_event_date():
    assert kalshi.event_date("KXHIGHNY-26NOV05") == date(2026, 11, 5)
    assert kalshi.event_date("KXHIGHCHI-27JAN31") == date(2027, 1, 31)
    assert kalshi.event_date("NODATE") is None


def test_orderbook_depth():
    raw = {"orderbook": {"yes": [[30, 100], [29, 50]], "no": [[66, 40], [65, 25], [60, 500]]}}
    ob = kalshi.parse_orderbook(raw)
    # Buy YES: NO bids 66,65,60 => YES asks .34,.35,.40
    qty, avg = kalshi.depth_to_buy(ob, "yes", 0.35)
    assert qty == 65 and avg == pytest.approx((40 * .34 + 25 * .35) / 65)
    raw2 = {"orderbook_fp": {"yes_dollars": [["0.30", 100]], "no_dollars": [["0.66", 40]]}}
    assert kalshi.depth_to_buy(kalshi.parse_orderbook(raw2), "yes", 0.34)[0] == 40


# ----------------------------- fees & sizing --------------------------

def test_fee():
    # 0.07 * 10 * 0.5 * 0.5 = 0.175 -> rounds up to $0.18 => $0.018/contract
    assert edge.taker_fee_per_contract(0.50, 10) == pytest.approx(0.018)
    assert edge.taker_fee_per_contract(0.50, 1) == pytest.approx(0.02)


def test_kelly():
    assert edge.kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert edge.kelly_fraction(0.4, 0.5) == 0.0


def test_evaluate_picks_yes_when_underpriced():
    m = kalshi.Market.from_api(RAW_BRACKET)   # YES ask 0.34
    r = edge.evaluate(m, 0.50, CFG, "weather", "test")
    assert r and r.side == "YES" and r.edge > 0.1 and r.contracts >= 1
    assert r.cost <= CFG["sizing"]["max_per_trade"] + 0.01
    assert r.limit_price >= r.price


def test_evaluate_picks_no_when_overpriced():
    m = kalshi.Market.from_api(RAW_BRACKET)   # NO ask 0.70 -> need P(NO) > ~0.77
    r = edge.evaluate(m, 0.10, CFG, "weather", "test")
    assert r and r.side == "NO"


def test_evaluate_none_when_fair():
    m = kalshi.Market.from_api(RAW_BRACKET)
    assert edge.evaluate(m, 0.32, CFG, "weather", "test") is None


def test_orderbook_caps_size():
    m = kalshi.Market.from_api(RAW_BRACKET)
    ob = kalshi.parse_orderbook({"orderbook": {"yes": [], "no": [[66, 3]]}})
    r = edge.evaluate(m, 0.55, CFG, "weather", "t", orderbook=ob)
    assert r.contracts == 3 and any("only 3" in w for w in r.warnings)


# ----------------------------- weather --------------------------------

def test_distribution_sums_to_one_and_truncates():
    d = weather.integer_distribution(55.0, 2.0)
    assert sum(d.values()) == pytest.approx(1.0)
    assert max(d, key=d.get) == 55
    t = weather.integer_distribution(55.0, 2.0, floor_obs=57)
    assert min(t) == 57 and sum(t.values()) == pytest.approx(1.0)
    # observed way above forecast: all mass at observed
    assert weather.integer_distribution(50, 1.0, floor_obs=80) == {80: 1.0}


def _mk(st, lo=None, hi=None):
    return kalshi.Market.from_api(dict(RAW_BRACKET, strike_type=st, floor_strike=lo, cap_strike=hi))


def test_brackets_cover_everything():
    dist = weather.integer_distribution(55.3, 2.5)
    ms = [_mk("less", hi=51)] + [_mk("between", lo, lo + 1) for lo in range(51, 60, 2)] + [_mk("greater", lo=60)]
    total = sum(weather.bracket_prob(dist, m) for m in ms)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_climate_day_uses_standard_time():
    # July (DST): NY climate day starts 05:00 UTC (= midnight EST), not 04:00
    s, e = weather.climate_day_window_utc(date(2026, 7, 4), "America/New_York")
    assert s == datetime(2026, 7, 4, 5, tzinfo=timezone.utc) and e - s == timedelta(days=1)


def test_weather_recommend_offline(monkeypatch):
    today = datetime.now(weather.ZoneInfo("America/New_York")).date()
    ev = f"KXHIGHNY-{today:%y}{today.strftime('%b').upper()}{today:%d}"
    def mkm(t, st, lo, hi, ask):
        return kalshi.Market.from_api(dict(RAW_BRACKET, ticker=f"{ev}-{t}", event_ticker=ev,
            strike_type=st, floor_strike=lo, cap_strike=hi, yes_ask=ask, yes_bid=ask - 3,
            no_ask=100 - ask + 3, no_bid=100 - ask))
    markets = [mkm("T51", "less", None, 51, 5), mkm("B51", "between", 51, 52, 10),
               mkm("B53", "between", 53, 54, 20), mkm("B55", "between", 55, 56, 20),  # model ~40%
               mkm("B57", "between", 57, 58, 30), mkm("T58", "greater", 58, None, 15)]
    cfg = dict(CFG, weather=dict(CFG["weather"], cities=[CFG["weather"]["cities"][0]]))
    monkeypatch.setattr(weather, "get_markets", lambda **k: markets)
    monkeypatch.setattr(weather, "build_forecast", lambda *a: {
        "ok": True, "mu": 55.5, "sigma": 1.8, "obs": None, "sources": {"NWS": 56, "gfs_seamless": 55},
        "lead": 0, "model_spread": 0.5, "notes": []})
    deep = {"orderbook": {"yes": [[x, 999] for x in range(1, 99)], "no": [[x, 999] for x in range(1, 99)]}}
    monkeypatch.setattr(weather, "get_orderbook", lambda t: kalshi.parse_orderbook(deep))
    recs, log = weather.recommend(cfg)
    tickers = {r.ticker.split("-")[-1]: r.side for r in recs}
    assert tickers.get("B55") == "YES"       # 40%+ model vs 20c
    assert "B57" not in tickers              # model 24% vs NO at 73c: edge too thin after fees
    assert tickers.get("T58") == "NO"        # model ~2% vs 15c YES -> NO is underpriced
    assert any("mu=55.5" in l for l in log)


# ----------------------------- sports ---------------------------------

ODDS_EVENT = {
    "home_team": "Dallas Cowboys", "away_team": "Washington Commanders",
    "commence_time": "2026-11-08T21:25:00Z",
    "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Dallas Cowboys", "price": 1.80}, {"name": "Washington Commanders", "price": 2.10}]}]},
        {"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Dallas Cowboys", "price": 1.75}, {"name": "Washington Commanders", "price": 2.05}]}]},
    ],
}


def test_devig():
    p = sports.devig([1.80, 2.10])
    assert sum(p) == pytest.approx(1.0) and p[0] == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 2.1))


def test_fair_probs_prefers_sharp():
    fair, src = sports.fair_probs(ODDS_EVENT, ["pinnacle"], 4)
    assert "pinnacle" in src and fair["Dallas Cowboys"] == pytest.approx(0.5385, abs=1e-3)


def test_match_market():
    m = kalshi.Market.from_api({"ticker": "KXNFLGAME-26NOV08WASDAL-WAS", "event_ticker": "KXNFLGAME-26NOV08WASDAL",
        "title": "Washington at Dallas Winner?", "yes_sub_title": "Washington",
        "close_time": "2026-11-22T21:25:00Z", "yes_ask": 40, "yes_bid": 38, "volume": 5000})
    ev, team, score = sports.match_market(m, [ODDS_EVENT])
    assert team == "Washington Commanders" and score >= 0.7
    # A different game should not match well
    other = dict(ODDS_EVENT, home_team="Green Bay Packers", away_team="Chicago Bears")
    ev2, team2, score2 = sports.match_market(m, [other])
    assert score2 < 0.7


# ----------------------------- politics -------------------------------

def test_similarity():
    assert politics.similarity("Who will win the 2028 Democratic nomination? Newsom",
                               "Will Gavin Newsom win the 2028 Democratic nomination?") > 0.45
    assert politics.similarity("Fed cuts rates in December", "Newsom nomination") < 0.1


def test_priors_file_loads(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("- {ticker: ABC, prob: 0.6}\n- {ticker: DEF}\n")
    assert politics.load_priors(str(f)) == [{"ticker": "ABC", "prob": 0.6}]
    assert politics.load_priors(str(ROOT / "politics_priors.yaml")) == []


# ----------------------------- tracker --------------------------------

def test_tracker_roundtrip(tmp_path, monkeypatch):
    con = tracker.connect(str(tmp_path / "t.db"))
    m = kalshi.Market.from_api(RAW_BRACKET)
    r = edge.evaluate(m, 0.50, CFG, "weather", "test")
    ids = tracker.log_recs(con, [r])
    assert tracker.log_recs(con, [r]) == ids          # deduped
    tracker.mark_taken(con, ids[0], 0.34, 5)
    monkeypatch.setattr(tracker, "get_market", lambda t: kalshi.Market.from_api(dict(RAW_BRACKET, result="yes")))
    assert "settled YES" in tracker.settle(con)[0]
    out = tracker.stats(con)
    assert "Hit rate" in out and "1/1" in out and "real P&L" in out
