"""Daily high-temperature markets (e.g. KXHIGHNY).

Model
-----
1. Forecast high (mu) = weighted blend of the official NWS point forecast and
   several global models via Open-Meteo (GFS, ECMWF, ICON).
2. Uncertainty (sigma) = baseline error for the lead time (config), widened when
   the models disagree with each other.
3. For same-day markets, the highest temperature already observed at the
   settlement station is a hard floor: the high can't come in below it.
4. P(official high == k degF) for every integer k, then sum over each bracket.

Kalshi settles on the NWS Daily Climate Report (CLI) for the station. The
climate "day" runs midnight-to-midnight LOCAL STANDARD TIME, so during
daylight saving time it's 1am-1am on the clock. That's handled below.
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .edge import Rec, evaluate, passes_liquidity
from .http import get_json
from .kalshi import Market, event_date, get_markets, get_orderbook


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def integer_distribution(mu: float, sigma: float, floor_obs: int | None = None,
                         lo: int = -40, hi: int = 130) -> dict[int, float]:
    """P(rounded high == k), optionally truncated below an observed max."""
    probs: dict[int, float] = {}
    for k in range(lo, hi + 1):
        p = norm_cdf((k + 0.5 - mu) / sigma) - norm_cdf((k - 0.5 - mu) / sigma)
        if floor_obs is not None and k < floor_obs:
            p = 0.0
        if p > 1e-9:
            probs[k] = p
    total = sum(probs.values())
    if total <= 0:  # observation far above forecast: all mass at the observed value
        return {floor_obs: 1.0} if floor_obs is not None else {round(mu): 1.0}
    return {k: v / total for k, v in probs.items()}


def bracket_prob(dist: dict[int, float], market: Market) -> float | None:
    """P(YES) for one Kalshi bracket market."""
    st = (market.strike_type or "").lower()
    lo, hi = market.floor_strike, market.cap_strike

    def ok(k: int) -> bool:
        if st == "between":
            return lo is not None and hi is not None and lo <= k <= hi
        if st == "greater":
            return lo is not None and k > lo
        if st == "greater_or_equal":
            return lo is not None and k >= lo
        if st == "less":
            return hi is not None and k < hi
        if st == "less_or_equal":
            return hi is not None and k <= hi
        return False

    if st not in {"between", "greater", "greater_or_equal", "less", "less_or_equal"}:
        return None
    return sum(p for k, p in dist.items() if ok(k))


# ------------------------------ data sources ------------------------------

def _utcnow() -> datetime:
    """Current time (a function so tests can freeze the clock)."""
    return datetime.now(timezone.utc)


def standard_time_now(tz_name: str, now: datetime | None = None) -> datetime:
    """Current time in the zone's LOCAL STANDARD TIME (no DST shift).

    Kalshi's climate day runs midnight-to-midnight standard time, so during DST
    12-1am on the wall clock still belongs to the previous climate day.
    """
    now = now or _utcnow()
    local = now.astimezone(ZoneInfo(tz_name))
    std_offset = local.utcoffset() - (local.dst() or timedelta(0))
    return now.astimezone(timezone(std_offset))


def climate_today(tz_name: str, now: datetime | None = None) -> date:
    """The climate day currently in progress at a station in `tz_name`."""
    return standard_time_now(tz_name, now).date()


def climate_day_window_utc(d: date, tz_name: str) -> tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    noon = datetime(d.year, d.month, d.day, 12, tzinfo=tz)
    std_offset = noon.utcoffset() - (noon.dst() or timedelta(0))
    start = datetime(d.year, d.month, d.day, tzinfo=timezone(std_offset))
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def nws_forecast_high(lat: float, lon: float, d: date, ua: str | None = None) -> float | None:
    pts = get_json(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", user_agent=ua)
    url = pts["properties"]["forecast"]
    fc = get_json(url, user_agent=ua)
    for period in fc["properties"]["periods"]:
        start = datetime.fromisoformat(period["startTime"])
        if period.get("isDaytime") and start.date() == d:
            t = float(period["temperature"])
            return t if period.get("temperatureUnit", "F") == "F" else t * 9 / 5 + 32
    return None


def open_meteo_highs(lat: float, lon: float, d: date, tz_name: str,
                     models: list[str]) -> dict[str, float]:
    data = get_json("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit", "timezone": tz_name,
        "models": ",".join(models), "forecast_days": 7,
    })
    daily = data.get("daily", {})
    days = daily.get("time", [])
    if d.isoformat() not in days:
        return {}
    i = days.index(d.isoformat())
    out = {}
    for m in models:
        series = daily.get(f"temperature_2m_max_{m}") or (daily.get("temperature_2m_max") if len(models) == 1 else None)
        if series and i < len(series) and series[i] is not None:
            out[m] = float(series[i])
    return out


def observed_max_so_far(station: str, d: date, tz_name: str, ua: str | None = None) -> int | None:
    start, end = climate_day_window_utc(d, tz_name)
    now = _utcnow()
    if now < start:
        return None
    data = get_json(f"https://api.weather.gov/stations/{station}/observations",
                    params={"start": start.isoformat().replace("+00:00", "Z"),
                            "end": min(now, end).isoformat().replace("+00:00", "Z")},
                    user_agent=ua)
    temps = []
    for f in data.get("features", []):
        v = ((f.get("properties") or {}).get("temperature") or {}).get("value")
        if v is not None:
            temps.append(v * 9 / 5 + 32)
    return round(max(temps)) if temps else None


# --------------------------------- model ----------------------------------

def build_forecast(city: dict, d: date, cfg: dict, today: date) -> dict:
    w = cfg["weather"]
    ua = cfg.get("user_agent")
    lead = (d - today).days
    notes, sources = [], {}

    try:
        nws = nws_forecast_high(city["lat"], city["lon"], d, ua)
        if nws is not None:
            sources["NWS"] = nws
    except Exception as e:
        notes.append(f"NWS forecast unavailable ({e.__class__.__name__})")
    try:
        sources.update(open_meteo_highs(city["lat"], city["lon"], d, city["tz"], w["open_meteo_models"]))
    except Exception as e:
        notes.append(f"Open-Meteo unavailable ({e.__class__.__name__})")

    if not sources:
        return {"ok": False, "notes": notes}

    weights = {k: (w["nws_weight"] if k == "NWS" else 1.0) for k in sources}
    mu = sum(sources[k] * weights[k] for k in sources) / sum(weights.values())
    mu += w.get("bias", {}).get(city["series"], 0.0)

    base = w["sigma_by_lead_days"].get(lead, w["sigma_by_lead_days"][max(w["sigma_by_lead_days"])])
    spread = statistics.pstdev(sources.values()) if len(sources) > 1 else 0.0
    sigma = math.sqrt(base ** 2 + (w["disagreement_multiplier"] * spread) ** 2)

    obs = None
    if lead == 0:
        try:
            obs = observed_max_so_far(city["station"], d, city["tz"], ua)
        except Exception as e:
            notes.append(f"obs unavailable ({e.__class__.__name__})")
        # Hour of the climate day in local standard time (11:30pm CST and
        # 12:30am CDT both count as late in the same day).
        if standard_time_now(city["tz"]).hour >= 17:
            sigma = min(sigma, 1.2)   # high is usually locked in by late afternoon
            notes.append("late day: high likely already set")

    return {"ok": True, "mu": mu, "sigma": sigma, "obs": obs, "sources": sources,
            "lead": lead, "model_spread": spread, "notes": notes}


def recommend(cfg: dict) -> tuple[list[Rec], list[str]]:
    recs: list[Rec] = []
    log: list[str] = []
    max_lead = cfg["weather"]["max_lead_days"]

    for city in cfg["weather"]["cities"]:
        today = climate_today(city["tz"])   # per city, not New York's date
        try:
            markets = get_markets(series_ticker=city["series"])
        except Exception as e:
            log.append(f"[weather] {city['series']}: could not load markets ({e})")
            continue
        by_event: dict[str, list[Market]] = {}
        for m in markets:
            by_event.setdefault(m.event_ticker, []).append(m)

        for ev, ms in sorted(by_event.items()):
            d = event_date(ev)
            if d is None or not (0 <= (d - today).days <= max_lead):
                continue
            fc = build_forecast(city, d, cfg, today)
            if not fc["ok"]:
                log.append(f"[weather] {ev}: no forecast ({'; '.join(fc['notes'])})")
                continue
            dist = integer_distribution(fc["mu"], fc["sigma"], fc["obs"])

            # Sanity check: brackets should cover ~100% of outcomes
            probs = {m.ticker: bracket_prob(dist, m) for m in ms}
            covered = sum(p for p in probs.values() if p is not None)
            warn = []
            if abs(covered - 1) > 0.05:
                warn.append(f"bracket probs sum to {covered:.2f} (check strike parsing)")

            src = ", ".join(f"{k} {v:.0f}" for k, v in fc["sources"].items())
            log.append(f"[weather] {ev}: mu={fc['mu']:.1f}F sigma={fc['sigma']:.1f} "
                       f"obs_max={fc['obs']} | {src}")
            conf = "high" if fc["lead"] == 0 and fc["model_spread"] < 1.5 else \
                   "medium" if fc["model_spread"] < 3 else "low"
            if fc["obs"] is not None and fc["obs"] >= fc["mu"]:
                # Already warmer than the forecast high (e.g. warm overnight, falling
                # temps): the daily-high forecast is stale. Don't trust brackets.
                conf = "low"
                warn.append(f"observed {fc['obs']}F already >= forecast high {fc['mu']:.1f}F; "
                            "forecast likely stale (high may already be set)")

            for m in ms:
                p = probs.get(m.ticker)
                if p is None:
                    continue
                ok, why = passes_liquidity(m, cfg)
                if not ok:
                    continue
                reason = (f"Forecast high {fc['mu']:.1f}°F ± {fc['sigma']:.1f} ({src}"
                          f"{'; observed so far ' + str(fc['obs']) + '°F' if fc['obs'] is not None else ''}). "
                          f"Model P(YES)={p:.0%}.")
                rec = evaluate(m, p, cfg, "weather", reason, conf, warnings=warn + fc["notes"])
                if rec:
                    try:  # confirm size against the real order book
                        rec = evaluate(m, p, cfg, "weather", reason, conf,
                                       orderbook=get_orderbook(m.ticker), warnings=warn + fc["notes"])
                    except Exception:
                        rec.warnings.append("order book not checked")
                    if rec:
                        recs.append(rec)
    return recs, log
