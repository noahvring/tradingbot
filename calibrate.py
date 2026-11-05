#!/usr/bin/env python3
"""Fit the weather model to real outcomes. Read-only; never edits config.

  python calibrate.py                    # all cities, last 75 days
  python calibrate.py --days 90 --city KXHIGHNY
  python calibrate.py --no-prices        # skip the market comparison (much faster)

For each city it:
  1. pulls settled Kalshi markets and works out each day's official high from
     the bracket that resolved YES (2-degree brackets -> midpoint, +-0.5F);
  2. pulls what GFS / ECMWF / ICON forecast for that day 0, 1 and 2 days ahead
     (Open-Meteo previous-runs API), as a daily max over the climate day in
     local standard time;
  3. prints bias (actual - forecast) and error spread (sigma) by lead day;
  4. scores the model vs Kalshi's own price (Brier) on the lead-1 markets,
     using the price at 10am local standard time the day before.

The NWS forecast isn't archived, so the forecast here is the equal-weight mean
of the global models, without the NWS blend the live model uses.
"""
from __future__ import annotations

import argparse
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from kscout.http import get_json
from kscout.kalshi import BASE, get_markets, event_date
from kscout.weather import bracket_prob, integer_distribution, standard_time_now

ROOT = Path(__file__).resolve().parent
PREV_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
LEAD_VARS = {0: "temperature_2m", 1: "temperature_2m_previous_day1", 2: "temperature_2m_previous_day2"}
QUANT_VAR = 0.25   # variance added by only knowing the high to +-0.5F (2-degree bracket)


def std_offset(tz_name: str, d: date) -> timedelta:
    noon = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
    return standard_time_now(tz_name, noon).utcoffset()


# ------------------------------- outcomes --------------------------------

def actual_highs(markets) -> dict[date, tuple[float, list]]:
    """{date: (official high estimate, markets of that day)} from settled brackets.

    Days where a tail bracket ("<55", ">85") won are skipped: the high is censored.
    """
    by_ev: dict[str, list] = {}
    for m in markets:
        by_ev.setdefault(m.event_ticker, []).append(m)
    out = {}
    for ev, ms in by_ev.items():
        d = event_date(ev)
        win = [m for m in ms if m.result == "yes"]
        if d is None or len(win) != 1:
            continue
        w = win[0]
        if (w.strike_type or "") == "between" and w.floor_strike is not None and w.cap_strike is not None:
            out[d] = ((w.floor_strike + w.cap_strike) / 2, ms)
    return out


# ------------------------------- forecasts -------------------------------

def daily_max_by_lead(hourly: dict, models: list[str], tz_name: str) -> dict[int, dict[date, float]]:
    """Mean-of-models daily max per lead, over each climate day (local standard time).

    `hourly` is an Open-Meteo hourly block in UTC (timezone=GMT).
    """
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly["time"]]
    result: dict[int, dict[date, float]] = {}
    for lead, var in LEAD_VARS.items():
        per_model: dict[date, list[float]] = {}
        for mdl in models:
            series = hourly.get(f"{var}_{mdl}") or (hourly.get(var) if len(models) == 1 else None)
            if not series:
                continue
            day_max: dict[date, list[float]] = {}
            for t, v in zip(times, series):
                if v is None:
                    continue
                d = (t + std_offset(tz_name, t.date())).date()
                day_max.setdefault(d, []).append(v)
            for d, vals in day_max.items():
                if len(vals) >= 20:            # need (nearly) the whole day
                    per_model.setdefault(d, []).append(max(vals))
        result[lead] = {d: statistics.fmean(v) for d, v in per_model.items()}
    return result


def fetch_forecasts(city: dict, start: date, end: date, models: list[str]) -> dict:
    data = get_json(PREV_RUNS, params={
        "latitude": city["lat"], "longitude": city["lon"],
        "hourly": ",".join(LEAD_VARS.values()), "models": ",".join(models),
        "temperature_unit": "fahrenheit", "timezone": "GMT",
        "start_date": (start - timedelta(days=1)).isoformat(),
        "end_date": (end + timedelta(days=1)).isoformat(),
    }, timeout=60)
    if data.get("error"):
        raise RuntimeError(data.get("reason"))
    return daily_max_by_lead(data["hourly"], models, city["tz"])


# ------------------------------ market prices -----------------------------

def price_at(series: str, ticker: str, when: datetime) -> float | None:
    """YES mid (or last trade) from the last hourly candle ending at/before `when`."""
    end = int(when.timestamp())
    data = get_json(f"{BASE}/series/{series}/markets/{ticker}/candlesticks",
                    params={"start_ts": end - 3 * 86400, "end_ts": end, "period_interval": 60})
    for c in reversed(data.get("candlesticks", [])):
        if c.get("end_period_ts", 0) > end:
            continue
        bid = float((c.get("yes_bid") or {}).get("close_dollars") or 0)
        ask = float((c.get("yes_ask") or {}).get("close_dollars") or 0)
        if 0 < bid < ask < 1:
            return (bid + ask) / 2
        last = (c.get("price") or {}).get("close_dollars")
        if last not in (None, ""):
            return float(last)
    return None


# --------------------------------- main ----------------------------------

def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=75)
    ap.add_argument("--city", action="append", help="series ticker, e.g. KXHIGHNY")
    ap.add_argument("--no-prices", action="store_true", help="skip the Brier comparison vs market")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / a.config).read_text())
    w = cfg["weather"]
    models = w["open_meteo_models"]
    cur_sigma = w["sigma_by_lead_days"]
    today = date.today()
    since = today - timedelta(days=a.days)
    suggest_bias, suggest_sigma = {}, {}
    all_model, all_mkt = [], []

    for city in w["cities"]:
        s = city["series"]
        if a.city and s not in a.city:
            continue
        print(f"\n=== {s} ({city['station']}) ===")
        try:
            ms = [m for m in get_markets(series_ticker=s, status="settled", max_pages=10)
                  if (event_date(m.event_ticker) or date.min) >= since]
        except Exception as e:
            print(f"  Kalshi settled markets unavailable: {e}")
            continue
        highs = actual_highs(ms)
        n_days = len({m.event_ticker for m in ms})
        print(f"  {n_days} settled days, {len(highs)} with an exact (non-tail) high")
        if not highs:
            continue
        try:
            fc = fetch_forecasts(city, min(highs), max(highs), models)
        except Exception as e:
            print(f"  forecasts unavailable: {e}")
            continue

        fitted = {}
        for lead in (0, 1, 2):
            errs = [highs[d][0] - fc[lead][d] for d in highs if d in fc.get(lead, {})]
            if len(errs) < 10:
                print(f"  lead {lead}: only {len(errs)} days, skipped")
                continue
            bias = statistics.fmean(errs)
            sd = math.sqrt(max(statistics.pvariance(errs) - QUANT_VAR, 0.25))
            mae = statistics.fmean(abs(e) for e in errs)
            fitted[lead] = (bias, sd)
            print(f"  lead {lead}: n={len(errs):3d}  bias {bias:+.2f}F (actual - forecast)  "
                  f"sigma {sd:.2f}F  MAE {mae:.2f}F   [config sigma {cur_sigma.get(lead)}]")
        if 1 in fitted:
            suggest_bias[s] = round(fitted[1][0], 1)
        for lead, (_, sd) in fitted.items():
            suggest_sigma.setdefault(lead, []).append(sd)

        if a.no_prices or 1 not in fc:
            continue
        # Lead-1 Brier: model (current config) vs Kalshi mid at 10am standard time the day before
        mp, kp = [], []
        for d, (_, dms) in sorted(highs.items()):
            mu = fc[1].get(d)
            if mu is None:
                continue
            mu += w.get("bias", {}).get(s, 0.0)
            dist = integer_distribution(mu, cur_sigma[1])
            prev = d - timedelta(days=1)
            when = datetime(prev.year, prev.month, prev.day, 10, tzinfo=timezone(std_offset(city["tz"], prev)))
            for m in dms:
                p = bracket_prob(dist, m)
                if p is None:
                    continue
                try:
                    q = price_at(s, m.ticker, when)
                except Exception:
                    q = None
                if q is None:
                    continue
                y = 1.0 if m.result == "yes" else 0.0
                mp.append((p, y)); kp.append((q, y))
        all_model += mp; all_mkt += kp
        if mp:
            print(f"  Brier, lead 1, {len(mp)} brackets:  model {brier(mp):.4f}   market {brier(kp):.4f}   "
                  f"-> {'MODEL BETTER' if brier(mp) < brier(kp) else 'market better'}")

    print("\n=== Suggested config (review before copying; not applied) ===")
    print("weather:\n  bias:")
    for s, b in suggest_bias.items():
        print(f"    {s}: {b}")
    print("  sigma_by_lead_days:   # median across cities")
    for lead in sorted(suggest_sigma):
        print(f"    {lead}: {statistics.median(suggest_sigma[lead]):.1f}")
    print("Note: lead-0 here is the model's same-day run without station observations;"
          " the live tool also uses observed temps, so its real same-day error is smaller.")
    if all_model:
        bm, bk = brier(all_model), brier(all_mkt)
        print(f"\n=== ALL CITIES, lead 1: model Brier {bm:.4f} vs market {bk:.4f} "
              f"over {len(all_model)} brackets ===")
        print("Model beats the market." if bm < bk else
              "The market beat the model. Don't trade weather on this model as-is.")


if __name__ == "__main__":
    main()
