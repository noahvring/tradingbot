"""Read-only HTTP layer.

Every network call in this project goes through `get_json`. It only issues
GET requests, only to an allowlist of public data hosts, and never attaches
credentials. There is intentionally no code anywhere in this project that can
place, modify, or cancel an order on Kalshi.
"""
from __future__ import annotations

import time
from urllib.parse import urlparse

import requests

ALLOWED_HOSTS = {
    "api.elections.kalshi.com",      # Kalshi public market data
    "api.weather.gov",               # National Weather Service
    "api.open-meteo.com",            # Open-Meteo multi-model forecasts
    "previous-runs-api.open-meteo.com",       # past forecasts by lead day (calibrate.py)
    "historical-forecast-api.open-meteo.com", # archived forecasts
    "api.the-odds-api.com",          # Sportsbook odds (free key)
    "gamma-api.polymarket.com",      # Polymarket public market data
}

USER_AGENT = "kalshi-scout/1.0 (personal read-only research tool)"

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
_last_call: dict[str, float] = {}
MIN_INTERVAL = 0.12  # seconds between calls to the same host (be polite)


class ReadOnlyViolation(RuntimeError):
    pass


def get_json(url: str, params: dict | None = None, timeout: float = 20.0,
             retries: int = 3, user_agent: str | None = None):
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise ReadOnlyViolation(f"Host not on read-only allowlist: {host}")

    wait = MIN_INTERVAL - (time.time() - _last_call.get(host, 0))
    if wait > 0:
        time.sleep(wait)

    headers = {"User-Agent": user_agent} if user_agent else None
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, timeout=timeout, headers=headers)
            _last_call[host] = time.time()
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                last_err = RuntimeError(f"HTTP {resp.status_code} from {host}")
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:  # network hiccup
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_err}")
