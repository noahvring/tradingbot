# Kalshi Scout

A read-only recommendation engine for Kalshi **weather, sports, and politics** markets.
It tells you what it would buy, at what price, how many, and why. **It cannot place trades.**
There is no login, no API key for Kalshi, and no order code anywhere in the project (a test
checks this). You place every order yourself in the Kalshi app.

---

## 1. Setup (one time, ~5 minutes)

You need Python 3.10+ (Mac: `python3 --version` in Terminal).

```bash
cd ~/Downloads/kalshi_scout          # wherever you unzipped it
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                  # should say "28 passed"
```

**Optional (for sports):** get a free key at <https://the-odds-api.com> (500 requests/month,
one request per league per run). Then either paste it into `config.yaml` → `sports.odds_api_key`,
or run `export ODDS_API_KEY=yourkey` before using the tool.

**Set your money limits** in `config.yaml` → `sizing` (bankroll, max per trade, max total).
The defaults are deliberately small.

---

## 2. Daily use

```bash
source .venv/bin/activate
python scout.py run                  # everything
python scout.py run -c weather       # one category
python scout.py run --top 5          # only the 5 best
python scout.py run -v               # show the model's numbers for every market it looked at
python scout.py run --watch 30       # refresh every 30 min (leave the terminal open)
```

Every run also saves `output/recs_<date>.md` (readable table) and `.csv` (spreadsheet).

### Reading a recommendation

```
#1  [id 14]  WEATHER  |  confidence: high
    Highest temperature in NYC on Nov 5? — 55° to 56°
    Ticker: KXHIGHNY-26NOV05-B55.5
    >>> BUY YES  x12  at $0.20  (limit order, max $0.33)  cost $2.52
    Model 42.1% vs market 20.0%  ->  edge +20.6 pts after fees, EV $+2.62
    Why: Forecast high 55.5°F ± 1.8 (NWS 56, gfs_seamless 55, ...). Model P(YES)=42%.
```

- **BUY YES/NO x12 at $0.20**: the side, the number of contracts, and the current best ask.
- **max $0.33**: the highest price where the trade still clears your minimum edge. Use a
  **limit order** at or below this price. Never use a market order.
- **edge**: the model's probability minus the price minus Kalshi's fee. +20 pts means the model
  thinks you're getting a big discount. Huge edges usually mean the *model* is missing something,
  so read the "Why" line before trusting it.
- **id**: use this to log the trade if you place it (below).
- **! lines** are warnings, such as a thin order book or an uncertain team match.

### After you place a trade
```bash
python scout.py took 14 --price 0.21 --contracts 10
```

### Grading the model (this is the important part)
```bash
python scout.py settle     # pulls results for settled markets (run daily)
python scout.py stats      # hit rate, Brier score, calibration, paper P&L
```
`stats` compares the model's **Brier score** to the **market price's Brier score** on the same
markets. If the model isn't lower (better) than the market after ~50+ settled recs in a category,
it has no real edge there. Stop trading that category or change the model.

---

## 3. How each model works

### Weather (strongest, start here)
Markets like `KXHIGHNY` settle on the official NWS daily high at a specific station (e.g.
Central Park, KNYC).
1. **Forecast high** = blend of the NWS point forecast (weighted 1.5x) and GFS, ECMWF, and ICON
   via Open-Meteo. No keys needed.
2. **Uncertainty** = typical forecast error for that lead time (`sigma_by_lead_days`), widened
   when the models disagree.
3. **Same-day trick:** for today's market it pulls the station's observations so far. The final
   high can't be lower than what's already been recorded, so lower brackets drop to ~0%.
4. It computes the probability of each whole-degree outcome, adds them up for each bracket,
   checks that all brackets sum to ~100%, and compares to prices.
5. It handles the "climate day", which runs midnight to midnight *standard* time (1am–1am during DST).

**Tuning:** after a few weeks, if `stats` shows a city's forecasts run consistently high or low,
add it to `weather.bias` (e.g. `KXHIGHNY: 0.7`). If calibration shows overconfidence
(predicted 70%, actual 50%), raise the `sigma` values.

### Sports (good signal, small edges)
Sportsbooks are very accurate. Instead of trying to beat them, this uses them:
1. Pulls moneylines from many books via The Odds API.
2. Takes **Pinnacle** (the sharpest book) if it's listed, otherwise the median across books.
3. **Devigs** the odds (removes the bookmaker's margin) to get a fair win probability.
4. Matches each Kalshi game market to the right game and team, then flags Kalshi prices that
   differ from fair by more than fees plus `min_edge` (3 points by default).

Edges here are usually small and short-lived. Games already in progress are skipped. Check that
sports contracts are available where you are. That varies by state and is changing in court.

### Politics (you supply the judgment)
There's no free, reliable feed of "fair" political probabilities, so the tool doesn't fake one.
- **Your priors → real recs.** Put your own probability for specific tickers in
  `politics_priors.yaml` (from polling averages, forecast models, your research). The tool
  compares them to live prices and sizes the trade.
- **Polymarket cross-check → research leads only.** It finds similar questions on Polymarket and
  lists big price gaps. These are *not* recommendations: the two sites often word or resolve
  questions differently. Read both sets of rules before acting.

---

### Model vs market blend
Every category's probability is blended with the market's own price before computing edge:
`p_final = w * p_model + (1 - w) * p_market` (market = bid/ask mid). `model_weight` in
`config.yaml` sets `w` per category (weather 0.3, sports 0.5, politics priors 0.5 to start).
`python calibrate.py` prints the best weather weight from past outcomes. `stats` shows the
Brier score of the raw model, the market, and the blend, so you can tune the others.

### Same-day weather
For today's market the high is `max(observed so far, NWS hourly forecast for the rest of the
climate day)`. If it was 54°F at 1am and the hourly forecast cools all day, "54° or below" is
very likely, whatever the daily-high forecast said. With 12+ hours left, the rest-of-day peak
is averaged with the multi-model daily high and keeps its wider uncertainty.

## 4. Sizing logic
- **Quarter-Kelly**: bet a quarter of what the Kelly formula says. Full Kelly assumes your model
  is perfect. It isn't.
- Capped by `max_per_trade`, by `max_total_exposure` per run, and by how many contracts are
  actually offered in the order book at or below the limit price.
- Filters skip: prices under 4¢ or over 94¢, bid/ask spreads over 8¢, and low-volume markets.

## 5. Things to verify once on kalshi.com
- **Series tickers** in `config.yaml` (open a market; the series is in the URL, e.g.
  `kalshi.com/markets/kxhighny`). Kalshi adds and renames series, so if a city or league shows
  0 markets, the ticker is the first thing to check.
- **Settlement station** for each weather city (it's in the market rules).
- **Fee rate** (`fees`) for each category on Kalshi's fee schedule.

## 6. Extending it with Claude
Upload the folder (or paste a file) and ask, for example:
- "Add Kalshi's daily low-temperature markets using the same weather model."
- "Add NFL spread/total markets using devigged spreads and totals from The Odds API."
- "Add an Elo model for college basketball to blend with the sportsbook probability."
- "Here's my `python scout.py stats` output. What should I tune?"

## 7. Honest caveats
- This is a research tool, not financial advice. Most simple models don't beat liquid markets.
- Paper trade (run it and `settle`/`stats` without placing orders) for 2–4 weeks first.
- Prediction-market gains are taxable. Keep the `scout.db` log.
