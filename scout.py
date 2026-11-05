#!/usr/bin/env python3
"""Kalshi Scout — recommends trades, never places them.

  python scout.py run                     # all categories, once
  python scout.py run -c weather          # just weather (also: sports, politics)
  python scout.py run --watch 30          # re-run every 30 minutes
  python scout.py took 12 --price 0.54 --contracts 10   # log a trade you placed
  python scout.py settle                  # pull results for past recs
  python scout.py stats                   # calibration + paper P&L
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from kscout import politics, report, sports, tracker, weather

ROOT = Path(__file__).resolve().parent


def load_cfg(path: str) -> dict:
    cfg = yaml.safe_load((ROOT / path).read_text())
    cfg["politics"]["priors_file"] = str(ROOT / cfg["politics"]["priors_file"])
    return cfg


def run_once(cfg: dict, cats: list[str], verbose: bool, top: int | None) -> None:
    recs, log, leads = [], [], []
    if "weather" in cats:
        r, l = weather.recommend(cfg); recs += r; log += l
    if "sports" in cats:
        r, l = sports.recommend(cfg); recs += r; log += l
    if "politics" in cats:
        r, l, leads = politics.recommend(cfg); recs += r; log += l

    # Rank by expected profit, cap total exposure
    recs.sort(key=lambda r: -r.expected_profit)
    budget = cfg["sizing"]["max_total_exposure"]
    kept, spent = [], 0.0
    for r in recs:
        if spent + r.cost > budget:
            r.warnings.append("dropped: over max_total_exposure")
            continue
        kept.append(r); spent += r.cost
    if top:
        kept = kept[:top]

    con = tracker.connect(str(ROOT / cfg["db_file"]))
    ids = tracker.log_recs(con, kept)
    print(f"\nKalshi Scout — {datetime.now():%a %b %d %I:%M %p}  (categories: {', '.join(cats)})")
    report.print_recs(kept, ids, leads, log, verbose)
    csv_p, md_p = report.save(kept, ids, leads, str(ROOT / cfg["output_dir"]))
    print(f"Saved: {md_p.name} and {csv_p.name} in {cfg['output_dir']}/")
    print("Reminder: this tool only recommends. You decide and place every order yourself.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="get recommendations")
    r.add_argument("-c", "--category", action="append", choices=["weather", "sports", "politics"])
    r.add_argument("--watch", type=float, metavar="MIN", help="repeat every MIN minutes")
    r.add_argument("--top", type=int, help="show only the best N")
    r.add_argument("-v", "--verbose", action="store_true", help="show model details for every market")

    t = sub.add_parser("took", help="log a trade you actually placed")
    t.add_argument("rec_id", type=int)
    t.add_argument("--price", type=float, required=True, help="fill price in dollars, e.g. 0.54")
    t.add_argument("--contracts", type=int, required=True)
    t.add_argument("--note", default="")

    sub.add_parser("settle", help="fetch results for past recommendations")
    sub.add_parser("stats", help="calibration and P&L report")

    a = ap.parse_args()
    cfg = load_cfg(a.config)
    con = tracker.connect(str(ROOT / cfg["db_file"]))

    if a.cmd == "run":
        cats = a.category or cfg["categories"]
        if not a.watch:
            run_once(cfg, cats, a.verbose, a.top)
            return
        print(f"Watching every {a.watch:g} min. Ctrl+C to stop.")
        while True:
            try:
                run_once(cfg, cats, a.verbose, a.top)
                time.sleep(a.watch * 60)
            except KeyboardInterrupt:
                sys.exit(0)
    elif a.cmd == "took":
        tracker.mark_taken(con, a.rec_id, a.price, a.contracts, a.note)
        print(f"Logged: rec {a.rec_id}, {a.contracts} @ ${a.price:.2f}")
    elif a.cmd == "settle":
        msgs = tracker.settle(con)
        print("\n".join(msgs) if msgs else "Nothing new settled yet.")
    elif a.cmd == "stats":
        print(tracker.stats(con))


if __name__ == "__main__":
    main()
