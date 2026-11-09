"""Rebuild every number of analytics/report.html from the raw data.

    python reproduce.py              # all steps
    python reproduce.py --skip-train # reuse models/*.parquet
    python reproduce.py --jobs 4     # fewer parallel backtests (each needs ~1.5 GB RAM)

Steps: 1) train the 1 s signal, 2) run the backtests behind the report, 3) train the 30 s / 60 min signals
(their report reads the fills of step 2) and backtest the 30 s signal, 4) build charts and summary.json.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OOS = "data.days=2026-03-20,2026-03-21"
FAIR_VALUE = ["strategy.name=fair_value", "strategy.signal_path=models/fair_value_signal.parquet"]
MID = ["strategy.name=mid"]
SPREADS = [0.15, 0.25, 0.35, 0.45, 0.65]


def backtest(out: str, sets: list[str]) -> list[str]:
    return ["run_backtest.py", *(a for s in [f"output.dir=output/{out}", "data.levels=12", *sets] for a in ("--set", s))]


def grid(out: str, sets: list[str], **values: list) -> list[list[str]]:
    """One command per grid point, so every run can go to its own process."""
    return [backtest(out, sets + [f"strategy.{k}={v}" for k, v in zip(values, point)]) for point in product(*values.values())]


def backtests() -> list[list[str]]:
    oos_limit2 = [OOS, "strategy.max_position=2"]
    latencies = {"latency_grid": [(1, 10, 10), (1, 50, 10)], "latency_hl": [(100, 150, 50), (180, 450, 450)]}
    return [
        # out of sample: ML fair value vs mid (the main result)
        *grid("fv_wide", oos_limit2 + FAIR_VALUE, half_spread=SPREADS, inventory_skew=[0.05, 0.1]),
        *grid("fv_wide", oos_limit2 + MID, half_spread=SPREADS),
        # latency sensitivity at half-spread 0.45
        *(backtest(out, oos_limit2 + strategy + ["strategy.half_spread=0.45", f"latency.market_data_ms={md}",
                                                 f"latency.order_ms={order}", f"latency.ack_ms={ack}"])
          for out, points in latencies.items() for md, order, ack in points for strategy in (FAIR_VALUE + ["strategy.inventory_skew=0.05"], MID)),
        # depth context and round trips: mid over 3 days
        *grid("spread_sweep", [], half_spread=[0.05, *SPREADS, 0.95]),
        *grid("spread_sweep_lowinv", ["strategy.max_position=2"], half_spread=[0.45, 0.65]),
    ]


def run(args: list[str]) -> None:
    started = time.perf_counter()
    done = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    label = " ".join(a for a in args if "=" in a and not a.startswith(("output", "data.levels")))
    if done.returncode:
        sys.exit(f"FAILED: {' '.join(args)}\n{done.stderr[-2000:]}")
    print(f"  {time.perf_counter() - started:5.0f} s  {args[0]} {label}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-train", action="store_true", help="reuse the signals in models/")
    p.add_argument("--jobs", type=int, default=6, help="parallel backtests")
    args = p.parse_args()

    if not (ROOT / "data/orderbook/2026-03-19.parquet").exists():
        sys.exit("data/ is missing: put the task parquet files in data/{orderbook,trades,fundings}/")
    if not args.skip_train:
        print("1/4 training the 1 s signal")
        run(["train_fair_value.py"])  # train 03-19 -> models/fair_value_signal.parquet
    print("2/4 backtests")
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(run, backtests()))
    print("3/4 30 s and 60 min signals")
    if not args.skip_train:
        run(["train_horizon_models.py"])  # -> models/fv_h30000.parquet, models/fv_h3600000.parquet
    run(backtest("fv_h30s", [OOS, "strategy.max_position=2", "strategy.name=fair_value", "strategy.inventory_skew=0.05",
                             "strategy.signal_path=models/fv_h30000.parquet", "strategy.half_spread=0.45"]))
    print("4/4 charts")
    run(["analytics/build_charts.py"])
    print("done: analytics/report.html, analytics/charts/, analytics/data/summary.json")


if __name__ == "__main__":
    main()
