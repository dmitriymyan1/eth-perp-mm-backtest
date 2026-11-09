"""Run backtests from a config file (all settings: see configs/default.toml and mm/config.py).

Usage:
  python run_backtest.py                                        # configs/default.toml
  python run_backtest.py --config my.toml
  python run_backtest.py --set fills.cross_rule=profitable --set fills.pickoff_fair_ms=200
  python run_backtest.py --sweep fills.cross_rule=never,confirmed,always
  python run_backtest.py --sweep fills.pickoff_fee_bps=1,2.4,4.5 --sweep fills.pickoff_fair_ms=0,200

Each run writes <output.dir>/<run name>_{pnl.png,metrics.json,fills.parquet}. The run name ends
with a hash of the config, so runs with different settings never overwrite each other;
metrics.json holds the full config of that run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mm.config import BacktestConfig
from mm.backtester.engine import BacktestResult
from mm.facade import run_backtest
from mm.analytics.metrics import compute_metrics, daily_pnl
from mm.analytics.plotting import plot_pnl

DEFAULT_CONFIG = Path("configs/default.toml")


def parse_value(text: str) -> Any:
    """CLI text -> None / bool / int / float / str."""
    lowered = text.strip().lower()
    if lowered in ("none", "null"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def parse_assignment(item: str) -> tuple[str, str]:
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected section.key=value, got {item!r}")
    return key.strip(), value


def expand(base: BacktestConfig, sweeps: list[tuple[str, str]]) -> list[BacktestConfig]:
    """One config per point of the sweep grid (cartesian product)."""
    keys = [key for key, _ in sweeps]
    grids = [[parse_value(v) for v in values.split(",")] for _, values in sweeps]
    return [base.with_overrides(dict(zip(keys, point))) for point in itertools.product(*grids)]


def report(cfg: BacktestConfig, result: BacktestResult) -> dict:
    name = cfg.run_name()
    out = Path(cfg.output.dir)
    metrics = compute_metrics(result)
    metrics["daily_pnl"] = {str(d.date()): v for d, v in daily_pnl(result.equity["equity"]).items()}
    metrics["config"] = cfg.to_dict()

    out.mkdir(parents=True, exist_ok=True)
    lat = cfg.latency
    plot_pnl(result, out / f"{name}_pnl.png",
             title=(f"MidQuoter | {name} | fee {cfg.maker_fee:.3%} | half-spread {cfg.strategy.half_spread} | "
                    f"max pos {cfg.max_position:g} | latency md/order/ack "
                    f"{lat.market_data_ms}/{lat.order_ms}/{lat.ack_ms} ms"))
    (out / f"{name}_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))
    result.fills.to_parquet(out / f"{name}_fills.parquet")
    return metrics


def _json_safe(value):
    """Strict JSON: NaN / inf become null, numpy scalars become floats."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value) if hasattr(value, "item") else value


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--set", dest="sets", action="append", default=[], type=parse_assignment,
                   metavar="SECTION.KEY=VALUE", help="override one config key (repeatable)")
    p.add_argument("--sweep", action="append", default=[], type=parse_assignment,
                   metavar="SECTION.KEY=V1,V2", help="run every value; several sweeps make a grid")
    args = p.parse_args(argv)

    try:
        base = BacktestConfig.from_toml(args.config).with_overrides(
            {key: parse_value(value) for key, value in args.sets})
        configs = expand(base, args.sweep)
    except (OSError, ValueError) as error:
        sys.exit(f"error: {error}")

    summary = {}
    for cfg in configs:
        started = time.perf_counter()
        try:
            summary[cfg.run_name()] = report(cfg, run_backtest(cfg))
        except ValueError as error:
            sys.exit(f"error: {error}")
        print(f"[{cfg.run_name()}] done in {time.perf_counter() - started:.0f} s")

    skip = {"daily_pnl", "config"}
    table = pd.DataFrame({n: {k: v for k, v in s.items() if k not in skip} for n, s in summary.items()})
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 250,
                           "display.max_columns", None, "display.max_rows", None):
        print("\n", table)


if __name__ == "__main__":
    main()
