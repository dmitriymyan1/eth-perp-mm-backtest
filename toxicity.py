"""Fill toxicity by horizon (CLI). Logic: mm/analytics/toxicity.py.

Usage: python toxicity.py output/<run>_fills.parquet [--data data] [--out output]
Writes <out>/<run>_toxicity.{png,csv} and prints the curve and the lead times.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mm.analytics.toxicity import KINDS, adverse_curve, lead_times, load_mid_path, plot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fills", type=Path)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("output"))
    args = p.parse_args()

    fills = pd.read_parquet(args.fills)
    ts, mid = load_mid_path(args.data)
    groups = {"all": fills} | {k: fills[fills["kind"] == k] for k in KINDS if (fills["kind"] == k).any()}
    curves = {name: adverse_curve(g, ts, mid) for name, g in groups.items()}
    leads = {name: lead_times(c) for name, c in curves.items()}
    shares = {name: g["qty"].sum() / fills["qty"].sum() for name, g in groups.items()}

    stem = args.fills.name.removesuffix("_fills.parquet")
    table = pd.DataFrame(curves)
    table.index.name = "horizon_ms"
    table.to_csv(args.out / f"{stem}_toxicity.csv")
    plot(curves, leads, shares, args.out / f"{stem}_toxicity.png", f"Fill toxicity by horizon - {stem}")

    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 200,
                           "display.max_columns", None):
        print("Adverse move, bp (< 0 = against us):\n", table, "\n")
        print("Lead times:\n", pd.DataFrame(leads).T.assign(volume_share=pd.Series(shares)))


if __name__ == "__main__":
    main()
