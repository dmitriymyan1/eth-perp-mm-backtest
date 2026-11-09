"""Trading economics of one or more runs (CLI). Logic: mm/analytics/economics.py.

Usage: python economics.py output/<run>_metrics.json [more runs ...] [--out output/economics] [--detail 0]

Prints for every run: markout curve (premium -> 1 h) vs fee, round-trip stats; for the run given
by --detail (index): breakdowns by fill kind, side, hour and volatility regime.
Writes a PNG of the markout curves and a CSV with all curves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mm.analytics.economics import (HORIZONS_S, activity_before, breakdown, fill_frame,  # noqa: E402
                                    markout_curve, round_trip_summary, round_trips)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED = "#0b0b0b", "#52514e"


def label(config: dict) -> str:
    s = config["strategy"]
    model = "mid" if s.get("name", "mid") == "mid" else "fair value" + (" + ML" if s.get("signal_path") else "")
    return f"{model}, half-spread {s['half_spread']:g}"


def load_mid(days: list[str]) -> tuple[np.ndarray, np.ndarray]:
    book = pd.concat(pd.read_parquet(f"data/orderbook/{d}.parquet", columns=["datetime", "bid_price_1", "ask_price_1"])
                     for d in days)
    return book["datetime"].astype("int64").to_numpy(), ((book["bid_price_1"] + book["ask_price_1"]) / 2).to_numpy()


def plot(curves: pd.DataFrame, fee_bp: float, path: Path, title: str) -> None:
    x_labels = ["0"] + [f"{h:g}s" if h < 60 else f"{h / 60:g}m" for h in HORIZONS_S]
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (name, row) in enumerate(curves.iterrows()):
        values = row[["premium"] + [f"{h:g}s" for h in HORIZONS_S]].to_numpy(float)
        ax.plot(x, values, color=COLORS[i % len(COLORS)], lw=2, marker="o", ms=4)
        ax.annotate(name, (x[-1], values[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axhline(fee_bp, color=INK, lw=1, ls="--")
    ax.annotate(f"maker fee {fee_bp:.2f} bp: a fill pays only above this line", (0.05, fee_bp), xytext=(0, 5),
                textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x, x_labels)
    ax.set_xlabel("time after the fill (0 = quoting premium at the fill)", color=MUTED)
    ax.set_ylabel("markout, bp of notional (gross of fees)", color=MUTED)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlim(right=x[-1] + 2.2)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("output/economics"))
    p.add_argument("--detail", type=int, default=0, help="index of the run for the breakdowns")
    p.add_argument("--title", default="Markout curves")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [(m, json.loads(m.read_text())) for m in args.metrics]
    days = sorted({d for _, r in runs for d in (r["config"]["data"].get("days") or "2026-03-19,2026-03-20,2026-03-21").split(",")})
    mid_ts, mid = load_mid(days)

    curves, trips, frames = {}, {}, {}
    for path, r in runs:
        name = label(r["config"])
        f = fill_frame(pd.read_parquet(str(path).replace("_metrics.json", "_fills.parquet")), mid_ts, mid)
        frames[name] = f
        curves[name] = markout_curve(f)
        trips[name] = round_trip_summary(round_trips(f))
    curves = pd.DataFrame(curves).T
    curves["net 1s"] = curves["1s"] - curves["fee"]
    curves["net 5m"] = curves["300s"] - curves["fee"]
    curves.to_csv(args.out / "markout_curves.csv")
    plot(curves, float(curves["fee"].mean()), args.out / "markout_curves.png", args.title)

    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 250,
                           "display.max_columns", None):
        print("MARKOUT CURVE, bp of notional (premium = at the fill; fee shown separately)\n", curves, "\n")
        print("ROUND TRIPS (FIFO)\n", pd.DataFrame(trips).T, "\n")
        name = list(frames)[args.detail]
        f = frames[name]
        print(f"BREAKDOWNS at 1 s for: {name}")
        print("\nby fill kind\n", breakdown(f, f["kind"]))
        print("\nby side\n", breakdown(f, f["side"]))
        regime = pd.qcut(activity_before(f, mid_ts, mid), 3, labels=["calm", "normal", "active"])
        print("\nby volatility regime (|mid move| over the previous 5 min, terciles)\n", breakdown(f, regime))
        hours = pd.to_datetime(f["ts"]).dt.hour
        print("\nby hour (UTC), 4-hour blocks\n", breakdown(f, (hours // 4 * 4).rename("hour")))


if __name__ == "__main__":
    main()
