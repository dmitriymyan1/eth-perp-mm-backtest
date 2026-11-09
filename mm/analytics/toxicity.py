"""Fill toxicity by horizon: how far the mid moves against our fills after (and before) each fill.

adverse(h) = side * (mid(t + h) - mid_before) / price, in bp, volume-weighted  (< 0 = against us)

delta_t (the lead of the informed flow / sniper signal) is read from the curve:
  - t50 / t90: first horizon where adverse(h) reaches 50 % / 90 % of its 1 s value
               (the fast, information part; the slow drift after 1 s is inventory risk);
  - steepest:  the horizon step with the largest adverse move per log-time step;
  - pre_fill_1s: the move 1 s BEFORE the fill (> 0 = the price was already running into us).
Mid moves only at book snapshots (~53 ms), so horizons below ~50 ms are not resolved.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HORIZONS_MS = [-2000, -1000, -500, -200, -100, -50, 0, 10, 25, 50, 75, 100, 150, 200, 300, 500,
               750, 1000, 1500, 2000, 3000, 5000, 10000, 30000, 60000]
KINDS = ["at_price", "through", "cross", "taker"]
COLORS = {"at_price": "#2a78d6", "through": "#eb6834", "cross": "#1baf7a", "taker": "#eda100", "all": "#0b0b0b"}
INK, MUTED = "#0b0b0b", "#52514e"


def load_mid_path(data: Path) -> tuple[np.ndarray, np.ndarray]:
    frames = [pd.read_parquet(p, columns=["datetime", "bid_price_1", "ask_price_1"])
              for p in sorted((data / "orderbook").glob("*.parquet"))]
    book = pd.concat(frames)
    return book["datetime"].astype("int64").to_numpy(), ((book["bid_price_1"] + book["ask_price_1"]) / 2).to_numpy()


def adverse_curve(fills: pd.DataFrame, ts: np.ndarray, mid: np.ndarray) -> pd.Series:
    """Volume-weighted adverse move (bp of notional) at each horizon."""
    sign = np.where(fills["side"] == "BUY", 1.0, -1.0)
    qty, price = fills["qty"].to_numpy(), fills["price"].to_numpy()
    notional = (qty * price).sum()
    out = {}
    for h in HORIZONS_MS:
        i = np.searchsorted(ts, fills["ts"].to_numpy() + h * 1_000_000, side="right") - 1
        out[h] = float((sign * (mid[np.clip(i, 0, None)] - fills["mid"].to_numpy()) * qty).sum() / notional * 1e4)
    return pd.Series(out)


def lead_times(curve: pd.Series, reference_ms: int = 1000) -> dict[str, float]:
    """t50 / t90 against the fast (information) part of the move, up to `reference_ms`;
    the slow drift after it is inventory risk, not the sniper's signal."""
    after = curve[(curve.index > 0) & (curve.index <= reference_ms)]
    final = after.iloc[-1]
    reach = lambda share: float(after.index[np.argmax(after.to_numpy() <= share * final)]) if final < 0 else np.nan
    steps = curve[curve.index >= 0]
    log_h = np.log10(np.maximum(steps.index.to_numpy(), 1))
    per_log_step = -np.diff(steps.to_numpy()) / np.diff(log_h)  # adverse bp per decade of time
    k = int(np.argmax(per_log_step))
    return {"t50_ms": reach(0.5), "t90_ms": reach(0.9),
            "steepest_from_ms": float(steps.index[k]), "steepest_to_ms": float(steps.index[k + 1]),
            "pre_fill_1s_bps": float(curve[-1000]), "adverse_1s_bps": float(final),
            "adverse_60s_bps": float(curve.iloc[-1])}


def plot(curves: dict[str, pd.Series], leads: dict[str, dict], shares: dict[str, float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for name, curve in curves.items():
        ax.plot(curve.index.to_numpy(), curve.to_numpy(), color=COLORS[name], lw=2.6 if name == "all" else 2,
                marker="o", ms=4, zorder=3 if name == "all" else 2)
    _label_line_ends(ax, curves, shares)
    t50 = leads["all"]["t50_ms"]
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(t50, color=INK, lw=1, ls="--")
    ax.annotate(f"all fills: half of the 1 s adverse move by {t50:.0f} ms", (t50, ax.get_ylim()[1]),
                xytext=(4, -14), textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_xticks([-2000, -500, -100, 0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 60000])
    ax.set_xticklabels(["-2s", "-0.5s", "-100", "0", "50", "100", "200", "500", "1s", "2s", "5s", "10s", "60s"])
    ax.set_xlabel("time from fill, ms (symlog; left of 0 = before the fill)", color=MUTED)
    ax.set_ylabel("mid move vs mid before fill, bp (< 0 = against us)", color=MUTED)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _label_line_ends(ax, curves: dict[str, pd.Series], shares: dict[str, float]) -> None:
    """Direct labels at the right end of each line, pushed apart so they never overlap."""
    low, high = ax.get_ylim()
    min_gap = 0.045 * (high - low)
    ends = sorted(((c.iloc[-1], name) for name, c in curves.items()), reverse=True)
    placed: list[float] = []
    for y, name in ends:
        y_text = min(y, placed[-1] - min_gap) if placed else y
        placed.append(y_text)
        label = "all fills" if name == "all" else f"{name} ({shares[name]:.0%} vol)"
        x_end = curves[name].index[-1]
        ax.annotate(label, (x_end, y), xytext=(x_end * 1.12, y_text), textcoords="data",
                    va="center", ha="left", fontsize=9, color=INK,
                    xycoords="data", annotation_clip=False)
        ax.annotate("", (x_end, y), xytext=(x_end * 1.05, y_text), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(right=curves["all"].index[-1] * 3)
