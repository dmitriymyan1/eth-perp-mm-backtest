"""Trading economics of a run: quoting premium, markouts by horizon, adverse selection, fees,
FIFO round trips and breakdowns. All per-fill values are volume-weighted, in bp of notional.

  premium      = side * (mid_before - price)          what we earn for providing liquidity
  markout(h)   = side * (mid(t + h) - price)          realized spread after h
  adverse(h)   = markout(h) - premium = side * (mid(t + h) - mid_before)
  net(h)       = markout(h) - fee                     profit if the position were closed at h
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

HORIZONS_S = (0.1, 0.5, 1, 5, 30, 60, 300, 900, 3600)
NS = 1_000_000_000


def mid_at(mid_ts: np.ndarray, mid: np.ndarray, at: np.ndarray) -> np.ndarray:
    return mid[np.clip(np.searchsorted(mid_ts, at, side="right") - 1, 0, None)]


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if weights.sum() > 0 else np.nan


def fill_frame(fills: pd.DataFrame, mid_ts: np.ndarray, mid: np.ndarray) -> pd.DataFrame:
    """Per-fill premium, fee and markouts in bp (one column per horizon)."""
    f = fills.copy()
    f["sign"] = np.where(f["side"] == "BUY", 1.0, -1.0)
    f["notional"] = f["price"] * f["qty"]
    f["premium_bp"] = f["sign"] * (f["mid"] - f["price"]) / f["price"] * 1e4
    f["fee_bp"] = f["fee"] / f["notional"] * 1e4
    for h in HORIZONS_S:
        later = mid_at(mid_ts, mid, f["ts"].to_numpy() + int(h * NS))
        f[f"mo_{h:g}s"] = f["sign"] * (later - f["price"]) / f["price"] * 1e4
    return f


def markout_curve(f: pd.DataFrame) -> pd.Series:
    """premium, markout at each horizon, fee (all weighted by notional)."""
    w = f["notional"].to_numpy()
    out = {"premium": _weighted(f["premium_bp"].to_numpy(), w)}
    out |= {f"{h:g}s": _weighted(f[f"mo_{h:g}s"].to_numpy(), w) for h in HORIZONS_S}
    out["fee"] = _weighted(f["fee_bp"].to_numpy(), w)
    return pd.Series(out)


def round_trips(f: pd.DataFrame) -> pd.DataFrame:
    """FIFO matching of opening and closing fills. One row per matched piece."""
    open_lots: deque = deque()  # [qty, price, ts, sign, fee_per_unit]
    rows = []
    for fill in f.itertuples():
        qty, fee_unit = fill.qty, fill.fee / fill.qty
        while qty > 1e-12 and open_lots and open_lots[0][3] != fill.sign:
            lot = open_lots[0]
            matched = min(qty, lot[0])
            gross = lot[3] * (fill.price - lot[1]) * matched
            rows.append((lot[2], fill.ts, matched, lot[1], gross, gross - (lot[4] + fee_unit) * matched))
            lot[0] -= matched
            qty -= matched
            if lot[0] <= 1e-12:
                open_lots.popleft()
        if qty > 1e-12:
            open_lots.append([qty, fill.price, fill.ts, fill.sign, fee_unit])
    trips = pd.DataFrame(rows, columns=["open_ts", "close_ts", "qty", "entry", "gross_usd", "net_usd"])
    trips["hold_s"] = (trips["close_ts"] - trips["open_ts"]) / NS
    trips["net_bp"] = trips["net_usd"] / (trips["entry"] * trips["qty"]) * 1e4
    return trips


def round_trip_summary(trips: pd.DataFrame) -> pd.Series:
    if trips.empty:
        return pd.Series(dtype=float)
    w = trips["qty"].to_numpy()
    return pd.Series({
        "round trips": len(trips),
        "matched ETH": w.sum(),
        "hold median s": float(np.median(trips["hold_s"])),
        "hold p90 s": float(np.percentile(trips["hold_s"], 90)),
        "win rate (net)": float(np.average(trips["net_usd"] > 0, weights=w)),
        "gross bp": _weighted(trips["gross_usd"] / (trips["entry"] * trips["qty"]) * 1e4, w),
        "net bp": _weighted(trips["net_bp"].to_numpy(), w),
        "net USD": trips["net_usd"].sum(),
    })


def breakdown(f: pd.DataFrame, by: pd.Series, horizon: str = "mo_1s") -> pd.DataFrame:
    """Volume share, premium, adverse and net at one horizon, per group."""
    def stats(g: pd.DataFrame) -> pd.Series:
        w = g["notional"].to_numpy()
        premium = _weighted(g["premium_bp"].to_numpy(), w)
        markout = _weighted(g[horizon].to_numpy(), w)
        return pd.Series({"volume share": w.sum() / f["notional"].sum(), "premium": premium,
                          "adverse": markout - premium, "markout": markout,
                          "net": markout - _weighted(g["fee_bp"].to_numpy(), w)})
    return f.groupby(by, observed=True)[["notional", "premium_bp", horizon, "fee_bp"]].apply(stats)


def activity_before(f: pd.DataFrame, mid_ts: np.ndarray, mid: np.ndarray, window_s: float = 300) -> np.ndarray:
    """|mid change| over the window before each fill, bp (a volatility regime proxy)."""
    before = mid_at(mid_ts, mid, f["ts"].to_numpy() - int(window_s * NS))
    return np.abs(f["mid"].to_numpy() / before - 1) * 1e4
