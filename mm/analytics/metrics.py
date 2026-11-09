"""Performance metrics computed from a BacktestResult.

PnL breakdown, in bp of traded notional (sums exactly to pnl_bps_of_notional):
    pnl = spread_capture + inventory - fees + funding
    spread_capture: fill price vs true mid just before the fill (edge at the moment of the fill)
    inventory:      everything after the fill (adverse selection + inventory price moves)
markout_<h>_bps shows how fast the edge disappears: fill price vs mid h after the fill;
adverse_1s_bps = markout_1s - spread_capture (adverse selection within 1 s).
For cross fills "mid before the fill" is the mid before the crossing book, so their spread
capture is a convention; see the per-kind numbers (<kind>_spread_capture_bps, <kind>_markout_1s_bps).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..core.domain import FillKind
from ..backtester.engine import BacktestResult

HOURS_PER_YEAR = 365 * 24  # crypto trades 24/7
NS_PER_SECOND = 10**9
MARKOUT_HORIZONS_S = (1, 60)


def period_pnl(equity: pd.Series, freq: str) -> pd.Series:
    """PnL per period; the first period counts from 0 (equity starts at 0)."""
    end_of_period = equity.resample(freq).last().ffill()
    return end_of_period.diff().fillna(end_of_period.iloc[0])


def daily_pnl(equity: pd.Series) -> pd.Series:
    return period_pnl(equity, "1D")


def sharpe_ratio(pnl_changes: pd.Series, periods_per_year: int) -> float:
    """Annualized Sharpe of PnL increments (scale-free, so no capital base is needed)."""
    std = pnl_changes.std()
    if len(pnl_changes) < 2 or not std > 0:
        return math.nan
    return pnl_changes.mean() / std * math.sqrt(periods_per_year)


def newey_west_t_stat(pnl_changes: pd.Series, lags: int) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) long-run variance."""
    x = pnl_changes.to_numpy(float)
    n = len(x)
    if n < 2:
        return math.nan
    d = x - x.mean()
    long_run_var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        long_run_var += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    if not long_run_var > 0:
        return math.nan
    return x.mean() / math.sqrt(long_run_var / n)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall, in USD (equity starts at 0)."""
    peak = equity.cummax().clip(lower=0.0)
    return float((peak - equity).max())


def inventory_stats(fills: pd.DataFrame, start_ns: int, end_ns: int) -> tuple[float, float]:
    """Exact (time-weighted mean |position|, max |position|) from the fill path."""
    if fills.empty or end_ns <= start_ns:
        return 0.0, 0.0
    signed = _signed_qty(fills)
    position = np.cumsum(signed)
    times = np.append(fills["ts"].to_numpy(), end_ns)
    held_for = np.diff(times)  # each position is held until the next fill (or the end)
    mean_abs = float(np.sum(np.abs(position) * held_for) / (end_ns - start_ns))
    return mean_abs, float(np.abs(position).max())


def edge_bps(fills: pd.DataFrame, mids: np.ndarray) -> float:
    """Sum of side * (mid - fill price) * qty, in bp of notional."""
    notional = float((fills["price"] * fills["qty"]).sum())
    if not notional:
        return math.nan
    gain = _signed_qty(fills) * (mids - fills["price"].to_numpy())
    return float(gain.sum() / notional * 1e4)


def mid_after(mid_path: pd.Series, ts: np.ndarray) -> np.ndarray:
    """True mid at each time in `ts` (last change at or before it)."""
    i = np.searchsorted(mid_path.index.to_numpy(), ts, side="right") - 1
    return mid_path.to_numpy()[np.clip(i, 0, None)]


def compute_metrics(result: BacktestResult) -> dict[str, float]:
    eq = result.equity
    last = eq.iloc[-1]
    stats = result.order_stats
    fills = result.fills
    hourly_pnl = period_pnl(eq["equity"], "1h")
    notional = float((fills["price"] * fills["qty"]).sum())
    to_bps = (lambda usd: usd / notional * 1e4) if notional else (lambda usd: math.nan)
    mean_abs_inv, max_abs_inv = inventory_stats(fills, eq.index[0].value, eq.index[-1].value)
    spread_bps = edge_bps(fills, fills["mid"].to_numpy()) if notional else math.nan

    metrics = {
        "total_pnl": last["equity"],
        "realized_pnl": last["realized"],
        "unrealized_pnl": last["unrealized"],
        "fees": last["fees"],
        "funding_pnl": last["funding"],
        "pnl_bps_of_notional": to_bps(last["equity"]),
        "spread_capture_bps": spread_bps,
        "inventory_bps": to_bps(last["equity"] + last["fees"] - last["funding"]) - spread_bps,
        "fees_bps": to_bps(last["fees"]),
        "funding_bps": to_bps(last["funding"]),
    }
    for horizon in MARKOUT_HORIZONS_S:
        metrics[f"markout_{horizon}s_bps"] = _markout_bps(result, fills, horizon)
    metrics["adverse_1s_bps"] = metrics["markout_1s_bps"] - spread_bps
    for kind in FillKind:
        part = fills[fills["kind"] == kind.value]
        metrics[f"{kind.value}_volume_share"] = part["qty"].sum() / stats.filled_qty if stats.filled_qty else math.nan
        metrics[f"{kind.value}_spread_capture_bps"] = edge_bps(part, part["mid"].to_numpy())
        metrics[f"{kind.value}_markout_1s_bps"] = _markout_bps(result, part, 1)
    metrics |= {
        "sharpe_hourly_annualized": sharpe_ratio(hourly_pnl, HOURS_PER_YEAR),
        "t_stat_hourly_newey_west": newey_west_t_stat(hourly_pnl, lags=5),
        "max_drawdown": max_drawdown(eq["equity"]),
        "orders_submitted": stats.submitted,
        "orders_cancelled": stats.cancelled,
        "fills": stats.fills,
        "taker_fills": stats.taker_fills,
        "filled_orders": stats.filled_orders,
        "fill_rate": stats.filled_orders / stats.submitted if stats.submitted else math.nan,  # incl. partial
        "volume_fill_rate": stats.filled_qty / stats.submitted_qty if stats.submitted_qty else math.nan,
        "filled_volume": stats.filled_qty,
        "market_volume": result.market_volume,
        "volume_share_of_market": stats.filled_qty / result.market_volume if result.market_volume else math.nan,
        "pnl_per_unit_volume": last["equity"] / stats.filled_qty if stats.filled_qty else math.nan,
        "mean_abs_inventory": mean_abs_inv,  # time-weighted
        "max_abs_inventory": max_abs_inv,    # exact, from fills
        "final_inventory": last["position"],
    }
    return metrics


def _markout_bps(result: BacktestResult, fills: pd.DataFrame, horizon_s: int) -> float:
    later = mid_after(result.mid_path, fills["ts"].to_numpy() + horizon_s * NS_PER_SECOND)
    return edge_bps(fills, later)


def _signed_qty(fills: pd.DataFrame) -> np.ndarray:
    return np.where(fills["side"] == "BUY", 1.0, -1.0) * fills["qty"].to_numpy()
