"""Causal microstructure features at every book snapshot, and the forward target.

Causality: a feature at snapshot time t uses only books and trades with timestamp <= t.
Scale rule: every feature window stays within MAX_SCALE_RATIO of the prediction horizon
(no 100 ms moving average for a 10 s prediction); check_scales() enforces it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

HORIZON_MS = 1000
MAX_SCALE_RATIO = 100
LEVELS = 5
NS_PER_MS = 1_000_000


def check_scales(windows_ms, horizon_ms: float, max_ratio: float = MAX_SCALE_RATIO) -> None:
    for window in windows_ms:
        ratio = max(window / horizon_ms, horizon_ms / window)
        if ratio > max_ratio:
            raise ValueError(f"feature window {window} ms is {ratio:.0f}x the {horizon_ms} ms horizon (max {max_ratio}x)")


@dataclass(frozen=True)
class FeatureSpec:
    """Feature windows for one prediction horizon. state_window_ms: average the book-state
    features (imbalance, microprice, spread) over this window; None = the value at t."""
    horizon_ms: int
    return_windows_ms: tuple[int, ...]
    ofi_windows_ms: tuple[int, ...]
    flow_windows_ms: tuple[int, ...]
    vol_windows_ms: tuple[int, ...]
    state_window_ms: Optional[int] = None

    @property
    def all_windows_ms(self) -> tuple[int, ...]:
        state = () if self.state_window_ms is None else (self.state_window_ms,)
        return self.return_windows_ms + self.ofi_windows_ms + self.flow_windows_ms + self.vol_windows_ms + state

    def check(self) -> "FeatureSpec":
        check_scales(self.all_windows_ms, self.horizon_ms)
        return self


def spec_for_horizon(horizon_ms: int, max_window_ms: Optional[int] = None) -> FeatureSpec:
    """Windows as fixed multiples of the horizon (the 1 s spec is the original one).
    Book state is averaged over 0.1 x horizon for horizons above 1 s.
    max_window_ms drops longer windows (for example, to stay inside one day of data)."""
    scale = lambda multipliers: tuple(int(horizon_ms * m) for m in multipliers
                                      if max_window_ms is None or horizon_ms * m <= max_window_ms)
    return FeatureSpec(horizon_ms, scale((0.1, 0.5, 2, 10)), scale((0.25, 1, 5)), scale((0.5, 2, 10)),
                       scale((2, 10)), None if horizon_ms <= 1000 else int(horizon_ms * 0.1)).check()


DEFAULT_SPEC = spec_for_horizon(HORIZON_MS)
ALL_WINDOWS_MS = DEFAULT_SPEC.all_windows_ms


def build_features(book: pd.DataFrame, trades: pd.DataFrame, tick: float,
                   spec: FeatureSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """One row per book snapshot: ts, mid and the features (float32)."""
    ts = _ns(book["datetime"])
    bp, bq = _levels(book, "bid")
    ap, aq = _levels(book, "ask")
    mid = (bp[:, 0] + ap[:, 0]) / 2
    out = {"ts": ts, "mid": mid}

    microprice = (bp[:, 0] * aq[:, 0] + ap[:, 0] * bq[:, 0]) / (bq[:, 0] + aq[:, 0])
    state = {"spread_ticks": (ap[:, 0] - bp[:, 0]) / tick,
             "imbalance_1": _ratio(bq[:, 0] - aq[:, 0], bq[:, 0] + aq[:, 0]),
             "imbalance_5": _ratio(bq.sum(1) - aq.sum(1), bq.sum(1) + aq.sum(1)),
             "microprice_ticks": (microprice - mid) / tick}
    for name, values in state.items():
        out[name] = values if spec.state_window_ms is None else _rolling_mean(ts, values, spec.state_window_ms)

    for w in spec.return_windows_ms:
        out[f"return_{w}ms_bp"] = (mid - mid[_index_at(ts, ts - w * NS_PER_MS)]) / mid * 1e4

    ofi = np.concatenate([[0.0], _order_flow_imbalance(bp, bq, ap, aq)]).cumsum()
    for w in spec.ofi_windows_ms:
        out[f"ofi_{w}ms"] = ofi - ofi[_index_at(ts, ts - w * NS_PER_MS)]

    signed, volume = _trade_cumsums(trades, ts)
    for w in spec.flow_windows_ms:
        past = _trade_cumsums(trades, ts - w * NS_PER_MS)
        vol_w = volume - past[1]
        out[f"flow_{w}ms"] = signed - past[0]
        out[f"flow_imbalance_{w}ms"] = _ratio(out[f"flow_{w}ms"], vol_w)
        out[f"volume_{w}ms"] = vol_w

    sq = np.concatenate([[0.0], (np.diff(mid) / mid[1:] * 1e4) ** 2]).cumsum()
    for w in spec.vol_windows_ms:
        out[f"volatility_{w}ms_bp"] = np.sqrt(np.maximum(sq - sq[_index_at(ts, ts - w * NS_PER_MS)], 0.0))

    frame = pd.DataFrame(out)
    features = [c for c in frame.columns if c not in ("ts", "mid")]
    frame[features] = frame[features].astype(np.float32)
    return frame


def forward_return_bp(ts: np.ndarray, mid: np.ndarray, horizon_ms: float = HORIZON_MS) -> np.ndarray:
    """Target: mid change over the next horizon, bp. NaN where the horizon passes the data end."""
    later = _index_at(ts, ts + int(horizon_ms * NS_PER_MS))
    target = (mid[later] - mid) / mid * 1e4
    target[ts + horizon_ms * NS_PER_MS > ts[-1]] = np.nan
    return target


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("ts", "mid")]


# ---------------------------------------------------------------- helpers

def _ns(column: pd.Series) -> np.ndarray:
    return pd.to_datetime(column).to_numpy("datetime64[ns]").astype(np.int64)


def _levels(book: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray]:
    prices = book[[f"{side}_price_{i}" for i in range(1, LEVELS + 1)]].to_numpy(float)
    sizes = book[[f"{side}_qty_{i}" for i in range(1, LEVELS + 1)]].to_numpy(float)
    return prices, sizes


def _index_at(ts: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Index of the last snapshot with ts <= at (clipped to the first snapshot)."""
    return np.clip(np.searchsorted(ts, at, side="right") - 1, 0, len(ts) - 1)


def _rolling_mean(ts: np.ndarray, values: np.ndarray, window_ms: int) -> np.ndarray:
    """Mean over the snapshots in (t - window, t] (causal)."""
    cum = np.concatenate([[0.0], np.cumsum(values)])
    end = np.arange(1, len(ts) + 1)
    start = np.searchsorted(ts, ts - window_ms * NS_PER_MS, side="right")
    return (cum[end] - cum[start]) / (end - start)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _order_flow_imbalance(bp, bq, ap, aq) -> np.ndarray:
    """Cont-Kukanov-Stoikov OFI at level 1 between consecutive snapshots (ETH)."""
    bid_in = np.where(bp[1:, 0] >= bp[:-1, 0], bq[1:, 0], 0.0) - np.where(bp[1:, 0] <= bp[:-1, 0], bq[:-1, 0], 0.0)
    ask_in = np.where(ap[1:, 0] <= ap[:-1, 0], aq[1:, 0], 0.0) - np.where(ap[1:, 0] >= ap[:-1, 0], aq[:-1, 0], 0.0)
    return bid_in - ask_in


def _trade_cumsums(trades: pd.DataFrame, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative signed (buyer aggressor = +) and total traded size up to each time in `at`."""
    t = _ns(trades["datetime"])
    size = trades["size"].to_numpy(float)
    sign = np.where(trades["is_maker_ask"].to_numpy() == 1, 1.0, -1.0)
    signed_cum = np.concatenate([[0.0], np.cumsum(sign * size)])
    volume_cum = np.concatenate([[0.0], np.cumsum(size)])
    i = np.searchsorted(t, at, side="right")
    return signed_cum[i], volume_cum[i]
