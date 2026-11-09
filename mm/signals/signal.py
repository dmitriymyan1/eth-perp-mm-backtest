"""Precomputed model predictions, looked up by the book snapshot the bot has observed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.domain import BookSnapshot


class SignalTable:
    """Prediction (bp of mid) per book snapshot time. The prediction for a snapshot uses only
    data up to that snapshot, so it is known when the bot observes the snapshot.
    A snapshot without a prediction gets 0 (no view)."""

    def __init__(self, ts: np.ndarray, prediction_bp: np.ndarray):
        order = np.argsort(ts, kind="stable")
        self._ts = np.asarray(ts, dtype=np.int64)[order]
        self._prediction = np.asarray(prediction_bp, dtype=float)[order]

    @classmethod
    def from_parquet(cls, path: Path) -> SignalTable:
        frame = pd.read_parquet(path, columns=["ts", "prediction_bp"])
        return cls(frame["ts"].to_numpy(), frame["prediction_bp"].to_numpy())

    def __call__(self, book: BookSnapshot) -> float:
        i = int(np.searchsorted(self._ts, book.ts))
        if i < len(self._ts) and self._ts[i] == book.ts:
            return float(self._prediction[i])
        return 0.0
