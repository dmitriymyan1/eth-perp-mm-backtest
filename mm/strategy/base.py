"""Strategy interface and shared quoting rules."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.domain import BookSnapshot, Quote


class Strategy(ABC):

    @abstractmethod
    def quote(self, book: BookSnapshot, position: float) -> Quote:
        ...

    def on_funding_rate(self, rate: float) -> None:
        """Hook: a new published funding rate (per settlement interval). Default: ignore."""


class InventoryLimit:
    """Soft position limit: a side is switched off if one more fill would break it.
    Orders in flight can still fill, so the position can go slightly over the limit."""

    def __init__(self, lot: float, max_position: float):
        if lot <= 0:
            raise ValueError("lot must be > 0")
        if max_position < lot:
            raise ValueError("max_position must be >= lot")
        self.lot = lot
        self.max_position = max_position

    def sides(self, position: float) -> tuple[bool, bool]:
        """(can_buy, can_sell)"""
        return (position + self.lot <= self.max_position + 1e-12,
                position - self.lot >= -self.max_position - 1e-12)
