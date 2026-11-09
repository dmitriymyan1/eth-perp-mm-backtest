"""Symmetric quoting around the mid price."""
from __future__ import annotations

from ..core.domain import BookSnapshot, Quote, ceil_to_tick, floor_to_tick
from .base import InventoryLimit, Strategy


class MidQuoter(Strategy):
    """Symmetric quote around the mid price, fixed size, soft inventory limit.

    bid = floor_to_tick(mid - half_spread), ask = ceil_to_tick(mid + half_spread).
    Rounding is outward, so with a 1-tick spread any half_spread in (0, tick/2] joins the touch.
    """

    def __init__(self, tick: float, half_spread: float, lot: float, max_position: float):
        if tick <= 0 or half_spread <= 0:
            raise ValueError("tick and half_spread must be > 0")
        self.tick = tick
        self.half_spread = half_spread
        self.limit = InventoryLimit(lot, max_position)

    def quote(self, book: BookSnapshot, position: float) -> Quote:
        can_buy, can_sell = self.limit.sides(position)
        bid = floor_to_tick(book.mid - self.half_spread, self.tick)
        ask = ceil_to_tick(book.mid + self.half_spread, self.tick)
        return Quote(bid if can_buy else None, ask if can_sell else None, self.limit.lot)
