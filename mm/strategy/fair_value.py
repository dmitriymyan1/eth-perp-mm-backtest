"""Quoting around a fair value: model signal + inventory skew toward a target + funding."""
from __future__ import annotations

import math
from typing import Callable, Optional

from ..core.domain import BookSnapshot, Quote, ceil_to_tick, floor_to_tick
from .base import InventoryLimit, Strategy

Signal = Callable[[BookSnapshot], float]  # predicted mid change, bp


class FairValueQuoter(Strategy):
    """fair = mid * (1 + signal_bp / 1e4 - funding_rate * funding_fair_hours)
           - inventory_skew * (position - target_position)
    bid  = floor_to_tick(fair - half_spread), ask = ceil_to_tick(fair + half_spread)

    - signal: model prediction of the mid change (bp); None = no model view.
    - inventory_skew (USD per ETH) leans quotes so the position moves to target_position.
    - Funding harvest, two ways (rate per settlement interval, > 0 = longs pay shorts):
      * target: hold -sign(rate) * funding_target_position when |rate| >= funding_threshold_bp;
      * fair-value shift: the expected funding over funding_fair_hours of holding.
    - Post-only: the bid never goes above the best bid, the ask never below the best ask.
    - pull_threshold_bp: if the model expects the price to move against a side by more than
      this, that side is not quoted at all. None = never pull.
    """

    def __init__(self, tick: float, half_spread: float, lot: float, max_position: float,
                 signal: Optional[Signal] = None, inventory_skew: float = 0.0, signal_cap_bp: float = 5.0,
                 pull_threshold_bp: Optional[float] = None, funding_target_position: float = 0.0,
                 funding_threshold_bp: float = 0.0, funding_fair_hours: float = 0.0,
                 signal_deadband_bp: float = 0.0, signal_release_ratio: float = 0.5):
        if tick <= 0 or half_spread <= 0 or inventory_skew < 0 or signal_cap_bp <= 0:
            raise ValueError("tick, half_spread, signal_cap_bp must be > 0 and inventory_skew >= 0")
        if pull_threshold_bp is not None and pull_threshold_bp < 0:
            raise ValueError("pull_threshold_bp must be >= 0")
        if not 0 <= funding_target_position <= max_position or funding_threshold_bp < 0 or funding_fair_hours < 0:
            raise ValueError("funding settings: 0 <= target <= max_position, threshold and hours >= 0")
        self.tick = tick
        self.half_spread = half_spread
        self.limit = InventoryLimit(lot, max_position)
        self.signal = signal or (lambda book: 0.0)
        self.inventory_skew = inventory_skew
        self.signal_cap_bp = signal_cap_bp
        self.pull_threshold_bp = pull_threshold_bp
        self.funding_target_position = funding_target_position
        self.funding_threshold_bp = funding_threshold_bp
        self.funding_fair_hours = funding_fair_hours
        self.funding_rate = 0.0
        if signal_deadband_bp < 0 or not 0 <= signal_release_ratio <= 1:
            raise ValueError("signal_deadband_bp must be >= 0 and signal_release_ratio in [0, 1]")
        self.signal_deadband_bp = signal_deadband_bp
        self.signal_release_ratio = signal_release_ratio
        self._signal_on = 0  # hysteresis state: -1 / 0 / +1

    def on_funding_rate(self, rate: float) -> None:
        self.funding_rate = rate

    def predicted_bp(self, book: BookSnapshot) -> float:
        """Clipped prediction after the dead zone with hysteresis: the signal switches on when
        |prediction| >= deadband and off only when it falls below deadband * release_ratio
        (or changes sign). Inside the dead zone the model has no view (0)."""
        raw = max(-self.signal_cap_bp, min(self.signal_cap_bp, self.signal(book)))
        if self.signal_deadband_bp == 0:
            return raw
        side = 1 if raw > 0 else -1 if raw < 0 else 0
        if abs(raw) >= self.signal_deadband_bp:
            self._signal_on = side
        elif self._signal_on != side or abs(raw) < self.signal_deadband_bp * self.signal_release_ratio:
            self._signal_on = 0
        return raw if self._signal_on else 0.0

    def target_position(self) -> float:
        if abs(self.funding_rate) * 1e4 < self.funding_threshold_bp or self.funding_rate == 0:
            return 0.0
        return -math.copysign(self.funding_target_position, self.funding_rate)

    def fair_value(self, book: BookSnapshot, position: float) -> float:
        drift = self.predicted_bp(book) / 1e4 - self.funding_rate * self.funding_fair_hours
        return book.mid * (1 + drift) - self.inventory_skew * (position - self.target_position())

    def quote(self, book: BookSnapshot, position: float) -> Quote:
        can_buy, can_sell = self.limit.sides(position)
        if self.pull_threshold_bp is not None:
            predicted = self.predicted_bp(book)
            can_buy = can_buy and predicted >= -self.pull_threshold_bp   # expected fall: no bid
            can_sell = can_sell and predicted <= self.pull_threshold_bp  # expected rise: no ask
        fair = self.fair_value(book, position)
        bid = min(floor_to_tick(fair - self.half_spread, self.tick), book.best_bid)
        ask = max(ceil_to_tick(fair + self.half_spread, self.tick), book.best_ask)
        return Quote(bid if can_buy else None, ask if can_sell else None, self.limit.lot)
