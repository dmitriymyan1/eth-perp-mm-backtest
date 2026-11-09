"""Inventory and PnL accounting (average-cost method)."""
from __future__ import annotations

from ..core.domain import QTY_EPS, Fill


class Portfolio:
    """Starts flat with zero cash.

    Identity: equity = cash + position * mark
                     = realized + unrealized - fees + funding
    """

    def __init__(self) -> None:
        self.position = 0.0
        self.avg_price = 0.0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.fees = 0.0
        self.funding_pnl = 0.0

    def on_fill(self, fill: Fill) -> None:
        signed_qty = fill.side.sign * fill.qty
        self.cash -= signed_qty * fill.price + fill.fee
        self.fees += fill.fee
        self._update_position(signed_qty, fill.price)

    def on_funding(self, rate: float, mark: float) -> None:
        """Positive rate: longs pay shorts."""
        payment = -self.position * mark * rate
        self.cash += payment
        self.funding_pnl += payment

    def unrealized_pnl(self, mark: float) -> float:
        return self.position * (mark - self.avg_price)

    def equity(self, mark: float) -> float:
        return self.cash + self.position * mark

    def _update_position(self, signed_qty: float, price: float) -> None:
        same_direction = self.position * signed_qty >= 0
        if same_direction:
            new_position = self.position + signed_qty
            self.avg_price = (self.avg_price * abs(self.position) + price * abs(signed_qty)) / abs(new_position)
            self.position = new_position
            return

        closed_qty = min(abs(signed_qty), abs(self.position))
        direction = 1 if self.position > 0 else -1
        self.realized_pnl += closed_qty * (price - self.avg_price) * direction
        self.position += signed_qty
        if abs(self.position) <= QTY_EPS:
            self.position, self.avg_price = 0.0, 0.0
        elif abs(signed_qty) > closed_qty:  # position flipped: rest opens at fill price
            self.avg_price = price
