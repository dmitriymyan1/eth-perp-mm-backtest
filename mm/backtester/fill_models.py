"""Fill models decide how much of a resting order a market event executes.

Common assumptions:
- Our orders are invisible in the historical data and do not change it.
- A maker fill is always at our limit price.
- The aggressor of a print has a fixed size. If our order had rested at a better price,
  the aggressor would have traded with us first. So a print THROUGH our price fills us
  for up to its size (a sweep prints one row per level/maker; the rows add up).
- Book crossing our price (opposite best at or through it), see CrossRule.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional

import numpy as np

from ..core.domain import QTY_EPS, BookSnapshot, Order, Trade, is_better


class CrossRule(str, Enum):
    """When does a book that crossed our price fill the rest of our order?"""
    NEVER = "never"            # only printed trades fill us (lower bound on volume)
    CONFIRMED = "confirmed"    # only if, since the last book, trades reached our price and cleared the queue ahead
    PROFITABLE = "profitable"  # CONFIRMED, or a sniper would profit from taking our stale quote (see below)
    ALWAYS = "always"          # any cross: a stale quote is always picked off (impossible upper bound)


class CancelRule(str, Enum):
    """Which part of the queue a shrinking level (cancels) is taken from."""
    BACK = "back"            # behind us first: most pessimistic
    PRO_RATA = "pro_rata"    # in proportion to volume ahead / behind us
    FRONT = "front"          # ahead of us first: most optimistic


class FuturePrice:
    """Fair price of a sniper with an information lead: the mid `lead_ns` after the book
    (it knows the price from other venues). Exchange-side counterparty model only:
    the bot never sees it, so this is not look-ahead for our strategy."""

    def __init__(self, ts: np.ndarray, mid: np.ndarray, lead_ns: int):
        self._ts, self._mid, self._lead_ns = ts, mid, lead_ns

    def __call__(self, book: BookSnapshot) -> float:
        i = int(np.searchsorted(self._ts, book.ts + self._lead_ns, side="right")) - 1
        return float(self._mid[i]) if i >= 0 else book.mid


class FillModel(ABC):
    """PROFITABLE: a sniper takes our stale quote only if that beats its taker fee, measured
    against its fair price (default: mid of the crossing book; see FuturePrice):
    (fair - ask) or (bid - fair) > pickoff_fee * our price.
    Without trades in the data there was no natural taker flow, so only a sniper can be left."""

    def __init__(self, cross_rule: CrossRule = CrossRule.CONFIRMED, pickoff_fee: float = 0.00045,
                 fair_price: Optional[Callable[[BookSnapshot], float]] = None):
        if pickoff_fee < 0:
            raise ValueError("pickoff fee must be >= 0")
        self.cross_rule = CrossRule(cross_rule)
        self.pickoff_fee = pickoff_fee
        self._fair_price = fair_price or (lambda book: book.mid)

    def on_activate(self, order: Order, book: Optional[BookSnapshot]) -> None:
        """Hook: the order has just reached the exchange (and does not cross the book)."""

    def on_book(self, order: Order, book: BookSnapshot) -> float:
        touched, order.touched = order.touched, False
        if not self.is_crossed(order, book):
            return 0.0
        confirmed = touched and self._queue_cleared(order)
        if (self.cross_rule is CrossRule.ALWAYS
                or (self.cross_rule is CrossRule.CONFIRMED and confirmed)
                or (self.cross_rule is CrossRule.PROFITABLE and (confirmed or self._sniper_profits(order, book)))):
            return order.remaining
        return 0.0

    def _sniper_profits(self, order: Order, book: BookSnapshot) -> bool:
        """The taker's gain vs its fair price is larger than its fee."""
        taker_gain = order.side.opposite.sign * (self._fair_price(book) - order.price)
        return taker_gain > self.pickoff_fee * order.price

    def on_trade(self, order: Order, trade: Trade, available: float) -> float:
        """Quantity of `order` that `trade` executes. `available` is what is left of the
        print after our better-priced orders took their part (see SimulatedExchange)."""
        if not self._reaches(order, trade):
            return 0.0
        if available > QTY_EPS:  # the print (what is left of it) really reached our price
            order.touched = True
        if is_better(order.side, order.price, trade.price):  # printed through our price
            self._on_level_swept(order)
            return min(order.remaining, available)
        return self._fill_at_our_price(order, trade, available)

    @abstractmethod
    def _fill_at_our_price(self, order: Order, trade: Trade, available: float) -> float:
        ...

    def _queue_cleared(self, order: Order) -> bool:
        return True

    def _on_level_swept(self, order: Order) -> None:
        """Hook: our whole price level has been traded."""

    @staticmethod
    def is_crossed(order: Order, book: BookSnapshot) -> bool:
        """The opposite best price is at or through our price."""
        return not is_better(order.side, book.best(order.side.opposite), order.price)

    @staticmethod
    def _reaches(order: Order, trade: Trade) -> bool:
        """Trade comes from the opposite side and prints at or through our price."""
        return (trade.aggressor is order.side.opposite
                and not is_better(order.side, trade.price, order.price))


class SimpleFillModel(FillModel):
    """Optimistic: we are always first in the queue at our price."""

    def _fill_at_our_price(self, order: Order, trade: Trade, available: float) -> float:
        return min(order.remaining, available)


class QueueAwareFillModel(FillModel):
    """We join the back of the queue at our price level.

    - Start: queue_ahead = visible qty at our price (unknown depth -> infinite).
    - Trade at our price: the print consumes queue_ahead first; the rest fills us.
    - Book update, level grew: new orders joined behind us, queue_ahead unchanged.
    - Book update, level shrank (cancels): queue_ahead falls per CancelRule,
      and never exceeds the visible level.
    """

    def __init__(self, cross_rule: CrossRule = CrossRule.CONFIRMED,
                 cancel_rule: CancelRule = CancelRule.PRO_RATA, pickoff_fee: float = 0.00045,
                 fair_price: Optional[Callable[[BookSnapshot], float]] = None):
        super().__init__(cross_rule, pickoff_fee, fair_price)
        self.cancel_rule = CancelRule(cancel_rule)

    def on_activate(self, order: Order, book: Optional[BookSnapshot]) -> None:
        visible = book.visible_qty(order.side, order.price) if book else None
        order.queue_ahead = order.level_qty = math.inf if visible is None else visible

    def on_book(self, order: Order, book: BookSnapshot) -> float:
        filled = super().on_book(order, book)  # cross check uses the queue BEFORE this book
        visible = book.visible_qty(order.side, order.price)
        if visible is not None:
            self._update_queue(order, visible)
        return filled

    def _fill_at_our_price(self, order: Order, trade: Trade, available: float) -> float:
        # Our fill: what is left of the print after the queue ahead of us.
        # The queue itself traded with the real print, so it shrinks by the full print size.
        fill = min(order.remaining, max(0.0, available - order.queue_ahead))
        order.queue_ahead = max(0.0, order.queue_ahead - trade.size)
        order.level_qty = max(0.0, order.level_qty - trade.size)
        return fill

    def _queue_cleared(self, order: Order) -> bool:
        return order.queue_ahead <= QTY_EPS

    def _on_level_swept(self, order: Order) -> None:
        order.queue_ahead = order.level_qty = 0.0

    def _update_queue(self, order: Order, visible: float) -> None:
        if not math.isinf(order.level_qty) and visible < order.level_qty:
            order.queue_ahead -= self._cancelled_ahead(order, order.level_qty - visible)
        order.queue_ahead = max(0.0, min(order.queue_ahead, visible))
        order.level_qty = visible

    def _cancelled_ahead(self, order: Order, cancelled: float) -> float:
        if self.cancel_rule is CancelRule.FRONT:
            return cancelled
        if self.cancel_rule is CancelRule.PRO_RATA:
            return cancelled * order.queue_ahead / order.level_qty
        return 0.0  # BACK: the cap at the visible level does the rest
