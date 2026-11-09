"""Bot side: acts only on what it has observed (delayed book, delayed fill reports, its clock)."""
from __future__ import annotations

from typing import Optional

from ..core.domain import BookSnapshot, Fill, Quote
from .order_manager import OrderManager
from ..strategy import Strategy

NO_QUOTE = Quote(None, None, 0.0)


class TradingBot:
    """If the last book is older than `max_book_age_ns`, the bot pulls its quotes
    instead of quoting on stale prices (feed outage). The engine wakes the bot
    (`on_clock`) when a book would become stale."""

    def __init__(self, strategy: Strategy, order_manager: OrderManager,
                 max_book_age_ns: Optional[int] = None):
        self.strategy = strategy
        self.orders = order_manager
        self.max_book_age_ns = max_book_age_ns
        self.book: Optional[BookSnapshot] = None
        self.position = 0.0  # position known from fill reports (lags the true position)

    def on_book(self, book: BookSnapshot, now: int) -> None:
        self.book = book
        self._requote(now)

    def on_funding_rate(self, rate: float, now: int) -> None:
        """Published funding rate (forecast of the next payment), seen with the market-data latency."""
        self.strategy.on_funding_rate(rate)
        self._requote(now)

    def on_fill_report(self, fill: Fill, now: int) -> None:
        self.position += fill.side.sign * fill.qty
        self.orders.on_fill_report(fill)
        self._requote(now)

    def on_clock(self, book_seen: BookSnapshot, now: int) -> None:
        """Timer set when `book_seen` arrived; nothing to do if a newer book came since."""
        if self.book is book_seen:
            self._requote(now)

    def _requote(self, now: int) -> None:
        if self.book is None:
            return
        if self._book_is_stale(now):
            self.orders.sync(NO_QUOTE, now)
        else:
            self.orders.sync(self.strategy.quote(self.book, self.position), now)

    def _book_is_stale(self, now: int) -> bool:
        return self.max_book_age_ns is not None and now - self.book.ts > self.max_book_age_ns
