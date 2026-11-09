"""Turns a desired quote into submit / cancel requests, based on the bot's own view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.domain import QTY_EPS, Fill, Quote, Side, same_price


class SubmittedOrder(Protocol):
    id: int


class OrderGateway(Protocol):
    """What the bot needs from an exchange. SimulatedExchange implements it;
    a live exchange adapter could too, without any change on the bot side."""

    def submit(self, side: Side, price: float, size: float, now: int) -> SubmittedOrder: ...

    def cancel(self, order_id: int, now: int) -> None: ...


@dataclass(frozen=True)
class OrderTicket:
    """What the bot knows about its own order (never the exchange's live state)."""
    id: int
    side: Side
    price: float
    size: float


class OrderManager:
    """Keeps at most one live order per side at the quoted price.

    - Quote price changed or side switched off -> cancel the live order.
    - No live order at the quoted price       -> submit a new one.
    A partially filled order stays in the book until its price changes (no top-up).

    An order counts as done only after its fill reports arrive (or after we ask to cancel it).
    """

    def __init__(self, exchange: OrderGateway):
        self._exchange = exchange
        self._live: dict[int, OrderTicket] = {}
        self._reported_qty: dict[int, float] = {}

    def on_fill_report(self, fill: Fill) -> None:
        ticket = self._live.get(fill.order_id)
        if ticket is None:
            return
        self._reported_qty[ticket.id] += fill.qty
        if self._reported_qty[ticket.id] >= ticket.size - QTY_EPS:
            self._forget(ticket)

    def sync(self, quote: Quote, now: int) -> None:
        for side in Side:
            target = quote.price(side)
            has_target_order = False
            for ticket in [t for t in self._live.values() if t.side is side]:
                if target is not None and same_price(ticket.price, target) and not has_target_order:
                    has_target_order = True
                else:
                    self._exchange.cancel(ticket.id, now)
                    self._forget(ticket)
            if target is not None and not has_target_order:
                order = self._exchange.submit(side, target, quote.size, now)
                self._live[order.id] = OrderTicket(order.id, side, target, quote.size)
                self._reported_qty[order.id] = 0.0

    def _forget(self, ticket: OrderTicket) -> None:
        del self._live[ticket.id]
        del self._reported_qty[ticket.id]
