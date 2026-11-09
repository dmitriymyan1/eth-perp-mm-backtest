"""Simulated exchange: order lifecycle (latency, cancels) and matching via a FillModel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.domain import QTY_EPS, BookSnapshot, Event, Fill, FillKind, Order, Side, Trade, is_better
from .fill_models import FillModel


@dataclass
class OrderStats:
    submitted: int = 0
    cancelled: int = 0
    filled_orders: int = 0   # orders with at least one fill
    fully_filled: int = 0
    fills: int = 0           # number of fill events
    taker_fills: int = 0     # orders that crossed the book on arrival
    submitted_qty: float = 0.0
    filled_qty: float = 0.0


class SimulatedExchange:
    """Order lifecycle and matching.

    - New orders and cancels take `latency_ns` (order entry, one way) to reach the exchange.
      Until a cancel arrives, the order can still be filled. An arrival at exactly the time
      of a market event is applied before that event.
    - An order that crosses the last book on arrival trades as a taker (GTC limit order):
      full size at the opposite best price, taker fee, time = arrival time.
      Trades between that book and the arrival are not checked (0 cases in the data).
    - A trade is matched against our orders in price-time priority. Each fill uses up part
      of the print, so the same print volume is never counted twice across our own orders.
    """

    def __init__(self, fill_model: FillModel, latency_ns: int = 0,
                 maker_fee: float = 0.0, taker_fee: float = 0.0):
        if latency_ns < 0:
            raise ValueError("order latency must be >= 0")
        self._fill_model = fill_model
        self._latency_ns = latency_ns
        self._maker_fee = maker_fee
        self._taker_fee = taker_fee
        self._orders: dict[int, Order] = {}
        self._book: Optional[BookSnapshot] = None
        self._next_id = 1
        self.stats = OrderStats()

    # ------------------------------------------------------------ client API

    def submit(self, side: Side, price: float, size: float, now: int) -> Order:
        order = Order(self._next_id, side, price, size, active_at=now + self._latency_ns)
        self._next_id += 1
        self._orders[order.id] = order
        self.stats.submitted += 1
        self.stats.submitted_qty += size
        return order

    def cancel(self, order_id: int, now: int) -> None:
        order = self._orders.get(order_id)
        if order is not None and order.cancel_at is None:
            order.cancel_at = now + self._latency_ns

    def open_orders(self) -> list[Order]:
        return list(self._orders.values())

    # ------------------------------------------------------------ market data

    def on_event(self, event: Event) -> list[Fill]:
        fills = self._process_arrivals(event.ts)
        if isinstance(event, BookSnapshot):
            self._book = event
            fills += self._match_book(event)
        elif isinstance(event, Trade):
            fills += self._match_trade(event)
        return fills

    def _process_arrivals(self, ts: int) -> list[Fill]:
        """Activate new orders and apply cancels whose latency has passed."""
        fills = []
        for order in list(self._orders.values()):
            if not order.is_active and order.active_at <= ts:
                order.is_active = True
                if self._book is not None and FillModel.is_crossed(order, self._book):
                    opposite_best = self._book.best(order.side.opposite)
                    fills.append(self._execute(order, order.remaining, order.active_at,
                                               FillKind.TAKER, price=opposite_best))
                    continue
                self._fill_model.on_activate(order, self._book)
            if order.cancel_at is not None and order.cancel_at <= ts:
                del self._orders[order.id]
                self.stats.cancelled += 1
        return fills

    def _match_book(self, book: BookSnapshot) -> list[Fill]:
        fills = []
        for order in [o for o in self._orders.values() if o.is_active]:
            qty = min(self._fill_model.on_book(order, book), order.remaining)
            if qty > QTY_EPS:
                fills.append(self._execute(order, qty, book.ts, FillKind.CROSS))
        return fills

    def _match_trade(self, trade: Trade) -> list[Fill]:
        fills = []
        available = trade.size
        for order in self._by_priority(trade.aggressor.opposite):
            qty = min(self._fill_model.on_trade(order, trade, available), order.remaining)
            if qty > QTY_EPS:
                kind = FillKind.THROUGH if is_better(order.side, order.price, trade.price) else FillKind.AT_PRICE
                fills.append(self._execute(order, qty, trade.ts, kind))
                available = max(0.0, available - qty)
        return fills

    def _by_priority(self, side: Side) -> list[Order]:
        """Active orders on `side`, best price first, then oldest first."""
        orders = [o for o in self._orders.values() if o.is_active and o.side is side]
        return sorted(orders, key=lambda o: (-side.sign * o.price, o.id))

    def _execute(self, order: Order, qty: float, ts: int, kind: FillKind,
                 price: Optional[float] = None) -> Fill:
        taker = kind is FillKind.TAKER
        price = order.price if price is None else price
        if order.filled == 0:
            self.stats.filled_orders += 1
        order.filled += qty
        self.stats.fills += 1
        self.stats.taker_fills += taker
        self.stats.filled_qty += qty
        if order.is_filled:
            self.stats.fully_filled += 1
            del self._orders[order.id]
        fee_rate = self._taker_fee if taker else self._maker_fee
        return Fill(ts, order.id, order.side, price, qty, fee=price * qty * fee_rate, kind=kind)
