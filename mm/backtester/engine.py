"""Event loop: market events drive the exchange in true time; the bot gets
delayed observations through a scheduler.

    exchange ──(market_data_ns)──► bot sees book
    bot      ──(order entry, in SimulatedExchange)──► exchange gets order / cancel
    exchange ──(ack_ns)──────────► bot sees fill report
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import pandas as pd

from ..trading.bot import TradingBot
from ..core.domain import BookSnapshot, Event, Fill, FundingRate, FundingSettlement, Trade
from .exchange import OrderStats, SimulatedExchange
from ..trading.order_manager import OrderManager
from .portfolio import Portfolio
from ..strategy import Strategy

NS_PER_SECOND = 10**9


@dataclass(frozen=True)
class FeedLatency:
    market_data_ns: int = 0  # exchange -> bot, book snapshots
    ack_ns: int = 0          # exchange -> bot, fill reports


@dataclass
class BacktestResult:
    equity: pd.DataFrame   # time-indexed: equity, position, realized, unrealized, fees, funding, mid
    fills: pd.DataFrame    # ts, order_id, side, price, qty, fee, kind, mid (true mid just before the event)
    order_stats: OrderStats
    market_volume: float   # total traded volume in the data
    mid_path: pd.Series    # true mid, one value per change, indexed by ts (ns)


class EquityRecorder:
    """Samples the portfolio state at most once per `interval_ns`."""

    COLUMNS = ["ts", "equity", "position", "realized", "unrealized", "fees", "funding", "mid"]

    def __init__(self, interval_ns: int):
        self._interval_ns = interval_ns
        self._next_ts: Optional[int] = None
        self._rows: list[tuple] = []

    def maybe_record(self, ts: int, portfolio: Portfolio, mark: float) -> None:
        if self._next_ts is None or ts >= self._next_ts:
            self.record(ts, portfolio, mark)
            self._next_ts = (ts // self._interval_ns + 1) * self._interval_ns

    def record(self, ts: int, p: Portfolio, mark: float) -> None:
        self._rows.append((ts, p.equity(mark), p.position, p.realized_pnl,
                           p.unrealized_pnl(mark), p.fees, p.funding_pnl, mark))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=self.COLUMNS)
        return df.set_index(pd.to_datetime(df.pop("ts")))


class Scheduler:
    """Time-ordered queue of delayed deliveries to the bot."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Callable[[int], None]]] = []
        self._seq = itertools.count()  # FIFO for equal times

    def schedule(self, ts: int, deliver: Callable[[int], None]) -> None:
        heapq.heappush(self._heap, (ts, next(self._seq), deliver))

    def pop_before(self, until: float) -> Iterable[tuple[int, Callable[[int], None]]]:
        while self._heap and self._heap[0][0] < until:
            ts, _, deliver = heapq.heappop(self._heap)
            yield ts, deliver


class Backtester:
    """funding_rate_scale converts the data rate to one settlement interval
    (settlement hours / hours the data rate is quoted for)."""

    def __init__(self, strategy: Strategy, exchange: SimulatedExchange,
                 latency: FeedLatency = FeedLatency(), sample_interval_ns: int = NS_PER_SECOND,
                 max_book_age_ns: Optional[int] = None, funding_rate_scale: float = 1.0):
        if latency.market_data_ns < 0 or latency.ack_ns < 0:
            raise ValueError("feed latencies must be >= 0")
        if max_book_age_ns is not None and max_book_age_ns <= latency.market_data_ns:
            raise ValueError("max book age must be larger than the market data latency")
        self.exchange = exchange
        self.bot = TradingBot(strategy, OrderManager(exchange), max_book_age_ns)
        self.portfolio = Portfolio()  # true state at the exchange
        self._latency = latency
        self._max_book_age_ns = max_book_age_ns
        self._funding_rate_scale = funding_rate_scale
        self._scheduler = Scheduler()
        self._recorder = EquityRecorder(sample_interval_ns)
        self._fills: list[tuple[Fill, float]] = []
        self._mid_ts: list[int] = []
        self._mids: list[float] = []
        self._book: Optional[BookSnapshot] = None  # true book (used as mark)
        self._funding_rate: Optional[float] = None
        self._market_volume = 0.0
        self._last_ts = 0

    def run(self, events: Iterable[Event]) -> BacktestResult:
        for event in events:
            self._deliver_before(event.ts)
            self._on_market_event(event)
        self._deliver_before(math.inf, record=False)  # bot reactions after the data end are not sampled
        if self._book is None:
            raise ValueError("no order book data")
        self._recorder.record(self._last_ts, self.portfolio, self._book.mid)
        return BacktestResult(self._recorder.to_frame(), self._fills_frame(), self.exchange.stats,
                              self._market_volume, pd.Series(self._mids, index=self._mid_ts, dtype=float))

    def _on_market_event(self, event: Event) -> None:
        mid_before = self._book.mid if self._book else math.nan
        fills = self.exchange.on_event(event)
        for fill in fills:
            self.portfolio.on_fill(fill)
            self._fills.append((fill, mid_before))

        if isinstance(event, BookSnapshot):  # scheduled first: at equal times the bot sees the book before the fill
            self._on_book(event)
        elif isinstance(event, Trade):
            self._market_volume += event.size
        elif isinstance(event, FundingRate):
            self._funding_rate = event.rate
            self._scheduler.schedule(event.ts + self._latency.market_data_ns,
                                     lambda now, r=event.rate: self.bot.on_funding_rate(r, now))
        elif isinstance(event, FundingSettlement) and self._funding_rate is not None and self._book:
            self.portfolio.on_funding(self._funding_rate * self._funding_rate_scale, self._book.mid)
        for fill in fills:
            ack_at = max(fill.ts + self._latency.ack_ns, event.ts)  # never before the current event
            self._scheduler.schedule(ack_at, lambda now, f=fill: self.bot.on_fill_report(f, now))
        self._advance_clock(event.ts)

    def _on_book(self, book: BookSnapshot) -> None:
        if self._book is None or book.mid != self._book.mid:
            self._mid_ts.append(book.ts)
            self._mids.append(book.mid)
        self._book = book
        self._scheduler.schedule(book.ts + self._latency.market_data_ns,
                                 lambda now, b=book: self._deliver_book(b, now))

    def _deliver_book(self, book: BookSnapshot, now: int) -> None:
        self.bot.on_book(book, now)
        if self._max_book_age_ns is not None:  # wake the bot when this book becomes stale
            self._scheduler.schedule(book.ts + self._max_book_age_ns + 1,
                                     lambda later, b=book: self.bot.on_clock(b, later))

    def _deliver_before(self, until: float, record: bool = True) -> None:
        for ts, deliver in self._scheduler.pop_before(until):
            deliver(ts)
            if record:
                self._advance_clock(ts)

    def _advance_clock(self, ts: int) -> None:
        self._last_ts = ts
        if self._book is not None:
            self._recorder.maybe_record(ts, self.portfolio, self._book.mid)

    def _fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(f.ts, f.order_id, f.side.name, f.price, f.qty, f.fee, f.kind.value, mid) for f, mid in self._fills],
            columns=["ts", "order_id", "side", "price", "qty", "fee", "kind", "mid"],
        )
