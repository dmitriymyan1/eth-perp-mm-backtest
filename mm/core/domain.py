"""Core domain objects shared by all backtester components."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional, Union

QTY_EPS = 1e-12
_REL_PRICE_EPS = 1e-12  # valid while tick / price >> 1e-12 (true for all real instruments)


class Side(Enum):
    BUY = 1
    SELL = -1

    @property
    def sign(self) -> int:
        return self.value

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class FillKind(str, Enum):
    AT_PRICE = "at_price"  # a trade printed at our price (after the queue ahead)
    THROUGH = "through"    # a trade printed through our price (level swept)
    CROSS = "cross"        # the book crossed our price (see CrossRule)
    TAKER = "taker"        # the order crossed the book on arrival


# ---------------------------------------------------------------- price helpers

def _price_eps(a: float, b: float) -> float:
    return _REL_PRICE_EPS * max(1.0, abs(a), abs(b))


def same_price(a: float, b: float) -> bool:
    return abs(a - b) <= _price_eps(a, b)


def is_better(side: Side, a: float, b: float) -> bool:
    """True if price `a` is strictly more aggressive than `b` for an order on `side`
    (higher for a bid, lower for an ask)."""
    return side.sign * (a - b) > _price_eps(a, b)


def _tick_decimals(tick: float) -> int:
    return max(0, -Decimal(repr(tick)).as_tuple().exponent)


def _to_tick(price: float, tick: float, direction) -> float:
    ticks = price / tick
    nearest = round(ticks)
    on_grid = abs(ticks - nearest) <= 1e-9 * max(1.0, abs(ticks))  # float noise, not a real offset
    return round((nearest if on_grid else direction(ticks)) * tick, _tick_decimals(tick))


def floor_to_tick(price: float, tick: float) -> float:
    return _to_tick(price, tick, math.floor)


def ceil_to_tick(price: float, tick: float) -> float:
    return _to_tick(price, tick, math.ceil)


# ---------------------------------------------------------------- market events

@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Top-N levels of the order book, best level first."""
    ts: int
    bid_prices: tuple[float, ...]
    bid_sizes: tuple[float, ...]
    ask_prices: tuple[float, ...]
    ask_sizes: tuple[float, ...]

    @property
    def best_bid(self) -> float:
        return self.bid_prices[0]

    @property
    def best_ask(self) -> float:
        return self.ask_prices[0]

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    def best(self, side: Side) -> float:
        return self.best_bid if side is Side.BUY else self.best_ask

    def visible_qty(self, side: Side, price: float) -> Optional[float]:
        """Quantity resting at `price` on `side`.

        0.0  -> price is better than a visible level but no one rests there;
        None -> price is deeper than the visible depth (unknown).
        """
        prices, sizes = (
            (self.bid_prices, self.bid_sizes) if side is Side.BUY
            else (self.ask_prices, self.ask_sizes)
        )
        for level_price, level_qty in zip(prices, sizes):
            if same_price(price, level_price):
                return level_qty
            if is_better(side, price, level_price):
                return 0.0
        return None


@dataclass(frozen=True, slots=True)
class Trade:
    ts: int
    price: float
    size: float
    aggressor: Side


@dataclass(frozen=True, slots=True)
class FundingRate:
    """Latest published funding rate (applied at the next settlement)."""
    ts: int
    rate: float


@dataclass(frozen=True, slots=True)
class FundingSettlement:
    """Moment when funding is exchanged between longs and shorts."""
    ts: int


Event = Union[BookSnapshot, Trade, FundingRate, FundingSettlement]


# ---------------------------------------------------------------- trading objects

@dataclass(slots=True)
class Order:
    id: int
    side: Side
    price: float
    size: float
    active_at: int                  # time the order reaches the exchange
    filled: float = 0.0
    cancel_at: Optional[int] = None  # time the cancel reaches the exchange
    is_active: bool = False
    # Fill-model state (exchange side only; the bot never reads it)
    queue_ahead: float = 0.0         # volume in front of us at our price level
    level_qty: float = math.inf      # last seen visible qty at our price (inf = unknown)
    touched: bool = False            # an opposite trade reached our price since the last book

    @property
    def remaining(self) -> float:
        return self.size - self.filled

    @property
    def is_filled(self) -> bool:
        return self.remaining <= QTY_EPS


@dataclass(frozen=True, slots=True)
class Fill:
    ts: int
    order_id: int
    side: Side
    price: float
    qty: float
    fee: float
    kind: FillKind = FillKind.AT_PRICE


@dataclass(frozen=True, slots=True)
class Quote:
    """Desired two-sided quote. `None` price means: do not quote this side."""
    bid: Optional[float]
    ask: Optional[float]
    size: float

    def price(self, side: Side) -> Optional[float]:
        return self.bid if side is Side.BUY else self.ask
