"""Market data loading and conversion into one time-ordered event stream."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ..core.domain import BookSnapshot, Event, FundingRate, FundingSettlement, Side, Trade

NS_PER_HOUR = 3_600 * 10**9
NS_PER_DAY = 24 * NS_PER_HOUR

# Tie-break for equal timestamps. A trade goes before a book snapshot with the same
# timestamp: the snapshot already shows the post-trade state (avoids double-counting queue).
_SETTLEMENT, _FUNDING, _TRADE, _BOOK = range(4)


@dataclass(frozen=True)
class MarketData:
    """One chunk of market data (raw parquet schema).

    book:    datetime, bid_price_i, bid_qty_i, ask_price_i, ask_qty_i  (i = 1..N, 1 = best)
    trades:  datetime, price, size, is_maker_ask (1 = buyer is the aggressor)
    funding: datetime, funding_rate
    start_ns / end_ns: window for funding settlements (default: data range).
    """
    book: pd.DataFrame
    trades: pd.DataFrame
    funding: pd.DataFrame
    start_ns: Optional[int] = None
    end_ns: Optional[int] = None


class ParquetDataSource:
    """Reads `<root>/{orderbook,trades,fundings}/<YYYY-MM-DD>.parquet`."""

    MAX_LEVELS = 20

    def __init__(self, root: Path, levels: int = 5):
        if not 1 <= levels <= self.MAX_LEVELS:
            raise ValueError(f"levels must be in 1..{self.MAX_LEVELS}")
        self.root = Path(root)
        self.levels = levels

    def days(self) -> list[str]:
        days = sorted(p.stem for p in (self.root / "orderbook").glob("*.parquet"))
        if not days:
            raise ValueError(f"no order book files in {self.root / 'orderbook'}")
        return days

    def load_day(self, day: str) -> MarketData:
        book = pd.read_parquet(self.root / "orderbook" / f"{day}.parquet", columns=self._book_columns())
        start = pd.Timestamp(day).value
        return MarketData(
            book=book,
            trades=pd.read_parquet(self.root / "trades" / f"{day}.parquet"),
            funding=pd.read_parquet(self.root / "fundings" / f"{day}.parquet"),
            start_ns=start,
            end_ns=start + NS_PER_DAY - 1,
        )

    def mid_path(self, days: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray]:
        """(ts, mid) of every snapshot of the given days (default: all); level 1 only."""
        books = [pd.read_parquet(self.root / "orderbook" / f"{day}.parquet",
                                 columns=["datetime", "bid_price_1", "ask_price_1"]) for day in days or self.days()]
        book = pd.concat(books)
        return _to_ns(book["datetime"]), ((book["bid_price_1"] + book["ask_price_1"]) / 2).to_numpy()

    def iter_events(self, funding_interval_ns: int, days: Optional[list[str]] = None) -> Iterator[Event]:
        """Loads one day at a time to keep memory low. The settlement at the midnight after the
        last day (funding for the last interval) is applied at the last ns of that day."""
        end_ns = None
        for day in days or self.days():
            data = self.load_day(day)
            end_ns = data.end_ns
            yield from build_events(data, funding_interval_ns)
        if end_ns is not None and (end_ns + 1) % funding_interval_ns == 0:
            yield FundingSettlement(end_ns)

    def _book_columns(self) -> list[str]:
        return ["datetime"] + [f"{side}_{field}_{i}" for i in range(1, self.levels + 1)
                               for side in ("bid", "ask") for field in ("price", "qty")]


def build_events(data: MarketData, funding_interval_ns: int) -> Iterator[Event]:
    if funding_interval_ns <= 0:
        raise ValueError("funding interval must be > 0")
    book_ts = _to_ns(data.book["datetime"])
    trade_ts = _to_ns(data.trades["datetime"])
    funding_ts = _to_ns(data.funding["datetime"])
    settlement_ts = _settlement_times(data, funding_interval_ns, book_ts, trade_ts, funding_ts)

    levels = range(1, _count_levels(data.book) + 1)
    bid_px, bid_qty, ask_px, ask_qty = (
        data.book[[f"{name}_{i}" for i in levels]].to_numpy(float)
        for name in ("bid_price", "bid_qty", "ask_price", "ask_qty")
    )
    trade_px = data.trades["price"].to_numpy(float)
    trade_size = data.trades["size"].to_numpy(float)
    buyer_aggressor = data.trades["is_maker_ask"].to_numpy() == 1
    rates = data.funding["funding_rate"].to_numpy(float)

    factories = {
        _SETTLEMENT: lambda i: FundingSettlement(int(settlement_ts[i])),
        _FUNDING: lambda i: FundingRate(int(funding_ts[i]), float(rates[i])),
        _BOOK: lambda i: BookSnapshot(int(book_ts[i]),
                                      tuple(bid_px[i].tolist()), tuple(bid_qty[i].tolist()),
                                      tuple(ask_px[i].tolist()), tuple(ask_qty[i].tolist())),
        _TRADE: lambda i: Trade(int(trade_ts[i]), float(trade_px[i]), float(trade_size[i]),
                                Side.BUY if buyer_aggressor[i] else Side.SELL),
    }
    sources = [(settlement_ts, _SETTLEMENT), (funding_ts, _FUNDING), (book_ts, _BOOK), (trade_ts, _TRADE)]
    all_ts = np.concatenate([ts for ts, _ in sources])
    kinds = np.concatenate([np.full(len(ts), kind) for ts, kind in sources])
    rows = np.concatenate([np.arange(len(ts)) for ts, _ in sources])

    for k in np.lexsort((kinds, all_ts)):  # primary: time, secondary: kind (stable)
        yield factories[kinds[k]](rows[k])


def _settlement_times(data, interval_ns, *timestamps) -> np.ndarray:
    present = [ts for ts in timestamps if len(ts)]
    start = data.start_ns if data.start_ns is not None else min(int(ts.min()) for ts in present)
    end = data.end_ns if data.end_ns is not None else max(int(ts.max()) for ts in present)
    first = -(-start // interval_ns) * interval_ns  # ceil to the interval grid
    return np.arange(first, end + 1, interval_ns, dtype=np.int64)


def _to_ns(column: pd.Series) -> np.ndarray:
    return pd.to_datetime(column).to_numpy("datetime64[ns]").astype(np.int64)


def _count_levels(book: pd.DataFrame) -> int:
    return sum(1 for c in book.columns if c.startswith("bid_price_"))
