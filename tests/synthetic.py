"""Builders for small hand-made market scenarios (1 book level, times in ms).

`run` goes through the same facade as the CLI (mm.facade.build_backtester), so the tests
also cover the config -> components wiring.
"""
from __future__ import annotations

import pandas as pd

from mm.config import BacktestConfig, FeeConfig, FundingConfig, LatencyConfig, StrategyConfig
from mm.data import NS_PER_HOUR, MarketData, build_events
from mm.backtester.engine import Backtester, BacktestResult
from mm.facade import build_backtester
from mm.backtester.fill_models import FillModel, SimpleFillModel

MS = 1_000_000


def market(books, trades=(), funding=(), end_ms=None) -> MarketData:
    """books:   (ms, bid, bid_qty, ask, ask_qty)
    trades:  (ms, price, size, aggressor) with aggressor 'buy' | 'sell'
    funding: (ms, rate)"""
    book = pd.DataFrame(books, columns=["ms", "bid_price_1", "bid_qty_1", "ask_price_1", "ask_qty_1"])
    trade = pd.DataFrame(trades, columns=["ms", "price", "size", "aggressor"])
    trade["is_maker_ask"] = (trade.pop("aggressor") == "buy").astype(int)
    fund = pd.DataFrame(funding, columns=["ms", "funding_rate"])
    for df in (book, trade, fund):
        df.insert(0, "datetime", pd.to_datetime(df.pop("ms").astype("int64") * MS))
    return MarketData(book, trade, fund, start_ns=0, end_ns=None if end_ms is None else end_ms * MS)


def config(*, latency_ms: float = 0, md_latency_ms: float = 0, ack_latency_ms: float = 0,
           maker_fee: float = 0.0, taker_fee: float = 0.0, half_spread: float = 0.05, lot: float = 1.0,
           max_position: float = 100.0, funding_interval_ns: int = NS_PER_HOUR,
           funding_rate_scale: float = 1.0, max_book_age_ms: float | None = None) -> BacktestConfig:
    """Test defaults: zero latency and fees, a wide position limit, no stale-book timer."""
    interval_hours = funding_interval_ns / NS_PER_HOUR
    return BacktestConfig(
        strategy=StrategyConfig(half_spread=half_spread, lot=lot, max_position=max_position),
        latency=LatencyConfig(market_data_ms=md_latency_ms, order_ms=latency_ms, ack_ms=ack_latency_ms,
                              max_book_age_s=(max_book_age_ms or 0) / 1000),
        fees=FeeConfig(maker_bps=maker_fee * 1e4, taker_bps=taker_fee * 1e4),
        funding=FundingConfig(interval_hours=interval_hours, rate_hours=interval_hours / funding_rate_scale),
    )


def run(data: MarketData, fill_model: FillModel | None = None, **settings) -> tuple[BacktestResult, Backtester]:
    cfg = config(**settings)
    bt = build_backtester(cfg, fill_model=fill_model or SimpleFillModel())
    return bt.run(build_events(data, cfg.funding_interval_ns)), bt


def fills(result: BacktestResult) -> list[tuple[float, str, float, float]]:
    """(ms, side, price, qty) per fill, rounded for exact comparison."""
    return [(row.ts / MS, row.side, round(row.price, 8), round(row.qty, 8))
            for row in result.fills.itertuples()]
