"""Facade: builds and runs a backtest from one BacktestConfig, so callers never wire
strategy, fill model, exchange, latencies and data source by hand."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .backtester.engine import Backtester, BacktestResult, FeedLatency
from .backtester.exchange import SimulatedExchange
from .backtester.fill_models import (CancelRule, CrossRule, FillModel, FuturePrice, QueueAwareFillModel,
                                     SimpleFillModel)
from .config import BacktestConfig
from .core.domain import BookSnapshot
from .data import MarketData, ParquetDataSource, build_events
from .signals import SignalTable
from .strategy import FairValueQuoter, MidQuoter, Strategy

FairPrice = Callable[[BookSnapshot], float]


def build_fill_model(cfg: BacktestConfig, fair_price: Optional[FairPrice] = None) -> FillModel:
    f = cfg.fills
    if f.model == "queue":
        return QueueAwareFillModel(CrossRule(f.cross_rule), CancelRule(f.cancel_rule), cfg.pickoff_fee, fair_price)
    return SimpleFillModel(CrossRule(f.cross_rule), cfg.pickoff_fee, fair_price)


def build_backtester(cfg: BacktestConfig, fill_model: Optional[FillModel] = None,
                     fair_price: Optional[FairPrice] = None) -> Backtester:
    lat = cfg.latency
    exchange = SimulatedExchange(fill_model or build_fill_model(cfg, fair_price), latency_ns=cfg.ms_to_ns(lat.order_ms),
                                 maker_fee=cfg.maker_fee, taker_fee=cfg.taker_fee)
    strategy = build_strategy(cfg)
    feed = FeedLatency(market_data_ns=cfg.ms_to_ns(lat.market_data_ms), ack_ns=cfg.ms_to_ns(lat.ack_ms))
    return Backtester(strategy, exchange, feed,
                      sample_interval_ns=round(cfg.output.sample_interval_s * 1e9),
                      max_book_age_ns=cfg.max_book_age_ns, funding_rate_scale=cfg.funding_rate_scale)


def build_strategy(cfg: BacktestConfig) -> Strategy:
    s = cfg.strategy
    if s.name == "fair_value":
        signal = SignalTable.from_parquet(s.signal_path) if s.signal_path else None
        return FairValueQuoter(cfg.data.tick, s.half_spread, s.lot, cfg.max_position, signal=signal,
                               inventory_skew=s.inventory_skew, signal_cap_bp=s.signal_cap_bp,
                               pull_threshold_bp=s.pull_threshold_bp,
                               funding_target_position=s.funding_target_position,
                               funding_threshold_bp=s.funding_threshold_bp,
                               funding_fair_hours=s.funding_fair_hours,
                               signal_deadband_bp=s.signal_deadband_bp,
                               signal_release_ratio=s.signal_release_ratio)
    return MidQuoter(cfg.data.tick, s.half_spread, s.lot, cfg.max_position)


def run_backtest(cfg: BacktestConfig, market: Optional[MarketData] = None) -> BacktestResult:
    """Run on the parquet data of cfg.data.path, or on an in-memory `market` (tests)."""
    if market is not None:
        events = build_events(market, cfg.funding_interval_ns)
        mid_path = _mid_path(market.book)
    else:
        source = ParquetDataSource(cfg.data.path, levels=cfg.data.levels)
        days = cfg.data.day_list
        events = source.iter_events(cfg.funding_interval_ns, days=days)
        mid_path = source.mid_path(days) if _needs_future_price(cfg) else None
    return build_backtester(cfg, fair_price=_fair_price(cfg, mid_path)).run(events)


def _needs_future_price(cfg: BacktestConfig) -> bool:
    return cfg.fills.cross_rule == CrossRule.PROFITABLE.value and cfg.fills.pickoff_fair_ms > 0


def _fair_price(cfg: BacktestConfig, mid_path) -> Optional[FairPrice]:
    if not _needs_future_price(cfg) or mid_path is None:
        return None  # fill model default: mid of the crossing book
    ts, mids = mid_path
    return FuturePrice(ts, mids, lead_ns=cfg.ms_to_ns(cfg.fills.pickoff_fair_ms))


def _mid_path(book) -> tuple[np.ndarray, np.ndarray]:
    ts = book["datetime"].astype("datetime64[ns]").astype(np.int64).to_numpy()
    return ts, ((book["bid_price_1"] + book["ask_price_1"]) / 2).to_numpy()
