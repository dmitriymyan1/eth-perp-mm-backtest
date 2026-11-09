"""Unit tests with hand-computed values for portfolio, prices, book and metrics."""
import math

import pandas as pd
import pytest

from mm.core.domain import BookSnapshot, Fill, Side, ceil_to_tick, floor_to_tick, is_better, same_price
from mm.analytics.metrics import daily_pnl, inventory_stats, max_drawdown, newey_west_t_stat, period_pnl, sharpe_ratio
from mm.backtester.portfolio import Portfolio

MINUTES_PER_YEAR = 365 * 24 * 60


def fill(side: Side, price: float, qty: float, fee: float = 0.0) -> Fill:
    return Fill(0, 0, side, price, qty, fee)


def test_portfolio_average_cost_and_position_flip():
    """Buy 1 @ 100, buy 1 @ 102 -> long 2 @ 101.
    Sell 3 @ 103 -> realized 2 * (103 - 101) = 4, now short 1 @ 103.
    Mark 104: unrealized = -1 * (104 - 103) = -1;
    cash = -100 - 102 + 309 = 107, equity = 107 - 104 = 3 = 4 + (-1)."""
    p = Portfolio()
    p.on_fill(fill(Side.BUY, 100, 1))
    p.on_fill(fill(Side.BUY, 102, 1))
    assert (p.position, p.avg_price) == (2, 101)

    p.on_fill(fill(Side.SELL, 103, 3))
    assert (p.position, p.avg_price, p.realized_pnl) == (-1, 103, 4)
    assert p.unrealized_pnl(104) == -1
    assert p.equity(104) == 3


def test_portfolio_funding_sign():
    """Short 2 @ 100, positive rate 0.01 at mark 100: shorts receive 2 * 100 * 0.01 = 2."""
    p = Portfolio()
    p.on_fill(fill(Side.SELL, 100, 2))
    p.on_funding(0.01, 100)
    assert p.funding_pnl == pytest.approx(2.0)


def test_tick_rounding_is_float_safe():
    # 100.05 - 0.05 = 99.99999999999999 in float: must still floor to 100.0
    assert floor_to_tick(100.05 - 0.05, 0.1) == 100.0
    assert ceil_to_tick(100.05 + 0.05, 0.1) == 100.1
    assert floor_to_tick(2146.75, 0.1) == 2146.7


def test_book_visible_qty():
    book = BookSnapshot(0, (100.0, 99.8), (5.0, 7.0), (100.1, 100.2), (3.0, 4.0))
    assert book.visible_qty(Side.BUY, 100.0) == 5.0
    assert book.visible_qty(Side.BUY, 100.05) == 0.0   # better than best bid
    assert book.visible_qty(Side.BUY, 99.9) == 0.0     # empty gap inside visible depth
    assert book.visible_qty(Side.BUY, 99.5) is None    # deeper than visible depth
    assert book.visible_qty(Side.SELL, 100.2) == 4.0


def test_sharpe_drawdown_daily_pnl():
    """Minute equity 0, 1, 3, 2, 4 -> diffs 1, 2, -1, 2: mean 1, std sqrt(2).
    Sharpe = 1 / sqrt(2) * sqrt(525600) ~ 512.64.  Max drawdown = 3 - 2 = 1."""
    idx = pd.date_range("2026-03-19 23:57", periods=5, freq="1min")
    equity = pd.Series([0.0, 1.0, 3.0, 2.0, 4.0], index=idx)

    assert sharpe_ratio(equity.diff().dropna(), MINUTES_PER_YEAR) == pytest.approx(math.sqrt(MINUTES_PER_YEAR / 2))
    assert max_drawdown(equity) == 1.0
    # day 19 ends at 3.0 (23:59), day 20 ends at 4.0 -> daily PnL 3, 1
    assert daily_pnl(equity).tolist() == [3.0, 1.0]


def test_tick_helpers_work_for_tiny_ticks():
    """1.2345678e-5 / 1e-9 = 12345.678 ticks -> floor 12345, ceil 12346.
    (Old code rounded to 8 decimals and gave 1.235e-05.)"""
    assert floor_to_tick(1.2345678e-5, 1e-9) == 1.2345e-05
    assert ceil_to_tick(1.2345678e-5, 1e-9) == 1.2346e-05
    assert not same_price(1.0e-9, 1.5e-9)
    assert is_better(Side.BUY, 1.5e-9, 1.0e-9)


def test_period_pnl_keeps_first_period():
    """Hourly equity at hour ends: 2, 1, 4 -> PnL 2, -1, 3 (first hour counts from 0)."""
    idx = pd.date_range("2026-03-19 00:30", periods=3, freq="1h")
    assert period_pnl(pd.Series([2.0, 1.0, 4.0], index=idx), "1h").tolist() == [2.0, -1.0, 3.0]


def test_inventory_stats_time_weighted_and_exact_max():
    """Buy 1 at t=0, buy 1 at t=10, sell 2 at t=20, end t=40.
    |pos|: 1 for 10, 2 for 10, 0 for 20 -> mean (10 + 20) / 40 = 0.75, max 2."""
    fills = pd.DataFrame({"ts": [0, 10, 20], "side": ["BUY", "BUY", "SELL"], "qty": [1.0, 1.0, 2.0]})
    assert inventory_stats(fills, 0, 40) == (0.75, 2.0)


def test_tick_snapping_at_large_price_over_tick():
    """9999999.986325 = 399999999453 * 2.5e-5 is on the grid and must not move."""
    assert floor_to_tick(9999999.986325, 2.5e-5) == 9999999.986325
    assert ceil_to_tick(9999999.986325, 2.5e-5) == 9999999.986325


def test_newey_west_t_stat_by_hand():
    """x = 1, 3, 2, 4, lags 1: mean 2.5, d = -1.5, .5, -.5, 1.5.
    g0 = 5/4 = 1.25; g1 = (-.75 - .25 - .75)/4 = -0.4375; LRV = 1.25 + 2 * 0.5 * g1 = 0.8125.
    t = 2.5 / sqrt(0.8125 / 4) = 5.5470."""
    assert newey_west_t_stat(pd.Series([1.0, 3.0, 2.0, 4.0]), lags=1) == pytest.approx(2.5 / math.sqrt(0.8125 / 4))


def test_last_interval_funding_settles_at_end_of_data(tmp_path):
    """One day of data: the 24:00 settlement (funding for 23:00-24:00) is applied at 23:59:59.999999999."""
    from mm.data import NS_PER_DAY, NS_PER_HOUR, ParquetDataSource
    from mm.core.domain import FundingSettlement
    day = pd.Timestamp("2026-03-19")
    for folder, frame in {
        "orderbook": pd.DataFrame({"datetime": [day + pd.Timedelta("1s")], "bid_price_1": [100.0], "bid_qty_1": [1.0],
                                   "ask_price_1": [100.1], "ask_qty_1": [1.0]}),
        "trades": pd.DataFrame({"datetime": pd.to_datetime([]), "price": [], "size": [], "is_maker_ask": []}),
        "fundings": pd.DataFrame({"datetime": pd.to_datetime([]), "funding_rate": []}),
    }.items():
        (tmp_path / folder).mkdir()
        frame.to_parquet(tmp_path / folder / "2026-03-19.parquet")
    events = list(ParquetDataSource(tmp_path, levels=1).iter_events(NS_PER_HOUR))
    settlements = [e.ts for e in events if isinstance(e, FundingSettlement)]
    assert settlements[-1] == day.value + NS_PER_DAY - 1
    assert len(settlements) == 25  # 00:00 ... 23:00 and the final one


def test_economics_premium_markout_and_round_trip():
    """Buy 1 @ 100.0 (mid before 100.05) at 0 s, sell 1 @ 100.1 (mid before 100.05) at 1 s.
    Premium: buy (100.05-100)/100*1e4 = 5.0 bp; sell (100.1-100.05)/100.1*1e4 = 4.995 bp.
    Mid path: 100.05 until 0.5 s, then 100.15. Markout 0.1 s of the buy = (100.05-100)/100 = 5 bp.
    FIFO round trip: 1 ETH, held 1 s, gross 0.1 USD, net 0.1 - fees (0.01 + 0.01) = 0.08 USD."""
    import numpy as np
    from mm.analytics.economics import fill_frame, round_trips
    fills = pd.DataFrame({"ts": [0, 10**9], "side": ["BUY", "SELL"], "price": [100.0, 100.1],
                          "qty": [1.0, 1.0], "fee": [0.01, 0.01], "kind": ["at_price"] * 2, "mid": [100.05, 100.05]})
    f = fill_frame(fills, np.array([0, 5 * 10**8]), np.array([100.05, 100.15]))
    assert f["premium_bp"].tolist() == pytest.approx([5.0, 0.05 / 100.1 * 1e4])
    assert f.loc[0, "mo_0.1s"] == pytest.approx(5.0)
    trip = round_trips(f).iloc[0]
    assert (trip["qty"], trip["hold_s"]) == (1.0, 1.0)
    assert trip["gross_usd"] == pytest.approx(0.1)
    assert trip["net_usd"] == pytest.approx(0.08)
