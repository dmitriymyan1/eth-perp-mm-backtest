"""Manual end-to-end scenarios: each expected value is computed by hand in the docstring.

Default setup: tick 0.1, lot 1, latency 0, quote = mid +/- 0.05
-> with book 100.0 / 100.1 we quote bid 100.0, ask 100.1.
A new order is live from the next event (no look-ahead on the event that created it).
"""
import math

import pandas as pd
import pytest

from mm.core.domain import BookSnapshot, Side, Trade
from mm.backtester.exchange import SimulatedExchange
from mm.backtester.fill_models import CancelRule, CrossRule, QueueAwareFillModel, SimpleFillModel
from mm.analytics.metrics import compute_metrics

from synthetic import MS, fills, market, run

BOOK = (0, 100.0, 5.0, 100.1, 5.0)


def test_1_round_trip_spread_capture_with_fees():
    """Buy 1 @ 100.0, sell 1 @ 100.1, maker fee 2 bps.
    realized = 0.1;  fees = 0.0002 * (100.0 + 100.1) = 0.04002;  total = 0.05998.
    Orders: 2 initial + 1 new bid after t=1 + 1 new ask after t=2 = 4 submitted,
    2 filled -> fill rate 0.5."""
    data = market([BOOK], trades=[(1, 100.0, 1.0, "sell"), (2, 100.1, 1.0, "buy")])
    result, bt = run(data, SimpleFillModel(), maker_fee=0.0002)

    assert fills(result) == [(1, "BUY", 100.0, 1.0), (2, "SELL", 100.1, 1.0)]
    m = compute_metrics(result)
    assert m["realized_pnl"] == pytest.approx(0.1)
    assert m["fees"] == pytest.approx(0.04002)
    assert m["total_pnl"] == pytest.approx(0.05998)
    assert bt.portfolio.position == 0
    assert (m["orders_submitted"], m["fills"], m["filled_orders"]) == (4, 2, 2)
    assert m["fill_rate"] == pytest.approx(0.5)


def test_2_partial_fill_unrealized_pnl_and_requote():
    """half_spread 0.25: mid 100.05 -> bid 99.8, ask 100.3.
    t=1 sell 0.4 @ 99.8 -> long 0.4 @ 99.8 (partial: 0.6 left on the bid).
    t=2 book 100.2/100.3 -> mid 100.25 -> new quote 100.0 / 100.5, both old orders cancelled.
    unrealized = 0.4 * (100.25 - 99.8) = 0.18 = total PnL.
    volume fill rate = 0.4 filled / 4 lots submitted = 0.1."""
    data = market([BOOK, (2, 100.2, 5.0, 100.3, 5.0), (3, 100.2, 5.0, 100.3, 5.0)],
                  trades=[(1, 99.8, 0.4, "sell")])
    result, bt = run(data, SimpleFillModel(), half_spread=0.25)

    assert fills(result) == [(1, "BUY", 99.8, 0.4)]
    m = compute_metrics(result)
    assert m["unrealized_pnl"] == pytest.approx(0.18)
    assert m["total_pnl"] == pytest.approx(0.18)
    assert m["realized_pnl"] == 0
    assert (m["orders_submitted"], m["orders_cancelled"]) == (4, 2)
    assert m["volume_fill_rate"] == pytest.approx(0.1)
    assert sorted(o.price for o in bt.exchange.open_orders()) == [100.0, 100.5]


def test_3_queue_position_delays_fill_vs_simple_model():
    """Bid 100.0 joins behind 5.0 visible.
    Queue model: t=1 sell 3.0 -> queue 2.0, no fill; t=2 book shows 2.0 (consistent, no change);
                 t=3 sell 2.5 -> 0.5 fills us, queue 0; t=5 sell 1.0 -> remaining 0.5 fills.
                 Position = 1.0.
    Simple model: every sell trade fills a full lot (min(1, size)) -> position = 3.0."""
    books = [BOOK, (2, 100.0, 2.0, 100.1, 5.0)]
    trades = [(1, 100.0, 3.0, "sell"), (3, 100.0, 2.5, "sell"), (5, 100.0, 1.0, "sell")]

    queue, bt_queue = run(market(books, trades), QueueAwareFillModel())
    assert fills(queue) == [(3, "BUY", 100.0, 0.5), (5, "BUY", 100.0, 0.5)]
    assert bt_queue.portfolio.position == pytest.approx(1.0)

    simple, bt_simple = run(market(books, trades), SimpleFillModel())
    assert [f[3] for f in fills(simple)] == [1.0, 1.0, 1.0]
    assert bt_simple.portfolio.position == pytest.approx(3.0)


def test_4_cancels_ahead_move_us_forward_new_orders_do_not_move_us_back():
    """Queue starts at 5.0. t=1 level shrinks to 2.0 (cancels) -> queue 2.0.
    t=2 sell 2.5 -> 0.5 fill, queue 0. t=3 level grows to 10.0 -> queue stays 0.
    t=4 sell 0.3 -> 0.3 fill. Total 0.8."""
    books = [BOOK, (1, 100.0, 2.0, 100.1, 5.0), (3, 100.0, 10.0, 100.1, 5.0)]
    trades = [(2, 100.0, 2.5, "sell"), (4, 100.0, 0.3, "sell")]
    result, _ = run(market(books, trades), QueueAwareFillModel())

    assert fills(result) == [(2, "BUY", 100.0, 0.5), (4, "BUY", 100.0, 0.3)]


def test_5_trade_through_our_price_fills_up_to_the_through_volume():
    """Queue 5.0 at 100.0, a sell prints 0.7 at 99.9 -> our level was swept, queue ignored.
    The aggressor has a fixed size: had our bid rested at 100.0, it would have traded the
    0.7 with us first -> fill 0.7 at OUR price 100.0 (not the whole lot)."""
    result, _ = run(market([BOOK], trades=[(1, 99.9, 0.7, "sell")]), QueueAwareFillModel())
    assert fills(result) == [(1, "BUY", 100.0, 0.7)]


def test_5b_sweep_prints_through_our_price_add_up():
    """One sweep at t=1: 5.0 @ 100.0 (eats the queue 5.0), then 0.4 and 0.3 @ 99.9 (through).
    Fills: 0.4 + 0.3 = 0.7 at 100.0."""
    trades = [(1, 100.0, 5.0, "sell"), (1, 99.9, 0.4, "sell"), (1, 99.9, 0.3, "sell")]
    result, _ = run(market([BOOK], trades), QueueAwareFillModel())
    assert fills(result) == [(1, "BUY", 100.0, 0.4), (1, "BUY", 100.0, 0.3)]


def test_6_book_crossing_our_order_fills_it_in_always_mode():
    """Our ask 100.1. t=1 book jumps to 100.3/100.4 -> best bid >= our ask -> ask fully filled at 100.1.
    Short 1 @ 100.1, mark 100.35 -> unrealized = -0.25 (adverse selection)."""
    data = market([BOOK, (1, 100.3, 5.0, 100.4, 5.0)])
    result, bt = run(data, QueueAwareFillModel(cross_rule=CrossRule.ALWAYS))

    assert fills(result) == [(1, "SELL", 100.1, 1.0)]
    assert bt.portfolio.position == -1.0
    assert compute_metrics(result)["total_pnl"] == pytest.approx(-0.25)


def test_6b_book_cross_without_trade_does_not_fill_in_never_mode():
    """Same book jump as test 6, cross rule NEVER -> no trade print -> no fill;
    the stale ask 100.1 is cancelled and re-quoted at 100.4."""
    data = market([BOOK, (1, 100.3, 5.0, 100.4, 5.0), (2, 100.3, 5.0, 100.4, 5.0)])
    result, bt = run(data, QueueAwareFillModel(cross_rule=CrossRule.NEVER))

    assert fills(result) == []
    assert sorted(o.price for o in bt.exchange.open_orders()) == [100.3, 100.4]


def test_7_latency_on_submit_and_cancel():
    """Latency 10 ms, half_spread 0.25 -> bid 99.8 sent at 0, live at 10.
    t=5  sell @ 99.8 -> order not live yet, no fill.
    t=20 book 99.9/100.0 -> new bid 99.7; cancel of 99.8 sent at 20, arrives at 30.
    t=25 sell @ 99.8 -> old bid still live -> fill 1 @ 99.8.
    t=40 sell @ 99.7 -> new bid (live from 30) fills 1 @ 99.7."""
    books = [BOOK, (20, 99.9, 5.0, 100.0, 5.0), (35, 99.9, 5.0, 100.0, 5.0)]
    trades = [(5, 99.8, 1.0, "sell"), (25, 99.8, 1.0, "sell"), (40, 99.7, 1.0, "sell")]
    result, _ = run(market(books, trades), SimpleFillModel(), latency_ms=10, half_spread=0.25)

    assert fills(result) == [(25, "BUY", 99.8, 1.0), (40, "BUY", 99.7, 1.0)]


def test_8_max_position_switches_off_bid():
    """max_position 1: after a buy of 1 the bid is not re-quoted, so the 2nd sell does not fill."""
    data = market([BOOK], trades=[(1, 100.0, 1.0, "sell"), (2, 100.0, 1.0, "sell")])
    result, bt = run(data, SimpleFillModel(), max_position=1.0)

    assert fills(result) == [(1, "BUY", 100.0, 1.0)]
    assert [o.side.name for o in bt.exchange.open_orders()] == ["SELL"]


def test_9_funding_settlement():
    """Long 1 @ 100.0, rate 0.001 published at 2 s, settlement every hour.
    At 1 h: payment = -1 * mid 100.05 * 0.001 = -0.10005.
    total = unrealized 0.05 + funding -0.10005 = -0.05005."""
    hour_ms = 3_600_000
    data = market([BOOK, (hour_ms + 1000, 100.0, 5.0, 100.1, 5.0)],
                  trades=[(1000, 100.0, 1.0, "sell")], funding=[(2000, 0.001)])
    result, _ = run(data, SimpleFillModel(), max_position=1.0)

    m = compute_metrics(result)
    assert m["funding_pnl"] == pytest.approx(-0.10005)
    assert m["total_pnl"] == pytest.approx(-0.05005)


def test_10_same_timestamp_trade_is_processed_before_book():
    """t=1: sell 3.0 and a snapshot showing the post-trade level 2.0.
    Trade first: queue 5 -> 2, then book min(2, 2) = 2 -> no fill.
    (Book first would give queue 2, then the trade 3.0 fills 1.0 -> double counting.)
    t=2 sell 2.5 -> fill 0.5."""
    books = [BOOK, (1, 100.0, 2.0, 100.1, 5.0)]
    trades = [(1, 100.0, 3.0, "sell"), (2, 100.0, 2.5, "sell")]
    result, _ = run(market(books, trades), QueueAwareFillModel())

    assert fills(result) == [(2, "BUY", 100.0, 0.5)]


def test_11_pnl_identity_holds_on_every_sample():
    """equity == realized + unrealized - fees + funding at every recorded point."""
    books = [BOOK, (10, 100.1, 3.0, 100.2, 4.0), (20, 99.9, 2.0, 100.0, 6.0), (30, 100.0, 5.0, 100.1, 5.0)]
    trades = [(5, 100.0, 2.0, "sell"), (12, 100.2, 0.5, "buy"), (25, 99.9, 1.5, "sell"), (32, 100.1, 3.0, "buy")]
    data = market(books, trades, funding=[(1, 0.0005)])
    result, _ = run(data, QueueAwareFillModel(), maker_fee=0.0001)

    eq = result.equity
    rebuilt = eq["realized"] + eq["unrealized"] - eq["fees"] + eq["funding"]
    assert (eq["equity"] - rebuilt).abs().max() < 1e-9
    assert not math.isnan(eq["equity"].iloc[-1])


def test_12_market_data_latency_leaves_a_stale_quote_to_be_picked_off():
    """Trades-only fills (no cross rule) to isolate the effect of a late book.
    t=20 book jumps to 100.3/100.4; t=22 a buyer lifts 1.0 at 100.4.
    md latency 0: bot re-quotes at 20 -> ask 100.4 -> fill at 100.4.
    md latency 5: bot sees the jump at 25 -> ask 100.1 is still live at 22
                  -> trade through it -> fill at 100.1 (0.3 worse).
    Also with md 5: the first orders go out at 5, so the sell at t=3 does not fill."""
    books = [BOOK, (20, 100.3, 5.0, 100.4, 5.0)]
    trades = [(3, 100.0, 1.0, "sell"), (22, 100.4, 1.0, "buy")]

    fast, _ = run(market(books, trades), SimpleFillModel(cross_rule=CrossRule.NEVER))
    assert fills(fast) == [(3, "BUY", 100.0, 1.0), (22, "SELL", 100.4, 1.0)]

    slow, _ = run(market(books, trades), SimpleFillModel(cross_rule=CrossRule.NEVER), md_latency_ms=5)
    assert fills(slow) == [(22, "SELL", 100.1, 1.0)]


def test_13_ack_latency_lets_the_bot_break_its_position_limit():
    """max_position 1, ack latency 5. Trades-only fills.
    t=1 sell fills our bid 1 @ 100.0 -> true position 1, bot learns it only at 6.
    t=3 book moves to 100.1/100.2 -> bot still thinks position 0 -> new bid 100.1.
    t=4 sell 1 @ 100.1 fills it -> true position 2 > limit 1.
    After the reports (t>=6) the bot knows position 2 and quotes no bid."""
    books = [BOOK, (3, 100.1, 5.0, 100.2, 5.0), (10, 100.1, 5.0, 100.2, 5.0)]
    trades = [(1, 100.0, 1.0, "sell"), (4, 100.1, 1.0, "sell")]
    result, bt = run(market(books, trades), SimpleFillModel(cross_rule=CrossRule.NEVER),
                     ack_latency_ms=5, max_position=1.0)

    assert fills(result) == [(1, "BUY", 100.0, 1.0), (4, "BUY", 100.1, 1.0)]
    assert bt.portfolio.position == 2.0 and bt.bot.position == 2.0
    assert [o.side.name for o in bt.exchange.open_orders()] == ["SELL"]


def test_14_hourly_funding_uses_last_rate_of_each_hour():
    """Long 1 @ 100.0 (mid 100.05), hourly settlement, hourly rate (no /8).
    1 h: last rate 0.001 -> -1 * 100.05 * 0.001 = -0.10005
    2 h: last rate 0.002 (the 0.003 at 2h+1s comes after settlement) -> -0.2001
    funding = -0.30015."""
    h = 3_600_000
    data = market([BOOK, (2 * h + 2000, 100.0, 5.0, 100.1, 5.0)],
                  trades=[(1000, 100.0, 1.0, "sell")],
                  funding=[(2000, 0.0005), (h - 1000, 0.001), (h + 1000, 0.002), (2 * h + 1000, 0.003)])
    result, _ = run(data, SimpleFillModel(), max_position=1.0)

    assert compute_metrics(result)["funding_pnl"] == pytest.approx(-0.30015)


def test_15_confirmed_cross_needs_trades_that_cleared_the_queue():
    """CONFIRMED (default), queue 5.0 ahead of our ask 100.1. Book jumps 100.3/100.4 at t=2.
    a) no trade before the jump              -> no fill (repricing makers do not take us).
    b) buy 0.5 @ 100.1: queue 4.5 still ahead -> no fill (the trades did not reach us).
    c) buy 5.2 @ 100.1: queue cleared, 0.2 fills us; the jump fills the rest 0.8."""
    jump = [BOOK, (2, 100.3, 5.0, 100.4, 5.0)]
    quiet, _ = run(market(jump), QueueAwareFillModel())
    assert fills(quiet) == []

    small, _ = run(market(jump, trades=[(1, 100.1, 0.5, "buy")]), QueueAwareFillModel())
    assert fills(small) == []

    big, _ = run(market(jump, trades=[(1, 100.1, 5.2, "buy")]), QueueAwareFillModel())
    assert fills(big) == [(1, "SELL", 100.1, 0.2), (2, "SELL", 100.1, 0.8)]


def test_16_one_print_is_not_counted_twice_across_our_orders():
    """Bids 100.0 and 99.9 (1.0 each), sell print 1.0 @ 99.9.
    Priority: 100.0 first -> traded through -> full fill 1.0, uses the whole print.
    99.9 at our price -> 0.0 of the print is left -> no fill. Total 1.0, not 2.0."""
    exchange = SimulatedExchange(SimpleFillModel())
    exchange.on_event(BookSnapshot(0, (100.0,), (0.0,), (100.1,), (5.0,)))
    exchange.submit(Side.BUY, 99.9, 1.0, 0)   # older order, worse price
    exchange.submit(Side.BUY, 100.0, 1.0, 0)
    got = exchange.on_event(Trade(1, 99.9, 1.0, Side.SELL))
    assert [(f.price, f.qty) for f in got] == [(100.0, 1.0)]


def test_17_sweep_with_one_timestamp_is_atomic_even_with_zero_latency():
    """All latencies 0. Three sell prints 1.0 @ 100.0 at t=10 (one aggressor).
    The bid fills on the 1st print; the bot can re-quote only after the whole sweep,
    so the 2nd and 3rd prints do not fill a new bid. Expected 1 fill."""
    trades = [(10, 100.0, 1.0, "sell")] * 3
    result, _ = run(market([BOOK], trades), SimpleFillModel())
    assert fills(result) == [(10, "BUY", 100.0, 1.0)]


@pytest.mark.parametrize("rule, expected_queue", [
    (CancelRule.BACK, 10.0),                  # cancels behind us first: 10 stays ahead
    (CancelRule.FRONT, 5.0),                  # cancels ahead of us first: 10 - 5
    (CancelRule.PRO_RATA, 10 - 5 * 10 / 15),  # 5 cancelled x share ahead 10/15 -> 6.667
])
def test_18_cancel_rules(rule, expected_queue):
    """Join behind 10.0; level grows to 15.0 (5 join behind); level shrinks to 10.0 (5 cancel)."""
    exchange = SimulatedExchange(QueueAwareFillModel(cancel_rule=rule))
    book = lambda ts, qty: BookSnapshot(ts, (100.0,), (qty,), (100.1,), (5.0,))
    exchange.on_event(book(0, 10.0))
    order = exchange.submit(Side.BUY, 100.0, 1.0, 0)
    for ts, qty in [(1, 10.0), (2, 15.0), (3, 10.0)]:
        exchange.on_event(book(ts, qty))
    assert order.queue_ahead == pytest.approx(expected_queue)


def test_19_stale_book_pulls_quotes_on_a_timer():
    """Max book age 5 s. Book at 0, then no book until 20 s.
    The bot wakes at 5 s + 1 ns and cancels both quotes (latency 0 -> gone at the next event).
    Trades at 7 s and 11 s find no orders. At 20 s a fresh book -> quotes again; a sell at 21 s fills."""
    data = market([BOOK, (20_000, 100.0, 5.0, 100.1, 5.0)],
                  trades=[(7_000, 100.0, 1.0, "sell"), (11_000, 100.1, 1.0, "buy"),
                          (21_000, 100.0, 1.0, "sell")])
    result, _ = run(data, SimpleFillModel(), max_book_age_ms=5_000)

    assert fills(result) == [(21_000, "BUY", 100.0, 1.0)]


def test_20_order_that_crosses_on_arrival_trades_as_taker():
    """Latency 10 ms: ask 100.1 sent at 0, arrives at 10. At 5 the book jumps to 100.3/100.4.
    On arrival (processed at the next event, t=12) the ask crosses the last book (bid 100.3)
    -> taker fill at 100.3 (opposite best) stamped at the ARRIVAL time 10,
    taker fee 4.5 bp = 100.3 * 0.00045 = 0.0451350."""
    data = market([BOOK, (5, 100.3, 5.0, 100.4, 5.0)], trades=[(12, 100.5, 0.1, "buy")])
    result, bt = run(data, SimpleFillModel(cross_rule=CrossRule.NEVER), latency_ms=10,
                     maker_fee=0.00015, taker_fee=0.00045)

    assert fills(result) == [(10, "SELL", 100.3, 1.0)]
    assert result.fills["kind"].tolist() == ["taker"]
    assert result.fills["fee"].iloc[0] == pytest.approx(0.045135)
    assert bt.exchange.stats.taker_fills == 1


def test_21_funding_rate_scale_for_longer_settlement_interval():
    """Hourly data rate, settlement every 2 h -> scale 2. Long 1 @ 100.0, mid 100.05.
    At 2 h: last rate 0.001 -> payment = -1 * 100.05 * 0.001 * 2 = -0.2001."""
    h = 3_600_000
    data = market([BOOK, (2 * h + 1000, 100.0, 5.0, 100.1, 5.0)],
                  trades=[(1000, 100.0, 1.0, "sell")], funding=[(h + 1000, 0.001)])
    result, _ = run(data, SimpleFillModel(), max_position=1.0,
                    funding_interval_ns=2 * h * MS, funding_rate_scale=2.0)

    assert compute_metrics(result)["funding_pnl"] == pytest.approx(-0.2001)


def test_22_pnl_breakdown_round_trip():
    """Test 1 numbers: buy 100.0 and sell 100.1, mid 100.05 at both fills, fee 2 bp.
    notional 200.1; spread capture = (0.05 + 0.05) / 200.1 * 1e4 = 4.99750 bp;
    fees = 0.04002 / 200.1 * 1e4 = 2.0 bp; pnl = 0.05998 / 200.1 * 1e4 = 2.99750 bp;
    inventory = pnl - spread + fees - funding = 0."""
    data = market([BOOK], trades=[(1, 100.0, 1.0, "sell"), (2, 100.1, 1.0, "buy")])
    result, _ = run(data, SimpleFillModel(), maker_fee=0.0002)
    m = compute_metrics(result)

    assert m["spread_capture_bps"] == pytest.approx(0.1 / 200.1 * 1e4)
    assert m["fees_bps"] == pytest.approx(2.0)
    assert m["pnl_bps_of_notional"] == pytest.approx(0.05998 / 200.1 * 1e4)
    assert m["inventory_bps"] == pytest.approx(0.0, abs=1e-9)
    assert m["markout_1s_bps"] == pytest.approx(m["spread_capture_bps"])  # mid never moves


@pytest.mark.parametrize("kwargs", [
    dict(half_spread=0.0), dict(lot=0.0), dict(lot=2.0, max_position=1.0),
    dict(latency_ms=-1), dict(md_latency_ms=60, max_book_age_ms=50),
])
def test_23_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        run(market([BOOK]), SimpleFillModel(), **kwargs)


def test_24_no_equity_sample_after_the_last_market_event():
    """The stale timer of the last book fires 5 s after the data end; it must not add a sample."""
    data = market([BOOK, (1000, 100.0, 5.0, 100.1, 5.0)])
    result, _ = run(data, SimpleFillModel(), max_book_age_ms=5_000)
    assert result.equity.index.max() == pd.Timestamp(1000 * MS)


def test_25_one_print_is_not_used_twice_by_two_bids_in_the_queue_model():
    """Bid A 100.1 (empty level, queue 0) and bid B 100.0 behind 2.0. Sell print 3.0 @ 100.0.
    A first (better price): through -> fills 1.0, 2.0 of the print is left.
    B: the 2.0 left goes to the real queue 2.0 ahead -> B fills 0 (not 1.0)."""
    exchange = SimulatedExchange(QueueAwareFillModel())
    exchange.on_event(BookSnapshot(0, (100.0,), (2.0,), (100.2,), (5.0,)))
    exchange.submit(Side.BUY, 100.1, 1.0, 0)
    exchange.submit(Side.BUY, 100.0, 1.0, 0)
    got = exchange.on_event(Trade(1, 100.0, 3.0, Side.SELL))
    assert [(f.price, f.qty, f.kind.value) for f in got] == [(100.1, 1.0, "through")]


def test_26_print_used_up_by_a_better_order_does_not_confirm_a_cross():
    """Bids 100.1 and 100.0; sell print 1.0 @ 99.9 fills only the 100.1 bid (nothing left).
    Then the ask drops to 100.0 (crosses the 100.0 bid) without a trade -> CONFIRMED: no fill."""
    exchange = SimulatedExchange(SimpleFillModel())
    exchange.on_event(BookSnapshot(0, (99.9,), (5.0,), (100.2,), (5.0,)))
    exchange.submit(Side.BUY, 100.1, 1.0, 0)
    exchange.submit(Side.BUY, 100.0, 1.0, 0)
    first = exchange.on_event(Trade(1, 99.9, 1.0, Side.SELL))
    second = exchange.on_event(BookSnapshot(2, (99.9,), (5.0,), (100.0,), (5.0,)))
    assert [(f.price, f.qty) for f in first] == [(100.1, 1.0)]
    assert second == []


def test_27_fill_kinds():
    """at_price (t=1), through (t=2, sell 99.9 under our new bid 100.0), cross (ALWAYS, t=3)."""
    books = [BOOK, (3, 100.3, 5.0, 100.4, 5.0)]
    trades = [(1, 100.0, 1.0, "sell"), (2, 99.9, 1.0, "sell")]
    result, _ = run(market(books, trades), SimpleFillModel(cross_rule=CrossRule.ALWAYS))
    assert result.fills["kind"].tolist() == ["at_price", "through", "cross"]


@pytest.mark.parametrize("pickoff_fee, expected", [
    (0.00045, [(1, "SELL", 100.1, 1.0)]),  # fee 100.1 * 4.5 bp = 0.045 < gain 0.05 -> sniper takes us
    (0.001, []),                            # fee 100.1 * 10 bp = 0.1001 > gain 0.05 -> nobody takes us
])
def test_28_profitable_cross_small_jump(pickoff_fee, expected):
    """Our ask 100.1. t=1 book 100.1/100.2 (best bid reached our ask), no trades. New mid 100.15:
    sniper gain = 100.15 - 100.1 = 0.05."""
    data = market([BOOK, (1, 100.1, 5.0, 100.2, 5.0)])
    result, _ = run(data, QueueAwareFillModel(CrossRule.PROFITABLE, pickoff_fee=pickoff_fee))
    assert fills(result) == expected


def test_29_profitable_cross_user_example():
    """Ask 100.1 at t=0; at t=1 best bid 100.3 (book 100.3/100.4), no trades. New mid 100.35.
    Sniper gain 0.25 - fee 100.1 * 4.5 bp (0.045) = 0.205 > 0 -> filled at 100.1."""
    data = market([BOOK, (1, 100.3, 5.0, 100.4, 5.0)])
    result, _ = run(data, QueueAwareFillModel(CrossRule.PROFITABLE, pickoff_fee=0.00045))
    assert fills(result) == [(1, "SELL", 100.1, 1.0)]


def test_30_profitable_includes_confirmed():
    """Same as test 15c (trades cleared the queue), but a sniper fee so high that sniping
    never pays: the CONFIRMED part still fills the rest 0.8 on the cross."""
    jump = [BOOK, (2, 100.3, 5.0, 100.4, 5.0)]
    result, _ = run(market(jump, trades=[(1, 100.1, 5.2, "buy")]),
                    QueueAwareFillModel(CrossRule.PROFITABLE, pickoff_fee=1.0))
    assert fills(result) == [(1, "SELL", 100.1, 0.2), (2, "SELL", 100.1, 0.8)]


def test_31_bot_requotes_on_the_newest_book_after_a_fill():
    """ALWAYS. t=1 book 100.3/100.4 crosses our ask 100.1 -> cross fill. The fill report and the
    t=1 book reach the bot at the same time: it must quote from the t=1 book (ask 100.4), not
    re-send ask 100.1 from the old book (which would cross on arrival and fill as a taker)."""
    data = market([BOOK, (1, 100.3, 5.0, 100.4, 5.0), (2, 100.3, 5.0, 100.4, 5.0)])
    result, bt = run(data, QueueAwareFillModel(cross_rule=CrossRule.ALWAYS))
    assert result.fills["kind"].tolist() == ["cross"]
    assert bt.exchange.stats.taker_fills == 0
