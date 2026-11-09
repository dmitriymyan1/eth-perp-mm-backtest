"""Fair-value pipeline: scale rule, causality, hand-computed features, signal lookup, strategy, facade."""
import numpy as np
import pandas as pd
import pytest

from mm.config import BacktestConfig
from mm.core.domain import BookSnapshot
from mm.facade import run_backtest
from mm.signals import SignalTable, build_features, check_scales, forward_return_bp
from mm.strategy import FairValueQuoter
from synthetic import MS, config, fills, market

TICK = 0.1


def book5(rows):
    """rows: (ms, bid, bid_qty, ask, ask_qty) -> 5-level book, levels 2..5 one tick apart, qty 1."""
    data = []
    for ms, bid, bq, ask, aq in rows:
        row = {"datetime": pd.Timestamp(ms * MS)}
        for i in range(1, 6):
            row[f"bid_price_{i}"] = round(bid - (i - 1) * TICK, 8)
            row[f"ask_price_{i}"] = round(ask + (i - 1) * TICK, 8)
            row[f"bid_qty_{i}"] = bq if i == 1 else 1.0
            row[f"ask_qty_{i}"] = aq if i == 1 else 1.0
        data.append(row)
    return pd.DataFrame(data)


def trades(rows):
    """rows: (ms, price, size, 'buy' | 'sell')"""
    return pd.DataFrame({"datetime": [pd.Timestamp(r[0] * MS) for r in rows], "price": [r[1] for r in rows],
                         "size": [r[2] for r in rows], "is_maker_ask": [int(r[3] == "buy") for r in rows]})


BOOKS = [(0, 100.0, 3.0, 100.1, 1.0), (50, 100.0, 3.0, 100.1, 1.0), (150, 100.1, 2.0, 100.2, 2.0),
         (700, 100.2, 1.0, 100.3, 4.0), (1700, 100.1, 2.0, 100.2, 2.0)]
TRADES = [(10, 100.1, 0.5, "buy"), (120, 100.1, 1.5, "buy"), (600, 100.2, 0.4, "sell")]


def test_scale_rule():
    check_scales([100], horizon_ms=10_000)            # 100x: allowed (limit)
    with pytest.raises(ValueError):
        check_scales([50], horizon_ms=10_000)         # 200x: a 50 ms window for a 10 s prediction
    with pytest.raises(ValueError):
        check_scales([200_000], horizon_ms=1_000)     # 200x the other way


def test_features_by_hand():
    """Row at 150 ms: bid 100.1 x2, ask 100.2 x2 -> mid 100.15, imbalance 0, microprice = mid.
    return_100ms: last snapshot at or before 50 ms has mid 100.05 -> (100.15-100.05)/100.15*1e4 = 9.985 bp.
    flow_500ms at 150 ms: trades in (-350, 150] -> +0.5 +1.5 = 2.0 (both buyers), imbalance 1.0.
    Row at 0 ms: imbalance_1 = (3-1)/(3+1) = 0.5; microprice = (100*1 + 100.1*3)/4 = 100.075 -> +0.25 tick."""
    f = build_features(book5(BOOKS), trades(TRADES), TICK)
    assert f.loc[0, "imbalance_1"] == pytest.approx(0.5)
    assert f.loc[0, "microprice_ticks"] == pytest.approx(0.25, abs=1e-5)
    row = f.loc[2]
    assert row["imbalance_1"] == pytest.approx(0.0)
    assert row["return_100ms_bp"] == pytest.approx(0.1 / 100.15 * 1e4, rel=1e-5)
    assert row["flow_500ms"] == pytest.approx(2.0)
    assert row["flow_imbalance_500ms"] == pytest.approx(1.0)


def test_features_are_causal():
    """Changing or adding data after a snapshot must not change features at or before it."""
    full = build_features(book5(BOOKS), trades(TRADES), TICK)
    future_changed = book5(BOOKS[:3] + [(700, 99.0, 50.0, 99.1, 0.1), (1700, 98.0, 9.0, 98.1, 9.0)])
    changed = build_features(future_changed, trades(TRADES[:2] + [(600, 99.0, 40.0, "sell")]), TICK)
    cols = [c for c in full.columns if c not in ("ts", "mid")]
    pd.testing.assert_frame_equal(full.loc[:2, cols], changed.loc[:2, cols])


def test_forward_target():
    """mid: 100.05 at 0 and 50, 100.15 at 150, 100.25 at 700, 100.15 at 1700. Horizon 1 s:
    at 0   -> last mid at or before 1000 = 100.25 -> +19.99 bp;
    at 700 -> 1700 is the last snapshot (still inside the data) -> (100.15-100.25)/100.25*1e4;
    at 1700 -> the horizon passes the data end -> NaN."""
    ts = np.array([0, 50, 150, 700, 1700]) * MS
    mid = np.array([100.05, 100.05, 100.15, 100.25, 100.15])
    target = forward_return_bp(ts, mid, 1000)
    assert target[0] == pytest.approx(0.2 / 100.05 * 1e4)
    assert target[3] == pytest.approx(-0.1 / 100.25 * 1e4)
    assert np.isnan(target[4])


def test_signal_table_lookup():
    table = SignalTable(np.array([30, 10, 20]), np.array([3.0, 1.0, 2.0]))
    snap = lambda ts: BookSnapshot(ts, (100.0,), (1.0,), (100.1,), (1.0,))
    assert [table(snap(t)) for t in (10, 20, 30, 25)] == [1.0, 2.0, 3.0, 0.0]


@pytest.mark.parametrize("signal_bp, position, expected", [
    (0.0, 0, (100.0, 100.1)),    # fair = mid 100.05 -> join both touches
    (2.0, 0, (100.0, 100.2)),    # fair 100.070 -> ask ceil(100.120) = 100.2; bid floor(100.020) = 100.0
    (-2.0, 0, (99.9, 100.1)),    # fair 100.030 -> bid floor(99.980) = 99.9 backs off; ask stays at touch
    (0.0, 2, (99.9, 100.1)),     # skew 0.05 x 2 = 0.1 -> fair 99.95 -> bid 99.9; ask 100.0 -> post-only 100.1
    (50.0, 0, (100.0, 100.2)),   # signal capped at 5 bp -> fair 100.100 -> bid floor(100.05)=100.0, ask 100.2
])
def test_fair_value_quoter(signal_bp, position, expected):
    book = BookSnapshot(0, (100.0,), (5.0,), (100.1,), (5.0,))
    quoter = FairValueQuoter(TICK, 0.05, 1.0, 10.0, signal=lambda b: signal_bp, inventory_skew=0.05, signal_cap_bp=5.0)
    q = quoter.quote(book, position)
    assert (q.bid, q.ask) == expected


def test_config_checks_signal_path_and_days(tmp_path):
    with pytest.raises(ValueError):
        BacktestConfig().with_overrides({"strategy.signal_path": str(tmp_path / "missing.parquet")})
    BacktestConfig().with_overrides({"strategy.name": "fair_value"})  # no model signal: allowed
    assert BacktestConfig().with_overrides({"data.days": "2026-03-20, 2026-03-21"}).data.day_list == \
        ["2026-03-20", "2026-03-21"]


def test_facade_fair_value_backs_off_on_negative_signal(tmp_path):
    """Book 100.0 (5.0 ahead of us) / 100.1, sell 6.0 @ 100.0 at t=1. Mid strategy: queue 5.0 eats 5.0,
    our bid 100.0 fills 1.0.
    Fair value with -3 bp on the first book: fair 100.02 -> bid 99.9 -> the sell does not reach us."""
    path = tmp_path / "signal.parquet"
    pd.DataFrame({"ts": [0], "prediction_bp": [-3.0]}).to_parquet(path)
    data = market([(0, 100.0, 5.0, 100.1, 5.0)], trades=[(1, 100.0, 6.0, "sell")])
    base = config()
    assert fills(run_backtest(base, market=data)) == [(1, "BUY", 100.0, 1.0)]
    fair = base.with_overrides({"strategy.name": "fair_value", "strategy.signal_path": str(path)})
    assert fills(run_backtest(fair, market=data)) == []



@pytest.mark.parametrize("signal_bp, expected", [
    (0.05, (100.0, 100.2)),    # inside the threshold: both sides (fair 100.055 -> ask ceil(100.105) = 100.2)
    (-0.2, (None, 100.1)),     # model expects a fall > 0.1 bp: no bid (not a bid one tick deeper)
    (0.2, (100.0, None)),      # model expects a rise > 0.1 bp: no ask
])
def test_fair_value_quoter_pulls_the_side_the_model_is_against(signal_bp, expected):
    book = BookSnapshot(0, (100.0,), (5.0,), (100.1,), (5.0,))
    quoter = FairValueQuoter(TICK, 0.05, 1.0, 10.0, signal=lambda b: signal_bp, pull_threshold_bp=0.1)
    q = quoter.quote(book, 0.0)
    assert (q.bid, q.ask) == expected



@pytest.mark.parametrize("rate, position, expected", [
    # rate in fraction per interval; target 2 ETH, threshold 0.3 bp, skew 0.05 USD/ETH; mid 100.05
    (-0.0001, 0.0, (100.0, 100.2)),   # longs receive: target +2 -> fair 100.15 -> bid 100.1 capped at 100.0, ask 100.2
    (0.0001, 0.0, (99.9, 100.1)),     # shorts receive: target -2 -> fair 99.95 -> bid 99.9, ask 100.0 -> 100.1
    (0.00002, 0.0, (100.0, 100.1)),   # |rate| 0.2 bp < 0.3 bp threshold: no target
    (-0.0001, 2.0, (100.0, 100.1)),   # already at target +2: no skew
])
def test_funding_target_skew(rate, position, expected):
    """fair = mid - skew * (position - target). Post-only caps the bid at the best bid 100.0
    and the ask at the best ask 100.1."""
    book = BookSnapshot(0, (100.0,), (5.0,), (100.1,), (5.0,))
    quoter = FairValueQuoter(TICK, 0.05, 1.0, 4.0, inventory_skew=0.05,
                             funding_target_position=2.0, funding_threshold_bp=0.3)
    quoter.on_funding_rate(rate)
    q = quoter.quote(book, position)
    assert (q.bid, q.ask) == expected


def test_funding_fair_value_shift():
    """rate -1 bp per hour, hold 2 h -> fair += mid * 2 bp = 100.05 * 0.0002 = +0.02 -> 100.07:
    bid floor(100.02) = 100.0, ask ceil(100.12) = 100.2."""
    book = BookSnapshot(0, (100.0,), (5.0,), (100.1,), (5.0,))
    quoter = FairValueQuoter(TICK, 0.05, 1.0, 4.0, funding_fair_hours=2.0)
    quoter.on_funding_rate(-0.0001)
    q = quoter.quote(book, 0.0)
    assert (q.bid, q.ask) == (100.0, 100.2)


def test_bot_receives_funding_rate_and_leans_inventory():
    """Funding -1 bp published at t=1 (longs receive), target 2 ETH, skew 0.05.
    t=0: quotes 100.0 / 100.1. After t=1 the bot sees the rate: fair 100.15 ->
    bid floor(100.10) = 100.1 -> capped at best bid 100.0; ask ceil(100.20) = 100.2 (leans long).
    t=2 buy 1.0 @ 100.1 does not fill us (our ask moved to 100.2)."""
    data = market([(0, 100.0, 0.0, 100.1, 0.0)], trades=[(2, 100.1, 1.0, "buy")], funding=[(1, -0.0001)])
    cfg = config(max_position=4.0).with_overrides({
        "strategy.name": "fair_value", "strategy.inventory_skew": 0.05,
        "strategy.funding_target_position": 2.0, "strategy.funding_threshold_bp": 0.3})
    assert fills(run_backtest(cfg, market=data)) == []
    neutral = config(max_position=4.0)
    assert fills(run_backtest(neutral, market=data)) == [(2, "SELL", 100.1, 1.0)]



def test_signal_deadband_with_hysteresis():
    """Dead zone 0.4 bp, release 0.5 -> on at |p| >= 0.4, off below 0.2 or on a sign change.
    Sequence: 0.3 (off: below 0.4) -> 0.5 (on) -> 0.3 (stays on: above 0.2) -> 0.1 (off)
    -> -0.5 (on, negative) -> 0.3 (sign change: off)."""
    values = iter([0.3, 0.5, 0.3, 0.1, -0.5, 0.3])
    quoter = FairValueQuoter(TICK, 0.05, 1.0, 10.0, signal=lambda b: next(values), signal_deadband_bp=0.4)
    book = BookSnapshot(0, (100.0,), (5.0,), (100.1,), (5.0,))
    assert [quoter.predicted_bp(book) for _ in range(6)] == [0.0, 0.5, 0.3, 0.0, -0.5, 0.0]


def test_spec_windows_scale_with_horizon_and_keep_the_rule():
    """30 s horizon: returns 3 / 15 / 60 / 300 s, state averaged over 3 s; all within 100x.
    60 min with a 2 h cap: the 10 h windows are dropped."""
    from mm.signals.features import DEFAULT_SPEC, spec_for_horizon
    assert DEFAULT_SPEC.return_windows_ms == (100, 500, 2000, 10000) and DEFAULT_SPEC.state_window_ms is None
    s30 = spec_for_horizon(30_000)
    assert s30.return_windows_ms == (3000, 15000, 60000, 300000) and s30.state_window_ms == 3000
    s60 = spec_for_horizon(3_600_000, max_window_ms=7_200_000)
    assert max(s60.all_windows_ms) == 7_200_000
    with pytest.raises(ValueError):
        check_scales([100], horizon_ms=3_600_000)    # a 100 ms window for a 60 min horizon


def test_averaged_state_features_are_causal():
    """With a state window, changing later books must not change earlier averaged features."""
    from mm.signals.features import FeatureSpec
    spec = FeatureSpec(1000, (500,), (500,), (500,), (500,), state_window_ms=200)
    full = build_features(book5(BOOKS), trades(TRADES), TICK, spec)
    changed = build_features(book5(BOOKS[:3] + [(700, 99.0, 50.0, 99.1, 0.1), (1700, 98.0, 9.0, 98.1, 9.0)]),
                             trades(TRADES), TICK, spec)
    cols = [c for c in full.columns if c not in ("ts", "mid")]
    pd.testing.assert_frame_equal(full.loc[:2, cols], changed.loc[:2, cols])
    # row 2 (150 ms): books at 0, 50, 150 ms are inside (−50, 150] -> imbalance_1 mean of 0.5, 0.5, 0.0
    assert full.loc[2, "imbalance_1"] == pytest.approx(1 / 3)
