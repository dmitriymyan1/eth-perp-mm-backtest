"""Config facade: loading, resolved values, validation, overrides, sweeps, and a facade run."""
from pathlib import Path

import pytest

import run_backtest
from mm.config import BacktestConfig
from mm.facade import run_backtest as run_facade
from synthetic import config, fills, market

ROOT = Path(__file__).parent.parent
BOOK = (0, 100.0, 5.0, 100.1, 5.0)


def test_default_toml_matches_code_defaults():
    assert BacktestConfig.from_toml(ROOT / "configs/default.toml") == BacktestConfig()


def test_resolved_values():
    """Default preset cex_pro: maker 0 bp, taker 1.7 bp; pickoff fee = taker fee; max position = 10 lots."""
    cfg = BacktestConfig()
    assert (cfg.maker_fee, cfg.taker_fee, cfg.pickoff_fee) == (0.0, 0.00017, 0.00017)
    assert BacktestConfig().with_overrides({"fees.venue": "cex_mm_rebate"}).maker_fee == pytest.approx(-0.00005)
    assert cfg.max_position == 10.0
    assert cfg.max_book_age_ns == 5_000_000_000

    custom = cfg.with_overrides({"fills.pickoff_fee_bps": 2.4, "strategy.lot": 0.5, "latency.max_book_age_s": 0,
                                 "fees.maker_bps": -0.5})
    assert custom.pickoff_fee == pytest.approx(0.00024)
    assert custom.max_position == 5.0
    assert custom.max_book_age_ns is None
    assert custom.maker_fee == pytest.approx(-0.00005)


@pytest.mark.parametrize("overrides", [
    {"fills.cross_rule": "sometimes"},
    {"fees.venue": "nowhere"},
    {"strategy.half_spread": float("inf")},
    {"strategy.lot": 2.0, "strategy.max_position": 1.0},
    {"latency.order_ms": -1.0},
    {"latency.market_data_ms": 60.0, "latency.max_book_age_s": 0.05},
    {"data.levels": 21},
    {"funding.interval_hours": 0.0},
    {"fills.pickoff_fair_ms": -5.0},
    {"strategy.no_such_key": 1},
    {"nosection.key": 1},
])
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        BacktestConfig().with_overrides(overrides)


def test_digest_tracks_settings_but_not_output_dir():
    base = BacktestConfig()
    assert base.digest() == BacktestConfig().digest()
    assert base.with_overrides({"fills.pickoff_fair_ms": 200.0}).digest() != base.digest()
    assert base.with_overrides({"output.dir": "elsewhere"}).digest() == base.digest()


def test_cli_parsing_and_sweep_grid():
    assert [run_backtest.parse_value(v) for v in ("none", "true", "3", "2.4", "never")] == [None, True, 3, 2.4, "never"]
    grid = run_backtest.expand(BacktestConfig(), [("fills.cross_rule", "never,confirmed"),
                                                  ("fills.pickoff_fair_ms", "0,200,500")])
    assert len(grid) == 6
    assert {(c.fills.cross_rule, c.fills.pickoff_fair_ms) for c in grid} == {
        (rule, ms) for rule in ("never", "confirmed") for ms in (0, 200, 500)}
    assert len({c.run_name() for c in grid}) == 6  # no output collisions


@pytest.mark.parametrize("fair_ms, expected", [(0.0, []), (200.0, [(1, "SELL", 100.1, 1.0)])])
def test_facade_profitable_with_sniper_information_lead(fair_ms, expected):
    """Our ask 100.1. t=1 ms book 100.1/100.2 (crosses us, mid 100.15), no trades; t=200 ms mid 100.25.
    Sniper fee 10 bp = 0.1001 (on ~100).
    fair = mid now:          t=1 gain 0.05 < 0.1001 -> no fill; the bot re-quotes the ask at 100.2;
                             t=200 the bid 100.2 crosses it, gain 100.25 - 100.2 = 0.05 -> no fill.
    fair = mid 200 ms later: t=1 gain 100.25 - 100.1 = 0.15 > 0.1001 -> filled at 100.1 at t=1;
                             t=200 gain on the new ask 100.2 is 0.05 -> no fill."""
    data = market([BOOK, (1, 100.1, 5.0, 100.2, 5.0), (200, 100.2, 5.0, 100.3, 5.0)])
    cfg = config().with_overrides({"fills.cross_rule": "profitable", "fills.pickoff_fee_bps": 10.0,
                                   "fills.pickoff_fair_ms": fair_ms})
    assert fills(run_facade(cfg, market=data)) == expected
