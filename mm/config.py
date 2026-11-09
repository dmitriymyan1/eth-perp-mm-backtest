"""Backtest configuration: one place for every setting.

Load from TOML (see configs/default.toml), override single keys with dotted names
("fills.cross_rule"), and read resolved values in engine units (fractions, ns).
Units are part of the key names: *_ms, *_s, *_hours, *_bps.
"""
from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Optional

from .backtester.fill_models import CancelRule, CrossRule
from .backtester.venues import VENUES

MODELS = ("queue", "simple")
STRATEGIES = ("mid", "fair_value")
MAX_LEVELS = 20
NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000
NS_PER_HOUR = 3_600 * NS_PER_SECOND


@dataclass(frozen=True)
class DataConfig:
    path: str = "data"
    levels: int = 5               # book depth used for the queue position (1..20)
    tick: float = 0.1             # price grid of the data
    days: Optional[str] = None    # comma-separated YYYY-MM-DD; None = all days

    @property
    def day_list(self) -> Optional[list[str]]:
        return None if self.days is None else [d.strip() for d in self.days.split(",") if d.strip()]


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "mid"                     # mid | fair_value
    half_spread: float = 0.05             # USD from mid (or fair value), rounded outward to the tick
    lot: float = 1.0                      # order size, ETH
    max_position: Optional[float] = None  # soft inventory limit, ETH (None = 10 lots)
    signal_path: Optional[str] = None     # fair_value: parquet with ts, prediction_bp
    inventory_skew: float = 0.0           # fair_value: USD per ETH of position
    signal_cap_bp: float = 5.0            # fair_value: clip of the predicted move
    pull_threshold_bp: Optional[float] = None  # fair_value: drop a side if the model is against it by more
    funding_target_position: float = 0.0  # fair_value: hold -sign(funding) * this, ETH (0 = off)
    funding_threshold_bp: float = 0.0     # fair_value: act only if |funding rate| >= this, bp per interval
    funding_fair_hours: float = 0.0       # fair_value: shift fair value by expected funding over this holding time
    signal_deadband_bp: float = 0.0       # fair_value: use the model only if |prediction| >= this (bp)
    signal_release_ratio: float = 0.5     # fair_value: hysteresis, switch off below deadband * ratio


@dataclass(frozen=True)
class LatencyConfig:
    market_data_ms: float = 1.0   # exchange -> bot, book snapshots
    order_ms: float = 2.0         # bot -> exchange, orders and cancels (one way)
    ack_ms: float = 2.0           # exchange -> bot, fill reports
    max_book_age_s: float = 5.0   # pull quotes if no new book arrives; 0 = off


@dataclass(frozen=True)
class FeeConfig:
    venue: str = "cex_pro"             # fee preset, see mm/backtester/venues.py
    maker_bps: Optional[float] = None  # None = venue preset; negative = rebate
    taker_bps: Optional[float] = None  # None = venue preset


@dataclass(frozen=True)
class FundingConfig:
    interval_hours: float = 1.0  # settlement interval
    rate_hours: float = 1.0      # period the data rate is quoted for


@dataclass(frozen=True)
class FillConfig:
    model: str = "queue"                     # queue | simple
    cross_rule: str = "confirmed"            # never | confirmed | profitable | always
    cancel_rule: str = "pro_rata"            # back | pro_rata | front (queue model only)
    pickoff_fee_bps: Optional[float] = None  # profitable: sniper taker fee (None = taker fee)
    pickoff_fair_ms: float = 0.0             # profitable: sniper fair price = mid this much later


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "output"
    sample_interval_s: float = 1.0  # equity sampling


SECTIONS = {"data": DataConfig, "strategy": StrategyConfig, "latency": LatencyConfig, "fees": FeeConfig,
            "funding": FundingConfig, "fills": FillConfig, "output": OutputConfig}


@dataclass(frozen=True)
class BacktestConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    fills: FillConfig = field(default_factory=FillConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        _validate(self)

    # ------------------------------------------------------------ loading and changing

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BacktestConfig:
        unknown = set(raw) - set(SECTIONS)
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(**{name: _make_section(SECTIONS[name], raw.get(name, {})) for name in SECTIONS})

    @classmethod
    def from_toml(cls, path: Path) -> BacktestConfig:
        with open(path, "rb") as file:
            return cls.from_dict(tomllib.load(file))

    def with_overrides(self, overrides: dict[str, Any]) -> BacktestConfig:
        """Return a copy with dotted keys changed, e.g. {"fills.cross_rule": "never"}."""
        raw = self.to_dict()
        for dotted, value in overrides.items():
            section, _, key = dotted.partition(".")
            if section not in raw or key not in raw[section]:
                raise ValueError(f"unknown config key: {dotted}")
            raw[section][key] = value
        return BacktestConfig.from_dict(raw)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: asdict(getattr(self, name)) for name in SECTIONS}

    def digest(self) -> str:
        """Short hash of everything that changes the result (not the output folder)."""
        identity = {k: v for k, v in self.to_dict().items() if k != "output"}
        return hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:6]

    def run_name(self) -> str:
        f = self.fills
        cancel = f"_{f.cancel_rule}" if f.model == "queue" else ""
        return f"{f.model}_{f.cross_rule}{cancel}_{self.fees.venue}_lot{self.strategy.lot:g}_{self.digest()}"

    # ------------------------------------------------------------ resolved values (engine units)

    @property
    def maker_fee(self) -> float:
        return _fee(self.fees.maker_bps, VENUES[self.fees.venue].maker_fee)

    @property
    def taker_fee(self) -> float:
        return _fee(self.fees.taker_bps, VENUES[self.fees.venue].taker_fee)

    @property
    def pickoff_fee(self) -> float:
        return _fee(self.fills.pickoff_fee_bps, self.taker_fee)

    @property
    def max_position(self) -> float:
        s = self.strategy
        return 10 * s.lot if s.max_position is None else s.max_position

    @property
    def funding_interval_ns(self) -> int:
        return round(self.funding.interval_hours * NS_PER_HOUR)

    @property
    def funding_rate_scale(self) -> float:
        return self.funding.interval_hours / self.funding.rate_hours

    @property
    def max_book_age_ns(self) -> Optional[int]:
        age = self.latency.max_book_age_s
        return max(1, round(age * NS_PER_SECOND)) if age > 0 else None

    def ms_to_ns(self, ms: float) -> int:
        return round(ms * NS_PER_MS)


def _fee(bps: Optional[float], default: float) -> float:
    return default if bps is None else bps / 1e4


def _make_section(cls, values: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown keys in [{_section_name(cls)}]: {sorted(unknown)}")
    return replace(cls(), **values)


def _section_name(cls) -> str:
    return next(name for name, section in SECTIONS.items() if section is cls)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(c: BacktestConfig) -> None:
    for name in SECTIONS:
        for key, value in asdict(getattr(c, name)).items():
            if isinstance(value, float):
                _require(math.isfinite(value), f"{name}.{key} must be finite")
    _require(c.fees.venue in VENUES, f"fees.venue must be one of {sorted(VENUES)}")
    _require(c.strategy.name in STRATEGIES, f"strategy.name must be one of {list(STRATEGIES)}")
    _require(c.strategy.signal_path is None or Path(c.strategy.signal_path).is_file(),
             "strategy.signal_path must point to a signal parquet file")
    s = c.strategy
    _require(0 <= s.funding_target_position <= c.max_position and s.funding_threshold_bp >= 0
             and s.funding_fair_hours >= 0,
             "strategy funding settings: 0 <= funding_target_position <= max_position, others >= 0")
    _require(s.signal_deadband_bp >= 0 and 0 <= s.signal_release_ratio <= 1,
             "strategy.signal_deadband_bp must be >= 0 and signal_release_ratio in [0, 1]")
    _require(c.strategy.inventory_skew >= 0 and c.strategy.signal_cap_bp > 0,
             "strategy.inventory_skew must be >= 0 and strategy.signal_cap_bp > 0")
    _require(c.data.days is None or bool(c.data.day_list), "data.days must list at least one day")
    _require(c.strategy.pull_threshold_bp is None or c.strategy.pull_threshold_bp >= 0,
             "strategy.pull_threshold_bp must be >= 0")
    _require(c.fills.model in MODELS, f"fills.model must be one of {list(MODELS)}")
    _require(c.fills.cross_rule in {r.value for r in CrossRule}, f"fills.cross_rule must be one of {[r.value for r in CrossRule]}")
    _require(c.fills.cancel_rule in {r.value for r in CancelRule}, f"fills.cancel_rule must be one of {[r.value for r in CancelRule]}")
    _require(1 <= c.data.levels <= MAX_LEVELS, f"data.levels must be in 1..{MAX_LEVELS}")
    _require(c.data.tick > 0, "data.tick must be > 0")
    _require(c.strategy.half_spread > 0 and c.strategy.lot > 0, "strategy.half_spread and strategy.lot must be > 0")
    _require(c.max_position >= c.strategy.lot, "strategy.max_position must be >= strategy.lot")
    lat = c.latency
    _require(min(lat.market_data_ms, lat.order_ms, lat.ack_ms, lat.max_book_age_s) >= 0, "latencies must be >= 0")
    _require(lat.max_book_age_s == 0 or lat.max_book_age_s * 1000 > lat.market_data_ms,
             "latency.max_book_age_s must be larger than the market data latency (or 0 = off)")
    _require(c.funding.interval_hours > 0 and c.funding.rate_hours > 0, "funding hours must be > 0")
    _require(c.fills.pickoff_fee_bps is None or c.fills.pickoff_fee_bps >= 0, "fills.pickoff_fee_bps must be >= 0")
    _require(c.fills.pickoff_fair_ms >= 0, "fills.pickoff_fair_ms must be >= 0")
    _require(c.output.sample_interval_s > 0, "output.sample_interval_s must be > 0")
