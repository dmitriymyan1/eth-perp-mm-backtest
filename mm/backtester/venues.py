"""Exchange fee presets.

The data looks like a continuous-matching (CEX-style) feed, and a strategy that needs
<= 5 ms order latency runs colocated, which in practice means a top fee tier:
- cex_pro: top VIP (Binance USD-M VIP 9 0 / 1.7 bp; Bybit Supreme VIP 0 / 3 bp) - default;
- cex_mm_rebate: market-maker rebate (OKX VIP 9 about -0.5 / 1.5 bp) - optimistic;
- hyperliquid / bybit: base tiers (no staking or referral discounts).

The funding schedule is a property of the data (the published rate resets every hour),
not of the fee preset: see --funding-hours / --funding-rate-hours in run_backtest.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    maker_fee: float  # fraction of notional
    taker_fee: float  # used when an order crosses the book on arrival


VENUES = {
    "cex_pro": Venue(maker_fee=0.0, taker_fee=0.00017),
    "cex_mm_rebate": Venue(maker_fee=-0.00005, taker_fee=0.00015),
    "hyperliquid": Venue(maker_fee=0.00015, taker_fee=0.00045),
    "bybit": Venue(maker_fee=0.0002, taker_fee=0.00055),
}
