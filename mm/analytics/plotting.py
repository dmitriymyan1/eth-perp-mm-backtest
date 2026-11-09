"""PnL and inventory chart saved as PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..backtester.engine import BacktestResult  # noqa: E402


def plot_pnl(result: BacktestResult, path: Path, title: str) -> None:
    eq = result.equity
    fig, (ax_pnl, ax_mid, ax_inv) = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    ax_pnl.plot(eq.index, eq["equity"], label="Total PnL", lw=1.2)
    ax_pnl.plot(eq.index, eq["realized"] - eq["fees"] + eq["funding"],
                label="Realized - fees + funding", lw=0.8, alpha=0.7)
    ax_pnl.axhline(0, color="grey", lw=0.5)
    ax_pnl.set_ylabel("PnL, USD")
    ax_pnl.set_title(title)
    ax_pnl.legend(loc="upper left")

    ax_mid.plot(eq.index, eq["mid"], color="tab:gray", lw=0.8)
    ax_mid.set_ylabel("Mid, USD")

    ax_inv.plot(eq.index, eq["position"], color="tab:orange", lw=0.8)
    ax_inv.axhline(0, color="grey", lw=0.5)
    ax_inv.set_ylabel("Inventory, ETH")

    for ax in (ax_pnl, ax_mid, ax_inv):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
