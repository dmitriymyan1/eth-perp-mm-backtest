"""Build the trading-economics charts and the numbers behind them.

Focus run: ML fair value + inventory skew 0.05, half-spread 0.45, limit 2 ETH, on the
out-of-sample days 03-20..21 (model trained on 03-19). Baseline: mid quoter with the same
half-spread on the same days. Context: quote-depth trade-off (mid, 3 days).

Usage (from the project root): python analytics/build_charts.py
Reads finished runs from output/ and raw books from data/; writes analytics/charts/*.png
and analytics/data/summary.json. Palette: validated categorical slots 1-4 + neutral ink.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mm.analytics.economics import (HORIZONS_S, fill_frame, markout_curve,  # noqa: E402
                                    round_trip_summary, round_trips)
from mm.analytics.toxicity import adverse_curve  # noqa: E402
from mm.backtester.venues import VENUES  # noqa: E402

OUT = ROOT / "analytics"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ALL_DAYS = ["2026-03-19", "2026-03-20", "2026-03-21"]
OOS_DAYS = ["2026-03-20", "2026-03-21"]
HALF_SPREAD = 0.45
FOCUS, BASE = "ML fair value 0.45", "mid 0.45"
FEE_VENUE = "cex_pro"  # every run is re-priced with these fees (fills do not depend on fees)
FEE_SCENARIOS = {"cex_mm_rebate": "rebate маркет-мейкера (−0.5 / 1.5 bp)", "cex_pro": "топ-VIP CEX (0 / 1.7 bp)",
                 "hyperliquid": "Hyperliquid, базовый (1.5 / 4.5 bp)", "bybit": "Bybit, базовый (2.0 / 5.5 bp)"}
HORIZON_LABELS = ["0"] + [f"{h:g} с" if h < 60 else f"{h / 60:g} мин" for h in HORIZONS_S]


# ---------------------------------------------------------------- data

def runs(folder: str) -> list[tuple[dict, Path]]:
    found = []
    for p in glob.glob(str(ROOT / folder / "*_metrics.json")):
        found.append((json.load(open(p)), Path(p.replace("_metrics.json", "_fills.parquet"))))
    return found


def pick(folder: str, name: str, half_spread: float, skew: float | None = None) -> tuple[dict, Path]:
    for m, fills in runs(folder):
        s = m["config"]["strategy"]
        if s.get("name", "mid") == name and s["half_spread"] == half_spread and (skew is None or s.get("inventory_skew") == skew):
            return m, fills
    raise SystemExit(f"run not found: {folder} {name} {half_spread}")


def fee_usd(fills: pd.DataFrame, venue: str) -> np.ndarray:
    v = VENUES[venue]
    rate = np.where(fills["kind"] == "taker", v.taker_fee, v.maker_fee)
    return (fills["price"] * fills["qty"]).to_numpy() * rate


def load_fills(path: Path, m: dict) -> tuple[pd.DataFrame, dict]:
    """Fills re-priced with FEE_VENUE, and the metrics with total PnL and fees adjusted to match."""
    fills = pd.read_parquet(path)
    old = float(fills["fee"].sum())
    fills["fee"] = fee_usd(fills, FEE_VENUE)
    new = float(fills["fee"].sum())
    return fills, {**m, "fees": new, "total_pnl": m["total_pnl"] + old - new}


def load_mid(days: list[str]) -> tuple[np.ndarray, np.ndarray]:
    book = pd.concat(pd.read_parquet(ROOT / f"data/orderbook/{d}.parquet", columns=["datetime", "bid_price_1", "ask_price_1"])
                     for d in days)
    return book["datetime"].astype("int64").to_numpy(), ((book["bid_price_1"] + book["ask_price_1"]) / 2).to_numpy()


def pnl_path(f: pd.DataFrame, mid_ts: np.ndarray, mid: np.ndarray, freq: str = "1min") -> pd.Series:
    """Mark-to-market PnL (USD, without funding) on a time grid, rebuilt from fills and the mid."""
    grid = pd.date_range(pd.Timestamp(mid_ts[0]).ceil(freq), pd.Timestamp(mid_ts[-1]), freq=freq).as_unit("ns")
    g = grid.asi8
    signed = (f["sign"] * f["qty"]).to_numpy()
    cash = -(f["sign"] * f["price"] * f["qty"]).to_numpy() - f["fee"].to_numpy()
    i = np.searchsorted(f["ts"].to_numpy(), g, side="right")
    pos = np.concatenate([[0.0], np.cumsum(signed)])[i]
    cash_t = np.concatenate([[0.0], np.cumsum(cash)])[i]
    mark = mid[np.clip(np.searchsorted(mid_ts, g, side="right") - 1, 0, None)]
    return pd.Series(cash_t + pos * mark, index=grid)


def interval_95(path: pd.Series) -> tuple[float, float, float]:
    """Total PnL and a 95 % interval from hourly increments (iid; t = mean / se)."""
    hourly = path.resample("1h").last().diff().dropna()
    total, se = hourly.sum(), hourly.std() * np.sqrt(len(hourly))
    return float(total), float(total - 1.96 * se), float(total + 1.96 * se)


def horizon_label(key: str) -> str:
    seconds = float(key.rstrip("s"))
    return f"{seconds:g} с" if seconds < 60 else f"{seconds / 60:g} мин"


def phases(curve: pd.Series) -> tuple[str, str]:
    """Trough (deepest mark-to-market loss) and the peak after it, as horizon keys like '30s'."""
    keys = [f"{h:g}s" for h in HORIZONS_S]
    trough = min(keys, key=lambda k: curve[k])
    peak = max(keys[keys.index(trough):], key=lambda k: curve[k])
    return trough, peak


# ---------------------------------------------------------------- chart helpers

def axes(title: str, size=(10, 5.2)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    style(ax)
    return fig, ax


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / "charts" / name, dpi=150)
    plt.close(fig)


def end_labels(ax, x_end, series: dict[str, tuple[float, str]], min_gap: float, dx: float = 0.25) -> None:
    """Direct labels at line ends, pushed apart so they never overlap."""
    placed = []
    for label, (y, _) in sorted(series.items(), key=lambda kv: -kv[1][0]):
        y_text = min(y, placed[-1] - min_gap) if placed else y
        placed.append(y_text)
        ax.annotate(label, (x_end, y), xytext=(x_end + dx, y_text), textcoords="data", va="center",
                    fontsize=9, color=INK, annotation_clip=False)


# ---------------------------------------------------------------- charts

def chart_waterfall(f: pd.DataFrame, m: dict, trough: str, peak: str) -> dict:
    usd = lambda col: float((f[col] * f["notional"]).sum() / 1e4)
    premium = usd("premium_bp")
    impact = usd(f"mo_{trough}") - premium
    reversion = usd(f"mo_{peak}") - usd(f"mo_{trough}")
    fees, funding, total = m["fees"], m["funding_pnl"], m["total_pnl"]
    after = total - premium - impact - reversion + fees - funding
    steps = [("Премия за\nкотирование", premium), (f"Влияние сделки\n0 → {horizon_label(trough)}", impact),
             (f"Возврат цены\n{horizon_label(trough)} → {horizon_label(peak)}", reversion),
             (f"После\n{horizon_label(peak)}", after), ("Комиссия\nmaker", -fees), ("Funding", funding)]
    fig, ax = axes(f"Куда уходят деньги: {FOCUS}, 03-20…21, USD")
    level, tops = 0.0, [0.0]
    for i, (_, value) in enumerate(steps):
        ax.bar(i, value, bottom=level, color=BLUE if value >= 0 else ORANGE, width=0.62)
        ax.annotate(f"{round(value) or 0:+,.0f}", (i, level + value), xytext=(0, 6 if value >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
        level += value
        tops.append(level)
    ax.bar(len(steps), total, color=INK, width=0.62)
    ax.annotate(f"{total:+,.0f}", (len(steps), total), xytext=(0, 6 if total >= 0 else -14),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
    span = max(tops) - min(tops + [total])
    ax.set_ylim(min(tops + [total]) - span * 0.12, max(tops) + span * 0.12)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(steps) + 1), [s[0] for s in steps] + ["Итого"], fontsize=9)
    ax.set_ylabel("USD", color=MUTED)
    save(fig, "01_waterfall.png")
    notional = float(f["notional"].sum())
    parts = {"premium": premium, "impact": impact, "reversion": reversion, "after_peak": after,
             "fees": -fees, "funding": funding, "total": total}
    return {"notional_usd": notional, "trough": trough, "peak": peak, **parts,
            "bp": {k: v / notional * 1e4 for k, v in parts.items()}}


def chart_waterfall_numbers(f: pd.DataFrame, m: dict, trough: str, peak: str) -> dict:
    usd = lambda col: float((f[col] * f["notional"]).sum() / 1e4)
    premium, impact = usd("premium_bp"), usd(f"mo_{trough}") - usd("premium_bp")
    reversion = usd(f"mo_{peak}") - usd(f"mo_{trough}")
    after = m["total_pnl"] - premium - impact - reversion + m["fees"] - m["funding_pnl"]
    notional = float(f["notional"].sum())
    parts = {"premium": premium, "impact": impact, "reversion": reversion, "after_peak": after,
             "fees": -m["fees"], "funding": m["funding_pnl"], "total": m["total_pnl"]}
    return {**parts, "bp": {k: v / notional * 1e4 for k, v in parts.items()}}


def chart_markout_pair(curves: dict[str, pd.Series], fee: float, trough: str, peak: str) -> None:
    x = np.arange(len(HORIZON_LABELS))
    fig, ax = axes("Markout по горизонтам: ML fair value против mid при half-spread 0.45 (bp, до комиссии)")
    ends = {}
    for (name, curve), color in zip(curves.items(), (BLUE, ORANGE)):
        y = curve[["premium"] + [f"{h:g}s" for h in HORIZONS_S]].to_numpy(float)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5)
        ends[name] = (y[-1], color)
    ax.axhline(fee, color=INK, lw=1, ls="--")
    ax.annotate(f"комиссия maker {fee:.2f} bp: fill прибылен только выше этой линии", (0.1, fee),
                xytext=(0, 6), textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    end_labels(ax, x[-1], ends, min_gap=0.9)
    keys = ["premium"] + [f"{h:g}s" for h in HORIZONS_S]
    for key, text in ((trough, "минимум: конец влияния сделки"), (peak, "максимум: конец возврата")):
        xi = keys.index(key)
        ax.axvline(xi, color=MUTED, lw=0.8, ls=":")
        ax.annotate(text, (xi, ax.get_ylim()[1]), xytext=(-12, -6), textcoords="offset points", fontsize=8.5,
                    color=MUTED, rotation=90, va="top")
    ax.set_xticks(x, HORIZON_LABELS)
    ax.set_xlim(-0.3, x[-1] + 2.6)
    ax.set_xlabel("время после fill (0 = премия в момент fill)", color=MUTED)
    ax.set_ylabel("bp от оборота", color=MUTED)
    save(fig, "02_markout.png")


def chart_edge_budget(rows: list[dict], trough: str) -> None:
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = axes(f"Контекст: бюджет edge по глубине котировки (mid, 3 дня, горизонт {horizon_label(trough)}, bp)")
    ax.bar(x - w, [r["premium"] for r in rows], w, color=BLUE, label="Премия за котирование")
    ax.bar(x, [r["adverse"] for r in rows], w, color=ORANGE, label=f"Влияние сделки 0 → {horizon_label(trough)}")
    ax.bar(x + w, [-r["fee"] for r in rows], w, color=MUTED, label="Комиссия maker")
    ax.plot(x, [r["net"] for r in rows], color=INK, lw=0, marker="D", ms=7, label=f"Итог на {horizon_label(trough)}")
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['net']:+.1f}", (xi, r["net"]), xytext=(9, -3), textcoords="offset points", fontsize=8, color=INK)
        if r["h"] == HALF_SPREAD:
            ax.axvspan(xi - 0.48, xi + 0.48, color=GRID, alpha=0.6, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"{r['h']:g}\n({r['ticks']} тик.)" for r in rows])
    ax.set_xlabel("half-spread, USD (тиков за лучшей ценой); серая полоса — выбранный 0.45", color=MUTED)
    ax.set_ylabel("bp от оборота", color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    save(fig, "03_edge_by_depth.png")


def chart_toxicity(curves: dict[str, pd.Series], shares: dict[str, float]) -> None:
    fig, ax = axes(f"Ход цены вокруг fill по виду исполнения: {FOCUS} (bp, относительно mid перед fill)")
    names = {"all": "все fills", "through": "sweep сквозь нас", "at_price": "по нашей цене", "cross": "cross стакана"}
    ends = {}
    for (key, curve), color in zip(curves.items(), (INK, BLUE, ORANGE, AQUA)):
        x = curve.index.to_numpy(float)
        ax.plot(x, curve.to_numpy(), color=color, lw=2.4 if key == "all" else 1.8, marker="o", ms=3.5)
        ends[names[key] + ("" if key == "all" else f" ({shares[key]:.0%} объёма)")] = (curve.iloc[-1], color)
    ax.set_xscale("symlog", linthresh=50)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([-2000, -500, -100, 0, 50, 100, 200, 500, 1000, 5000, 60000],
                  ["−2 с", "−0.5 с", "−100", "0", "50", "100", "200", "500", "1 с", "5 с", "60 с"])
    span = np.ptp(np.concatenate([c.to_numpy() for c in curves.values()]))
    placed = []
    for label, (y, _) in sorted(ends.items(), key=lambda kv: -kv[1][0]):
        y_text = min(y, placed[-1] - span * 0.06) if placed else y
        placed.append(y_text)
        ax.annotate(label, (60000, y), xytext=(80000, y_text), textcoords="data", va="center", fontsize=9,
                    color=INK, annotation_clip=False)
    ax.set_xlim(right=600000)
    ax.set_xlabel("мс от fill (symlog; слева от 0 — до fill)", color=MUTED)
    ax.set_ylabel("bp, < 0 = против нас", color=MUTED)
    save(fig, "04_toxicity_by_kind.png")


def chart_round_trips(rows: list[dict]) -> None:
    fig, ax = axes("Round trips (FIFO) при half-spread 0.45: итог после комиссий")
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["net_bp"] for r in rows], color=[r["color"] for r in rows], height=0.55)
    for yi, r in zip(y, rows):
        ax.annotate(f"{r['net_bp']:+.1f} bp · удержание {r['hold_min']:.0f} мин · win {r['win']:.0%} · {r['trips']} сделок",
                    (max(r["net_bp"], 0), yi), xytext=(6, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    lo, hi = min(r["net_bp"] for r in rows), max(r["net_bp"] for r in rows)
    ax.set_xlim(min(lo, 0) * 1.3 - 1, max(hi, 0) + (max(hi, 0) - min(lo, 0)) * 1.6 + 2)
    ax.set_xlabel("net bp на round trip (> 0 — прибыль)", color=MUTED)
    save(fig, "05_round_trips.png")


def phase_table(f: pd.DataFrame, groups: pd.Series, trough: str, peak: str) -> pd.DataFrame:
    """Per group: volume share, impact (0 -> trough) and reversion (trough -> peak), bp."""
    def stats(g: pd.DataFrame) -> pd.Series:
        w = g["notional"].to_numpy()
        avg = lambda col: float(np.average(g[col], weights=w))
        return pd.Series({"volume share": w.sum() / f["notional"].sum(), "premium": avg("premium_bp"),
                          "impact": avg(f"mo_{trough}") - avg("premium_bp"),
                          "reversion": avg(f"mo_{peak}") - avg(f"mo_{trough}")})
    return f.groupby(groups, observed=True)[["notional", "premium_bp", f"mo_{trough}", f"mo_{peak}"]].apply(stats)


def chart_breakdowns(by_kind: pd.DataFrame, by_day: pd.DataFrame, trough: str, peak: str) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(SURFACE)
    names = {"at_price": "по нашей\nцене", "cross": "cross", "through": "sweep", "taker": "taker"}
    w = 0.36
    for ax, frame, title, labels in (
            (a, by_kind, "Влияние и возврат по виду fill", [names.get(k, k) for k in by_kind.index]),
            (b, by_day, "Влияние и возврат по дням", list(by_day.index))):
        ax.set_facecolor(SURFACE)
        style(ax)
        x = np.arange(len(frame))
        ax.bar(x - w / 2, frame["impact"], w, color=ORANGE, label=f"влияние 0 → {horizon_label(trough)}")
        ax.bar(x + w / 2, frame["reversion"], w, color=BLUE, label=f"возврат {horizon_label(trough)} → {horizon_label(peak)}")
        for xi, (_, row) in zip(x, frame.iterrows()):
            ax.annotate(f"{row['impact']:+.1f}", (xi - w / 2, row["impact"]), xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
            ax.annotate(f"{row['reversion']:+.1f}", (xi + w / 2, row["reversion"]),
                        xytext=(0, 4 if row["reversion"] >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, [f"{lab}\n{share:.0%} объёма" for lab, share in zip(labels, frame["volume share"])], fontsize=9)
        span = max(abs(frame[["impact", "reversion"]].to_numpy()).max(), 1.0)
        ax.set_ylim(-span * 1.3, span * 1.3)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylabel("bp", color=MUTED)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "06_breakdowns.png", dpi=150)
    plt.close(fig)


def chart_pnl_path(paths: dict[str, pd.Series], intervals: dict[str, tuple[float, float, float]]) -> None:
    fig, ax = axes("Путь PnL вне выборки, USD (mark-to-market, без funding)")
    ends = {}
    for (name, path), color in zip(paths.items(), (BLUE, ORANGE)):
        ax.plot(path.index, path.to_numpy(), color=color, lw=1.8)
        total, lo, hi = intervals[name]
        ends[f"{name}: {total:+,.0f} (95%: {lo:+,.0f} … {hi:+,.0f})"] = (path.iloc[-1], color)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(pd.Timestamp("2026-03-21"), color=MUTED, lw=0.8, ls=":")
    ax.annotate("03-21", (pd.Timestamp("2026-03-21"), ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                fontsize=9, color=MUTED)
    span = np.ptp(np.concatenate([p.to_numpy() for p in paths.values()]))
    placed = []
    for label, (y, _) in sorted(ends.items(), key=lambda kv: -kv[1][0]):
        y_text = min(y, placed[-1] - span * 0.08) if placed else y
        placed.append(y_text)
        ax.annotate(label, (paths[next(iter(paths))].index[-1], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, annotation_clip=False)
    ax.set_xlim(right=paths[next(iter(paths))].index[-1] + pd.Timedelta("16h"))
    ax.set_ylabel("USD", color=MUTED)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m\n%H:%M"))
    save(fig, "07_pnl_path.png")


# ---------------------------------------------------------------- main

def main() -> None:
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "charts").glob("*.png"):
        old.unlink()
    summary: dict = {"focus": FOCUS, "baseline": BASE, "days": OOS_DAYS}

    oos_ts, oos_mid = load_mid(OOS_DAYS)
    focus_raw, focus_m = load_fills(*reversed(pick("output/fv_wide", "fair_value", HALF_SPREAD, 0.05)))
    base_raw, base_m = load_fills(*reversed(pick("output/fv_wide", "mid", HALF_SPREAD)))
    focus = fill_frame(focus_raw, oos_ts, oos_mid)
    base = fill_frame(base_raw, oos_ts, oos_mid)
    summary["fee_venue"] = FEE_VENUE

    curves = {FOCUS: markout_curve(focus), BASE: markout_curve(base)}
    trough, peak = phases(curves[FOCUS])
    summary["phases"] = {"trough": trough, "peak": peak}
    summary["waterfall"] = chart_waterfall(focus, focus_m, trough, peak)
    summary["waterfall_baseline"] = chart_waterfall_numbers(base, base_m, trough, peak)
    chart_markout_pair(curves, float(np.mean([c["fee"] for c in curves.values()])), trough, peak)
    summary["markout_curves"] = {k: c.round(3).to_dict() for k, c in curves.items()}
    summary["runs"] = {name: {"fills": m["fills"], "fills_per_day": m["fills"] / 2, "pnl": m["total_pnl"],
                              "bp": m["total_pnl"] / f["notional"].sum() * 1e4, "fees": m["fees"],
                              "volume_eth": m["filled_volume"], "mean_abs_inventory": m["mean_abs_inventory"]}
                       for name, m, f in ((FOCUS, focus_m, focus), (BASE, base_m, base))}
    summary["fee_scenarios"] = {
        name: {label: m["total_pnl"] + m["fees"] - float(fee_usd(f, venue).sum()) for venue, label in FEE_SCENARIOS.items()}
        for name, m, f in ((FOCUS, focus_m, focus), (BASE, base_m, base))}

    all_ts, all_mid = load_mid(ALL_DAYS)
    depth = []
    for m, p in sorted(runs("output/spread_sweep"), key=lambda r: r[0]["config"]["strategy"]["half_spread"]):
        h = m["config"]["strategy"]["half_spread"]
        c = markout_curve(fill_frame(load_fills(p, m)[0], all_ts, all_mid))
        depth.append({"h": h, "ticks": int(round((h - 0.05) / 0.1)), "premium": c["premium"],
                      "adverse": c[trough] - c["premium"], "fee": c["fee"], "net": c[trough] - c["fee"],
                      "fills_per_day": m["fills"] / 3})
    chart_edge_budget(depth, trough)
    summary["edge_by_depth"] = depth

    groups = {"all": focus} | {k: focus[focus["kind"] == k] for k in ("through", "at_price", "cross") if (focus["kind"] == k).any()}
    tox = {k: adverse_curve(g, oos_ts, oos_mid) for k, g in groups.items()}
    shares = {k: g["qty"].sum() / focus["qty"].sum() for k, g in groups.items()}
    chart_toxicity(tox, shares)
    summary["toxicity"] = {k: c.round(3).to_dict() for k, c in tox.items()}
    summary["kind_share"] = shares

    trip_specs = [(FOCUS + ", лимит 2 ETH, 03-20…21", focus, BLUE), (BASE + ", лимит 2 ETH, 03-20…21", base, ORANGE)]
    for label, folder, days in (("mid 0.45, лимит 2 ETH, 3 дня", "output/spread_sweep_lowinv", ALL_DAYS),
                                ("mid 0.45, лимит 10 ETH, 3 дня", "output/spread_sweep", ALL_DAYS)):
        m, p = pick(folder, "mid", HALF_SPREAD)
        trip_specs.append((label, fill_frame(load_fills(p, m)[0], all_ts, all_mid), MUTED))
    trip_rows = []
    for label, f, color in trip_specs:
        s = round_trip_summary(round_trips(f))
        trip_rows.append({"label": label, "color": color, "net_bp": s["net bp"], "hold_min": s["hold median s"] / 60,
                          "win": s["win rate (net)"], "trips": int(s["round trips"]), "net_usd": s["net USD"]})
    chart_round_trips(trip_rows)
    summary["round_trips"] = [{k: v for k, v in r.items() if k != "color"} for r in trip_rows]

    by_kind = phase_table(focus, focus["kind"], trough, peak).drop(index="taker", errors="ignore")
    by_day = phase_table(focus, pd.to_datetime(focus["ts"]).dt.strftime("%m-%d").rename("day"), trough, peak)
    chart_breakdowns(by_kind, by_day, trough, peak)
    summary["by_kind"] = by_kind.round(3).reset_index().to_dict(orient="records")
    summary["by_day"] = by_day.round(3).reset_index().to_dict(orient="records")

    paths = {FOCUS: pnl_path(focus, oos_ts, oos_mid), BASE: pnl_path(base, oos_ts, oos_mid)}
    intervals = {k: interval_95(p) for k, p in paths.items()}
    chart_pnl_path(paths, intervals)
    summary["pnl_interval_95"] = intervals
    for name, path in paths.items():
        hourly = path.resample("1h").last().diff().dropna()
        summary["runs"][name]["t_hourly"] = float(hourly.mean() / hourly.std() * np.sqrt(len(hourly)))

    latency = []
    for folder in ("output/fv_wide", "output/latency_grid", "output/latency_hl"):
        for m, p in runs(folder):
            st, lat = m["config"]["strategy"], m["config"]["latency"]
            if st["half_spread"] != HALF_SPREAD or (st.get("name") == "fair_value" and st.get("inventory_skew") != 0.05):
                continue
            fills, adj = load_fills(p, m)
            latency.append({"strategy": "ML fair value" if st.get("name") == "fair_value" else "mid",
                            "md_ms": lat["market_data_ms"], "order_ms": lat["order_ms"], "ack_ms": lat["ack_ms"],
                            "fills_per_day": m["fills"] / 2, "taker_fills": int((fills["kind"] == "taker").sum()),
                            "pnl": adj["total_pnl"]})
    summary["latency"] = sorted(latency, key=lambda r: (r["strategy"], r["order_ms"]))

    (OUT / "data" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("charts:", sorted(p.name for p in (OUT / "charts").glob("*.png")))


if __name__ == "__main__":
    main()
