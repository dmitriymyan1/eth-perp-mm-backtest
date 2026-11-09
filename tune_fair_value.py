"""Choose the prediction horizon: train on the train day, compare on the validation day only.

Usage: python tune_fair_value.py [--train 2026-03-19] [--valid 2026-03-20] [--test 2026-03-21]
       [--horizons 250,500,1000,2000,5000] [--fills <mid strategy fills parquet>]

Per horizon: boosting on the train day (trees chosen on its last 20 %), then on the validation day:
corr / R2, size of the prediction vs one tick, and (if --fills is given) how well the model separates
toxic from good fills of a plain mid strategy. Writes models/fv_h<ms>.parquet with predictions for
the validation AND test days, so the strategy can be tuned on one and tested on the other.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mm.data import ParquetDataSource
from mm.signals import build_features, check_scales, feature_columns, forward_return_bp
from mm.signals.features import ALL_WINDOWS_MS
from train_fair_value import TICK, evaluate, fit_boosting, time_split

NS_PER_MS = 1_000_000


def with_target(frame: pd.DataFrame, horizon_ms: int) -> pd.DataFrame:
    out = frame.copy()
    out["target_bp"] = forward_return_bp(frame["ts"].to_numpy(), frame["mid"].to_numpy(), horizon_ms)
    return out.dropna(subset=["target_bp"])


def fill_separation(fills: pd.DataFrame, ts: np.ndarray, pred: np.ndarray, mid_ts: np.ndarray, mid: np.ndarray) -> dict:
    """Markout 1 s of mid-strategy fills, split by the model view on the last book before each fill."""
    sign = np.where(fills["side"] == "BUY", 1.0, -1.0)
    i = np.clip(np.searchsorted(ts, fills["ts"].to_numpy(), side="left") - 1, 0, None)
    view = sign * pred[i]  # > 0: the model expects the price to move our way
    later = mid[np.clip(np.searchsorted(mid_ts, fills["ts"].to_numpy() + 1_000 * NS_PER_MS, side="right") - 1, 0, None)]
    markout = sign * (later - fills["price"].to_numpy()) / fills["price"].to_numpy() * 1e4
    w = fills["qty"].to_numpy()
    against, toward = view < 0, view >= 0
    avg = lambda mask: float(np.average(markout[mask], weights=w[mask])) if mask.any() else np.nan
    return {"fills_against_share": float(w[against].sum() / w.sum()), "markout_against_bp": avg(against),
            "markout_toward_bp": avg(toward), "separation_bp": avg(toward) - avg(against),
            "corr_view_markout": float(np.corrcoef(view, markout)[0, 1])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--train", default="2026-03-19")
    p.add_argument("--valid", default="2026-03-20")
    p.add_argument("--test", default="2026-03-21")
    p.add_argument("--horizons", default="250,500,1000,2000,5000")
    p.add_argument("--fills", type=Path, help="fills of a mid-strategy run (for the toxicity separation)")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    for h in horizons:
        check_scales(ALL_WINDOWS_MS, h)

    source = ParquetDataSource(args.data, levels=5)
    feats = {}
    for day in (args.train, args.valid, args.test):
        data = source.load_day(day)
        feats[day] = build_features(data.book, data.trades, TICK)
    cols = feature_columns(feats[args.train])
    valid = feats[args.valid]
    fills = None
    if args.fills:
        fills = pd.read_parquet(args.fills)
        day_start = pd.Timestamp(args.valid).value
        fills = fills[(fills["ts"] >= day_start) & (fills["ts"] < day_start + 86_400 * 10**9)]

    rows = []
    for h in horizons:
        train = with_target(feats[args.train], h)
        fit_part, valid_part = time_split(train)
        model, info = fit_boosting(fit_part, valid_part, cols)
        v = with_target(valid, h)
        pred_v = model.predict(v[cols])
        row = {"horizon_ms": h, "trees": info["best_trees"], **evaluate("boosting", v["target_bp"].to_numpy(), pred_v)}
        row["share_abs_pred_over_0.47bp"] = float(np.mean(np.abs(pred_v) > 0.47))  # one tick at ~2150
        if fills is not None and len(fills):
            row |= fill_separation(fills, valid["ts"].to_numpy(), model.predict(valid[cols]),
                                   valid["ts"].to_numpy(), valid["mid"].to_numpy())
        rows.append(row)
        signal = pd.concat([pd.DataFrame({"ts": feats[d]["ts"].to_numpy(),
                                          "prediction_bp": model.predict(feats[d][cols]).astype(np.float32)})
                            for d in (args.valid, args.test)], ignore_index=True)
        signal.to_parquet(args.out / f"fv_h{h}.parquet")
        print(f"[{h} ms] done: corr {row['corr']:.3f}, R2 {row['r2']:.3f}", flush=True)

    table = pd.DataFrame(rows).drop(columns=["model"]).set_index("horizon_ms")
    (args.out / "horizon_report.json").write_text(json.dumps(table.reset_index().to_dict(orient="records"), indent=2))
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 220, "display.max_columns", None):
        print("\nVALIDATION DAY", args.valid, "\n", table)


if __name__ == "__main__":
    main()
