"""Fair-value models for long horizons: the impact horizon (deepest markout, 30 s) and the
reversion horizon (best markout, 60 min) of deep fills.

Usage: python train_horizon_models.py [--horizons 30000,3600000]

- Feature windows scale with the horizon (mm/signals/features.py: spec_for_horizon), and the
  <= 100x rule is checked. Windows longer than 2 h are dropped (one data file = one day).
- Train 03-19, validate 03-20, test 03-21. Training rows are subsampled in time, because
  neighbouring targets overlap almost fully at long horizons.
- Model: boosting or robust Ridge (clipped features), chosen on the last 20 % of the train day.
- Fill diagnostic: does the model view on the last book before a fill predict the part of the
  markout that this horizon is about? 30 s -> impact (markout 30 s - premium);
  60 min -> reversion (markout 60 min - markout 30 s). Fills: half-spread 0.45 runs, 03-20..21.
- Writes models/fv_h<ms>.parquet: predictions for every snapshot of 03-20..21.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mm.analytics.economics import fill_frame
from mm.data import ParquetDataSource
from mm.signals import build_features, feature_columns, forward_return_bp
from mm.signals.features import spec_for_horizon

TICK = 0.1
TRAIN, VALID, TEST = "2026-03-19", "2026-03-20", "2026-03-21"
MAX_WINDOW_MS = 2 * 3_600_000
NS_PER_MS = 1_000_000


class RobustRidge:
    """Clip each feature at its 1st / 99th train percentile, standardize, Ridge."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RobustRidge":
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        Z = X.clip(self.lo, self.hi, axis=1)
        self.mean, self.std = Z.mean(), Z.std().replace(0, 1)
        self.model = Ridge(alpha=10.0).fit((Z - self.mean) / self.std, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict((X.clip(self.lo, self.hi, axis=1) - self.mean) / self.std)


def r2(y, p) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def subsample(frame: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    """First snapshot of every step: fewer, less overlapping rows."""
    bucket = frame["ts"] // (step_ms * NS_PER_MS)
    return frame[~bucket.duplicated()]


def choose_model(train: pd.DataFrame, cols: list[str], horizon_ms: int):
    cut = train["ts"].iloc[int(len(train) * 0.8)]
    fit = train[train["ts"] < cut - horizon_ms * NS_PER_MS]
    val = train[train["ts"] >= cut]
    y_fit, y_val = fit["target_bp"].to_numpy(), val["target_bp"].to_numpy()
    candidates = {"robust_ridge": RobustRidge().fit(fit[cols], y_fit)}
    leaf = max(20, len(fit) // 50)
    best_trees, best_score = 25, -np.inf
    boost = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=leaf,
                                          l2_regularization=1.0, max_iter=25, warm_start=True, early_stopping=False,
                                          random_state=0)
    for n in range(25, 301, 25):
        boost.max_iter = n
        boost.fit(fit[cols], y_fit)
        score = r2(y_val, boost.predict(val[cols]))
        if score > best_score:
            best_trees, best_score = n, score
    candidates["boosting"] = HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=leaf, l2_regularization=1.0,
        max_iter=best_trees, random_state=0).fit(fit[cols], y_fit)
    scores = {name: r2(y_val, m.predict(val[cols])) for name, m in candidates.items()}
    name = max(scores, key=scores.get)
    final = (RobustRidge() if name == "robust_ridge" else HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=max(20, len(train) // 50), l2_regularization=1.0,
        max_iter=best_trees, random_state=0)).fit(train[cols], train["target_bp"].to_numpy())
    return name, final, scores, best_trees


def evaluate(y: np.ndarray, p: np.ndarray, horizon_ms: int, hours: float) -> dict:
    moved = np.abs(y) > 1e-9
    return {"r2": r2(y, p), "corr": float(np.corrcoef(y, p)[0, 1]),
            "sign_hit": float(np.mean(np.sign(p[moved]) == np.sign(y[moved]))),
            "pred_std_bp": float(p.std()), "target_std_bp": float(y.std()),
            "independent_samples": hours * 3_600_000 / horizon_ms}


def fill_diagnostic(fills: pd.DataFrame, frame: pd.DataFrame, pred: np.ndarray, horizon_ms: int) -> dict:
    """Corr of the model view (on the last book before each fill) with this horizon's markout part."""
    i = np.clip(np.searchsorted(frame["ts"].to_numpy(), fills["ts"].to_numpy(), side="left") - 1, 0, None)
    view = fills["sign"].to_numpy() * pred[i]  # > 0: the model expects a move in our favour
    part = (fills["mo_30s"] - fills["premium_bp"] if horizon_ms <= 30_000 else fills["mo_3600s"] - fills["mo_30s"]).to_numpy()
    w = fills["notional"].to_numpy()
    avg = lambda mask: float(np.average(part[mask], weights=w[mask])) if mask.any() else np.nan
    return {"fills": len(fills), "corr_view_part": float(np.corrcoef(view, part)[0, 1]),
            "part_when_view_for_bp": avg(view > 0), "part_when_view_against_bp": avg(view <= 0),
            "share_view_for": float(np.mean(view > 0))}


def half_spread_fills(name: str, skew: float | None, mid_ts, mid) -> pd.DataFrame:
    for p in glob.glob("output/fv_wide/*_metrics.json"):
        s = json.load(open(p))["config"]["strategy"]
        if s.get("name") == name and s["half_spread"] == 0.45 and (skew is None or s.get("inventory_skew") == skew):
            return fill_frame(pd.read_parquet(p.replace("_metrics.json", "_fills.parquet")), mid_ts, mid)
    raise SystemExit(f"fills not found: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", default="30000,3600000")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    source = ParquetDataSource(Path("data"), levels=5)
    raw = {day: source.load_day(day) for day in (TRAIN, VALID, TEST)}
    report = {}

    for horizon in (int(h) for h in args.horizons.split(",")):
        spec = spec_for_horizon(horizon, max_window_ms=MAX_WINDOW_MS)
        step_ms = max(1000, horizon // 60)
        frames = {}
        for day, data in raw.items():
            f = build_features(data.book, data.trades, TICK, spec)
            f["target_bp"] = forward_return_bp(f["ts"].to_numpy(), f["mid"].to_numpy(), horizon)
            frames[day] = f
        cols = feature_columns(frames[TRAIN].drop(columns=["target_bp"]))
        train = subsample(frames[TRAIN].dropna(subset=["target_bp"]), step_ms)
        name, model, val_scores, trees = choose_model(train, cols, horizon)

        result = {"spec": spec.__dict__, "train_rows": len(train), "step_ms": step_ms, "chosen": name,
                  "train_day_validation_r2": val_scores, "boosting_trees": trees}
        for day in (VALID, TEST):
            s = subsample(frames[day].dropna(subset=["target_bp"]), step_ms)
            result[day] = evaluate(s["target_bp"].to_numpy(), model.predict(s[cols]), horizon, 24)

        oos = pd.concat([frames[VALID], frames[TEST]], ignore_index=True)
        pred = model.predict(oos[cols])
        pd.DataFrame({"ts": oos["ts"].to_numpy(), "prediction_bp": pred.astype(np.float32)}).to_parquet(
            args.out / f"fv_h{horizon}.parquet")
        mid_ts, mid = oos["ts"].to_numpy(), oos["mid"].to_numpy()
        for label, (strategy, skew) in {"fair_value_1s_0.45": ("fair_value", 0.05), "mid_0.45": ("mid", None)}.items():
            result[f"fills_{label}"] = fill_diagnostic(half_spread_fills(strategy, skew, mid_ts, mid), oos, pred, horizon)
        report[horizon] = result
        print(f"[{horizon} ms] {name}: valid corr {result[VALID]['corr']:.3f}, test corr {result[TEST]['corr']:.3f}", flush=True)

    (args.out / "horizon_models_report.json").write_text(json.dumps(report, indent=2, default=float))
    for h, r in report.items():
        print(f"\n===== horizon {int(h) / 1000:g} s | model {r['chosen']} | train rows {r['train_rows']} (every {r['step_ms'] / 1000:g} s)")
        print("train-day validation R2:", {k: round(v, 3) for k, v in r["train_day_validation_r2"].items()})
        print(pd.DataFrame({d: r[d] for d in (VALID, TEST)}).round(3).to_string())
        print(pd.DataFrame({k[6:]: v for k, v in r.items() if k.startswith("fills_")}).round(3).to_string())


if __name__ == "__main__":
    main()
