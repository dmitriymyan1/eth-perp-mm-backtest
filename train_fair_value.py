"""Train the fair-value model (mid change over the next 1 s), evaluate it out of sample,
and write the signal table the fair_value strategy uses.

Usage: python train_fair_value.py [--train 2026-03-19] [--test 2026-03-20,2026-03-21] [--out models]

No look-ahead:
- features at t use only data with ts <= t (mm/signals/features.py);
- model choice uses a time-ordered split of the TRAIN days (last 20 % as validation,
  with a gap of one horizon so overlapping targets do not leak);
- the signal file holds predictions for the TEST days only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mm.data import ParquetDataSource  # noqa: E402
from mm.signals import HORIZON_MS, build_features, feature_columns, forward_return_bp  # noqa: E402
from mm.signals.features import ALL_WINDOWS_MS, MAX_SCALE_RATIO  # noqa: E402

TICK = 0.1
VALIDATION_SHARE = 0.2
NS_PER_MS = 1_000_000


def load(source: ParquetDataSource, day: str) -> pd.DataFrame:
    data = source.load_day(day)
    frame = build_features(data.book, data.trades, TICK)
    frame["target_bp"] = forward_return_bp(frame["ts"].to_numpy(), frame["mid"].to_numpy(), HORIZON_MS)
    frame["day"] = day
    return frame


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last VALIDATION_SHARE of the rows as validation, with a one-horizon gap."""
    cut_ts = frame["ts"].iloc[int(len(frame) * (1 - VALIDATION_SHARE))]
    train = frame[frame["ts"] < cut_ts - HORIZON_MS * NS_PER_MS]
    valid = frame[frame["ts"] >= cut_ts]
    return train, valid


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_boosting(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str]) -> tuple[HistGradientBoostingRegressor, dict]:
    """Grow trees in steps; keep the number of trees with the best validation R2."""
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=2000,
                                          l2_regularization=1.0, max_iter=25, warm_start=True,
                                          early_stopping=False, random_state=0)
    history, best = [], (-np.inf, 0)
    for n_trees in range(25, 401, 25):
        model.max_iter = n_trees
        model.fit(train[cols], train["target_bp"])
        score = r2(valid["target_bp"].to_numpy(), model.predict(valid[cols]))
        history.append((n_trees, score))
        if score > best[0]:
            best = (score, n_trees)
        elif n_trees - best[1] >= 75:
            break
    model.set_params(max_iter=best[1], warm_start=False)
    model.fit(train[cols], train["target_bp"])
    return model, {"best_trees": best[1], "valid_r2": best[0], "history": history}


def evaluate(name: str, y: np.ndarray, pred: np.ndarray) -> dict:
    moved = np.abs(y) > 1e-9
    return {"model": name, "r2": r2(y, pred), "corr": float(np.corrcoef(y, pred)[0, 1]),
            "sign_hit_rate": float(np.mean(np.sign(pred[moved]) == np.sign(y[moved]))),
            "pred_std_bp": float(pred.std()), "target_std_bp": float(y.std())}


def decile_table(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    bins = pd.qcut(pred, 10, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": pred, "y": y, "bin": bins}).groupby("bin").agg(
        mean_pred_bp=("pred", "mean"), mean_realized_bp=("y", "mean"), count=("y", "size"))


def plot_deciles(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    lim = float(np.abs(table[["mean_pred_bp", "mean_realized_bp"]].to_numpy()).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="#52514e", lw=0.8, ls="--")
    ax.plot(table["mean_pred_bp"], table["mean_realized_bp"], color="#2a78d6", lw=2, marker="o", ms=6)
    ax.annotate("perfect calibration", (lim * 0.55, lim * 0.62), color="#52514e", fontsize=9)
    ax.set_xlabel("predicted 1 s mid change, bp (decile mean)", color="#52514e")
    ax.set_ylabel("realized 1 s mid change, bp", color="#52514e")
    ax.set_title("Fair-value model, out of sample: calibration by decile", loc="left", color="#0b0b0b")
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--train", default="2026-03-19")
    p.add_argument("--test", default="2026-03-20,2026-03-21")
    p.add_argument("--out", type=Path, default=Path("models"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = ParquetDataSource(args.data, levels=5)
    train_days, test_days = args.train.split(","), args.test.split(",")
    assert not set(train_days) & set(test_days), "train and test days must not overlap"

    train_all = pd.concat([load(source, d) for d in train_days]).dropna(subset=["target_bp"])
    cols = feature_columns(train_all.drop(columns=["target_bp", "day"]))
    fit_part, valid_part = time_split(train_all)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(fit_part[cols], fit_part["target_bp"])
    micro = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(fit_part[["microprice_ticks"]], fit_part["target_bp"])
    boosting, boost_info = fit_boosting(fit_part, valid_part, cols)
    valid_scores = {name: r2(valid_part["target_bp"].to_numpy(), m.predict(valid_part[c]))
                    for name, m, c in (("ridge", ridge, cols), ("microprice_only", micro, ["microprice_ticks"]),
                                       ("boosting", boosting, cols))}
    best_name = max(valid_scores, key=valid_scores.get)
    del train_all, fit_part, valid_part

    test_scores, signal_parts, deciles = [], [], []
    for day in test_days:
        frame = load(source, day)
        predictions = {name: m.predict(frame[c]) for name, (m, c) in
                       {"ridge": (ridge, cols), "microprice_only": (micro, ["microprice_ticks"]),
                        "boosting": (boosting, cols)}.items()}
        known = frame["target_bp"].notna().to_numpy()
        y = frame["target_bp"].to_numpy()[known]
        for name, pred in predictions.items():
            test_scores.append({"day": day, **evaluate(name, y, pred[known])})
        chosen = predictions[best_name]
        signal_parts.append(pd.DataFrame({"ts": frame["ts"].to_numpy(), "prediction_bp": chosen.astype(np.float32)}))
        deciles.append(pd.DataFrame({"pred": chosen[known], "y": y}))
        del frame

    signal = pd.concat(signal_parts, ignore_index=True)
    signal.to_parquet(args.out / "fair_value_signal.parquet")
    all_test = pd.concat(deciles)
    table = decile_table(all_test["y"].to_numpy(), all_test["pred"].to_numpy())
    plot_deciles(table, args.out / "fair_value_deciles.png")

    report = {"horizon_ms": HORIZON_MS, "feature_windows_ms": list(ALL_WINDOWS_MS), "max_scale_ratio": MAX_SCALE_RATIO,
              "features": cols, "train_days": train_days, "test_days": test_days,
              "validation_r2": valid_scores, "chosen_model": best_name, "boosting": boost_info,
              "test": test_scores, "test_deciles": table.reset_index().to_dict(orient="records"),
              "ridge_coefficients_std": dict(zip(cols, ridge[-1].coef_.round(4).tolist()))}
    (args.out / "fair_value_report.json").write_text(json.dumps(report, indent=2, default=float))

    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 200):
        print("validation R2 (train day, last 20 %):", {k: round(v, 4) for k, v in valid_scores.items()})
        print("chosen model:", best_name, "| boosting trees:", boost_info["best_trees"])
        print("\nOUT OF SAMPLE:\n", pd.DataFrame(test_scores).set_index(["day", "model"]))
        print("\nOut-of-sample deciles (chosen model):\n", table)
    print(f"\nsignal: {args.out / 'fair_value_signal.parquet'} ({len(signal):,} rows, test days only)")


if __name__ == "__main__":
    main()
