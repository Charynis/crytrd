"""
Precision of high-confidence picks vs the base rate, from the walk-forward
predictions SwingFreqAI dumps during backtesting.

Run (from the project folder):
    docker compose run --rm --entrypoint python freqtrade user_data/scripts/evaluate_signals.py
"""

import sys
from pathlib import Path

import pandas as pd


PRED_DIR = Path(__file__).resolve().parents[1] / "backtest_predictions"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load() -> pd.DataFrame:
    files = sorted(PRED_DIR.glob("*.feather"))
    if not files:
        sys.exit(f"No prediction files in {PRED_DIR}. Run a backtest first.")
    df = pd.concat([pd.read_feather(f) for f in files], ignore_index=True)
    # Keep only candles the model actually scored and whose outcome is known.
    df = df[(df["do_predict"] == 1) & df["label"].notna() & df["swing"].notna()]
    df = df[df["swing"] > 0]  # rows before the first trained model are filled with 0
    df["hit"] = (df["label"] == "swing").astype(int)
    return df


def threshold_table(df: pd.DataFrame, base: float) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        picks = df[df["swing"] >= t]
        n = len(picks)
        prec = picks["hit"].mean() if n else float("nan")
        rows.append({
            "threshold": t,
            "signals": n,
            "signals/day": n / max(df["date"].dt.normalize().nunique(), 1),
            "pairs": picks["pair"].nunique(),
            "precision": prec,
            "lift": prec / base if n else float("nan"),
        })
    return pd.DataFrame(rows)


def top_k_per_day(df: pd.DataFrame, k: int) -> float:
    """Each day, take the k highest-probability pairs (best candle per pair)."""
    daily = (df.assign(day=df["date"].dt.normalize())
               .sort_values("swing", ascending=False)
               .drop_duplicates(["day", "pair"]))
    top = daily.groupby("day").head(k)
    return top["hit"].mean()


def main() -> None:
    df = load()
    base = df["hit"].mean()
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(f"Scored candles: {len(df):,}  pairs: {df['pair'].nunique()}  "
          f"range: {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(f"Base rate (any candle hits the target within the horizon): {base:.3f}\n")

    print("Precision by probability threshold")
    print(threshold_table(df, base).to_string(index=False))
    print("\nNote: hourly candles overlap heavily, so signal counts overstate independent trades.\n")

    print("Top-k picks per day")
    for k in (1, 3, 5):
        p = top_k_per_day(df, k)
        print(f"  top {k}: precision {p:.3f}  lift {p / base:.2f}")

    print("\nStability by month (threshold 0.65)")
    monthly = df.assign(month=df["date"].dt.tz_localize(None).dt.to_period("M"))
    summary = monthly.groupby("month").apply(
        lambda g: pd.Series({
            "base": g["hit"].mean(),
            "signals": int((g["swing"] >= 0.65).sum()),
            "precision": g.loc[g["swing"] >= 0.65, "hit"].mean(),
        }),
        include_groups=False,
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
