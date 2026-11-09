"""Context-as-covariates smoke test on SWARM-Lab/CoCo-Eval.

The point is not to squeeze alpha out of one stock.
The point is to make the data-alignment problem explicit:
- load a real dataset with text and targets,
- construct a one-step-ahead forecasting task,
- compare persistence against a covariate-aware baseline,
- print the exact wrong number on the first Boeing step,
- save a small diagnostic plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_SOURCE_NAME = "SWARM-Lab/CoCo-Eval"
DATA_SOURCE_URL = "https://huggingface.co/datasets/SWARM-Lab/CoCo-Eval"
DATASET_CONFIG = "fnspid"
SERIES_ID = "fnspid/ba"
OUTPUT_PLOT = Path("coco_eval_smoke_plot.png")


@dataclass
class ForecastResult:
    y_true: np.ndarray
    y_naive: np.ndarray
    y_model: np.ndarray
    test_frame: pd.DataFrame


def load_ba_series() -> pd.DataFrame:
    """Load and sort the Boeing series from the public CoCo-Eval release."""
    print(f"Loading dataset from: {DATA_SOURCE_URL}")
    ds = load_dataset(DATA_SOURCE_NAME, DATASET_CONFIG, split="test")
    df = ds.to_pandas()
    df = df.loc[df["dataset_id"] == SERIES_ID, ["timestamp", "target", "text"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["text"] = df["text"].fillna("").astype(str)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def make_supervised(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["y_next"] = out["target"].shift(-1)
    out["lag_1"] = out["target"]
    out["lag_5"] = out["target"].shift(4)
    out["delta_1"] = out["target"].diff()
    out["text_len"] = out["text"].str.len()
    out["word_count"] = out["text"].str.split().map(len)
    out = out.dropna(subset=["y_next", "lag_5", "delta_1"]).reset_index(drop=True)
    return out


def fit_and_forecast(df: pd.DataFrame) -> ForecastResult:
    feature_cols = ["lag_1", "lag_5", "delta_1", "text_len", "word_count", "text"]
    train_end = int(len(df) * 0.8)
    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:].copy()

    numeric_cols = ["lag_1", "lag_5", "delta_1", "text_len", "word_count"]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler(with_mean=False))]), numeric_cols),
            ("txt", TfidfVectorizer(max_features=250, ngram_range=(1, 2), min_df=2), "text"),
        ]
    )
    model = Pipeline(
        steps=[
            ("features", pre),
            ("ridge", Ridge(alpha=2.0, random_state=0)),
        ]
    )

    model.fit(train[feature_cols], train["y_next"])
    y_model = model.predict(test[feature_cols])
    y_naive = test["lag_1"].to_numpy()
    y_true = test["y_next"].to_numpy()

    return ForecastResult(y_true=y_true, y_naive=y_naive, y_model=y_model, test_frame=test)


def save_diagnostic_plot(result: ForecastResult) -> None:
    horizon = min(30, len(result.y_true))
    x = np.arange(horizon)
    plt.figure(figsize=(10, 4))
    plt.plot(x, result.y_true[:horizon], marker="o", linewidth=1.5, label="actual next target")
    plt.plot(x, result.y_naive[:horizon], marker="o", linewidth=1.2, label="naive persistence")
    plt.plot(x, result.y_model[:horizon], marker="o", linewidth=1.2, label="text + lag ridge")
    plt.title("CoCo-Eval smoke test: Boeing one-step forecast")
    plt.xlabel("test step")
    plt.ylabel("target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=160)
    plt.close()


def main() -> None:
    df = load_ba_series()
    supervised = make_supervised(df)

    first_naive = supervised.loc[0, "lag_1"]
    first_actual = supervised.loc[0, "y_next"]
    first_gap = abs(first_actual - first_naive)

    print("--- first observable wrong number ---")
    print(f"timestamp: {supervised.loc[0, 'timestamp'].date()}")
    print(f"naive carry-forward forecast: {first_naive:.6f}")
    print(f"actual next value:            {first_actual:.6f}")
    print(f"absolute miss:                {first_gap:.6f}")

    result = fit_and_forecast(supervised)
    naive_mae = mean_absolute_error(result.y_true, result.y_naive)
    model_mae = mean_absolute_error(result.y_true, result.y_model)

    print("\n--- out-of-sample smoke test ---")
    print(f"naive persistence MAE: {naive_mae:.6f}")
    print(f"text + lag ridge MAE:  {model_mae:.6f}")
    print(f"plot saved to: {OUTPUT_PLOT.resolve()}")

    save_diagnostic_plot(result)


if __name__ == "__main__":
    main()
