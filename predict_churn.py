"""Predict churn probability for new customer rows.

Train/save the model in Garrett_Wesley_Assignment5.ipynb.
This file loads that model and scores a pandas DataFrame.

The new LMS CSV does not match week 2 column names, so I align features first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
WEEK05_DATA = WORK_DIR.parents[2] / "data"
DEFAULT_NEW_DATA = WEEK05_DATA / "week_05_data_new_churn.csv"
DEFAULT_UNMOD_DATA = WEEK05_DATA / "week_05_data_new_churn_unmodified.csv"
DEFAULT_MODEL_NAME = str(WORK_DIR / "churn_pycaret_model")

# Same columns the week 2 cleaned training file uses (minus id/target)
FEATURE_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "PhoneService_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "avg_charge_per_tenure",
]


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load a CSV of new customers."""
    return pd.read_csv(filepath)


def _week2_features_from_string_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Turn Yes/No text categories into week-2 style columns."""
    out = df.copy()
    out["avg_charge_per_tenure"] = out["TotalCharges"] / out["tenure"]
    out["PhoneService_Yes"] = out["PhoneService"] == "Yes"
    out["Contract_One year"] = out["Contract"] == "One year"
    out["Contract_Two year"] = out["Contract"] == "Two year"
    out["PaymentMethod_Credit card (automatic)"] = (
        out["PaymentMethod"] == "Credit card (automatic)"
    )
    out["PaymentMethod_Electronic check"] = (
        out["PaymentMethod"] == "Electronic check"
    )
    out["PaymentMethod_Mailed check"] = out["PaymentMethod"] == "Mailed check"
    return out[FEATURE_COLS]


def align_features_like_week2(
    df: pd.DataFrame,
    unmodified_path: str | Path | None = None,
) -> pd.DataFrame:
    """Make new rows match the week 2 training columns.

    If the frame already looks like week 2, keep those columns.
    If it looks like new_churn_data.csv, pull Yes/No text from the unmodified
    companion file by customerID, then rebuild the week 2 columns.
    """
    if all(c in df.columns for c in FEATURE_COLS):
        return df[FEATURE_COLS].copy()

    if "charge_per_tenure" in df.columns or (
        "PhoneService" in df.columns and df["PhoneService"].dtype != object
    ):
        unmod_path = Path(unmodified_path) if unmodified_path else DEFAULT_UNMOD_DATA
        if "customerID" not in df.columns:
            raise ValueError(
                "new_churn rows need customerID so I can merge unmodified categories"
            )
        unmod = pd.read_csv(unmod_path)
        merged = df[
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]
        ].merge(
            unmod[["customerID", "PhoneService", "Contract", "PaymentMethod"]],
            on="customerID",
            how="left",
        )
        if merged[["PhoneService", "Contract", "PaymentMethod"]].isna().any().any():
            raise ValueError(
                "Could not merge unmodified categories for every customerID"
            )
        return _week2_features_from_string_cats(merged)

    if {"PhoneService", "Contract", "PaymentMethod"}.issubset(df.columns):
        return _week2_features_from_string_cats(df)

    raise ValueError(
        "Columns do not match week 2 training or new_churn format. "
        f"Got: {list(df.columns)}"
    )


def predict_churn_proba(
    df: pd.DataFrame,
    model_name: str | Path | None = None,
    unmodified_path: str | Path | None = None,
) -> pd.Series:
    """Return the chance each row churns (class 1 probability)."""
    from pycaret.classification import load_model, predict_model

    model_path = str(model_name or DEFAULT_MODEL_NAME)
    features = align_features_like_week2(df, unmodified_path=unmodified_path)
    model = load_model(model_path)
    preds = predict_model(model, data=features, raw_score=True)

    if "prediction_score_1" in preds.columns:
        probs = preds["prediction_score_1"]
    elif "Score_1" in preds.columns:
        probs = preds["Score_1"]
    else:
        score_cols = [c for c in preds.columns if "score" in c.lower()]
        if not score_cols:
            raise KeyError(
                f"No probability column in predict_model output: {preds.columns.tolist()}"
            )
        probs = preds[score_cols[-1]]

    return pd.Series(probs.to_numpy(), index=df.index, name="churn_probability")


if __name__ == "__main__":
    # Prefer a relative path when this file sits in the assignment work/ folder
    data_path = Path("../../../data/week_05_data_new_churn.csv")
    if not data_path.exists():
        data_path = DEFAULT_NEW_DATA
    print(f"Loading: {data_path.as_posix()}")
    new_df = load_data(data_path)
    print("Rows:", len(new_df))
    print("Columns:", list(new_df.columns))
    probs = predict_churn_proba(new_df)
    print("Churn probabilities:")
    print(probs)
