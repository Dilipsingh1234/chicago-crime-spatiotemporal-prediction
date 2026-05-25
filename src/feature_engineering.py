"""Feature engineering for Chicago crime prediction.

This file creates two clean model-ready datasets:
1. Crime occurrence prediction: binary label for each Community Area × Hour.
2. Crime type forecasting: grouped category label for hours where crime occurred.

Important: all lag and rolling features are computed using past values only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Simple, paper-friendly grouping. Anything not listed becomes "Other".
VIOLENT_CRIMES = {
    "HOMICIDE",
    "CRIMINAL SEXUAL ASSAULT",
    "SEX OFFENSE",
    "ASSAULT",
    "BATTERY",
    "ROBBERY",
    "KIDNAPPING",
    "INTIMIDATION",
    "OFFENSE INVOLVING CHILDREN",
    "STALKING",
}

PROPERTY_CRIMES = {
    "THEFT",
    "BURGLARY",
    "MOTOR VEHICLE THEFT",
    "CRIMINAL DAMAGE",
    "ARSON",
    "DECEPTIVE PRACTICE",
    "CRIMINAL TRESPASS",
}

OTHER_CRIMES = {
    "NARCOTICS",
    "WEAPONS VIOLATION",
    "OTHER OFFENSE",
    "PUBLIC PEACE VIOLATION",
    "INTERFERENCE WITH PUBLIC OFFICER",
    "PROSTITUTION",
    "GAMBLING",
    "LIQUOR LAW VIOLATION",
    "OBSCENITY",
    "CONCEALED CARRY LICENSE VIOLATION",
    "NON-CRIMINAL",
    "PUBLIC INDECENCY",
    "HUMAN TRAFFICKING",
    "OTHER NARCOTIC VIOLATION",
}


def group_primary_type(primary_type: str) -> str:
    """Map Chicago Primary Type to broad grouped categories."""
    value = str(primary_type).strip().upper()
    if value in VIOLENT_CRIMES:
        return "Violent"
    if value in PROPERTY_CRIMES:
        return "Property"
    if value in OTHER_CRIMES:
        return "Other"
    return "Other"


def _read_base_columns(file_path: str | Path, usecols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file_path, usecols=usecols)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Community Area"] = pd.to_numeric(df["Community Area"], errors="coerce")
    df = df.dropna(subset=["Date", "Community Area"]).copy()
    df["Community Area"] = df["Community Area"].astype(int)
    return df


def _build_full_area_hour_grid(df: pd.DataFrame, time_freq: str = "h") -> pd.DataFrame:
    """Create all Community Area × Hour rows, including no-crime hours."""
    all_areas = sorted(df["Community Area"].unique())
    start_day = df["Date"].dt.floor("D").min()
    end_day = df["Date"].dt.floor("D").max()
    all_hours = pd.date_range(start=start_day, end=end_day + pd.Timedelta(hours=23), freq=time_freq)
    return pd.MultiIndex.from_product(
        [all_areas, all_hours], names=["Community Area", "hour_slot"]
    ).to_frame(index=False)


def _add_time_lag_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add time, cyclical, lag, and rolling-window features.

    Rolling means use grouped.shift(1), so the current hour's crime count is not leaked
    into the feature value.
    """
    data = data.copy()
    data["hour"] = data["hour_slot"].dt.hour
    data["day_of_week"] = data["hour_slot"].dt.dayofweek
    data["month"] = data["hour_slot"].dt.month
    data["day"] = data["hour_slot"].dt.day
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    data = data.sort_values(["Community Area", "hour_slot"]).reset_index(drop=True)
    grouped = data.groupby("Community Area", sort=False)["crime_count"]

    for lag in [1, 2, 3, 24]:
        data[f"lag_{lag}h"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in [3, 6, 12, 24]:
        data[f"rolling_{window}h_mean"] = shifted.rolling(window).mean().reset_index(level=0, drop=True)

    lag_roll_cols = [
        "lag_1h",
        "lag_2h",
        "lag_3h",
        "lag_24h",
        "rolling_3h_mean",
        "rolling_6h_mean",
        "rolling_12h_mean",
        "rolling_24h_mean",
    ]
    data[lag_roll_cols] = data[lag_roll_cols].fillna(0)

    int_cols = ["Community Area", "crime_occurrence", "hour", "day_of_week", "month", "day", "is_weekend"]
    for col in int_cols:
        data[col] = data[col].astype(int)

    return data


def build_crime_occurrence_features(
    file_path: str | Path,
    output_path: Optional[str | Path] = None,
    time_freq: str = "h",
) -> pd.DataFrame:
    """Create the binary crime occurrence dataset."""
    df = _read_base_columns(file_path, usecols=["Date", "Community Area"])
    df["hour_slot"] = df["Date"].dt.floor(time_freq)

    crime_counts = (
        df.groupby(["Community Area", "hour_slot"])
        .size()
        .reset_index(name="crime_count")
    )
    crime_counts["crime_occurrence"] = (crime_counts["crime_count"] > 0).astype(int)

    data = _build_full_area_hour_grid(df, time_freq=time_freq)
    data = data.merge(crime_counts, on=["Community Area", "hour_slot"], how="left")
    data["crime_count"] = data["crime_count"].fillna(0).astype(int)
    data["crime_occurrence"] = data["crime_occurrence"].fillna(0).astype(int)
    data = _add_time_lag_rolling_features(data)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)

    return data


def build_crime_type_features(
    file_path: str | Path,
    output_path: Optional[str | Path] = None,
    time_freq: str = "h",
) -> pd.DataFrame:
    """Create grouped crime type data.

    For each Community Area × Hour with crime, the dominant grouped category is used.
    No-crime rows are retained as NO_CRIME during feature creation, but the training
    script removes them before fitting the crime type model.
    """
    df = _read_base_columns(file_path, usecols=["Date", "Community Area", "Primary Type"])
    df["Primary Type"] = df["Primary Type"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["Primary Type"]).copy()
    df["crime_category_group"] = df["Primary Type"].apply(group_primary_type)
    df["hour_slot"] = df["Date"].dt.floor(time_freq)

    crime_counts = (
        df.groupby(["Community Area", "hour_slot"])
        .size()
        .reset_index(name="crime_count")
    )
    crime_counts["crime_occurrence"] = (crime_counts["crime_count"] > 0).astype(int)

    type_counts = (
        df.groupby(["Community Area", "hour_slot", "crime_category_group"])
        .size()
        .reset_index(name="type_count")
    )
    dominant_type = (
        type_counts.sort_values(
            ["Community Area", "hour_slot", "type_count", "crime_category_group"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=["Community Area", "hour_slot"], keep="first")
        [["Community Area", "hour_slot", "crime_category_group"]]
    )

    data = _build_full_area_hour_grid(df, time_freq=time_freq)
    data = data.merge(crime_counts, on=["Community Area", "hour_slot"], how="left")
    data = data.merge(dominant_type, on=["Community Area", "hour_slot"], how="left")
    data["crime_count"] = data["crime_count"].fillna(0).astype(int)
    data["crime_occurrence"] = data["crime_occurrence"].fillna(0).astype(int)
    data["crime_category_group"] = data["crime_category_group"].fillna("NO_CRIME")
    data = _add_time_lag_rolling_features(data)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)

    return data
