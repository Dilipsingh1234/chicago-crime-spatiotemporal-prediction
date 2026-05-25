"""Time-based splitting helpers.

We avoid random shuffling because this is a forecasting problem.
"""

from __future__ import annotations

import pandas as pd


def chronological_train_test_split(
    df: pd.DataFrame,
    time_col: str = "hour_slot",
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split data by time without shuffling.

    The last `test_size` fraction of unique timestamps is used as the test set.
    This avoids mixing future records into training.
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)

    unique_times = data[time_col].drop_duplicates().sort_values().reset_index(drop=True)
    split_index = int(len(unique_times) * (1 - test_size))
    split_time = unique_times.iloc[split_index]

    train_df = data[data[time_col] < split_time].copy()
    test_df = data[data[time_col] >= split_time].copy()
    return train_df, test_df, split_time
