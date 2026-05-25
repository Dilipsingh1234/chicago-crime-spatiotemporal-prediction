"""Project configuration.

Keep paths and feature lists in one place so the notebooks and scripts stay simple.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

RAW_CRIME_FILE = RAW_DATA_DIR / "Chicago_Crimes_SEIS764.csv"
OCCURRENCE_FEATURE_FILE = PROCESSED_DATA_DIR / "crime_occurrence_features.csv"
CRIME_TYPE_FEATURE_FILE = PROCESSED_DATA_DIR / "crime_type_grouped_features.csv"

OCCURRENCE_TARGET = "crime_occurrence"
CRIME_TYPE_TARGET = "crime_category_group"
TIME_COLUMN = "hour_slot"

FEATURE_COLUMNS = [
    "Community Area",
    "hour",
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "rolling_3h_mean",
    "rolling_6h_mean",
    "rolling_12h_mean",
    "rolling_24h_mean",
]

RANDOM_STATE = 42
TEST_SIZE_BY_TIME = 0.20
