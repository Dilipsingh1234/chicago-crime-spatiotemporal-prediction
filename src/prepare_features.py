"""Prepare model-ready datasets.

Run from the project root:
    python -m src.prepare_features
"""

from __future__ import annotations

from .config import CRIME_TYPE_FEATURE_FILE, OCCURRENCE_FEATURE_FILE, RAW_CRIME_FILE
from .feature_engineering import build_crime_occurrence_features, build_crime_type_features


def main() -> None:
    if not RAW_CRIME_FILE.exists():
        raise FileNotFoundError(
            f"Raw file not found: {RAW_CRIME_FILE}\n"
            "Place your CSV at data/raw/Chicago_Crimes_SEIS764.csv"
        )

    print("Creating crime occurrence features...")
    occurrence_df = build_crime_occurrence_features(RAW_CRIME_FILE, OCCURRENCE_FEATURE_FILE)
    print(f"Saved: {OCCURRENCE_FEATURE_FILE} | shape={occurrence_df.shape}")

    print("Creating grouped crime type features...")
    type_df = build_crime_type_features(RAW_CRIME_FILE, CRIME_TYPE_FEATURE_FILE)
    print(f"Saved: {CRIME_TYPE_FEATURE_FILE} | shape={type_df.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
