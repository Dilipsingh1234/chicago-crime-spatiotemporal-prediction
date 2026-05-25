"""Create basic EDA figures for the paper.

Run after `python -m src.prepare_features`:
    python -m src.make_eda_figures
"""

from __future__ import annotations

import pandas as pd

from .config import CRIME_TYPE_FEATURE_FILE, FIGURES_DIR, OCCURRENCE_FEATURE_FILE
from .plotting import ensure_dir, save_bar_chart, save_line_chart


def main() -> None:
    ensure_dir(FIGURES_DIR)

    occurrence = pd.read_csv(OCCURRENCE_FEATURE_FILE, parse_dates=["hour_slot"])
    crime_type = pd.read_csv(CRIME_TYPE_FEATURE_FILE, parse_dates=["hour_slot"])

    # 1. Binary class distribution
    occurrence_counts = occurrence["crime_occurrence"].map({0: "No Crime", 1: "Crime"}).value_counts()
    save_bar_chart(
        occurrence_counts,
        "Crime Occurrence Class Distribution",
        "Class",
        "Number of Area-Hour Rows",
        FIGURES_DIR / "01_occurrence_class_distribution.png",
    )

    # 2. Grouped crime category distribution among crime hours only
    type_counts = crime_type.loc[
        crime_type["crime_category_group"] != "NO_CRIME", "crime_category_group"
    ].value_counts()
    save_bar_chart(
        type_counts,
        "Grouped Crime Type Distribution",
        "Grouped Category",
        "Number of Area-Hour Rows",
        FIGURES_DIR / "02_grouped_crime_type_distribution.png",
    )

    # 3. Crime count by hour of day
    hourly = occurrence.groupby("hour", as_index=True)["crime_count"].sum().sort_index()
    save_bar_chart(
        hourly,
        "Total Crime Count by Hour of Day",
        "Hour of Day",
        "Crime Count",
        FIGURES_DIR / "03_crime_count_by_hour.png",
    )

    # 4. Crime count by day of week
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    dow = occurrence.groupby("day_of_week", as_index=True)["crime_count"].sum().rename(index=day_map)
    save_bar_chart(
        dow,
        "Total Crime Count by Day of Week",
        "Day of Week",
        "Crime Count",
        FIGURES_DIR / "04_crime_count_by_day_of_week.png",
    )

    # 5. Monthly trend
    monthly = (
        occurrence.assign(month_slot=occurrence["hour_slot"].dt.to_period("M").dt.to_timestamp())
        .groupby("month_slot", as_index=False)["crime_count"]
        .sum()
    )
    save_line_chart(
        monthly,
        x="month_slot",
        y="crime_count",
        title="Monthly Crime Count Trend",
        xlabel="Month",
        ylabel="Crime Count",
        output_path=FIGURES_DIR / "05_monthly_crime_trend.png",
    )

    # 6. Community area totals, top 15
    top_areas = occurrence.groupby("Community Area")["crime_count"].sum().sort_values(ascending=False).head(15)
    save_bar_chart(
        top_areas,
        "Top 15 Community Areas by Crime Count",
        "Community Area",
        "Crime Count",
        FIGURES_DIR / "06_top_community_areas.png",
    )

    print(f"Saved EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
