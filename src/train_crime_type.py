"""Train multiple simple models for grouped crime type forecasting.

This predicts the grouped crime category only for rows where a crime occurred:
Property, Violent, or Other.

Run from the project root:
    python -m src.train_crime_type
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import (
    CRIME_TYPE_FEATURE_FILE,
    CRIME_TYPE_TARGET,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE_BY_TIME,
    TIME_COLUMN,
)
from .evaluate import evaluate_multiclass, save_json, save_metrics
from .plotting import ensure_dir, save_confusion_matrix, save_feature_importance, save_model_comparison
from .splitting import chronological_train_test_split


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
        "neural_network_mlp": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.0001,
                        batch_size=256,
                        learning_rate_init=0.001,
                        max_iter=35,
                        shuffle=False,
                        random_state=RANDOM_STATE,
                        early_stopping=False,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    ensure_dir(METRICS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(PREDICTIONS_DIR)
    ensure_dir(RESULTS_DIR / "models")

    df = pd.read_csv(CRIME_TYPE_FEATURE_FILE, parse_dates=[TIME_COLUMN])

    # Train crime type only for area-hour rows where at least one crime occurred.
    df = df[df[CRIME_TYPE_TARGET] != "NO_CRIME"].copy()
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    label_encoder = LabelEncoder()
    df["crime_category_encoded"] = label_encoder.fit_transform(df[CRIME_TYPE_TARGET])
    class_names = label_encoder.classes_.tolist()

    train_df, test_df, split_time = chronological_train_test_split(
        df, time_col=TIME_COLUMN, test_size=TEST_SIZE_BY_TIME
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["crime_category_encoded"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["crime_category_encoded"]

    split_info = {
        "split_type": "chronological_by_hour_no_shuffle",
        "split_time": str(split_time),
        "classes": class_names,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df[TIME_COLUMN].min()),
        "train_end": str(train_df[TIME_COLUMN].max()),
        "test_start": str(test_df[TIME_COLUMN].min()),
        "test_end": str(test_df[TIME_COLUMN].max()),
    }
    save_json(split_info, METRICS_DIR / "crime_type_split_info.json")

    summary_rows = []
    models = build_models()

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_multiclass(
            y_test,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
        )
        metrics["classes"] = class_names
        save_metrics(metrics, METRICS_DIR / f"crime_type_{name}.json")

        pred_labels = label_encoder.inverse_transform(y_pred)
        true_labels = label_encoder.inverse_transform(y_test)
        predictions = pd.DataFrame(
            {
                "hour_slot": test_df[TIME_COLUMN].values,
                "Community Area": test_df["Community Area"].values,
                "y_true": true_labels,
                "y_pred": pred_labels,
            }
        )
        predictions.to_csv(PREDICTIONS_DIR / f"crime_type_{name}_predictions.csv", index=False)

        save_confusion_matrix(
            true_labels,
            pred_labels,
            labels=class_names,
            title=f"Grouped Crime Type Confusion Matrix - {name}",
            output_path=FIGURES_DIR / f"crime_type_{name}_confusion_matrix.png",
        )

        raw_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
        save_feature_importance(
            raw_model,
            FEATURE_COLUMNS,
            title=f"Feature Importance - {name}",
            output_path=FIGURES_DIR / f"crime_type_{name}_feature_importance.png",
        )

        joblib.dump(model, RESULTS_DIR / "models" / f"crime_type_{name}.joblib")

        summary_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(METRICS_DIR / "crime_type_model_comparison.csv", index=False)

    save_model_comparison(
        summary,
        metric="macro_f1",
        title="Grouped Crime Type Model Comparison - Macro F1",
        output_path=FIGURES_DIR / "crime_type_model_comparison_macro_f1.png",
    )
    save_model_comparison(
        summary,
        metric="weighted_f1",
        title="Grouped Crime Type Model Comparison - Weighted F1",
        output_path=FIGURES_DIR / "crime_type_model_comparison_weighted_f1.png",
    )

    print("Done. Saved crime type metrics, predictions, models, and figures.")


if __name__ == "__main__":
    main()
