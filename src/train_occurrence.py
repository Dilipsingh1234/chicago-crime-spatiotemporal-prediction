"""Train multiple simple models for crime occurrence prediction.

This script is designed for paper-ready comparison:
- no random train/test shuffling
- chronological split by hour
- multiple models
- saved metrics, predictions, confusion matrices, ROC/PR curves, and feature importance

Run from the project root:
    python -m src.train_occurrence
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_DIR,
    OCCURRENCE_FEATURE_FILE,
    OCCURRENCE_TARGET,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE_BY_TIME,
    TIME_COLUMN,
)
from .evaluate import evaluate_binary, find_best_threshold, save_json, save_metrics
from .plotting import (
    ensure_dir,
    save_binary_curves,
    save_confusion_matrix,
    save_feature_importance,
    save_model_comparison,
)
from .splitting import chronological_train_test_split


def build_models() -> dict[str, object]:
    """Models kept intentionally simple and explainable."""
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
            min_samples_leaf=20,
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


def predict_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    # Fallback for models with decision_function only.
    scores = model.decision_function(X_test)
    return (scores - scores.min()) / (scores.max() - scores.min())


def main() -> None:
    ensure_dir(METRICS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(PREDICTIONS_DIR)
    ensure_dir(RESULTS_DIR / "models")

    df = pd.read_csv(OCCURRENCE_FEATURE_FILE, parse_dates=[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    train_df, test_df, split_time = chronological_train_test_split(
        df, time_col=TIME_COLUMN, test_size=TEST_SIZE_BY_TIME
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[OCCURRENCE_TARGET]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[OCCURRENCE_TARGET]

    split_info = {
        "split_type": "chronological_by_hour_no_shuffle",
        "split_time": str(split_time),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df[TIME_COLUMN].min()),
        "train_end": str(train_df[TIME_COLUMN].max()),
        "test_start": str(test_df[TIME_COLUMN].min()),
        "test_end": str(test_df[TIME_COLUMN].max()),
    }
    save_json(split_info, METRICS_DIR / "occurrence_split_info.json")

    summary_rows = []
    models = build_models()

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_score = predict_scores(model, X_test)

        default_metrics = evaluate_binary(y_test, y_score, threshold=0.5)
        best_metrics = find_best_threshold(y_test, y_score)

        save_metrics(default_metrics, METRICS_DIR / f"occurrence_{name}_threshold_050.json")
        save_metrics(best_metrics, METRICS_DIR / f"occurrence_{name}_best_threshold.json")

        y_pred_default = (y_score >= 0.5).astype(int)
        y_pred_best = (y_score >= best_metrics["threshold"]).astype(int)

        predictions = pd.DataFrame(
            {
                "hour_slot": test_df[TIME_COLUMN].values,
                "Community Area": test_df["Community Area"].values,
                "y_true": y_test.values,
                "probability_crime": y_score,
                "prediction_threshold_050": y_pred_default,
                "prediction_best_threshold": y_pred_best,
            }
        )
        predictions.to_csv(PREDICTIONS_DIR / f"occurrence_{name}_predictions.csv", index=False)

        save_confusion_matrix(
            y_test,
            y_pred_default,
            labels=["No Crime", "Crime"],
            title=f"Crime Occurrence Confusion Matrix - {name}",
            output_path=FIGURES_DIR / f"occurrence_{name}_confusion_matrix.png",
        )
        save_binary_curves(
            y_test,
            y_score,
            model_name=name,
            output_prefix=FIGURES_DIR / f"occurrence_{name}",
        )

        # Feature importance for tree model where available.
        raw_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
        save_feature_importance(
            raw_model,
            FEATURE_COLUMNS,
            title=f"Feature Importance - {name}",
            output_path=FIGURES_DIR / f"occurrence_{name}_feature_importance.png",
        )

        joblib.dump(model, RESULTS_DIR / "models" / f"occurrence_{name}.joblib")

        summary_rows.append(
            {
                "model": name,
                "accuracy": default_metrics["accuracy"],
                "precision": default_metrics["precision"],
                "recall": default_metrics["recall"],
                "f1": default_metrics["f1"],
                "roc_auc": default_metrics["roc_auc"],
                "pr_auc": default_metrics["pr_auc"],
                "best_threshold": best_metrics["threshold"],
                "best_threshold_f1": best_metrics["f1"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    summary.to_csv(METRICS_DIR / "occurrence_model_comparison.csv", index=False)

    save_model_comparison(
        summary,
        metric="f1",
        title="Crime Occurrence Model Comparison - F1 Score",
        output_path=FIGURES_DIR / "occurrence_model_comparison_f1.png",
    )
    save_model_comparison(
        summary,
        metric="pr_auc",
        title="Crime Occurrence Model Comparison - PR-AUC",
        output_path=FIGURES_DIR / "occurrence_model_comparison_pr_auc.png",
    )

    print("Done. Saved occurrence metrics, predictions, models, and figures.")


if __name__ == "__main__":
    main()
