"""Reusable plotting functions for paper-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    ax = series.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_line_chart(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    ax = df.plot(x=x, y=y, kind="line", figsize=(12, 5), legend=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_model_comparison(metrics_df: pd.DataFrame, metric: str, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plot_df = metrics_df.sort_values(metric, ascending=False)
    ax = plot_df.plot(x="model", y=metric, kind="bar", legend=False, figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(1.0, float(plot_df[metric].max()) + 0.05))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, labels, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax, xticks_rotation=30)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_binary_curves(y_true, y_score, model_name: str, output_prefix: str | Path) -> None:
    output_prefix = Path(output_prefix)
    ensure_dir(output_prefix.parent)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_pr_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, feature_names: list[str], title: str, output_path: str | Path) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    values = np.asarray(model.feature_importances_)
    importance = pd.Series(values, index=feature_names).sort_values(ascending=False).head(15)
    ax = importance.sort_values().plot(kind="barh", figsize=(8, 6))
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
