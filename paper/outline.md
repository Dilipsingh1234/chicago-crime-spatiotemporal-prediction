# Paper Outline

## Title

Spatio-Temporal Crime Occurrence Prediction and Crime Type Forecasting in Chicago Using Interpretable Machine Learning

## Abstract

Briefly summarize the dataset, two prediction tasks, feature engineering, models, results, and limitations.

## I. Introduction

- Introduce urban crime forecasting as a spatio-temporal prediction problem.
- Explain why predicting whether crime may occur in a community area and hour can support analytical planning.
- Clarify that the work is not intended for direct policing deployment.
- Present the two tasks:
  1. Crime occurrence prediction
  2. Grouped crime type forecasting

## II. Related Work

- Prior crime prediction studies using spatial and temporal features.
- Chicago crime prediction studies.
- Deep learning and graph-based approaches.
- Interpretability and ethical concerns in predictive policing.

## III. Dataset and Preprocessing

- Chicago Crime Dataset.
- Date range: January 1, 2024 to December 31, 2025.
- Raw records: approximately 494,273.
- Usable records after removing missing coordinates: approximately 492,776.
- Community Area × Hour grid creation.
- Crime/no-crime target creation.
- Grouped crime category creation.

## IV. Feature Engineering

Features:
- Community area
- Hour
- Day of week
- Month
- Day
- Weekend indicator
- Cyclical hour encoding
- Lag features
- Rolling mean features

## V. Methodology

- Chronological non-shuffled train-test split.
- Models:
  - Logistic Regression
  - Random Forest
  - Histogram Gradient Boosting
  - Neural Network MLP
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
  - Macro-F1
  - Weighted-F1

## VI. Experimental Results

### A. Crime Occurrence Prediction

- Present Table I.
- Discuss Histogram Gradient Boosting as best overall.
- Discuss Random Forest as best high-recall model.
- Discuss threshold tuning.

### B. Grouped Crime Type Forecasting

- Present Table II.
- Discuss Random Forest as best balanced model.
- Discuss why accuracy alone is misleading.

## VII. Interpretability

- Feature importance.
- Lag and rolling features.
- Temporal patterns.
- Community-area effects.

## VIII. Limitations and Ethical Considerations

- Dataset reflects reported crime, not all crime.
- Possible enforcement and reporting bias.
- No demographic, weather, event, mobility, or POI features included.
- Not intended for direct individual-level policing decisions.
- Results should be interpreted as analytical forecasting, not deterministic prediction.

## IX. Conclusion and Future Work

- Summarize findings.
- Occurrence prediction performed better than crime type forecasting.
- Tree-based models performed strongly.
- Future work should include external context and stronger spatio-temporal models.