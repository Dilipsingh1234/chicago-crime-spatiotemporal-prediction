# Results Summary

## Dataset

The project uses the Chicago Crime Dataset filtered from January 1, 2024 to December 31, 2025. The raw dataset contained approximately 494,273 incident records. After removing records with missing coordinates, approximately 492,776 usable records remained.

For crime occurrence prediction, the original event-only data was transformed into a supervised Community Area × Hour dataset. This produced 1,352,736 hourly community-area observations with both crime and no-crime examples.

## Task 1: Crime Occurrence Prediction

Crime occurrence prediction was formulated as a binary classification task. The target variable indicates whether at least one crime occurred in a given community area during a given hour.

A chronological, non-shuffled train-test split was used to avoid future data leaking into the training period.

Models evaluated:
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
- Neural Network MLP

Best overall model:
- Histogram Gradient Boosting

Key results:
- Accuracy: 0.7675
- ROC-AUC: 0.7484
- PR-AUC: 0.5160
- Best threshold: 0.27
- Best threshold F1-score: 0.5246

Random Forest achieved the strongest default-threshold F1-score and highest recall:
- F1-score: 0.5128
- Recall: 0.6288

This suggests that Histogram Gradient Boosting was better for overall ranking and threshold-tuned performance, while Random Forest was better when the goal was to capture more crime occurrence cases.

## Task 2: Grouped Crime Type Forecasting

Crime type forecasting was formulated as a multiclass classification task. Original Chicago crime categories were grouped into broader categories:
- Property
- Violent
- Other

Only rows where a crime occurred were used for this task.

Models evaluated:
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
- Neural Network MLP

Best balanced model:
- Random Forest

Key results for Random Forest:
- Accuracy: 0.4389
- Macro-F1: 0.4004
- Weighted-F1: 0.4578

Histogram Gradient Boosting achieved the highest accuracy:
- Accuracy: 0.5699

However, its Macro-F1 score was lower:
- Macro-F1: 0.2728

This suggests that the model may have favored the majority class. For this reason, Random Forest is treated as the best balanced model for crime type forecasting.

## Main Research Finding

Crime occurrence prediction produced stronger results than grouped crime type forecasting. Tree-based models performed better than the neural network baseline in the final experiments. The crime type forecasting task remained challenging, suggesting that additional external context such as weather, public events, holidays, population density, demographics, and points of interest may be needed for stronger category-level prediction.