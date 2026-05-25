# Related Work Notes

## Purpose of this section

The related work section should show that this project is connected to existing research in spatio-temporal crime prediction, but it should not overclaim novelty.

This project should be positioned as a reproducible empirical study rather than a new state-of-the-art model.

## Main research areas to cite

### 1. Spatio-temporal crime prediction

Prior work has studied crime forecasting as a spatial and temporal prediction problem. Many studies model crime at grid, district, neighborhood, or community-area level.

A 2023 systematic review of multi-scale spatio-temporal crime prediction summarizes common temporal/spatial methods, evaluation metrics, and limitations in the field. This can be cited in the introduction or related work section.

How this project connects:
- This project also treats crime prediction as a spatio-temporal problem.
- It uses Chicago Community Areas as spatial units.
- It uses hourly time slots as temporal units.
- It creates lag and rolling-window features to capture temporal dependence.

How this project differs:
- It focuses on a simple, reproducible supervised ML pipeline.
- It compares classical ML, gradient boosting, and neural network baselines.
- It uses a chronological non-shuffled evaluation setup.

### 2. Deep learning and graph-based crime prediction

Recent studies use advanced deep learning architectures such as Informer, ST-GCN, CNN-LSTM, and graph convolutional networks for crime forecasting.

One recent Chicago-focused study proposed an Informer + ST-GCN hybrid model for community-level crime prediction in Chicago. Another recent work used graph convolutional networks for crime hotspot prediction using the Chicago Crime Dataset.

How this project connects:
- The project uses the same broad idea of spatio-temporal crime forecasting.
- It also works with Chicago crime data and community-level patterns.

How this project differs:
- It does not propose a complex graph neural network.
- It emphasizes interpretability, simple reproducibility, and model comparison.
- It uses lag/rolling features and threshold tuning instead of advanced spatial graph modeling.

### 3. Interpretable machine learning for crime prediction

Interpretable ML is important because crime prediction is sensitive and cannot be treated as a black-box task.

Prior work has used XGBoost and SHAP to explain crime prediction models. Other work has used SHAP to interpret spatial effects from machine learning models.

How this project connects:
- This project includes feature importance / interpretability outputs.
- It uses feature importance to understand the role of temporal, lag, rolling, and community-area features.

How this project differs:
- The project focuses on a student/reproducible pipeline using public Chicago crime data.
- It compares multiple simple models rather than focusing only on one interpretable model.

### 4. Ethics and predictive policing bias

Crime prediction can be risky because reported crime data may reflect enforcement bias, reporting bias, and unequal policing patterns.

Prior work and policy discussions show that predictive policing systems can reinforce historical inequities if deployed without fairness review and community oversight.

How this project handles this:
- The paper should clearly state that the model predicts patterns in reported crime, not true crime.
- The model is not intended for individual-level policing.
- The results are for academic aggregate-level analysis only.
- Future deployment would require fairness, bias, privacy, and community-impact evaluation.

## How to position our contribution

Avoid saying:
- "This paper introduces a novel state-of-the-art algorithm."
- "This model can prevent crime."
- "This system should be used for policing deployment."

Use this instead:

This study presents a reproducible empirical pipeline for transforming event-only Chicago crime records into a supervised Community Area × Hour dataset and compares linear, tree-based, gradient boosting, and neural network models for crime occurrence prediction and grouped crime type forecasting under a chronological non-shuffled evaluation setup.

## Related work paragraph draft

Crime prediction has often been studied as a spatio-temporal forecasting problem, where historical crime records are aggregated over spatial units and time intervals to estimate future risk. Prior research has explored a range of spatial and temporal modeling strategies, including statistical approaches, machine learning baselines, deep neural networks, and graph-based methods. Recent work has also applied advanced architectures such as graph convolutional networks and transformer-style temporal models to Chicago crime data. Compared with these approaches, the present study does not aim to introduce a new deep architecture. Instead, it focuses on a reproducible empirical pipeline that converts event-only incident records into a Community Area × Hour supervised-learning dataset, compares several model families, and evaluates both crime occurrence prediction and grouped crime type forecasting using a chronological non-shuffled split.

## Ethics paragraph draft

Because crime datasets represent reported incidents rather than all crime, they may contain enforcement and reporting biases. Predictive models trained on such data can reproduce historical inequalities if they are used without careful review. For this reason, this study treats crime forecasting as an aggregate analytical task rather than a tool for individual-level policing or automated deployment. The results should be interpreted as patterns in reported crime data, and any real-world use would require fairness evaluation, privacy safeguards, transparency, human oversight, and community review.