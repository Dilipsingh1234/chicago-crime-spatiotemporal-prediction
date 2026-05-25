# Spatio-Temporal Crime Occurrence Prediction and Crime Type Forecasting in Chicago

This project presents a reproducible machine learning pipeline for short-term crime occurrence prediction and grouped crime type forecasting using the Chicago Crime Dataset.

The original event-level crime records were transformed into a supervised Community Area × Hour dataset, allowing models to learn from both crime and no-crime observations.

## Project Goals

This project addresses two prediction tasks:

1. **Crime Occurrence Prediction**  
   Predict whether at least one crime will occur in a given Chicago community area during a given hour.

2. **Grouped Crime Type Forecasting**  
   Predict the broader crime category after a crime occurrence is observed.

The grouped crime categories are:

- Property
- Violent
- Other

## Dataset

Dataset source: Chicago Data Portal — Crimes 2001 to Present

Date range used:

- January 1, 2024 to December 31, 2025

Dataset summary:

- Raw records: approximately 494,273
- Usable records after removing missing coordinates: approximately 492,776
- Processed Community Area × Hour observations: 1,352,736

The raw dataset is not included in this repository because of file size. Download the dataset from the Chicago Data Portal and place it here:

```text
data/raw/Chicago_Crimes_SEIS764.csv
Feature Engineering

Features include:

Community area
Hour
Day of week
Month
Day of month
Weekend indicator
Cyclical hour encoding using sine and cosine
Lag features: 1h, 2h, 3h, 24h
Rolling mean features: 3h, 6h, 12h, 24h
Models Evaluated

For both tasks, the following models were evaluated:

Logistic Regression
Random Forest
Histogram Gradient Boosting
Neural Network MLP

A chronological non-shuffled train-test split was used to avoid future data leakage.

Results Summary
Crime Occurrence Prediction

Best overall ranking and threshold-tuned model:

Histogram Gradient Boosting
Accuracy: 0.767
ROC-AUC: 0.748
PR-AUC: 0.516
Best threshold F1-score: 0.525

Best high-recall model:

Random Forest
Recall: 0.629
Default-threshold F1-score: 0.513
Grouped Crime Type Forecasting

Best balanced model:

Random Forest
Accuracy: 0.439
Macro-F1: 0.400
Weighted-F1: 0.458

Histogram Gradient Boosting achieved higher accuracy, but lower Macro-F1, suggesting weaker class-balanced performance.

Run Instructions

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Prepare features:

python -m src.prepare_features

Generate EDA figures:

python -m src.make_eda_figures

Train crime occurrence models:

python -m src.train_occurrence

Train grouped crime type models:

python -m src.train_crime_type
Project Structure
crime_prediction_chicago_publishable/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── config.py
│   ├── feature_engineering.py
│   ├── prepare_features.py
│   ├── make_eda_figures.py
│   ├── train_occurrence.py
│   ├── train_crime_type.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── predictions/
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE