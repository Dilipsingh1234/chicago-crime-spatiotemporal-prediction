# Run Order

This project is intentionally kept simple and paper-friendly.

## 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Place the raw CSV

Put your original CSV here:

```text
data/raw/Chicago_Crimes_SEIS764.csv
```

## 3. Prepare features

```bash
python -m src.prepare_features
```

This creates:

```text
data/processed/crime_occurrence_features.csv
data/processed/crime_type_grouped_features.csv
```

## 4. Create EDA figures for the paper

```bash
python -m src.make_eda_figures
```

Figures are saved to:

```text
results/figures/
```

## 5. Train crime occurrence models

```bash
python -m src.train_occurrence
```

This compares:

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
- Neural Network MLP

No random train/test shuffle is used. The last 20% of timestamps are used as the test set.

## 6. Train grouped crime type models

```bash
python -m src.train_crime_type
```

Crime types are grouped before training:

- Property
- Violent
- Other

Rows with `NO_CRIME` are removed only for the crime type task.

## 7. Use outputs in the paper

Important outputs:

```text
results/metrics/occurrence_model_comparison.csv
results/metrics/crime_type_model_comparison.csv
results/figures/*.png
results/predictions/*.csv
```
