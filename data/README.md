# Data Folder

Place the raw Chicago crime CSV here:

```text
data/raw/Chicago_Crimes_SEIS764.csv
```

The raw dataset is not included in GitHub because it is large.

Generated processed files will be created here after running:

```bash
python -m src.prepare_features
```

Generated files:

```text
data/processed/crime_occurrence_features.csv
data/processed/crime_type_grouped_features.csv
```
