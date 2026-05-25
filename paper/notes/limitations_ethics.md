# Limitations and Ethical Considerations

## Dataset limitations

The Chicago Crime Dataset contains reported crime incidents. It does not represent all crimes that occurred in the city. Some crimes may be underreported, while others may be more visible due to policing patterns, neighborhood conditions, or reporting behavior.

Because of this, the model learns patterns from recorded incidents rather than true underlying crime occurrence.

## Spatial limitations

The project uses Chicago Community Areas as the spatial unit. This is useful for aggregation and interpretation, but it hides smaller block-level differences. Two locations within the same community area may have very different risk patterns.

## Temporal limitations

The project uses hourly aggregation. This helps convert event data into a supervised forecasting task, but some crime types may follow longer-term patterns that are not fully captured by hourly features.

## Feature limitations

The current model uses spatial and temporal features, lag features, and rolling-window features. It does not include external context such as:
- Weather
- Public events
- Holidays
- School schedules
- Transit activity
- Population density
- Demographics
- Points of interest
- Street-level land use
- Police deployment patterns

These missing features may limit performance, especially for crime type forecasting.

## Modeling limitations

Crime occurrence prediction performed better than grouped crime type forecasting. The crime type task remained difficult because many crime categories overlap spatially and temporally.

The grouped labels Property, Violent, and Other reduce class imbalance, but they also simplify the complexity of real crime categories.

## Evaluation limitations

The project uses a chronological non-shuffled split, which is more realistic than random splitting. However, the model was evaluated on one city and one time period only. Results may not generalize to other cities or years without additional validation.

## Ethical considerations

Crime prediction can be sensitive because historical crime data may reflect enforcement bias, reporting bias, and unequal policing patterns. A model trained on this data could reproduce or amplify those patterns if used carelessly.

This project should be interpreted as an academic and analytical forecasting study. It is not designed for individual-level prediction, officer deployment, surveillance, or direct policing decisions.

Any real-world use would require:
- Bias and fairness testing
- Community review
- Transparency
- Privacy safeguards
- Human oversight
- Clear limits on operational use

## Responsible use statement

The results should be used only for understanding aggregate spatio-temporal patterns in reported crime. They should not be used to label individuals, justify aggressive policing, or make automated public-safety decisions.