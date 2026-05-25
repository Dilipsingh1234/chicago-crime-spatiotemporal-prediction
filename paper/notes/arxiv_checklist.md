# arXiv Readiness Checklist

## Target Submission Type

This manuscript is intended as an empirical machine learning study / technical research report.

Possible arXiv category:
- cs.LG: Machine Learning

Possible secondary categories:
- cs.AI: Artificial Intelligence
- cs.CY: Computers and Society, if ethics and public safety discussion is emphasized

## arXiv Readiness Requirements

### 1. Scholarly contribution

Status: In progress

The paper should not claim to introduce a new state-of-the-art crime prediction algorithm. Instead, it should present a reproducible empirical pipeline for transforming event-only crime records into a supervised spatio-temporal forecasting dataset.

Planned contribution statement:

This study transforms event-only Chicago crime incident records into a Community Area × Hour supervised learning dataset and compares machine learning and neural network models for crime occurrence prediction and grouped crime type forecasting under a chronological non-shuffled evaluation setup.

### 2. Originality and novelty

Status: Moderate

The novelty is not the model architecture. The contribution is the full reproducible pipeline:
- Event-only to crime/no-crime supervised dataset conversion
- Community Area × Hour aggregation
- Lag and rolling-window feature engineering
- Grouped crime type forecasting
- Chronological non-shuffled evaluation
- Model comparison across linear, tree-based, gradient boosting, and neural network models
- Threshold tuning and feature importance interpretation

### 3. Data and reproducibility

Status: Good

Dataset:
- Chicago Crime Dataset
- Date range: January 1, 2024 to December 31, 2025
- Raw records: approximately 494,273
- Usable records after coordinate filtering: approximately 492,776
- Processed occurrence dataset: 1,352,736 rows

Reproducibility requirements:
- Public data source must be cited.
- Code should be available on GitHub.
- Raw data should not be committed if too large.
- README should explain how to download/place the dataset.
- requirements.txt should be included.
- Non-shuffled train-test split should be clearly described.

### 4. Real results only

Status: Good

Use only actual generated results from:
- paper/tables/occurrence_model_comparison.csv
- paper/tables/crime_type_model_comparison.csv
- paper/tables/occurrence_split_info.json
- paper/tables/crime_type_split_info.json

No placeholder values should be included in the final paper.

### 5. References

Status: Not complete

Before submission:
- Add 12-20 verified references.
- Verify every author/title/venue/year.
- Do not use AI-generated references without checking.
- Prefer papers on spatio-temporal crime prediction, Chicago crime forecasting, graph/deep learning crime prediction, interpretability, and predictive policing ethics.

### 6. AI use and authorship

Status: Must be handled carefully

AI tools may be used for editing and drafting assistance, but:
- AI cannot be listed as an author.
- The human author is fully responsible for the manuscript.
- All claims, citations, tables, and figures must be verified by the author.
- No AI meta-comments or placeholders should remain.
- If submitting to an IEEE venue, AI-assisted writing must be disclosed in the acknowledgments section according to IEEE policy.

Possible disclosure wording if needed:

The author used ChatGPT for language editing, organization, and drafting assistance. All experimental design, code execution, result verification, analysis, and final manuscript review were performed by the author.

### 7. Ethical considerations

Status: Required

The paper must clearly state:
- Crime data represents reported crime, not all crime.
- Reported crime data can reflect enforcement and reporting bias.
- The model should not be used for individual-level policing decisions.
- The work is intended for academic analysis and aggregate-level forecasting.
- Future deployment would require fairness, bias, privacy, and community-impact evaluation.

### 8. Submission risk

Current risk level: Medium

Reasons:
- This began as a course project.
- The modeling approach is empirical rather than highly novel.
- Crime prediction is ethically sensitive.
- arXiv may reject weak or non-scholarly manuscripts.

Risk reduction:
- Strong related work section
- Honest contribution statement
- Clear methodology
- Real tables and figures
- Ethics and limitations section
- Clean GitHub repository
- Optional professor review before arXiv submission

## Current Decision

Do not submit to arXiv yet.

First:
1. Finish literature review.
2. Write IEEE-style paper.
3. Verify all references.
4. Review results and figures.
5. Publish GitHub repository.
6. Create Zenodo DOI.
7. Then consider arXiv submission.