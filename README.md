# PET-KLS brain-map pipeline

Regional PET-KLS topology is associated with Schaefer-100 normative brain maps. Computational pipeline only; no figures are generated.

## Inputs

Set the file locations in `config.yaml` (copy from `config.example.yaml`):

- `staging_metrics`: one observation per participant, with `subject_id`, `estdtt0`, `age`, `sex`, and 100 ROI columns per metric.
- `group_metrics`: one observation per participant, with `subject_id`, `cdr_glb_score`, `pet_cl_refwc_gca_gaain`, and the same ROI metrics. Use the baseline group-comparison sample, not the SILA-only sample.
- `brainmaps`: 100-row Schaefer-Yeo17 CSVs, each containing an ROI identifier and numeric biological features.

Recognized regional metric prefixes:

```text
participation_coeff_17Networks_
betweenness_17Networks_
clustering_coeff_17Networks_
strength_17Networks_
```

Supported ROI identifier columns include `ROI.Name_17_mod`, `ROI.Name_17`, and `roi`. Each input must contain exactly 100 unique matching parcels. A microstructure map can be added as an optional fourth modality.

## Run

```bash
python -m venv .venv
pip install -r requirements.txt
cp config.example.yaml config.yaml
python brainmaps_pipeline.py --config config.yaml
```

To run components independently:

```bash
python brainmaps_pipeline.py --config config.yaml --step temporal
python brainmaps_pipeline.py --config config.yaml --step group
```

The temporal analysis fits parcel-wise Gaussian GAMs (cubic B-splines, age and sex covariates) and computes Pearson correlations with normative maps at each estimated disease time. Group analysis computes Healthy and EarlyAD regional means using CDR 0 / Centiloid <24 and CDR 0.5 / Centiloid ≥24, respectively.

## Outputs

| File | Contents |
| --- | --- |
| `regional_gam_predictions.csv` | GAM-predicted regional graph properties |
| `temporal_spatial_correlations.csv` | Parcelwise Pearson r by time, graph property and biological feature |
| `temporal_feature_summary.csv` | Mean and maximum absolute correlation across the fitted time grid |
| `group_spatial_correlations.csv` | Healthy and EarlyAD spatial r and signed Δr |
| `group_sample_sizes.csv` | Data-derived participant counts |

Correlations are descriptive. This pipeline does **not** calculate spin-test p-values or false-discovery-rate-adjusted q-values. Corrected spatial inference requires a separate validated spatial-null analysis; do not equate ordinary parcelwise Pearson p-values with spatial-null significance.

Input data and generated results are excluded from version control by `.gitignore`.
