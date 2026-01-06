# Data Directory

This directory contains raw input data and processed outputs for the Volve Production Analytics pipeline.

## Directory Structure

```
data/
├── README.md           # This file
├── raw/                # Raw input data (gitignored)
│   └── Volve production data.csv
└── processed/          # Pipeline outputs (gitignored, except samples)
    ├── volve_monthly.parquet
    ├── volve_monthly.csv
    ├── forecasts.csv
    ├── metrics.json
    └── email_summary.txt
```

## Obtaining the Data

### Option 1: From Course Repository

If you have the Applied Data Science Course repository:

```bash
cp "Course Notebooks/Data/Volve production data.csv" data/raw/
```

### Option 2: From Equinor/NPD

The Volve dataset is publicly available from Equinor's open data initiative:

1. Visit: https://www.equinor.com/energy/volve-data-sharing
2. Download the production data files
3. Place the monthly production CSV in `data/raw/`

### Option 3: SharePoint Integration

For enterprise deployments, configure SharePoint integration:

1. Copy `.env.example` to `.env`
2. Configure Azure AD credentials
3. Set SharePoint folder paths
4. Run the pipeline with `--sharepoint` flag

## Expected Input Format

The pipeline expects a CSV file with these columns:

| Column | Type | Description |
|--------|------|-------------|
| Wellbore name | string | Well identifier (e.g., "15/9-F-11") |
| NPDCode | integer | Norwegian Petroleum Directorate code |
| Year | integer | Production year |
| Month | integer | Production month (1-12) |
| On Stream | float | Operating hours |
| Oil | float | Oil production volume (Sm³) |
| Gas | float | Gas production volume (Sm³) |
| Water | float | Water production volume (Sm³) |
| GI | float | Gas injection (optional) |
| WI | float | Water injection (optional) |

## Output Files

After running the pipeline, these files are generated:

### volve_monthly.parquet / volve_monthly.csv
Cleaned and standardized production data with columns:
- `date`: Monthly datetime
- `wellbore`: Well identifier
- `oil`, `gas`, `water`: Production volumes
- `on_stream`: Operating hours
- `gas_injection`, `water_injection`: Injection volumes

### forecasts.csv
Production forecasts with columns:
- `date`: Forecast month
- `yhat`: Point forecast
- `yhat_lower`, `yhat_upper`: Confidence bounds
- `series_id`: Wellbore or "TOTAL"
- `model`: Model used ("baseline" or "ets")

### metrics.json
Model evaluation metrics from backtesting:
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)

### email_summary.txt
Human-readable summary for email distribution.

## Data Size

The Volve production dataset is relatively small:
- ~500 records
- ~7 wellbores
- Date range: 2007-2016

Large binary files (seismic, well logs) are NOT used in this project.
