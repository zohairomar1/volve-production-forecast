# Data Dictionary

Reference for all data columns, units, and source information used in the Volve Production Analytics pipeline.

## Data Source

The dataset originates from the **Volve Field**, a North Sea oil field operated by Equinor (formerly Statoil). Equinor released the full Volve dataset under a CC BY-NC-SA 4.0 license for research and educational use.

- **Location**: North Sea, Norwegian Continental Shelf
- **Operator**: Equinor ASA
- **Production Period**: 2008 to 2016
- **Wellbores**: 7 production wells
- **Granularity**: Monthly production volumes

## Core Columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| date | datetime | YYYY-MM-DD | First day of the production month |
| wellbore | string | -- | Well identifier (e.g., 15/9-F-11) |
| oil | float | Sm3 | Monthly oil production volume |
| gas | float | Sm3 | Monthly gas production volume |
| water | float | Sm3 | Monthly produced water volume |
| on_stream | float | hours | Hours the well was producing during the month |

## Optional Columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| gas_injection | float | Sm3 | Gas injected for pressure maintenance |
| water_injection | float | Sm3 | Water injected for pressure maintenance |

## Engineered Features

These columns are added by the feature engineering pipeline (`src/features.py`):

| Column | Description |
|--------|-------------|
| year, month, quarter | Calendar components extracted from date |
| month_index | Sequential month counter from start of dataset |
| oil_rolling_3m, oil_rolling_6m | 3-month and 6-month rolling average of oil production |
| gas_rolling_3m, gas_rolling_6m | Rolling averages for gas |
| water_rolling_3m, water_rolling_6m | Rolling averages for water |
| oil_yoy_change | Year-over-year percentage change in oil |
| oil_mom_change | Month-over-month percentage change in oil |
| uptime_rate | Fraction of month the well was on-stream |

## Data Processing Pipeline

1. **Load** raw CSV from `data/raw/` (or SharePoint if configured)
2. **Standardize** column names via COLUMN_MAPPING in config.py
3. **Create** date column from year/month components or DATEPRD
4. **Clean** numeric values (coerce errors to NaN, remove negatives)
5. **Fill** missing injection values with 0; keep production NaN
6. **Select** output columns per OUTPUT_COLUMNS schema
7. **Save** to `data/processed/volve_monthly.parquet`

## Refresh Process

Data is static (historical Volve dataset). To update:
- Place new CSV in `data/raw/`
- Run `python -m src.scripts.run_pipeline`
- Optionally sync to SharePoint with `--sync-sharepoint`
