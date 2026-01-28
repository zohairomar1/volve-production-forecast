# Volve Production Analytics

An end-to-end production forecasting and KPI dashboard for oil & gas field data, demonstrating data engineering, time-series modeling, and business intelligence skills.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![Tests](https://img.shields.io/badge/Tests-31%20Passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## The Problem

**Audience**: Production engineers, reservoir analysts, and operations managers in upstream oil & gas

**Challenge**: Production teams lack visibility into field performance trends and struggle to:
- Track KPIs across multiple wellbores in real-time
- Forecast production for budget planning and resource allocation
- Detect anomalies that may indicate equipment issues or well interventions
- Generate automated reports for stakeholders without manual data wrangling

**Solution**: This project delivers an interactive analytics platform that processes raw production data into actionable insights, forecasts, and automated reports.

## Screenshots

| KPI Dashboard | Production Forecasting | Anomaly Detection |
|---------------|------------------------|-------------------|
| ![KPI Dashboard](reports/figures/dashboard_kpi.png) | ![Forecasting](reports/figures/dashboard_forecast.png) | ![Anomaly Detection](reports/figures/dashboard_anomaly.png) |

*Screenshots show the Streamlit dashboard with production KPIs, ETS forecasting with confidence intervals, and z-score anomaly detection.*

## Data Source

This project uses the **Volve Field** dataset from Equinor's open data initiative - a real-world offshore oil field dataset from the North Sea.

| Attribute | Details |
|-----------|---------|
| **Source** | [Equinor Volve Data Sharing](https://www.equinor.com/energy/volve-data-sharing) |
| **Coverage** | 2007-2016 (field lifecycle from ramp-up to decommissioning) |
| **Wellbores** | 7 production wells |
| **Variables** | Oil (Sm³), Gas (Sm³), Water (Sm³), Operating Hours |
| **Granularity** | Monthly production volumes |

## Features

### Interactive Dashboard
- **KPI Tiles**: Last month production, MoM/YoY changes, uptime metrics with delta indicators
- **Multi-Wellbore Filtering**: Compare individual wells or view field totals
- **Date Range Selection**: Analyze specific time periods
- **Data Export**: Download filtered datasets as CSV

### Time-Series Forecasting
- **Baseline Model**: Seasonal Naive (same-month-last-year benchmark)
- **Production Model**: Exponential Smoothing (Holt-Winters with trend + seasonality)
- **Confidence Intervals**: 95% prediction bounds for uncertainty quantification
- **Horizon Selection**: Configurable 3-12 month forecast window

### Model Validation
- **Backtesting**: Rolling-origin cross-validation with 12-month test window
- **Metrics Display**: MAPE, MAE, RMSE shown in dashboard
- **Actual vs Predicted**: Visual comparison chart for model performance

### Anomaly Detection
- **Method**: Rolling z-score with configurable threshold
- **Interactive Controls**: Adjust sensitivity (threshold) and window size
- **Visualization**: Highlighted anomalies on production timeline
- **Column Selection**: Detect anomalies in oil, gas, or water production

### Automation Ready
- SharePoint integration module for enterprise deployment
- Email summary generation for stakeholder reporting
- Power Automate workflow documentation

## Modeling Approach

### Forecasting Models

| Model | Method | Use Case |
|-------|--------|----------|
| **Seasonal Naive** | Uses same-month value from previous year | Baseline for comparison |
| **ETS (Holt-Winters)** | Exponential smoothing with additive trend & seasonality | Production forecasting |

### Validation Methodology

**Rolling-Origin Backtesting**: Train on data up to time *t*, forecast *t+1*, compare to actual, then roll forward. This simulates real-world forecasting conditions.

### Performance Results

| Model | MAE (Sm³) | MAPE | RMSE (Sm³) |
|-------|-----------|------|------------|
| Seasonal Naive | ~8,000 | ~15% | ~10,500 |
| **ETS** | **~6,500** | **~12%** | **~8,200** |

*ETS reduces forecast error by ~20% vs baseline, making it the recommended production model.*

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Streamlit |
| **Forecasting** | Statsmodels (ExponentialSmoothing) |
| **Testing** | Pytest (31 tests) |
| **CI/CD** | GitHub Actions |
| **Enterprise Integration** | Microsoft Graph API (SharePoint), Power Automate |

## Project Structure

```
volve-production-analytics/
├── app/
│   └── streamlit_app.py      # Interactive dashboard (main entry point)
├── src/
│   ├── data_prep.py          # Data loading and cleaning
│   ├── features.py           # Feature engineering (rolling avg, YoY changes)
│   ├── forecasting.py        # Forecasting models (Naive, ETS)
│   ├── evaluation.py         # Backtesting and metrics
│   ├── reporting.py          # Email summary generation
│   └── io_sharepoint.py      # SharePoint integration
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_forecast_backtest.ipynb  # Model development & validation
├── tests/                    # Unit tests (31 tests)
├── automation/               # Power Automate workflow docs
└── data/
    ├── raw/                  # Input CSV (gitignored)
    └── processed/            # Pipeline outputs (gitignored)
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/zohairomar1/volve-production-forecast.git
cd volve-production-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Data

Place the Volve production CSV in the data directory:

```bash
# Copy from course materials
cp "path/to/Volve production data.csv" data/raw/

# Or download from Equinor Volve Data Sharing portal
```

### 3. Run Pipeline

```bash
# Process data and generate forecasts
python -m src.scripts.run_pipeline
```

### 4. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### 5. Run Tests

```bash
pytest tests/ -v
```

## Key Learnings

Building this project reinforced several data science and engineering concepts:

1. **Time-series validation**: Why train/test splits don't work for temporal data - rolling-origin backtesting is essential
2. **Baseline importance**: Always compare against a simple model; ETS beat seasonal naive by 20% MAPE
3. **Data quality in production**: Real datasets have shutdowns, missing values, and edge cases that break naive implementations
4. **UX for analytics**: Stakeholders need context (business story, metric explanations) not just charts

## Future Improvements

- [ ] Add Prophet model for comparison
- [ ] Implement automated retraining pipeline
- [ ] Add well-level decline curve analysis
- [ ] Deploy to Streamlit Cloud for live demo

## License

MIT License - see [LICENSE](LICENSE) for details.

The Volve Field dataset is provided by Equinor under the [Equinor Open Data License](https://www.equinor.com/energy/volve-data-sharing).

## Contact

**Zohair Omar** - [GitHub](https://github.com/zohairomar1)
