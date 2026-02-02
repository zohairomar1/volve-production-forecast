# Volve Production Analytics

A production forecasting and KPI dashboard for oil & gas operations, built to support planning, maintenance triage, and stakeholder reporting.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![Tests](https://img.shields.io/badge/Tests-31%20Passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Problem Statement

Production teams in upstream oil & gas face recurring challenges:

- **Operational Planning**: Forecasting future production for budgeting and resource allocation
- **Maintenance Triage**: Identifying wells with anomalous behavior that may need intervention
- **Stakeholder Reporting**: Communicating KPIs without manual data wrangling

This dashboard addresses these needs with an automated analytics pipeline that transforms raw production data into actionable insights.

---

## What's Inside

| Feature | Description |
|---------|-------------|
| **KPI Dashboard** | Monthly production totals, MoM/YoY changes, top wellbores |
| **Time-Series Forecasting** | ETS (Holt-Winters) with 95% confidence intervals |
| **Model Validation** | Rolling-origin backtesting with MAE, RMSE, MAPE, WAPE |
| **Anomaly Detection** | Rolling z-score method with configurable sensitivity |
| **Data Export** | Download filtered production, forecasts, and anomaly reports |

---

## Screenshots

| KPI Overview | Forecast & Validation | Anomaly Detection |
|--------------|----------------------|-------------------|
| ![KPI](reports/figures/dashboard_kpi.png) | ![Forecast](reports/figures/dashboard_forecast.png) | ![Anomaly](reports/figures/dashboard_anomaly.png) |

---

## Skills Demonstrated

- **Time-Series Forecasting**: Implemented ETS (exponential smoothing) with seasonal decomposition
- **Backtesting Methodology**: Rolling-origin cross-validation to simulate real forecast conditions
- **Metric Tradeoffs**: MAPE vs WAPE—understanding when percentage errors become unstable
- **Statistical Anomaly Detection**: Z-score normalization with rolling windows
- **Data Visualization**: Interactive Plotly charts with confidence intervals
- **Dashboard Development**: Streamlit with caching, filters, and responsive layout
- **Software Engineering**: Modular codebase, unit tests (31 passing), CI/CD with GitHub Actions

---

## Data Source

**Volve Field** dataset from Equinor's open data initiative—real offshore production data from the North Sea.

| Attribute | Value |
|-----------|-------|
| Coverage | 2008–2016 (field lifecycle) |
| Wellbores | 7 production wells |
| Variables | Oil, Gas, Water (Sm³), Operating Hours |
| Granularity | Monthly aggregates |

Source: [Equinor Volve Data Sharing](https://www.equinor.com/energy/volve-data-sharing)

---

## Analytical Decisions

Key choices made in the pipeline design:

1. **Field-level aggregation by default** — Shows macro trends; wellbore mode enables diagnostics
2. **Zero-production months excluded** — Prevents distortion from shut-in periods in training and MAPE
3. **Forecast floor at zero** — Production cannot be negative; model outputs are clipped
4. **12-month backtest window** — Provides statistically meaningful validation while preserving training data
5. **WAPE alongside MAPE** — More robust when actuals vary widely or approach zero
6. **Rolling 6-month window for anomalies** — Balances responsiveness vs. noise reduction
7. **Baseline as benchmark** — Seasonal naive model sets the bar ETS must beat

---

## Assumptions & Limitations

| Assumption | Implication |
|------------|-------------|
| 12-month seasonality | May not hold for all fields |
| No external factors | Drilling schedules, maintenance, reservoir pressure not incorporated |
| Statistical anomalies only | Flags indicate unusual behavior, not confirmed operational failures |
| Declining production trend | Model captures trend but may lag sharp structural breaks |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/zohairomar1/volve-production-forecast.git
cd volve-production-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add data
# Download from Equinor Volve Data Sharing and place in data/raw/
cp "path/to/Volve production data.csv" data/raw/

# Run pipeline
python -m src.scripts.run_pipeline

# Launch dashboard
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
volve-production-analytics/
├── app/
│   └── streamlit_app.py      # Dashboard entry point
├── src/
│   ├── data_prep.py          # Data loading & cleaning
│   ├── features.py           # Feature engineering
│   ├── forecasting.py        # ETS & baseline models
│   ├── evaluation.py         # Backtesting & metrics (MAE, MAPE, WAPE, RMSE)
│   ├── reporting.py          # Email summaries
│   └── io_sharepoint.py      # SharePoint integration
├── tests/                    # Unit tests (31 passing)
├── notebooks/                # EDA & model development
└── automation/               # Power Automate workflow docs
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.9+ |
| Data | Pandas, NumPy |
| Visualization | Plotly, Streamlit |
| Forecasting | statsmodels (ExponentialSmoothing) |
| Testing | Pytest |
| CI/CD | GitHub Actions |

---

## Future Improvements

- [ ] Incorporate maintenance schedules as exogenous variables
- [ ] Add change-point detection for structural break identification
- [ ] Implement well-level hierarchical forecasting
- [x] ~~Deploy to Streamlit Cloud for live demo~~
- [ ] Add Prophet model for comparison

---

## Contact

**Zohair Omar** — [GitHub](https://github.com/zohairomar1)
