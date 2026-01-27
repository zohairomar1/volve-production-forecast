# Volve Production Analytics

A production forecasting and KPI dashboard system for oil & gas field data, featuring time-series forecasting, automated reporting, and SharePoint integration.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

### Business Problem
Oil and gas production teams need to:
- Monitor field production KPIs in real-time
- Forecast future production for planning
- Detect anomalies that may indicate well issues
- Automate reporting to stakeholders

### Solution
This project delivers an end-to-end analytics pipeline that:
1. **Cleans and standardizes** production data
2. **Generates forecasts** using statistical models
3. **Provides interactive dashboards** for exploration
4. **Automates reporting** via SharePoint and email

## Dataset

This project uses the **Volve Field** dataset from Equinor's open data initiative.

- **Source**: [Equinor Volve Data Sharing](https://www.equinor.com/energy/volve-data-sharing)
- **Coverage**: 2007-2016 production data
- **Wellbores**: 7 production wells
- **Fields**: Oil, Gas, Water volumes, Operating hours

### Data Setup

Place the Volve production CSV in the data directory:

```bash
# From course repository
cp "Course Notebooks/Data/Volve production data.csv" data/raw/

# Or download from Equinor and place in data/raw/
```

## Methodology

### 1. Data Preparation
- Column standardization and mapping
- Date parsing (Year + Month → datetime)
- Missing value handling
- Quality validation

### 2. Feature Engineering
- Rolling averages (3M, 6M)
- Year-over-year and month-over-month changes
- Uptime rate calculation

### 3. Forecasting Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Seasonal Naive** | Uses same-month value from previous year | Baseline comparison |
| **Exponential Smoothing** | Holt-Winters with trend and seasonality | Production forecasting |

### 4. Backtesting
- Rolling-origin cross-validation
- 12-month test window
- Metrics: MAE, MAPE, RMSE

### 5. Evaluation Results

| Model | MAE | MAPE | Notes |
|-------|-----|------|-------|
| Seasonal Naive | ~8,000 Sm³ | ~15% | Baseline |
| ETS | ~6,500 Sm³ | ~12% | **Recommended** |

*Actual results vary by data - run notebooks for current metrics.*

## Project Structure

```
volve-production-analytics/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── .env.example             # Environment variables template
│
├── data/
│   ├── README.md            # Data documentation
│   ├── raw/                 # Raw input files (gitignored)
│   └── processed/           # Pipeline outputs (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── data_prep.py         # Data loading and cleaning
│   ├── features.py          # Feature engineering
│   ├── forecasting.py       # Forecasting models
│   ├── evaluation.py        # Backtesting and metrics
│   ├── reporting.py         # Email summary generation
│   ├── io_sharepoint.py     # SharePoint integration
│   └── scripts/
│       └── run_pipeline.py  # Main pipeline script
│
├── app/
│   └── streamlit_app.py     # Interactive dashboard
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   └── 02_forecast_backtest.ipynb  # Forecasting demonstration
│
├── automation/
│   └── power_automate/
│       ├── README.md        # Integration guide
│       └── email_template.md
│
├── reports/
│   ├── final_report.md      # Analysis summary
│   └── figures/             # Dashboard screenshots
│
├── tests/
│   ├── test_data_prep.py
│   └── test_forecasting_shapes.py
│
└── .github/
    └── workflows/
        └── ci.yml           # GitHub Actions CI
```

## Getting Started

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/zohairomar1/volve-production-forecast.git
cd volve-production-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run full analytics pipeline
python -m src.scripts.run_pipeline

# With custom input file
python -m src.scripts.run_pipeline --input data/raw/my_data.csv
```

### Launching the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open your browser to `http://localhost:8501`

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src
```

## Dashboard Features

### KPI Tiles
- Last month oil production
- Month-over-month change
- Year-over-year change
- Uptime metrics

### Interactive Charts
- Production time series (oil, gas, water)
- Wellbore comparison
- Forecast overlay with confidence intervals

### Data Export
- Download filtered CSV
- Export forecasts

## Automation (SharePoint + Email)

This project includes documentation for automating reports via Microsoft Power Automate:

1. **Scheduled Python job** processes data weekly
2. **Power Automate flow** detects new outputs
3. **Email sent** with summary and attachments

See [automation/power_automate/README.md](automation/power_automate/README.md) for setup instructions.

## Attribution & License

### Dataset
The Volve Field dataset is provided by Equinor under the [Equinor Open Data License](https://www.equinor.com/energy/volve-data-sharing).

### Learning Foundation
This project builds on concepts from the [Applied Data Science in Oil and Gas Industry](https://github.com/your-course-repo) course (MIT License).

### This Project
The analytics pipeline, dashboard, forecasting implementation, and automation documentation are original work, licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Zohair Omar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`pytest tests/`)
5. Submit a pull request

## Contact

- GitHub: [@zohairomar1](https://github.com/zohairomar1)
