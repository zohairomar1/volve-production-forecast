# Volve Production Analytics

A production forecasting and KPI dashboard for oil & gas operations, built to support planning, maintenance triage, and stakeholder reporting.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![Tests](https://img.shields.io/badge/Tests-105%20Passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**[Live Demo](https://volve-analytics.streamlit.app/)**

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
| **Operations Copilot** | Rule-based Q&A over KPIs, anomalies, and forecasts with pluggable provider interface |
| **Knowledge Base** | In-app documentation with keyword search and cited answers |
| **Automation Status** | Pipeline health monitoring with test harness and failure simulation |
| **JSON Export** | Machine-readable summary payload for Power Automate / UiPath consumption |
| **SharePoint Integration** | Dual-mode I/O via Microsoft Graph API with local fallback |
| **Power Automate** | Automated weekly report distribution via Outlook |

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
- **Software Engineering**: Modular codebase, unit tests (105 passing), CI/CD with GitHub Actions
- **Enterprise Integration**: SharePoint I/O via Microsoft Graph API (Azure AD OAuth2, dual-mode with local fallback), Power Automate report automation
- **Copilot Architecture**: Vendor-agnostic provider interface (rule-based engine active; Azure OpenAI and Google Vertex AI stubs ready)
- **RAG Prototype**: Keyword-based document retrieval with cited answers from an in-app knowledge base
- **Automation Observability**: Pipeline health monitoring, test harness with failure simulation, JSON export for RPA tool consumption

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
│   └── io_sharepoint.py      # SharePoint I/O (Microsoft Graph API, dual-mode)
├── src/copilot/              # Operations Copilot (provider interface + rule engine)
│   ├── provider.py           # Abstract CopilotProvider base class
│   ├── rule_engine.py        # Rule-based engine (pattern match + data lookup)
│   └── placeholders.py       # AzureOpenAI / GoogleVertex stubs (TODO)
├── src/knowledge/            # Knowledge base (document loading + keyword retrieval)
├── src/automation/           # Automation status tracking + test harness
├── docs/                     # Knowledge base Markdown (KPIs, data dictionary, troubleshooting)
├── tests/                    # Unit tests (105 passing)
├── notebooks/                # EDA & model development
└── automation/               # Power Automate flow definition & docs
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
| Integration | Microsoft Graph API, Azure AD (OAuth2), Power Automate |

---

## Enterprise Integration

The pipeline supports optional SharePoint and Power Automate integration for enterprise data workflows.

```
SharePoint (Raw Data)  -->  Python Pipeline  -->  SharePoint (Processed)
                                                         |
                                                         v
                            Email (Recipients)  <--  Power Automate Flow
```

| Component | Details |
|-----------|---------|
| **Authentication** | Azure AD app registration (client credentials flow) |
| **API** | Microsoft Graph API v1.0 |
| **Operations** | List, download, upload files to SharePoint document libraries |
| **Fallback** | Local filesystem — works without any credentials |
| **Pipeline Flag** | `--sync-sharepoint` on `run_pipeline.py` |
| **Dashboard** | Sidebar shows connection status (Local / SharePoint) |
| **Report Distribution** | Power Automate flow sends weekly email with attachments |

```bash
# Run pipeline with SharePoint sync (falls back to local if no credentials)
python -m src.scripts.run_pipeline --sync-sharepoint

# Run pipeline without SharePoint (default)
python -m src.scripts.run_pipeline
```

Without credentials, the `--sync-sharepoint` flag copies outputs via the local fallback path, demonstrating the integration pattern without requiring a SharePoint tenant.

---

## Operations Copilot

The dashboard includes a rule-based Operations Copilot panel that answers questions about production data without requiring any external AI service.

**Capabilities:**
- Weekly operations summary (5-8 data-driven bullets)
- Forecast driver analysis (ETS vs Baseline comparison)
- Draft email summary generation (matches Power Automate report style)
- KPI, anomaly, and model validation Q&A
- Knowledge base search with cited answers

**Provider Architecture:**
The copilot uses an abstract `CopilotProvider` interface (`src/copilot/provider.py`). The active implementation is `RuleBasedProvider`, which generates answers deterministically from computed data. Placeholder classes exist for `AzureOpenAIProvider` and `GoogleVertexProvider` but are not enabled. To add an LLM provider, implement the `CopilotProvider.answer()` method and set `is_available = True` when credentials are configured.

No LLM or external AI service is called. All responses are generated from the same data visible in the dashboard.

---

## 5-Minute Demo Script

1. **Launch**: `streamlit run app/streamlit_app.py` -- point out the data source sidebar (Local / SharePoint)
2. **KPIs**: Highlight the 4 metric cards (oil, gas, water, YoY) and MoM sparkline
3. **Wellbore mode**: Switch to Single Wellbore -- show per-well diagnostics
4. **Forecast**: Toggle ETS vs Baseline, highlight the 95% confidence interval and model comparison table
5. **Anomaly Detection**: Adjust the z-score threshold slider -- show the dual-panel chart and sensitivity table
6. **Operations Copilot**: Click "Weekly Ops Summary", then type "What is WAPE?" to demo knowledge base retrieval
7. **Automation Status**: Expand the panel, run a test report, then simulate a credential failure
8. **JSON Export**: Click "Summary (JSON)" -- show the schema and explain how Power Automate / UiPath would consume it
9. **Knowledge Base**: Browse the data dictionary and KPI definitions

---

## Interview Talking Points

- **Forecasting**: "I chose ETS over ARIMA because the Volve data has clear additive seasonality and a declining trend. I validate with rolling-origin backtesting, not a single train/test split, because it better simulates real operational forecasting conditions."

- **Metric selection**: "MAPE is intuitive but breaks down with zero-production months. I added WAPE as a complementary metric and explain the tradeoff in the UI."

- **Copilot architecture**: "I built the copilot with a provider interface (abstract base class) so the rule-based engine can be swapped for Azure OpenAI or Vertex AI without changing the dashboard code. The rule engine is deterministic and uses only the data already computed by the pipeline."

- **RAG prototype**: "The knowledge base uses keyword-overlap scoring against Markdown docs. It's a lightweight RAG pattern -- no embeddings in this iteration, but the retrieval interface is the same shape you'd use with a vector store."

- **Enterprise integration**: "The SharePoint module uses a dual-mode pattern: Graph API when credentials are configured, local filesystem otherwise. This means the demo works anywhere without requiring a tenant."

- **Automation observability**: "The test harness lets me simulate failure modes (missing credentials, network errors) and verify the fallback behavior works. The JSON export is structured so Power Automate or UiPath can consume it directly."

---

## Future Improvements

- [ ] Incorporate maintenance schedules as exogenous variables
- [ ] Add change-point detection for structural break identification
- [ ] Implement well-level hierarchical forecasting
- [x] ~~Deploy to Streamlit Cloud~~ — [Live](https://volve-analytics.streamlit.app/)
- [x] ~~SharePoint integration~~ — Dual-mode I/O with local fallback
- [x] ~~Power Automate flow~~ — Weekly report distribution
- [ ] Add Prophet model for comparison
- [x] ~~Operations Copilot~~ — Rule-based engine with provider interface
- [x] ~~Knowledge Base~~ — In-app docs with keyword retrieval
- [x] ~~Automation Status~~ — Test harness with failure simulation
- [x] ~~JSON Export~~ — Machine-readable summary for RPA tools

---

## Future Project Ideas (TODO)

- [ ] **Ticket Triage RAG Bot**: Build a support ticket classifier using embeddings + vector search over historical tickets, with an LLM summarizer for resolution suggestions
- [ ] **Well Intervention Optimizer**: Combine production decline curves with maintenance cost data to recommend optimal intervention timing using survival analysis
- [ ] **LLM-Powered Copilot**: Swap in Azure OpenAI or Gemini via the existing provider interface for natural-language report generation
- [ ] **Real-Time Streaming**: Replace batch CSV ingestion with Kafka / Azure Event Hub for live production monitoring

---

## Contact

**Zohair Omar** — [GitHub](https://github.com/zohairomar1)
