"""
Volve Production Analytics Dashboard
====================================

Portfolio-ready interactive dashboard for oil & gas production monitoring,
forecasting, and anomaly detection.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_prep import prepare_data, load_processed_data, aggregate_total_production
from src.features import engineer_features
from src.forecasting import forecast_series, get_historical_with_forecast
from src.reporting import get_last_month_summary, get_top_wellbores

# Page config
st.set_page_config(
    page_title="Volve Production Analytics",
    page_icon=":material/oil_barrel:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Metric styling */
    [data-testid="stMetricValue"] { color: #1f2937 !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #374151 !important; font-size: 0.9rem !important; }
    [data-testid="stMetricDelta"] svg { display: none; }
    .stMetric { background-color: #f8fafc; border-radius: 8px; padding: 15px; border: 1px solid #e2e8f0; }

    /* Info boxes */
    .info-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; }
    .info-box h4 { margin: 0 0 0.5rem 0; }
    .info-box ul { margin: 0; padding-left: 1.2rem; }

    /* Section headers */
    .section-header { border-left: 4px solid #667eea; padding-left: 10px; margin: 1.5rem 0 1rem 0; }

    /* Tooltip styling */
    .tooltip { font-size: 0.8rem; color: #6b7280; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load and cache production data."""
    processed_path = PROCESSED_DATA_DIR / "volve_monthly.parquet"
    if processed_path.exists():
        df = load_processed_data(processed_path)
    else:
        raw_paths = [
            RAW_DATA_DIR / "Volve production data.csv",
            project_root.parent / "Course Notebooks" / "Data" / "Volve production data.csv",
        ]
        for path in raw_paths:
            if path.exists():
                df = prepare_data(path, save_output=True)
                break
        else:
            st.error("No data found. Please place Volve production data in data/raw/")
            return None
    df = engineer_features(df)
    return df


@st.cache_data
def get_active_production_data(df):
    """Filter to only include periods with active production."""
    monthly_totals = df.groupby("date")["oil"].sum()
    active_months = monthly_totals[monthly_totals > 0]
    if len(active_months) == 0:
        return df
    last_active_date = active_months.index.max()
    return df[df["date"] <= last_active_date]


@st.cache_data
def generate_forecast(df, series_id, model, horizon=6):
    """Generate forecast using only active production periods."""
    try:
        df_active = get_active_production_data(df)
        forecast_df = forecast_series(df_active, target_col="oil", series_id=series_id, model=model, horizon=horizon)
        return forecast_df
    except Exception:
        return None


@st.cache_data
def run_backtest(df, series_id, model, test_periods=12):
    """Run backtesting and return metrics."""
    from src.evaluation import rolling_origin_backtest, compute_backtest_metrics

    df_active = get_active_production_data(df)

    try:
        backtest_results = rolling_origin_backtest(
            df_active, target_col="oil", series_id=series_id,
            model=model, test_periods=test_periods, forecast_horizon=1
        )
        metrics = compute_backtest_metrics(backtest_results)
        return backtest_results, metrics
    except Exception:
        return None, None


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def detect_anomalies_zscore(df, column="oil", window=6, threshold=2.0):
    """Detect anomalies using rolling z-score method."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df["rolling_mean"] = df[column].rolling(window=window, min_periods=3).mean()
    df["rolling_std"] = df[column].rolling(window=window, min_periods=3).std()

    df["zscore"] = np.where(
        df["rolling_std"] > 0,
        (df[column] - df["rolling_mean"]) / df["rolling_std"],
        0
    )
    df["is_anomaly"] = np.abs(df["zscore"]) > threshold
    df["residual"] = df[column] - df["rolling_mean"]

    return df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_number(value, suffix=""):
    """Format large numbers with K/M suffixes."""
    if pd.isna(value) or value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M{suffix}"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K{suffix}"
    else:
        return f"{value:.0f}{suffix}"


def create_sparkline(series, height=50):
    """Create a simple sparkline figure."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=series.values, mode="lines",
        line=dict(color="#667eea", width=2),
        fill="tozeroy", fillcolor="rgba(102, 126, 234, 0.1)"
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # =========================================================================
    # HEADER & BUSINESS CONTEXT
    # =========================================================================
    st.title("Volve Production Analytics")

    # Business Story Section
    with st.expander("About This Dashboard", expanded=False, icon=":material/info:"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### What This Dashboard Helps You Decide
            - **Production Planning**: Identify declining wells that may need intervention
            - **Resource Allocation**: Focus maintenance on highest-producing wellbores
            - **Forecasting**: Plan for future production levels with statistical models
            - **Anomaly Detection**: Catch unexpected production drops early
            """)

        with col2:
            st.markdown("""
            #### How to Use It
            1. **Select View Mode**: Total Field aggregates all wells; Single Wellbore drills down
            2. **Adjust Date Range**: Focus on specific time periods
            3. **Choose Forecast Model**: ETS (flexible trend/seasonality) or Baseline (seasonal naive benchmark)
            4. **Review Anomalies**: Adjust z-score threshold for sensitivity
            5. **Export Data**: Download filtered data or forecasts for further analysis
            """)

        st.markdown("""
        #### Key Insights to Look For
        - **Declining Trends**: Sustained MoM decreases may indicate reservoir depletion
        - **Anomaly Flags**: Large z-scores highlight unusual production behavior worth investigating
        - **Model Performance**: Lower MAPE generally indicates better forecast reliability (thresholds vary by context)

        ---
        *Data: Volve Field (North Sea), operated by Equinor. Production period: 2008-2016.*
        """)

        # Key Analytical Decisions (portfolio signal)
        st.markdown("#### Key Analytical Decisions")
        st.markdown("""
        - **Field-level aggregation** is default to show macro production behavior; wellbore mode enables diagnostics
        - **Zero-production months** are excluded from training data and MAPE calculation (WAPE handles them more robustly)
        - **Forecast floor at 0**: Production cannot be negative; model outputs are clipped accordingly
        - **Rolling 6-month window** for anomaly detection balances responsiveness vs. noise reduction
        - **12-month backtest window** provides statistically meaningful validation while preserving training data
        - **Baseline model (seasonal naive)** serves as a benchmark, not a competitor—ETS should outperform it
        """)

    st.markdown('<p class="tooltip">Sm³ = Standard cubic meters (volume at standard temperature &amp; pressure)</p>',
                unsafe_allow_html=True)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    df = load_data()
    if df is None:
        return

    df_active = get_active_production_data(df)

    # =========================================================================
    # SIDEBAR FILTERS
    # =========================================================================
    st.sidebar.header("Filters", anchor=False)
    st.sidebar.markdown('<p class="tooltip">Configure your analysis view</p>', unsafe_allow_html=True)

    mode = st.sidebar.radio(
        "View Mode", ["Total Field", "Single Wellbore"],
        help="Total Field: Aggregated production across all wells\nSingle Wellbore: Individual well analysis"
    )

    wellbore = None
    if mode == "Single Wellbore":
        wellbores = sorted(df["wellbore"].unique())
        wellbore = st.sidebar.selectbox("Select Wellbore", wellbores, help="Choose a specific wellbore to analyze")
        st.sidebar.info(f"Viewing: **{wellbore}**")

    min_date = df["date"].min()
    max_date = df_active["date"].max()

    date_range = st.sidebar.date_input(
        "Date Range", value=(min_date, max_date),
        min_value=min_date, max_value=df["date"].max(),
        help="Filter production data to this period"
    )

    st.sidebar.header("Forecasting", anchor=False)
    forecast_model = st.sidebar.selectbox(
        "Model", ["ets", "baseline"],
        format_func=lambda x: "Exponential Smoothing (ETS)" if x == "ets" else "Seasonal Naive (Baseline)",
        help="ETS: Flexible model capturing trend and seasonality (Holt-Winters)\nSeasonal Naive: Benchmark using same month from prior year"
    )

    st.sidebar.header("Anomaly Detection", anchor=False)
    anomaly_threshold = st.sidebar.slider(
        "Z-Score Threshold", min_value=1.0, max_value=4.0, value=2.5, step=0.5,
        help="Higher = fewer anomalies flagged. 2-3 is typical."
    )
    anomaly_column = st.sidebar.selectbox(
        "Monitor Column", ["oil", "gas", "water"],
        help="Which production metric to monitor for anomalies"
    )

    # =========================================================================
    # FILTER DATA
    # =========================================================================
    if len(date_range) == 2:
        mask = (df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))
        df_filtered = df[mask]
        df_active_filtered = df_active[
            (df_active["date"] >= pd.Timestamp(date_range[0])) &
            (df_active["date"] <= pd.Timestamp(date_range[1]))
        ]
    else:
        df_filtered = df
        df_active_filtered = df_active

    if mode == "Total Field":
        series_id = "TOTAL"
        series_df = aggregate_total_production(df_filtered)
        series_df_active = aggregate_total_production(df_active_filtered)
        series_title = "Total Field"
    else:
        series_id = wellbore
        series_df = df_filtered[df_filtered["wellbore"] == wellbore].copy()
        series_df_active = df_active_filtered[df_active_filtered["wellbore"] == wellbore].copy()
        series_title = wellbore

    # =========================================================================
    # KPI SECTION
    # =========================================================================
    st.header("Key Performance Indicators", anchor=False, divider="gray")
    if mode == "Single Wellbore":
        st.markdown(f"**Currently viewing: {wellbore}**")

    summary = get_last_month_summary(df_filtered)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mom_pct = summary.get("mom_change_pct")
        st.metric(
            "Oil Production",
            format_number(summary["total_oil"], " Sm³"),
            f"{mom_pct:+.1f}% MoM" if mom_pct else None,
            help="Last month with production data"
        )
        prev_oil = summary.get("prev_month_oil")
        if prev_oil and prev_oil > 0:
            st.caption(f"Prior month: {format_number(prev_oil, ' Sm³')}")
        if len(series_df_active) > 3 and "oil" in series_df_active.columns:
            st.plotly_chart(create_sparkline(series_df_active["oil"].tail(12)), use_container_width=True)

    with col2:
        gas_mom = summary.get("gas_mom_change_pct")
        st.metric(
            "Gas Production",
            format_number(summary["total_gas"], " Sm³"),
            f"{gas_mom:+.1f}% MoM" if gas_mom else None,
            help="Associated gas production"
        )

    with col3:
        water_mom = summary.get("water_mom_change_pct")
        st.metric(
            "Water Production",
            format_number(summary["total_water"], " Sm³"),
            f"{water_mom:+.1f}% MoM" if water_mom else None,
            help="Produced water volume"
        )

    with col4:
        yoy = summary.get("yoy_change_pct")
        st.metric("YoY Change (Oil)", f"{yoy:+.1f}%" if yoy else "N/A", help="Year-over-year change in oil production")

    st.caption(f"Report period: {summary.get('report_month', 'N/A')}")

    # =========================================================================
    # TOP WELLBORES
    # =========================================================================
    if mode == "Total Field":
        st.header("Top Producing Wellbores", anchor=False, divider="gray")
        top_wellbores = get_top_wellbores(df_filtered, n=5)

        if len(top_wellbores) > 0 and top_wellbores["oil"].sum() > 0:
            fig_top = px.bar(
                top_wellbores, x="wellbore", y="oil", color="oil",
                color_continuous_scale="Blues", labels={"oil": "Oil (Sm³)", "wellbore": "Wellbore"},
            )
            fig_top.update_layout(showlegend=False, height=280, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No production data available for the selected period.")

    # =========================================================================
    # PRODUCTION TRENDS
    # =========================================================================
    st.header(f"Production Trends — {series_title}", anchor=False, divider="gray")

    tab1, tab2, tab3 = st.tabs(["Oil", "Gas", "Water"])

    with tab1:
        if len(series_df) > 0 and "oil" in series_df.columns:
            fig = px.line(series_df, x="date", y="oil", labels={"oil": "Oil (Sm³)", "date": ""})
            fig.update_traces(line_color="#2E86AB", line_width=2)
            fig.update_layout(height=350, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if len(series_df) > 0 and "gas" in series_df.columns:
            fig = px.line(series_df, x="date", y="gas", labels={"gas": "Gas (Sm³)", "date": ""})
            fig.update_traces(line_color="#A23B72", line_width=2)
            fig.update_layout(height=350, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if len(series_df) > 0 and "water" in series_df.columns:
            fig = px.line(series_df, x="date", y="water", labels={"water": "Water (Sm³)", "date": ""})
            fig.update_traces(line_color="#F18F01", line_width=2)
            fig.update_layout(height=350, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # FORECASTING WITH VALIDATION
    # =========================================================================
    st.header("Production Forecast & Model Validation", anchor=False, divider="gray")

    forecast_df = generate_forecast(df_active, series_id, forecast_model, horizon=6)
    backtest_results, metrics = run_backtest(df_active, series_id, forecast_model, test_periods=12)

    # Get comparison model metrics for benchmark table
    comparison_model = "baseline" if forecast_model == "ets" else "ets"
    _, comparison_metrics = run_backtest(df_active, series_id, comparison_model, test_periods=12)

    col_forecast, col_metrics = st.columns([2, 1])

    with col_forecast:
        if forecast_df is not None and len(series_df_active) > 0:
            historical, forecast = get_historical_with_forecast(df_active, forecast_df, series_id, "oil")

            # Clip negative forecasts to 0 (production cannot be negative)
            forecast = forecast.copy()
            forecast["yhat"] = forecast["yhat"].clip(lower=0)
            if "yhat_lower" in forecast.columns:
                forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
            if "yhat_upper" in forecast.columns:
                forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical["date"], y=historical["oil"],
                mode="lines", name="Historical", line=dict(color="#2E86AB", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast["date"], y=forecast["yhat"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#E94F37", width=2, dash="dash"), marker=dict(size=8)
            ))

            if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast["date"], forecast["date"][::-1]]),
                    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(233, 79, 55, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"), name="95% CI", showlegend=True
                ))

            model_name = "Exponential Smoothing" if forecast_model == "ets" else "Seasonal Naive"
            fig.update_layout(
                title=f"Oil Forecast — {model_name}",
                xaxis_title="", yaxis_title="Oil (Sm³)",
                height=400, hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Forecast Values**")
            forecast_display = forecast[["date", "yhat"]].copy()
            forecast_display["date"] = forecast_display["date"].dt.strftime("%Y-%m")
            forecast_display["yhat"] = forecast_display["yhat"].apply(lambda x: f"{x:,.0f}")
            forecast_display.columns = ["Month", "Forecast (Sm³)"]
            st.dataframe(forecast_display, hide_index=True, use_container_width=True)
            st.caption("Note: Negative forecasts clipped to 0 (production floor)")
        else:
            st.warning("Unable to generate forecast. Check data availability.")

    with col_metrics:
        st.markdown("### Model Performance")

        if metrics and backtest_results is not None and len(backtest_results) > 0:
            mape = metrics.get("mape")
            wape = metrics.get("wape")
            mae = metrics.get("mae", 0)
            rmse = metrics.get("rmse", 0)
            n_obs = metrics.get("n_observations", 0)

            # Helper to format metric values
            def fmt_pct(val):
                if val is None:
                    return "N/A"
                try:
                    if np.isnan(val):
                        return "N/A"
                except (TypeError, ValueError):
                    pass
                return f"{val:.1f}%"

            def fmt_vol(val):
                if val is None:
                    return "N/A"
                try:
                    if np.isnan(val):
                        return "N/A"
                except (TypeError, ValueError):
                    pass
                return f"{val:,.0f}"

            def is_valid_metric(val):
                """Check if metric value is valid (not None, not NaN)."""
                if val is None:
                    return False
                try:
                    return not np.isnan(val)
                except (TypeError, ValueError):
                    return True  # Non-numeric is treated as valid

            # Display primary metrics in compact layout
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                wape_display = fmt_pct(wape)
                st.metric("WAPE", wape_display, help="Weighted Absolute Percentage Error (robust to varying magnitudes)")
                if wape_display == "N/A":
                    st.caption("Undefined: total actual ≈ 0")
            with col_m2:
                mape_display = fmt_pct(mape)
                st.metric("MAPE", mape_display, help="Mean Absolute Percentage Error (excludes zero actuals)")
                if mape_display == "N/A":
                    st.caption("Undefined: no non-zero actuals")

            st.metric("MAE", format_number(mae, " Sm³"), help="Mean Absolute Error in volume units")
            st.metric("RMSE", format_number(rmse, " Sm³"), help="Root Mean Squared Error")
            st.caption(f"Based on {n_obs} rolling-origin backtest predictions")

            # Expanded model comparison table (ETS vs Baseline)
            if comparison_metrics:
                st.markdown("**Model Comparison (ETS vs Baseline)**")
                current_name = "ETS" if forecast_model == "ets" else "Baseline"
                other_name = "Baseline" if forecast_model == "ets" else "ETS"

                c_wape = wape
                c_mape = mape
                c_mae = mae
                c_rmse = rmse
                o_wape = comparison_metrics.get("wape")
                o_mape = comparison_metrics.get("mape")
                o_mae = comparison_metrics.get("mae")
                o_rmse = comparison_metrics.get("rmse")

                comp_data = {
                    "Metric": ["MAE (Sm³)", "RMSE (Sm³)", "MAPE", "WAPE"],
                    current_name: [fmt_vol(c_mae), fmt_vol(c_rmse), fmt_pct(c_mape), fmt_pct(c_wape)],
                    other_name: [fmt_vol(o_mae), fmt_vol(o_rmse), fmt_pct(o_mape), fmt_pct(o_wape)]
                }
                st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

                # Interpretation note
                st.caption("""
                **Reading this table:** Lower values = better. Baseline is a benchmark (same-month-last-year).
                MAPE can be unstable with zeros; WAPE is more robust.
                """)

            if len(backtest_results) > 3:
                st.markdown("**Backtest: Actual vs Predicted**")
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=backtest_results["date"], y=backtest_results["actual"],
                    mode="lines+markers", name="Actual", line=dict(color="#2E86AB")
                ))
                fig_bt.add_trace(go.Scatter(
                    x=backtest_results["date"], y=backtest_results["predicted"],
                    mode="lines+markers", name="Predicted", line=dict(color="#E94F37", dash="dash")
                ))
                fig_bt.update_layout(
                    height=200, margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    xaxis=dict(title=""), yaxis=dict(title="")
                )
                st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.info("Insufficient data for backtest metrics.")

        with st.expander("Assumptions & Limitations", icon=":material/rule:"):
            st.markdown("""
            - **Seasonality**: 12-month cycle assumed
            - **Structural breaks**: This series has declining production; model captures trend but may lag sharp changes
            - **Zero-production months**: Excluded from training and MAPE; WAPE is more robust here
            - **Negative forecasts**: Clipped to 0 (production floor)
            - **External factors**: No drilling/maintenance schedules incorporated
            - **Anomaly flags**: Statistical indicators only, not confirmed operational events
            """)

    # =========================================================================
    # ANOMALY DETECTION
    # =========================================================================
    st.header("Anomaly Detection", anchor=False, divider="gray")

    st.markdown(f'<p class="tooltip">Method: Rolling 6-month Z-score | Threshold: |z| > {anomaly_threshold} | Monitoring: {anomaly_column.upper()}</p>', unsafe_allow_html=True)

    anomaly_df = detect_anomalies_zscore(series_df_active.copy(), column=anomaly_column, window=6, threshold=anomaly_threshold)

    if len(anomaly_df) > 0:
        anomalies_flagged = anomaly_df[anomaly_df["is_anomaly"] == True]
        n_anomalies = len(anomalies_flagged)

        col_chart, col_summary = st.columns([3, 1])

        with col_chart:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)

            fig.add_trace(go.Scatter(
                x=anomaly_df["date"], y=anomaly_df[anomaly_column],
                mode="lines", name=f"Actual {anomaly_column.title()}",
                line=dict(color="#2E86AB", width=2)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=anomaly_df["date"], y=anomaly_df["rolling_mean"],
                mode="lines", name="Rolling Mean (6M)",
                line=dict(color="#6b7280", width=1, dash="dot")
            ), row=1, col=1)

            if n_anomalies > 0:
                fig.add_trace(go.Scatter(
                    x=anomalies_flagged["date"], y=anomalies_flagged[anomaly_column],
                    mode="markers", name="Anomaly",
                    marker=dict(color="#E94F37", size=12, symbol="x")
                ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=anomaly_df["date"], y=anomaly_df["zscore"],
                mode="lines", name="Z-Score",
                line=dict(color="#667eea", width=2),
                fill="tozeroy", fillcolor="rgba(102, 126, 234, 0.1)"
            ), row=2, col=1)

            fig.add_hline(y=anomaly_threshold, line_dash="dash", line_color="red", annotation_text=f"+{anomaly_threshold}", row=2, col=1)
            fig.add_hline(y=-anomaly_threshold, line_dash="dash", line_color="red", annotation_text=f"-{anomaly_threshold}", row=2, col=1)

            fig.update_layout(height=450, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=40))
            fig.update_yaxes(title_text=f"{anomaly_column.title()} (Sm³)", row=1, col=1)
            fig.update_yaxes(title_text="Z-Score", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

        with col_summary:
            st.markdown("### Summary")

            if n_anomalies > 0:
                st.warning(f"**{n_anomalies}** anomalies flagged")

                # Anomaly table
                st.markdown("**Flagged Points**")
                anomaly_table = anomalies_flagged[["date", anomaly_column, "rolling_mean", "zscore"]].copy()
                anomaly_table["date"] = anomaly_table["date"].dt.strftime("%Y-%m")
                anomaly_table[anomaly_column] = anomaly_table[anomaly_column].apply(lambda x: f"{x:,.0f}")
                anomaly_table["rolling_mean"] = anomaly_table["rolling_mean"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
                anomaly_table["zscore"] = anomaly_table["zscore"].apply(lambda x: f"{x:+.1f}")
                anomaly_table.columns = ["Date", "Value", "Rolling Avg", "Z-Score"]
                st.dataframe(anomaly_table.head(10), hide_index=True, use_container_width=True)
            else:
                st.success("**0 anomalies** flagged")
                st.markdown("All observations within normal range.")

                # Empty state table
                st.markdown("**Flagged Points**")
                empty_df = pd.DataFrame(columns=["Date", "Value", "Rolling Avg", "Z-Score"])
                st.dataframe(empty_df, hide_index=True, use_container_width=True)
                st.caption("No points exceed threshold")

            st.markdown("---")
            st.markdown("**Interpretation:**")
            st.markdown("- Z > 0: Above average\n- Z < 0: Below average\n- |Z| > threshold: Flagged")

            # Anomaly Sensitivity Analysis
            st.markdown("---")
            st.markdown("**Threshold Sensitivity**")
            sensitivity_data = []
            for thresh in [2.0, 2.5, 3.0]:
                count = len(anomaly_df[np.abs(anomaly_df["zscore"]) > thresh])
                sensitivity_data.append({"Threshold": f"|z| > {thresh}", "Flagged": count})
            st.dataframe(pd.DataFrame(sensitivity_data), hide_index=True, use_container_width=True)
            st.caption("Statistical anomalies are not confirmed operational failures; use for triage.")
    else:
        st.info("Insufficient data for anomaly detection.")

    # =========================================================================
    # DATA EXPORT
    # =========================================================================
    st.header("Data Export", anchor=False, divider="gray")
    st.caption("Exports reflect current filter selections (date range, wellbore, model)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            ":material/download: Production Data",
            data=df_filtered.to_csv(index=False),
            file_name="volve_production.csv",
            mime="text/csv",
            help="Download filtered production data as CSV",
            key="download_production"  # Unique key prevents duplication
        )

    with col2:
        if forecast_df is not None:
            st.download_button(
                ":material/download: Forecasts",
                data=forecast_df.to_csv(index=False),
                file_name="volve_forecast.csv",
                mime="text/csv",
                help="Download forecast data as CSV",
                key="download_forecast"  # Unique key prevents duplication
            )

    with col3:
        if len(anomaly_df) > 0:
            st.download_button(
                ":material/download: Anomaly Report",
                data=anomaly_df.to_csv(index=False),
                file_name="volve_anomalies.csv",
                mime="text/csv",
                help="Download anomaly detection results",
                key="download_anomaly"  # Unique key prevents duplication
            )

    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.85rem;">
        <strong>Volve Production Analytics</strong> | Data: Equinor Volve Field (CC BY-NC-SA 4.0)<br>
        Built with Python, Streamlit, Plotly, and statsmodels
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
