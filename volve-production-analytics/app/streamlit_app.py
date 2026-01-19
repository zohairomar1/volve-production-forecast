"""
Volve Production Analytics Dashboard
====================================

Interactive Streamlit dashboard for production KPIs and forecasting.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_prep import prepare_data, load_processed_data, aggregate_total_production
from src.features import engineer_features
from src.forecasting import forecast_series, get_historical_with_forecast
from src.reporting import get_last_month_summary, get_top_wellbores, detect_anomalies

# Page config
st.set_page_config(
    page_title="Volve Production Analytics",
    page_icon="🛢️",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache production data."""
    # Try processed data first
    processed_path = PROCESSED_DATA_DIR / "volve_monthly.parquet"
    if processed_path.exists():
        df = load_processed_data(processed_path)
    else:
        # Try to find raw data and process it
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

    # Add features
    df = engineer_features(df)
    return df


@st.cache_data
def load_forecasts(df, series_id, model):
    """Generate and cache forecasts."""
    try:
        forecast_df = forecast_series(df, target_col="oil", series_id=series_id, model=model)
        return forecast_df
    except Exception as e:
        st.warning(f"Could not generate forecast: {e}")
        return None


def format_number(value, suffix=""):
    """Format large numbers with K/M suffixes."""
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M{suffix}"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K{suffix}"
    else:
        return f"{value:.0f}{suffix}"


def main():
    # Header
    st.title("🛢️ Volve Production Analytics")
    st.markdown("*Interactive dashboard for oil & gas production monitoring and forecasting*")

    # Load data
    df = load_data()
    if df is None:
        return

    # Sidebar
    st.sidebar.header("Filters")

    # Mode selection
    mode = st.sidebar.radio(
        "View Mode",
        ["Total Field", "Single Wellbore"],
        help="View aggregated field production or individual wellbore"
    )

    # Wellbore selection (when in single mode)
    wellbore = None
    if mode == "Single Wellbore":
        wellbores = sorted(df["wellbore"].unique())
        wellbore = st.sidebar.selectbox("Select Wellbore", wellbores)

    # Date range filter
    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Model selection
    st.sidebar.header("Forecasting")
    forecast_model = st.sidebar.selectbox(
        "Model",
        ["ets", "baseline"],
        format_func=lambda x: "Exponential Smoothing" if x == "ets" else "Seasonal Naive"
    )

    # Filter data
    if len(date_range) == 2:
        mask = (df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))
        df_filtered = df[mask]
    else:
        df_filtered = df

    # Prepare series data based on mode
    if mode == "Total Field":
        series_id = "TOTAL"
        series_df = aggregate_total_production(df_filtered)
        series_title = "Total Field Production"
    else:
        series_id = wellbore
        series_df = df_filtered[df_filtered["wellbore"] == wellbore].copy()
        series_title = f"Wellbore: {wellbore}"

    # KPI Section
    st.header("📊 Key Performance Indicators")

    summary = get_last_month_summary(df_filtered)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Oil (Last Month)",
            format_number(summary["total_oil"], " Sm³"),
            f"{summary['mom_change_pct']:+.1f}% MoM" if summary["mom_change_pct"] else None
        )

    with col2:
        st.metric(
            "Total Gas (Last Month)",
            format_number(summary["total_gas"], " Sm³")
        )

    with col3:
        st.metric(
            "Total Water (Last Month)",
            format_number(summary["total_water"], " Sm³")
        )

    with col4:
        yoy = summary["yoy_change_pct"]
        st.metric(
            "YoY Change (Oil)",
            f"{yoy:+.1f}%" if yoy else "N/A"
        )

    # Top Wellbores
    st.subheader("🏆 Top Producing Wellbores")
    top_wellbores = get_top_wellbores(df_filtered, n=5)
    if len(top_wellbores) > 0:
        fig_top = px.bar(
            top_wellbores,
            x="wellbore",
            y="oil",
            color="oil",
            color_continuous_scale="Viridis",
            labels={"oil": "Oil (Sm³)", "wellbore": "Wellbore"},
        )
        fig_top.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # Production Charts
    st.header(f"📈 Production Trends - {series_title}")

    # Time series chart
    tab1, tab2, tab3 = st.tabs(["Oil", "Gas", "Water"])

    with tab1:
        fig_oil = px.line(
            series_df,
            x="date",
            y="oil",
            title="Oil Production",
            labels={"oil": "Oil (Sm³)", "date": "Date"},
        )
        fig_oil.update_traces(line_color="#2E86AB")
        st.plotly_chart(fig_oil, use_container_width=True)

    with tab2:
        if "gas" in series_df.columns:
            fig_gas = px.line(
                series_df,
                x="date",
                y="gas",
                title="Gas Production",
                labels={"gas": "Gas (Sm³)", "date": "Date"},
            )
            fig_gas.update_traces(line_color="#A23B72")
            st.plotly_chart(fig_gas, use_container_width=True)

    with tab3:
        if "water" in series_df.columns:
            fig_water = px.line(
                series_df,
                x="date",
                y="water",
                title="Water Production",
                labels={"water": "Water (Sm³)", "date": "Date"},
            )
            fig_water.update_traces(line_color="#F18F01")
            st.plotly_chart(fig_water, use_container_width=True)

    # Forecast Section
    st.header("🔮 Production Forecast")

    forecast_df = load_forecasts(df, series_id, forecast_model)

    if forecast_df is not None:
        # Get historical data for this series
        historical, forecast = get_historical_with_forecast(df, forecast_df, series_id, "oil")

        # Create combined chart
        fig_forecast = go.Figure()

        # Historical
        fig_forecast.add_trace(go.Scatter(
            x=historical["date"],
            y=historical["oil"],
            mode="lines",
            name="Historical",
            line=dict(color="#2E86AB", width=2)
        ))

        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast["date"],
            y=forecast["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#E94F37", width=2, dash="dash"),
            marker=dict(size=8)
        ))

        # Confidence interval (if available)
        if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
            fig_forecast.add_trace(go.Scatter(
                x=pd.concat([forecast["date"], forecast["date"][::-1]]),
                y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(233, 79, 55, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True
            ))

        fig_forecast.update_layout(
            title=f"Oil Production Forecast ({forecast_model.upper()})",
            xaxis_title="Date",
            yaxis_title="Oil (Sm³)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Forecast table
        st.subheader("Forecast Details")
        forecast_display = forecast[["date", "yhat"]].copy()
        forecast_display["date"] = forecast_display["date"].dt.strftime("%Y-%m")
        forecast_display["yhat"] = forecast_display["yhat"].apply(lambda x: f"{x:,.0f}")
        forecast_display.columns = ["Month", "Forecast (Sm³)"]
        st.dataframe(forecast_display, hide_index=True)

    # Anomaly Detection
    st.header("⚠️ Anomaly Detection")

    anomalies = detect_anomalies(df_filtered, threshold_pct=30)
    if len(anomalies) > 0:
        st.warning(f"Detected {len(anomalies)} wellbore(s) with significant production drops:")
        for _, row in anomalies.iterrows():
            st.markdown(
                f"- **{row['wellbore']}**: {row['pct_change']:.1f}% vs 3-month average "
                f"(Current: {row['current_oil']:,.0f}, Avg: {row['rolling_avg']:,.0f})"
            )
    else:
        st.success("No significant production anomalies detected.")

    # Data Download
    st.header("📥 Data Export")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Production Data (CSV)",
            data=csv_data,
            file_name="volve_production_filtered.csv",
            mime="text/csv"
        )

    with col2:
        if forecast_df is not None:
            forecast_csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast Data (CSV)",
                data=forecast_csv,
                file_name="volve_forecast.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "*Data source: Volve Field Dataset (Equinor). "
        "Dashboard built with Streamlit and Plotly.*"
    )


if __name__ == "__main__":
    main()
