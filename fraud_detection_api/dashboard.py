"""
Fraud Detection Monitoring Dashboard
Real-time monitoring with live fraud rate, drift detection, and prediction history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from src.monitoring.drift_detector import DriftDetector, load_reference_data

st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure paths
BASE_DIR = Path(__file__).parent
PREDICTIONS_CSV = BASE_DIR / "logs" / "predictions.csv"
REFERENCE_DATA_PATH = BASE_DIR / "data" / "train_processed.csv"
DRIFT_REPORT_PATH = BASE_DIR / "reports" / "drift_report.html"
DRIFT_SUMMARY_PATH = BASE_DIR / "reports" / "drift_report_summary.html"


@st.cache_data(ttl=30)
def load_predictions():
    """Load predictions CSV with caching (30s TTL)."""
    if not PREDICTIONS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()


@st.cache_data
def load_ref_data():
    """Load reference data for drift comparison."""
    try:
        if REFERENCE_DATA_PATH.exists():
            return load_reference_data(str(REFERENCE_DATA_PATH))
        else:
            st.warning("Reference data not found for drift analysis")
            return None
    except Exception as e:
        st.warning(f"Could not load reference data: {e}")
        return None


def calculate_fraud_rate(df: pd.DataFrame, time_window: str = "1h") -> tuple:
    """Calculate fraud rate for given time window."""
    if df.empty:
        return 0.0, 0

    if time_window == "all":
        fraud_rate = df['prediction'].mean()
        count = len(df)
    else:
        # Parse time window
        if time_window == "1h":
            cutoff = datetime.utcnow() - timedelta(hours=1)
        elif time_window == "24h":
            cutoff = datetime.utcnow() - timedelta(hours=24)
        elif time_window == "7d":
            cutoff = datetime.utcnow() - timedelta(days=7)
        else:
            cutoff = datetime.utcnow() - timedelta(hours=1)

        filtered = df[df['timestamp'] >= cutoff]
        if filtered.empty:
            return 0.0, 0
        fraud_rate = filtered['prediction'].mean()
        count = len(filtered)

    return fraud_rate, count


def plot_fraud_rate_timeseries(df: pd.DataFrame) -> go.Figure:
    """Plot fraud rate over time."""
    if df.empty:
        st.info("No prediction data available")
        return go.Figure()

    # Resample to 1-hour fraud rate
    df_copy = df.copy()
    df_copy['hour'] = df_copy['timestamp'].dt.floor('1h')
    hourly = df_copy.groupby('hour').agg({
        'prediction': ['sum', 'count']
    }).reset_index()
    hourly.columns = ['hour', 'fraud_count', 'total_count']
    hourly['fraud_rate'] = hourly['fraud_count'] / hourly['total_count']
    hourly['fraud_rate_pct'] = hourly['fraud_rate'] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly['hour'],
        y=hourly['fraud_rate_pct'],
        mode='lines+markers',
        name='Fraud Rate',
        line=dict(color='#EF553B', width=2),
        fill='tozeroy'
    ))

    fig.update_layout(
        title="Fraud Rate Over Time (Hourly)",
        xaxis_title="Time",
        yaxis_title="Fraud Rate (%)",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_prediction_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of predictions."""
    if df.empty:
        return go.Figure()

    fraud_count = (df['prediction'] == 1).sum()
    legit_count = (df['prediction'] == 0).sum()

    fig = go.Figure(data=[
        go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=[legit_count, fraud_count],
            marker=dict(colors=['#00CC96', '#EF553B']),
            textposition='auto',
            textinfo='label+percent+value'
        )
    ])

    fig.update_layout(
        title=f"Transaction Distribution (Total: {len(df):,})",
        height=400
    )

    return fig


def plot_probability_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of prediction probabilities."""
    if df.empty or 'probability' not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['probability'],
        nbinsx=30,
        name='Probability Distribution',
        marker_color='#636EFA'
    ))

    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )

    return fig


def plot_prediction_history(df: pd.DataFrame, n_rows: int = 100) -> pd.DataFrame:
    """Display recent prediction history."""
    if df.empty:
        st.info("No predictions logged yet")
        return pd.DataFrame()

    display_df = df.tail(n_rows).copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['prediction_label'] = display_df['prediction'].map({1: '🚨 Fraud', 0: '✓ Legit'})
    display_df['probability_pct'] = (display_df['probability'] * 100).round(2).astype(str) + '%'

    return display_df[['timestamp', 'log_id', 'prediction_label', 'probability_pct', 'model_version']]


def display_drift_analysis(ref_data, current_data: pd.DataFrame):
    """Display drift analysis using Evidently."""
    if ref_data is None or current_data.empty:
        st.warning("Insufficient data for drift analysis")
        return

    try:
        detector = DriftDetector(ref_data, str(PREDICTIONS_CSV))

        # Prediction drift
        predictions = current_data['prediction'].astype(int)
        drift_metrics = detector.detect_prediction_drift(predictions, threshold=0.05)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Reference Fraud Rate",
                f"{drift_metrics['reference_fraud_rate']*100:.2f}%"
            )

        with col2:
            st.metric(
                "Current Fraud Rate",
                f"{drift_metrics['current_fraud_rate']*100:.2f}%"
            )

        with col3:
            drift_pct = drift_metrics['drift_percentage']
            status = "🚨 DRIFTED" if drift_metrics['is_drifted'] else "✓ STABLE"
            st.metric(
                "Drift Status",
                status,
                f"{drift_pct*100:.2f}% (threshold: {drift_metrics['threshold']*100:.1f}%)"
            )

        # Generate report button
        if st.button("Generate Detailed Drift Report", key="drift_report"):
            with st.spinner("Generating drift report..."):
                try:
                    report, summary = detector.generate_drift_report(
                        current_data,
                        output_path=str(DRIFT_REPORT_PATH)
                    )
                    st.success("✓ Drift report generated!")
                    st.info(f"Report saved to: {DRIFT_REPORT_PATH}")

                    # Display summary
                    with st.expander("View Report Summary"):
                        st.write(f"**Timestamp:** {summary['timestamp']}")
                        st.write(f"**Reference rows:** {summary['reference_rows']:,}")
                        st.write(f"**Current rows:** {summary['current_rows']:,}")

                except Exception as e:
                    st.error(f"Error generating report: {e}")

    except Exception as e:
        st.error(f"Error in drift analysis: {e}")


def main():
    st.title("📊 Fraud Detection Monitoring Dashboard")
    st.markdown("Real-time monitoring of fraud detection model predictions and data drift")

    # Load data
    predictions_df = load_predictions()
    ref_data = load_ref_data()

    if predictions_df.empty:
        st.warning("⚠️ No prediction data available. Start making predictions to populate this dashboard.")
        st.info("Use `/predict` endpoint or `simulate_drift.py` to generate predictions.")
        return

    # Sidebar - filters
    st.sidebar.header("Filters")
    time_window = st.sidebar.selectbox(
        "Time Window",
        options=["1h", "24h", "7d", "all"],
        index=0
    )

    refresh_interval = st.sidebar.slider(
        "Auto-refresh (seconds)",
        min_value=0,
        max_value=60,
        value=30,
        step=5
    )

    # Top metrics
    st.subheader("📈 Key Metrics")
    fraud_rate, pred_count = calculate_fraud_rate(predictions_df, time_window)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Fraud Rate",
            f"{fraud_rate*100:.2f}%",
            f"{int(fraud_rate * pred_count)} fraud cases"
        )

    with col2:
        st.metric(
            "Predictions",
            f"{pred_count:,}",
            f"in {time_window}"
        )

    with col3:
        latest = predictions_df.iloc[-1]
        st.metric(
            "Latest Model",
            latest['model_version'] if 'model_version' in latest else "Unknown"
        )

    with col4:
        total_fraud = (predictions_df['prediction'] == 1).sum()
        st.metric(
            "Total Fraud Cases",
            f"{total_fraud:,}",
            f"out of {len(predictions_df):,}"
        )

    # Charts
    st.subheader("📉 Trend Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            plot_fraud_rate_timeseries(predictions_df),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            plot_prediction_distribution(predictions_df),
            use_container_width=True
        )

    # Probability distribution
    st.plotly_chart(
        plot_probability_distribution(predictions_df),
        use_container_width=True
    )

    # Drift analysis
    st.subheader("🔍 Data Drift Detection")
    display_drift_analysis(ref_data, predictions_df)

    # Prediction history
    st.subheader("📋 Recent Predictions")
    n_display = st.slider("Show last N predictions", 10, 1000, 100)
    history_df = plot_prediction_history(predictions_df, n_display)

    if not history_df.empty:
        st.dataframe(
            history_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
    else:
        st.info("No prediction history available")

    # Footer
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    with col3:
        st.caption("Use `/log-prediction` endpoint to log predictions")


if __name__ == "__main__":
    main()
