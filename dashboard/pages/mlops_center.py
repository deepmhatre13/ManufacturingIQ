"""
ManufacturingIQ - MLOps Center Page
Operational monitoring dashboard for production model management
"""

import streamlit as st
from components.cards import kpi_card, section_header
from components.charts import model_evolution_chart
from utils.api import (
    get_model_metadata,
    get_drift_status,
    get_retraining_status,
    get_mlflow_status,
    get_model_evolution
)


def _status_color(status: str) -> str:
    """Map status to color"""
    mapping = {
        "Production": "#10B981",
        "Stable": "#10B981",
        "Drift Detected": "#F59E0B",
        "Approved": "#10B981",
        "Pending Review": "#F59E0B",
        "Rejected": "#EF4444",
        "Healthy": "#10B981",
        "Warning": "#F59E0B",
        "Critical": "#EF4444",
        "Active": "#10B981",
        "High": "#EF4444",
        "Medium": "#F59E0B",
        "Low": "#10B981"
    }
    return mapping.get(status, "#64748B")


def _status_badge_html(status: str) -> str:
    """Generate a status badge HTML"""
    color = _status_color(status)
    return f'<span class="ops-badge" style="background:{color}15;color:{color};border-color:{color}30;">{status}</span>'


def render():
    """Render the MLOps Center page"""

    # =============================================
    # SECTION 1: SYSTEM OVERVIEW
    # =============================================
    section_header(
        "System Overview",
        "Production model health and deployment status"
    )

    model_data = get_model_metadata()
    drift_data = get_drift_status()
    retrain_data = get_retraining_status()

    # Top row: 4 operational KPIs
    perf_cols = st.columns(4, gap="medium")
    with perf_cols[0]:
        kpi_card("Model Status", "Active", value_color="#10B981", subtitle=f"Version {model_data.get('model_version', 'N/A')}")
    with perf_cols[1]:
        drift_status = drift_data.get("drift_status", "Stable")
        drift_color = "#F59E0B" if drift_status == "Drift Detected" else "#10B981"
        kpi_card("Drift Status", drift_status, value_color=drift_color, subtitle=f"{drift_data.get('drifted_features', 0)} features drifted")
    with perf_cols[2]:
        promo = retrain_data.get("promotion_decision", "Pending Review")
        promo_color = _status_color(promo)
        kpi_card("Deployment Status", promo, value_color=promo_color, subtitle=f"Last: {retrain_data.get('last_retraining', 'N/A')[:10]}")
    with perf_cols[3]:
        kpi_card("System Health", "Operational", value_color="#10B981", subtitle="All systems nominal")

    # =============================================
    # SECTION 2: MODEL PERFORMANCE (condensed)
    # =============================================
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Model Performance",
        "Production model quality metrics"
    )

    perf_cols = st.columns(4, gap="medium")
    with perf_cols[0]:
        kpi_card("Accuracy", f"{model_data.get('roc_auc', 0)*100:.1f}%", subtitle="ROC-AUC")
    with perf_cols[1]:
        kpi_card("Precision", f"{model_data.get('precision', 0)*100:.1f}%")
    with perf_cols[2]:
        kpi_card("Recall", f"{model_data.get('recall', 0)*100:.1f}%")
    with perf_cols[3]:
        kpi_card("F1 Score", f"{model_data.get('f1_score', 0)*100:.1f}%")

    # =============================================
    # SECTION 3: MONITORING STATUS (Drift + Retrain combined)
    # =============================================
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Monitoring Status",
        "Drift detection and model refresh pipeline"
    )

    mon_col1, mon_col2 = st.columns(2, gap="large")

    with mon_col1:
        drift_status = drift_data.get("drift_status", "Stable")
        is_drifted = drift_status == "Drift Detected"
        drift_class = "warning" if is_drifted else "healthy"
        drift_severity = drift_data.get("drift_severity", "Low")

        st.markdown(f"""
        <div class="ops-card {drift_class}">
            <div class="ops-card-header">
                <div class="ops-card-title">Feature Drift Monitor</div>
                {_status_badge_html(drift_status)}
            </div>
            <div class="ops-card-body">
                <div class="ops-metric-row">
                    <div class="ops-metric">
                        <div class="ops-metric-value">{drift_data.get("features_checked", 0)}</div>
                        <div class="ops-metric-label">Features Checked</div>
                    </div>
                    <div class="ops-metric">
                        <div class="ops-metric-value" style="color:{'#F59E0B' if drift_data.get('drifted_features', 0) > 0 else '#10B981'};">{drift_data.get("drifted_features", 0)}</div>
                        <div class="ops-metric-label">Drifted Features</div>
                    </div>
                    <div class="ops-metric">
                        <div class="ops-metric-value">{drift_data.get("max_drift", 0):.1f}%</div>
                        <div class="ops-metric-label">Max Drift</div>
                    </div>
                    <div class="ops-metric">
                        <div class="ops-metric-value" style="color:{_status_color(drift_severity)};">{drift_severity}</div>
                        <div class="ops-metric-label">Severity</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mon_col2:
        promotion = retrain_data.get("promotion_decision", "Pending Review")
        promo_color = _status_color(promotion)

        st.markdown(f"""
        <div class="ops-card">
            <div class="ops-card-header">
                <div class="ops-card-title">Model Refresh Pipeline</div>
                {_status_badge_html(promotion)}
            </div>
            <div class="ops-card-body">
                <div class="ops-metric-row">
                    <div class="ops-metric">
                        <div class="ops-metric-value">{retrain_data.get("production_version", "N/A")}</div>
                        <div class="ops-metric-label">Active Model</div>
                    </div>
                    <div class="ops-metric">
                        <div class="ops-metric-value">{retrain_data.get("last_retraining", "N/A")[:10]}</div>
                        <div class="ops-metric-label">Last Refresh</div>
                    </div>
                    <div class="ops-metric">
                        <div class="ops-metric-value">{retrain_data.get("retraining_count", 0)}</div>
                        <div class="ops-metric-label">Total Refreshes</div>
                    </div>
                    <div class="ops-metric">
                        <div class="ops-metric-value" style="color:{promo_color};">{promotion}</div>
                        <div class="ops-metric-label">Pipeline Status</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =============================================
    # SECTION 4: MODEL EVOLUTION (compact)
    # =============================================
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Model Evolution",
        "Performance trend across model versions"
    )

    evolution_data = get_model_evolution()
    fig = model_evolution_chart(evolution_data)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # =============================================
    # SECTION 5: MLFLOW (collapsible, de-emphasized)
    # =============================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("MLflow Experiment Tracking (Engineering)", expanded=False):
        mlflow_data = get_mlflow_status()
        mlflow_cols = st.columns(4, gap="medium")
        with mlflow_cols[0]:
            kpi_card("Total Runs", str(mlflow_data.get("total_runs", 0)))
        with mlflow_cols[1]:
            kpi_card("Best Score", f"{mlflow_data.get('best_roc_auc', 0):.4f}")
        with mlflow_cols[2]:
            kpi_card("Latest Version", mlflow_data.get("latest_version", "N/A"))
        with mlflow_cols[3]:
            kpi_card("Production", mlflow_data.get("current_production", "N/A"))