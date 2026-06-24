"""
ManufacturingIQ - Predictive Intelligence Page
INPUTS → PREDICTION → ACTIONABLE INSIGHTS
Clean product layout with compact gauge embedded in results
"""

import streamlit as st
from components.cards import kpi_card, status_badge
from components.charts import feature_importance_chart
from components.gauges import health_gauge
from components.tables import prediction_history_table
from utils.api import (
    predict_health,
    get_feature_importance,
    get_prediction_history
)


def _get_risk_drivers(importance_data, top_n=3):
    """Extract top risk drivers from feature importance data"""
    sorted_features = sorted(importance_data, key=lambda x: x["importance"], reverse=True)
    drivers = []
    for f in sorted_features[:top_n]:
        name = f["feature"].split("[")[0].strip() if "[" in f["feature"] else f["feature"]
        drivers.append(name)
    return drivers


def _generate_assessment(result):
    """Generate dynamic machine assessment from prediction result"""
    status = result.get("machine_status", "Unknown")
    health = result.get("health_score", 0)
    
    if status == "Critical":
        return "CRITICAL", "#EF4444"
    elif status == "Warning" or health < 80:
        return "WARNING", "#F59E0B"
    else:
        return "HEALTHY", "#10B981"


def _generate_recommendation(result):
    """Generate dynamic recommended action based on prediction result"""
    status = result.get("machine_status", "Healthy")
    health = result.get("health_score", 0)
    
    if status == "Critical" or health < 50:
        return "Schedule maintenance inspection within 24 hours. Immediate attention required."
    elif status == "Warning" or health < 80:
        return "Monitor machine closely. Plan inspection within the next 72 hours."
    else:
        return "No action required. Machine is operating within normal parameters."


def render():
    """Render the Predictive Intelligence page"""

    # =============================================
    # SECTION 1: COMPACT HERO
    # =============================================
    st.markdown("""
    <div class="pi-hero">
        <div class="pi-hero-row">
            <div class="pi-hero-left">
                <div class="pi-hero-title">ManufacturingIQ</div>
                <div class="pi-hero-subtitle">AI Powered Predictive Maintenance Platform</div>
            </div>
            <div class="pi-hero-badges">
                <span class="status-badge success"><span class="dot"></span>Production Model Active</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =============================================
    # SECTION 2: PREDICTION WORKSPACE (INPUTS → PREDICTION → ACTIONABLE INSIGHTS)
    # =============================================
    st.markdown("""
    <div class="section-header">
        <h2>Prediction Workspace</h2>
        <p>Configure machine parameters and analyze health in real-time</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="pi-input-header">
            <div class="pi-input-label">Machine Configuration</div>
        </div>
        """, unsafe_allow_html=True)

        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            help="L: Low capacity | M: Medium capacity | H: High capacity"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=290.0,
                max_value=320.0,
                value=300.0,
                step=0.5,
                format="%.1f"
            )
            rpm = st.number_input(
                "RPM",
                min_value=500,
                max_value=3500,
                value=1500,
                step=50
            )
            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=500,
                value=200,
                step=5
            )

        with col_b:
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=300.0,
                max_value=360.0,
                value=320.0,
                step=0.5,
                format="%.1f"
            )
            torque = st.number_input(
                "Torque (Nm)",
                min_value=10.0,
                max_value=150.0,
                value=75.0,
                step=1.0,
                format="%.1f"
            )

        analyze_clicked = st.button("Analyze Machine", type="primary", use_container_width=True)

    with col_right:
        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None
        if "prediction_error" not in st.session_state:
            st.session_state.prediction_error = None

        if analyze_clicked:
            with st.spinner("Analyzing machine health..."):
                try:
                    result = predict_health(
                        machine_type=machine_type,
                        air_temp=air_temp,
                        process_temp=process_temp,
                        rpm=rpm,
                        torque=torque,
                        tool_wear=tool_wear
                    )
                    st.session_state.prediction_result = result
                    st.session_state.prediction_error = None
                except Exception as e:
                    error_msg = str(e)
                    print(f"[DEBUG] Prediction failed: {error_msg}")
                    st.session_state.prediction_error = error_msg
                    st.session_state.prediction_result = None

        if st.session_state.get("prediction_error"):
            st.error(f"API Error: {st.session_state.prediction_error}")

        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            health_score = result.get("health_score", 0)
            failure_prob = result.get("failure_probability", 0)
            status = result.get("machine_status", "Unknown")
            
            assessment_label, assessment_color = _generate_assessment(result)
            confidence = min(99.5, 85 + health_score * 0.15)
            
            # Determine risk level
            if health_score >= 80:
                risk_level = "LOW"
                risk_color = "#10B981"
            elif health_score >= 50:
                risk_level = "MEDIUM"
                risk_color = "#F59E0B"
            else:
                risk_level = "HIGH"
                risk_color = "#EF4444"

            # Get top risk drivers dynamically
            importance_data = get_feature_importance()
            top_drivers = _get_risk_drivers(importance_data, 3)
            
            # Generate recommendation dynamically
            recommendation = _generate_recommendation(result)

            st.markdown(f"""
            <div class="pi-result-card">
            <div class="pi-result-status">
                <div>
                    <div class="pi-result-assessment" style="color: {assessment_color};">{assessment_label}</div>
                    <div class="pi-result-status-label">Machine Assessment</div>
                </div>
                <div style="text-align:right;">
                    <div class="pi-result-metric-value" style="color: {health_score >= 80 and '#10B981' or health_score >= 50 and '#F59E0B' or '#EF4444'}; font-size:1.75rem;">{health_score:.1f}</div>
                    <div class="pi-result-metric-label">Health Score</div>
                </div>
            </div>
            <div class="pi-result-metrics">
                <div class="pi-result-metric">
                    <span class="pi-result-metric-label">Failure Probability</span>
                    <span class="pi-result-metric-value">{failure_prob*100:.1f}%</span>
                </div>
                <div class="pi-result-metric">
                    <span class="pi-result-metric-label">Risk Level</span>
                    <span class="pi-result-metric-value" style="color: {risk_color};">{risk_level}</span>
                </div>
                <div class="pi-result-metric">
                    <span class="pi-result-metric-label">Model Confidence</span>
                    <span class="pi-result-metric-value">{confidence:.1f}%</span>
                </div>
                <div class="pi-result-metric">
                    <span class="pi-result-metric-label">Assessment</span>
                    <span class="pi-result-metric-value" style="color: {assessment_color};">{assessment_label}</span>
                </div>
            </div>
            <div class="pi-result-drivers">
                <div class="pi-result-drivers-label">Primary Risk Drivers</div>
                <div class="pi-result-drivers-list">
                    {''.join(f'<span class="pi-result-driver-tag">{d}</span>' for d in top_drivers)}
                </div>
            </div>
            <div class="pi-result-action">
                <div class="pi-result-action-label">Recommended Action</div>
                <div class="pi-result-action-text">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)

            # Health Score Gauge — rendered inside result card for unified panel
            st.markdown('<div class="gauge-wrapper">', unsafe_allow_html=True)
            gauge_fig = health_gauge(health_score)
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pi-result-card">
            <div class="pi-result-empty">
                <div class="pi-result-empty-icon">⚙️</div>
                <p>Configure machine parameters and click <strong>Analyze Machine</strong> to see prediction results</p>
            </div>
            </div>
            """, unsafe_allow_html=True)

    # =============================================
    # SECTION 3: MODEL INTELLIGENCE (compact)
    # =============================================
    st.markdown("""
    <hr style="margin: 0.75rem 0; border-color: #F1F5F9;">
    <div class="section-header">
        <h2>Model Intelligence</h2>
        <p>Top factors influencing the current prediction</p>
    </div>
    """, unsafe_allow_html=True)

    importance_data = get_feature_importance()
    top_drivers = _get_risk_drivers(importance_data, 3)
    drivers_html = ''.join(f'<span class="pi-driver-badge">{d}</span>' for d in top_drivers)
    st.markdown(f"""
    <div class="pi-top-drivers-bar">
        <span class="pi-top-drivers-label">Top Drivers:</span>
        {drivers_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_importance = feature_importance_chart(importance_data)
    st.plotly_chart(fig_importance, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # =============================================
    # SECTION 4: PREDICTION HISTORY (compact - 5 records)
    # =============================================
    history_data = get_prediction_history()
    
    if history_data and len(history_data) > 0:
        st.markdown("""
        <hr style="margin: 0.75rem 0; border-color: #F1F5F9;">
        <div class="section-header">
            <h2>Prediction History</h2>
            <p>Latest machine health predictions</p>
        </div>
        """, unsafe_allow_html=True)

        history_df = prediction_history_table(history_data)

        if not history_df.empty:
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
                height=220,
                column_config={
                    "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                    "Machine": st.column_config.TextColumn("Machine", width="small"),
                    "Health": st.column_config.TextColumn("Health", width="small"),
                    "Risk": st.column_config.TextColumn("Risk", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small")
                }
            )