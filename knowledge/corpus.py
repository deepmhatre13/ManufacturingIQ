"""
ManufacturingIQ - Knowledge Base Corpus  (H-3)

Domain knowledge for RAG retrieval over the AI4I 2020 Predictive Maintenance Dataset.

H-3 additions vs. original:
  - Expanded each document to several substantive sentences so semantic search
    has real signal to match against.
  - Added 6 new documents: overstrain failure mode, power failure mode,
    bearing degradation, lubrication best practices, inspection checklist, and
    a root-cause analysis guide.  This brings the corpus to 17 documents.
  - Added `tags` metadata field to every document to support keyword (BM25-style)
    hybrid retrieval in retriever.py.
  - Added `citation` field containing a formatted, source-attributed string
    for inclusion in engineering reports.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# AI4I 2020 Dataset documentation
# ---------------------------------------------------------------------------

AI4I_DATASET_DOCS: Dict[str, Dict[str, Any]] = {
    "ai4i_2020_overview": {
        "title": "AI4I 2020 Predictive Maintenance Dataset",
        "source": "AI4I 2020 Dataset Documentation",
        "section": "Overview",
        "citation": "Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI ML Repository.",
        "tags": ["dataset", "predictive maintenance", "machine failure", "synthetic", "overview"],
        "text": (
            "The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset that reflects "
            "real-world industrial machine failure scenarios across 10,000 production cycles. "
            "It models five distinct failure types — tool wear failure, heat dissipation failure, "
            "power failure, overstrain failure, and random failures — each triggered by specific "
            "combinations of operating parameters. "
            "The dataset is deliberately imbalanced (~3.4% failure rate) to mirror industrial "
            "conditions where failures are rare but costly. "
            "Predictive maintenance models trained on this dataset learn to detect early degradation "
            "signals before catastrophic failure occurs."
        ),
    },
    "ai4i_2020_features": {
        "title": "AI4I 2020 Feature Descriptions",
        "source": "AI4I 2020 Dataset Documentation",
        "section": "Features",
        "citation": "Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI ML Repository.",
        "tags": ["features", "sensor", "type", "temperature", "torque", "tool wear", "rotational speed"],
        "text": (
            "Raw sensor features: Type (L/M/H for low/medium/high quality grade machines), "
            "Air temperature [K] (ambient temperature, normally distributed ~300 K ± 2 K), "
            "Process temperature [K] (generated from air temperature plus a heat source, ~310 K), "
            "Rotational speed [rpm] (calculated from a 2860 W power, normally distributed ~1500 rpm), "
            "Torque [Nm] (normally distributed ~40 Nm, no negative values), "
            "Tool wear [min] (cumulative wear time, H/M/L machines add 5/3/2 minutes per cycle). "
            "Engineered features derived from these raw values include: temperature_difference "
            "(process minus air temperature, indicator of thermal load), torque_speed_ratio "
            "(mechanical load per unit speed), wear_intensity (torque × wear, cumulative stress), "
            "machine_stress_index (composite normalized index on ~[0,3] scale), "
            "thermal_risk_index (process/air temperature ratio), and "
            "wear_efficiency_index (speed normalized by wear time)."
        ),
    },
    "ai4i_2020_target": {
        "title": "AI4I 2020 Failure Types and Target Variable",
        "source": "AI4I 2020 Dataset Documentation",
        "section": "Target",
        "citation": "Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI ML Repository.",
        "tags": ["failure", "target", "tool wear failure", "heat dissipation", "power failure", "overstrain"],
        "text": (
            "The binary target variable 'Machine failure' is 1 if any failure mode occurred, 0 otherwise. "
            "Five labeled failure sub-types exist: "
            "Tool Wear Failure (TWF) — occurs when tool wear exceeds a random threshold between 200–240 min; "
            "Heat Dissipation Failure (HDF) — triggered when air-process temperature difference < 8.6 K and "
            "rotational speed < 1380 rpm; "
            "Power Failure (PWF) — occurs when torque × rotational speed (in rad/s) falls below 3500 W "
            "or exceeds 9000 W; "
            "Overstrain Failure (OSF) — triggered when tool_wear × torque exceeds a type-specific threshold "
            "(H: 11000 Nm·min, M: 12000, L: 13000); "
            "Random Failures (RNF) — independent 0.1% probability per cycle regardless of parameters."
        ),
    },
}

# ---------------------------------------------------------------------------
# Industrial maintenance knowledge
# ---------------------------------------------------------------------------

MAINTENANCE_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "predictive_maintenance": {
        "title": "Predictive Maintenance Principles",
        "source": "Industrial Maintenance Reference",
        "section": "Concepts",
        "citation": "ISO 13381-1:2015. Condition monitoring and diagnostics of machines — Prognostics.",
        "tags": ["predictive maintenance", "condition monitoring", "sensor", "anomaly", "degradation"],
        "text": (
            "Predictive maintenance (PdM) uses real-time sensor data and machine learning to predict "
            "equipment failures before they cause unplanned downtime. "
            "Key leading indicators of deterioration include: abnormal temperature rise beyond normal operating range, "
            "unusual vibration or rotational speed instability, increased torque requirements indicating "
            "mechanical resistance, and cumulative tool wear beyond manufacturer specifications. "
            "PdM typically achieves 10–25% reduction in maintenance costs and 25–30% reduction in breakdowns "
            "compared to time-based preventive maintenance. "
            "Effective PdM programs establish baseline normal operating envelopes and trigger alerts "
            "when sensor readings deviate significantly from those baselines."
        ),
    },
    "tool_wear": {
        "title": "Tool Wear Failure Mode",
        "source": "Maintenance Engineering Handbook",
        "section": "Failure Modes",
        "citation": "Boothroyd, G. & Knight, W.A. (2006). Fundamentals of Machining and Machine Tools. CRC Press.",
        "tags": ["tool wear", "tool replacement", "cutting tool", "wear", "inspection", "TWF"],
        "text": (
            "Tool wear is the gradual degradation of cutting or processing tool geometry through abrasion, "
            "adhesion, diffusion, and fatigue mechanisms. "
            "In the AI4I dataset, Tool Wear Failure (TWF) is triggered when cumulative wear exceeds a "
            "random threshold between 200–240 minutes, making proactive replacement critical near the 200-minute mark. "
            "Early indicators: increased cutting forces (rising torque), surface finish degradation, "
            "elevated temperatures at the cutting zone, and abnormal vibration signatures. "
            "Recommended actions: (1) inspect tooling at each production run boundary near the 180-minute threshold, "
            "(2) replace worn tools before reaching 200 min when process conditions are demanding (high torque + high temperature), "
            "(3) verify tool material grade suitability for the work material, "
            "(4) review cutting parameters — reducing feed rate or depth of cut can extend tool life. "
            "Machine type affects wear rate: H-type machines accumulate 5 min/cycle vs 2 min/cycle for L-type."
        ),
    },
    "thermal_failure": {
        "title": "Thermal Failure Mode (Heat Dissipation Failure)",
        "source": "Maintenance Engineering Handbook",
        "section": "Failure Modes",
        "citation": "ISO 13373-1:2002. Condition monitoring and diagnostics of machines — Vibration.",
        "tags": ["thermal", "heat dissipation", "temperature", "cooling", "HDF", "overheating"],
        "text": (
            "Heat Dissipation Failure (HDF) occurs when the machine cannot dissipate heat fast enough, "
            "causing thermal stress on components. "
            "In the AI4I dataset, HDF is triggered when both conditions hold simultaneously: "
            "air-to-process temperature difference < 8.6 K AND rotational speed < 1380 rpm. "
            "Low speed reduces cooling airflow while insufficient temperature differential indicates "
            "inadequate heat dissipation. "
            "The Thermal Risk Index (process_temp / air_temp) captures this ratio — values significantly "
            "above 1.03 suggest elevated thermal loading. "
            "Recommended actions: inspect and clean cooling vents and heat exchangers, verify lubrication "
            "is adequate (dried lubricant dramatically increases friction heat), check for blocked coolant passages, "
            "review operating speed — increasing RPM within specification can restore cooling airflow, "
            "and monitor bearing temperatures as overheating accelerates bearing fatigue."
        ),
    },
    "mechanical_stress": {
        "title": "Mechanical Stress and Overstrain Failure",
        "source": "Reliability Engineering Reference",
        "section": "Failure Modes",
        "citation": "Shigley, J.E. (2011). Shigley's Mechanical Engineering Design. McGraw-Hill.",
        "tags": ["mechanical stress", "overstrain", "torque", "bearing", "OSF", "machine stress index"],
        "text": (
            "Overstrain Failure (OSF) occurs when accumulated mechanical stress exceeds component design limits. "
            "In the AI4I dataset, OSF is triggered when tool_wear × torque exceeds type-specific thresholds "
            "(H-type: 11,000 Nm·min, M-type: 12,000, L-type: 13,000). "
            "High torque combined with high tool wear is therefore a compounded risk signal. "
            "The Machine Stress Index combines normalized temperature differential, normalized torque, and "
            "normalized tool wear into a composite [0,3] score — values approaching 3.0 indicate compounded "
            "stress across all three dimensions simultaneously. "
            "High torque at low rotational speed indicates mechanical resistance, potentially from failing bearings, "
            "misalignment, or excessive cutting loads. "
            "Recommended actions: inspect bearings for pitting, spalling, or unusual noise, "
            "verify shaft alignment with dial indicator, check torque calibration against reference load, "
            "review load profiles and reduce feed rate if overstrain threshold is being approached."
        ),
    },
    "power_failure": {
        "title": "Power Failure Mode",
        "source": "Maintenance Engineering Handbook",
        "section": "Failure Modes",
        "citation": "IEEE Std 3001.5-2013. Power Systems Reliability in Industrial and Commercial Facilities.",
        "tags": ["power failure", "power", "PWF", "rotational speed", "torque", "electrical"],
        "text": (
            "Power Failure (PWF) in the AI4I dataset occurs when the product of torque and angular velocity "
            "(torque × rotational_speed × π/30 W) falls below 3,500 W or exceeds 9,000 W. "
            "Sub-threshold power indicates stalled or heavily loaded operation; "
            "above-threshold power indicates runaway or overload conditions. "
            "Both extremes cause component stress and eventual motor or drive failure. "
            "Recommended actions: verify motor drive settings and current limits, "
            "inspect motor windings for signs of overheating (discolouration, burnt odour), "
            "check power supply quality (voltage sags, harmonics), "
            "verify that torque and speed setpoints are within the machine's rated power envelope, "
            "and install power monitoring to detect transient overloads that do not trigger alarms."
        ),
    },
    "operational_impact": {
        "title": "Operational Impact Classification",
        "source": "Operations Risk Framework",
        "section": "Impact",
        "citation": "IEC 60300-3-11:2009. Dependability management — Application guide — Reliability centred maintenance.",
        "tags": ["operational impact", "priority", "downtime", "urgency", "severity", "risk classification"],
        "text": (
            "Operational impact categories are classified by failure probability and predicted health score: "
            "Low impact (health ≥ 80) — routine scheduled maintenance acceptable, no production disruption expected; "
            "Medium/High impact (health 50–79) — plan maintenance within 24–72 hours to prevent escalation; "
            "Critical operational impact (health < 50 or probability > 0.7) — immediate maintenance intervention "
            "required; continued operation risks unplanned line stoppage, secondary component damage, and "
            "safety hazards. "
            "Severity scores reflect urgency and direct actionability, not fabricated financial estimates. "
            "Deferring maintenance beyond recommended windows typically doubles repair cost and triples downtime "
            "due to cascading secondary damage."
        ),
    },
}

# ---------------------------------------------------------------------------
# Additional domain knowledge (H-3 additions)
# ---------------------------------------------------------------------------

ADVANCED_MAINTENANCE_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "bearing_degradation": {
        "title": "Bearing Degradation and Failure Progression",
        "source": "Reliability Engineering Reference",
        "section": "Component Failure Modes",
        "citation": "SKF Group (2014). Bearing Failure and Diagnosis. SKF Technical Publication.",
        "tags": ["bearing", "degradation", "vibration", "fatigue", "lubrication", "RPM"],
        "text": (
            "Bearing failures progress through four stages: Stage 1 (ultrasonic stress waves, no audible noise), "
            "Stage 2 (vibration detectable in high-frequency spectrum, slight temperature rise), "
            "Stage 3 (sideband frequencies visible, audible at low speeds, significant temperature increase), "
            "Stage 4 (random broadband noise, rapid temperature rise, imminent failure). "
            "In predictive maintenance terms, high torque at low RPM is a key Stage 2–3 indicator, "
            "as increased bearing friction resists rotation. "
            "Temperature monitoring near the bearing housing provides early warning — a sustained 10°C rise "
            "above baseline warrants inspection. "
            "Recommended actions: ultrasonic or vibration analysis to confirm bearing condition, "
            "re-lubrication (correct type and quantity — over-greasing is as damaging as under-greasing), "
            "and planned replacement at Stage 2 to avoid Stage 4 catastrophic failure."
        ),
    },
    "lubrication_maintenance": {
        "title": "Lubrication Best Practices for Industrial Machines",
        "source": "Industrial Maintenance Reference",
        "section": "Lubrication",
        "citation": "STLE (2008). Lubrication Fundamentals. Society of Tribologists and Lubrication Engineers.",
        "tags": ["lubrication", "grease", "oil", "friction", "temperature", "maintenance", "viscosity"],
        "text": (
            "Inadequate lubrication is responsible for approximately 40–50% of all bearing failures. "
            "Correct lubrication requires matching lubricant viscosity to operating temperature and speed "
            "(higher speed → lower viscosity; higher load → higher viscosity). "
            "Signs of lubrication failure: increased operating temperature (friction heat), "
            "abnormal noise (metal-to-metal contact), increased torque requirements (higher mechanical resistance), "
            "and accelerated wear particle generation. "
            "Best practices: establish re-lubrication intervals based on operating hours and temperature "
            "(halve the interval for every 15°C above rated temperature), "
            "use automatic lubrication systems for continuous operation machines, "
            "sample lubricant for particle analysis (ferrography) as part of a condition monitoring programme, "
            "and never mix different lubricant types or grades without manufacturer approval."
        ),
    },
    "inspection_checklist": {
        "title": "Pre-Failure Inspection Checklist for CNC and Machining Centers",
        "source": "Industrial Maintenance Reference",
        "section": "Inspection",
        "citation": "NIST (2019). Smart Manufacturing Systems for Advanced Manufacturing. NIST Technical Note.",
        "tags": ["inspection", "checklist", "CNC", "machining", "procedure", "maintenance action"],
        "text": (
            "When a predictive maintenance alert is raised, the following inspection sequence is recommended: "
            "1. Record baseline sensor readings (temperature, vibration, torque, RPM) before shutdown. "
            "2. Visually inspect the work area for chips, coolant leaks, or unusual deposits. "
            "3. Check tool holder runout (> 0.01 mm indicates holder damage or contamination). "
            "4. Inspect cutting tool for chipping, flank wear, or crater wear; measure tool wear with micrometer if possible. "
            "5. Check spindle for play, noise, or temperature rise (run at 50% speed for 2 min with no load). "
            "6. Verify coolant flow rate and temperature at nozzle. "
            "7. Check lubrication reservoirs and auto-lube system indicators. "
            "8. Inspect drive belts/chains for tension, wear, and fraying. "
            "9. Log all findings and compare against baseline; escalate to engineering if multiple findings coincide."
        ),
    },
    "root_cause_analysis": {
        "title": "Root Cause Analysis for Machine Failures",
        "source": "Reliability Engineering Reference",
        "section": "Diagnostics",
        "citation": "Latino, R.J. & Latino, K.C. (2006). Root Cause Analysis. CRC Press.",
        "tags": ["root cause", "RCA", "fault tree", "diagnosis", "failure analysis", "why"],
        "text": (
            "Root Cause Analysis (RCA) identifies the fundamental cause of failures rather than treating symptoms. "
            "For predictive maintenance alerts, a structured 5-Why approach is effective: "
            "Why did the machine fail? → A tool wore beyond threshold. "
            "Why did the tool wear excessively? → Operating torque was 15% above specification for 200 cycles. "
            "Why was torque excessive? → Workpiece material hardness was outside expected range. "
            "Why was material hardness out of range? → Incoming material inspection was not performed. "
            "Why was inspection skipped? → Inspection SOP was not updated after supplier change. "
            "Common root causes in CNC machining: incorrect cutting parameters, material variability, "
            "inadequate tooling for the application, deferred maintenance creating cascading failures, "
            "and insufficient operator training. "
            "RCA findings should feed back into preventive maintenance schedules and engineering change orders."
        ),
    },
    "confidence_interpretation": {
        "title": "Interpreting Prediction Confidence and Uncertainty",
        "source": "ManufacturingIQ Model Documentation",
        "section": "Model Interpretation",
        "citation": "ManufacturingIQ v3 Engineering Report — Model Confidence Methodology.",
        "tags": ["confidence", "uncertainty", "prediction margin", "reliability", "decision boundary", "model"],
        "text": (
            "The ManufacturingIQ confidence score represents the model's certainty about its prediction, "
            "computed as the prediction margin: |failure_probability - 0.5| / 0.5 × 100. "
            "A confidence of 100% means the model assigns probability near 0 or near 1 (maximally certain). "
            "A confidence of 0% means the model assigns probability near 0.5 (maximally uncertain — "
            "the prediction is near the decision boundary and should be treated with caution). "
            "Predictions with confidence below 70% are flagged for human review. "
            "High confidence with low health score is the most actionable outcome: "
            "the model is certain this machine is at risk. "
            "Low confidence with medium health score warrants collecting additional sensor readings "
            "or running a secondary inspection before committing to a maintenance action."
        ),
    },
    "shap_interpretation": {
        "title": "Interpreting SHAP Feature Contributions",
        "source": "ManufacturingIQ Model Documentation",
        "section": "Explainability",
        "citation": "Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
        "tags": ["SHAP", "explainability", "feature importance", "contribution", "XGBoost", "interpretation"],
        "text": (
            "SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to a specific prediction. "
            "A positive SHAP value for a feature means that feature pushed the prediction toward failure (higher risk). "
            "A negative SHAP value means that feature reduced the failure probability (protective effect). "
            "The magnitude of the SHAP value indicates the strength of influence. "
            "For ManufacturingIQ predictions, top positive contributors typically include: "
            "high Tool wear [min] (proximity to wear threshold), high Torque [Nm] combined with high wear "
            "(overstrain signal), and small temperature difference (heat dissipation concern). "
            "Negative contributors often include adequate rotational speed (good cooling, normal power range) "
            "and normal process-to-air temperature ratio. "
            "Maintenance engineers should focus corrective action on the top positive contributors "
            "as these represent the primary failure drivers for this specific prediction."
        ),
    },
}

# ---------------------------------------------------------------------------
# Model documentation
# ---------------------------------------------------------------------------

MODEL_DOCUMENTATION: Dict[str, Dict[str, Any]] = {
    "xgboost_model": {
        "title": "XGBoost Failure Prediction Model",
        "source": "ManufacturingIQ Model Documentation",
        "section": "Model",
        "citation": "ManufacturingIQ v3 Engineering Report — XGBoost Model Card v1.0.0.",
        "tags": ["XGBoost", "model", "classifier", "ROC-AUC", "prediction", "machine learning"],
        "text": (
            "The production model is an XGBoost classifier (version 1.0.0) trained on the AI4I 2020 dataset "
            "with hyperparameter optimization via Optuna. "
            "Model performance: ROC-AUC 0.9953 on held-out test set, indicating near-perfect discrimination "
            "between failure and non-failure conditions. "
            "The model uses 12 features: 6 raw sensor features (Type, Air temperature, Process temperature, "
            "Rotational speed, Torque, Tool wear) and 6 engineered features "
            "(temperature_difference, torque_speed_ratio, wear_intensity, machine_stress_index, "
            "thermal_risk_index, wear_efficiency_index). "
            "The model outputs a failure probability in [0,1]; the confidence score is the prediction margin "
            "— distance from the 0.5 decision boundary — not a calibrated probability."
        ),
    },
    "reliability_centered_maintenance": {
        "title": "Reliability-Centered Maintenance (RCM) Overview",
        "source": "Industrial Maintenance Reference",
        "section": "Strategy",
        "citation": "ISO 55001:2014. Asset management — Management systems — Requirements.",
        "tags": ["RCM", "maintenance strategy", "asset management", "reliability", "lifecycle", "cost optimization"],
        "text": (
            "Reliability-Centered Maintenance (RCM) is a systematic approach for defining what must be done "
            "to ensure that physical assets continue to do what their users require in the most efficient way. "
            "RCM builds on the concept that maintenance actions should be prioritized based on the consequence "
            "of failure, not just the likelihood. "
            "The RCM process asks seven key questions: What is the asset and what does it do? "
            "How can it fail to provide the required function? "
            "What causes each failure? What happens when each failure occurs? "
            "How does each failure matter? What can be done to predict or prevent each failure? "
            "What should be done if no proactive task is suitable? "
            "For ManufacturingIQ, RCM informs how the agentic pipeline ranks maintenance recommendations: "
            "high-consequence failure modes (e.g., catastrophic bearing seizure) receive higher priority "
            "than low-consequence modes (e.g., cosmetic surface finish degradation), even if the probability "
            "is similar. "
            "Effective RCM balances preventive maintenance costs against unplanned downtime costs, "
            "typically achieving 20–40% cost savings over reactive-only strategies."
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_documents() -> List[Dict[str, Any]]:
    """Return all knowledge documents as a flat list."""
    docs: List[Dict[str, Any]] = []
    for collection in [
        AI4I_DATASET_DOCS,
        MAINTENANCE_KNOWLEDGE,
        ADVANCED_MAINTENANCE_KNOWLEDGE,
        MODEL_DOCUMENTATION,
    ]:
        for doc in collection.values():
            docs.append(doc)
    return docs


def get_document_tags() -> Dict[str, List[str]]:
    """Return a mapping of document title → tags for keyword filtering."""
    return {
        doc["title"]: doc.get("tags", [])
        for doc in get_all_documents()
    }