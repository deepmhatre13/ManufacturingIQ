"""
ManufacturingIQ Agentic AI - Typed State for LangGraph

Uses TypedDict for LangGraph compatibility (Pydantic models don't support .get() or item assignment).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class MachineInput(TypedDict, total=False):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float


class EngineeredFeatures(TypedDict, total=False):
    temperature_difference: float
    torque_speed_ratio: float
    wear_intensity: float
    machine_stress_index: float
    thermal_risk_index: float
    wear_efficiency_index: float


class PredictionResult(TypedDict, total=False):
    failure_prediction: int
    failure_probability: float
    health_score: float
    machine_status: str
    confidence: float
    risk_level: str


class ShapExplanation(TypedDict, total=False):
    top_contributors: List[str]
    positive_contributors: List[str]
    negative_contributors: List[str]
    explanation_text: Optional[str]
    confidence: float


class RetrievedDocument(TypedDict, total=False):
    title: str
    source: str
    section: Optional[str]
    excerpt: str
    confidence: float
    citation: Optional[str]   # H-3: formatted source attribution, e.g. "Author (Year). Title."
    tags: List[str]           # H-3: keyword tags for hybrid retrieval scoring


class MaintenanceRecommendation(TypedDict, total=False):
    action: str
    priority: str
    rationale: str
    references: List[RetrievedDocument]


class RiskAssessment(TypedDict, total=False):
    risk_level: str
    severity: str
    business_impact: str
    rationale: str
    urgency: str


class TrendAnalysis(TypedDict, total=False):
    direction: str
    health_trend: str
    risk_trend: str
    summary: str


class OperationalImpact(TypedDict, total=False):
    impact_category: str
    estimated_priority: str
    severity_score: float
    notes: str


class EngineeringReport(TypedDict, total=False):
    prediction_summary: str
    technical_explanation: str
    primary_drivers: List[str]
    retrieved_evidence: List[RetrievedDocument]
    maintenance_recommendations: List[MaintenanceRecommendation]
    trend_analysis: Optional[TrendAnalysis]
    risk_assessment: RiskAssessment
    confidence: float
    final_recommendation: str


class GraphExecutionLog(TypedDict, total=False):
    node: str
    status: str
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_ms: float
    retries: int
    error: Optional[str]
    metadata: Dict[str, Any]


class ManufacturingIQState(TypedDict, total=False):
    raw_input: MachineInput
    engineered_features: Optional[EngineeredFeatures]
    prediction: Optional[PredictionResult]
    shap_explanation: Optional[ShapExplanation]
    retrieved_documents: List[RetrievedDocument]
    maintenance_recommendations: List[MaintenanceRecommendation]
    risk_assessment: Optional[RiskAssessment]
    trend_analysis: Optional[TrendAnalysis]
    operational_impact: Optional[OperationalImpact]
    engineering_report: Optional[EngineeringReport]
    prediction_history: List[Dict[str, Any]]
    execution_logs: List[GraphExecutionLog]
    node_status: Dict[str, str]
    next: Optional[str]
    error: Optional[str]