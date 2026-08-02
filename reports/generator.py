"""
ManufacturingIQ Agentic AI - Report Generator

Generates structured reports in JSON/Markdown/PDF.
EngineeringReport is now a TypedDict (plain dict at runtime) — use .get() for all field access.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _gd(obj: Any, key: str, default=None):
    """Get a field from a dict or object safely."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_json(report: Any) -> Dict[str, Any]:
    """Return the report as a plain dict. Handles both TypedDict and Pydantic models."""
    if isinstance(report, dict):
        return report
    if hasattr(report, "model_dump"):
        return report.model_dump()
    if hasattr(report, "dict"):
        return report.dict()
    return {}


def to_markdown(report: Any) -> str:
    lines = [
        f"# Engineering Report - {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Prediction Summary",
        _gd(report, "prediction_summary") or "",
        "",
        "## Technical Explanation",
        _gd(report, "technical_explanation") or "",
        "",
        "## Primary Drivers",
    ]
    for driver in (_gd(report, "primary_drivers") or []):
        lines.append(f"- {driver}")

    lines.extend([
        "",
        "## Recommendations",
    ])
    for rec in (_gd(report, "maintenance_recommendations") or []):
        action = _gd(rec, "action", "")
        priority = _gd(rec, "priority", "")
        rationale = _gd(rec, "rationale", "")
        lines.append(f"- **{action}** ({priority}): {rationale}")

    risk = _gd(report, "risk_assessment") or {}
    lines.extend([
        "",
        "## Risk Assessment",
        f"- Risk Level: {_gd(risk, 'risk_level', 'Unknown')}",
        f"- Severity: {_gd(risk, 'severity', 'Unknown')}",
        f"- Business Impact: {_gd(risk, 'business_impact', 'Unknown')}",
        f"- Urgency: {_gd(risk, 'urgency', 'Unknown')}",
        f"- Rationale: {_gd(risk, 'rationale', '')}",
    ])

    # H-3: Evidence & Citations section
    citations = _gd(report, "retrieved_evidence_citations") or []
    if citations:
        lines.extend(["", "## Evidence & Citations"])
        for i, cite in enumerate(citations, 1):
            lines.append(f"{i}. {cite}")

    lines.extend([
        "",
        f"**Final Recommendation:** {_gd(report, 'final_recommendation') or ''}",
        "",
        f"**Confidence:** {float(_gd(report, 'confidence') or 0):.1f}%",
    ])
    return "\n".join(lines)


def to_pdf(report: Any) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        logger.error("fpdf2 is required for PDF generation: %s", exc)
        return b""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ManufacturingIQ Engineering Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, datetime.now().isoformat(timespec="seconds"), ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _gd(report, "prediction_summary") or "")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Technical Explanation", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _gd(report, "technical_explanation") or "")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Primary Drivers", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for driver in (_gd(report, "primary_drivers") or []):
        pdf.cell(10)
        pdf.cell(0, 6, f"- {driver}", ln=True)
    pdf.ln(2)

    risk = _gd(report, "risk_assessment") or {}
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Risk Assessment", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        f"Risk Level: {_gd(risk, 'risk_level', 'Unknown')}\n"
        f"Urgency: {_gd(risk, 'urgency', 'Unknown')}\n"
        f"Rationale: {_gd(risk, 'rationale', '')}"
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Final Recommendation", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _gd(report, "final_recommendation") or "")

    # H-3: Evidence & Citations
    citations = _gd(report, "retrieved_evidence_citations") or []
    if citations:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Evidence & Citations", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for i, cite in enumerate(citations, 1):
            safe_cite = cite.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, f"{i}. {safe_cite}")

    return bytes(pdf.output())