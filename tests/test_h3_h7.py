"""
Regression tests for H-3 (RAG improvements) and H-7 (test coverage expansion)
===============================================================================

H-3 tests:
  - Corpus has ≥ 17 documents with tags and citations on every entry
  - Keyword scorer correctly handles overlap / empty sets
  - Hybrid retriever returns citation + tags on each result
  - Min-score filter removes low-relevance documents
  - Retrieval agent builds context-driven queries (failure mode hints)
  - Markdown report includes Evidence & Citations section when docs available
  - `retrieved_evidence_citations` is present in engineering_report dict

H-7 tests (broader coverage):
  - graph.py structure: node names are prefixed with `node_`, no state-key collisions
  - graph.py routing: _route_from_supervisor returns expected strings
  - report generator: to_markdown produces well-formed Markdown
  - report generator: to_json round-trips correctly
  - End-to-end: critical input produces High-risk report with citations

Run with:
    cd d:/jupyter_notebook/ManufacturingIQ
    python -m pytest tests/test_h3_h7.py -v
"""

import re
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEALTHY_INPUT = {
    "Type": "L",
    "Air_temperature_K": 298.1,
    "Process_temperature_K": 308.6,
    "Rotational_speed_rpm": 1551.0,
    "Torque_Nm": 42.8,
    "Tool_wear_min": 0.0,
}

_CRITICAL_INPUT = {
    "Type": "H",
    "Air_temperature_K": 298.1,
    "Process_temperature_K": 305.5,   # temp_diff ~7.4 K < 8.6 → HDF hint
    "Rotational_speed_rpm": 1168.0,   # < 1380 → HDF hint, low RPM
    "Torque_Nm": 76.6,
    "Tool_wear_min": 253.0,           # > 200 → TWF hint; 253×76.6 = 19,379 Nm·min > OSF threshold
}


def _make_full_state(raw_input=None, status="Healthy", prob=0.05, health=95.0,
                     risk="Low", confidence=90.0) -> Dict[str, Any]:
    pred = {
        "failure_prediction": 1 if prob > 0.5 else 0,
        "failure_probability": prob,
        "health_score": health,
        "machine_status": status,
        "confidence": confidence,
        "risk_level": risk,
    }
    return {
        "raw_input": raw_input or _HEALTHY_INPUT,
        "prediction": pred,
        "retrieved_documents": [],
        "maintenance_recommendations": [],
        "execution_logs": [],
        "node_status": {},
    }


# ---------------------------------------------------------------------------
# H-3: Corpus quality
# ---------------------------------------------------------------------------

class TestH3Corpus:
    """H-3: Every corpus document must have required fields and metadata."""

    def test_corpus_has_17_or_more_documents(self):
        from knowledge.corpus import get_all_documents
        docs = get_all_documents()
        assert len(docs) >= 17, (
            f"Expected ≥ 17 corpus documents after H-3 expansion, got {len(docs)}"
        )

    def test_every_document_has_required_fields(self):
        from knowledge.corpus import get_all_documents
        required = {"title", "source", "text"}
        for doc in get_all_documents():
            missing = required - set(doc.keys())
            assert not missing, (
                f"Document {doc.get('title', '?')!r} missing required fields: {missing}"
            )

    def test_every_document_has_tags(self):
        """H-3: all documents must have tags for keyword scoring."""
        from knowledge.corpus import get_all_documents
        for doc in get_all_documents():
            assert "tags" in doc and isinstance(doc["tags"], list) and len(doc["tags"]) > 0, (
                f"Document {doc.get('title', '?')!r} is missing non-empty tags list"
            )

    def test_every_document_has_citation(self):
        """H-3: all documents must have a citation string."""
        from knowledge.corpus import get_all_documents
        for doc in get_all_documents():
            assert "citation" in doc and isinstance(doc["citation"], str) and len(doc["citation"]) > 5, (
                f"Document {doc.get('title', '?')!r} is missing a valid citation string"
            )

    def test_documents_have_substantive_text(self):
        """Text must be at least 100 characters to be useful for semantic retrieval."""
        from knowledge.corpus import get_all_documents
        for doc in get_all_documents():
            assert len(doc["text"]) >= 100, (
                f"Document {doc.get('title', '?')!r} text is too short ({len(doc['text'])} chars); "
                "too sparse for meaningful semantic retrieval"
            )

    def test_ai4i_failure_modes_all_documented(self):
        """All five AI4I failure modes must be mentioned in the corpus."""
        from knowledge.corpus import get_all_documents
        all_text = " ".join(d["text"] for d in get_all_documents()).lower()
        for mode in ["tool wear failure", "heat dissipation", "power failure", "overstrain", "random"]:
            assert mode in all_text, f"AI4I failure mode {mode!r} not found in corpus"


# ---------------------------------------------------------------------------
# H-3: Hybrid retrieval
# ---------------------------------------------------------------------------

class TestH3HybridRetrieval:
    """H-3: Retriever must use hybrid scoring, enforce min-score, forward citation/tags."""

    def test_keyword_score_overlap(self):
        from retriever.retriever import _keyword_score, _tokenize
        doc = {"title": "Tool Wear Failure Mode", "tags": ["tool wear", "TWF", "inspection"]}
        q_tokens = _tokenize("tool wear inspection")
        score = _keyword_score(q_tokens, doc)
        assert score > 0.0, "Expected positive keyword score for overlapping terms"
        assert score <= 1.0, "Keyword score must be capped at 1.0"

    def test_keyword_score_empty_query(self):
        from retriever.retriever import _keyword_score, _tokenize
        doc = {"title": "Tool Wear", "tags": ["tool", "wear"]}
        score = _keyword_score(set(), doc)
        assert score == 0.0, "Empty query should return 0.0 keyword score"

    def test_keyword_score_no_overlap(self):
        from retriever.retriever import _keyword_score, _tokenize
        doc = {"title": "Lubrication Best Practices", "tags": ["grease", "oil", "viscosity"]}
        q_tokens = _tokenize("power failure electrical motor PWF")
        score = _keyword_score(q_tokens, doc)
        # Very different domains — expect low but not necessarily 0 due to
        # common stop words; just verify it's lower than a relevant pair
        from retriever.retriever import _tokenize as tok
        doc2 = {"title": "Power Failure Mode", "tags": ["power failure", "PWF", "electrical"]}
        score2 = _keyword_score(q_tokens, doc2)
        assert score2 >= score, (
            f"Relevant doc score ({score2}) should be ≥ irrelevant doc score ({score})"
        )

    def test_retrieve_returns_citation_and_tags(self):
        """Each retrieved document must carry citation and tags from corpus."""
        from retriever.retriever import FaissRetriever
        r = FaissRetriever()
        results = r.retrieve("tool wear failure inspection", top_k=3)
        assert len(results) > 0, "Retriever returned no results for a valid query"
        for doc in results:
            assert doc.get("citation"), (
                f"Retrieved doc {doc.get('title')!r} has no citation (H-3)"
            )
            assert isinstance(doc.get("tags"), list) and len(doc["tags"]) > 0, (
                f"Retrieved doc {doc.get('title')!r} has no tags (H-3)"
            )

    def test_retrieve_min_score_filters_noise(self):
        """A very high min_score should return fewer (or zero) results."""
        from retriever.retriever import FaissRetriever
        r = FaissRetriever()
        all_results = r.retrieve("machine failure", top_k=5, min_score=0.0)
        filtered = r.retrieve("machine failure", top_k=5, min_score=0.99)
        assert len(filtered) <= len(all_results), (
            "min_score=0.99 returned MORE results than min_score=0.0"
        )

    def test_retrieve_tool_wear_query_hits_relevant_doc(self):
        """A 'tool wear' query must retrieve the Tool Wear Failure Mode document."""
        from retriever.retriever import FaissRetriever
        r = FaissRetriever()
        results = r.retrieve("tool wear failure inspection replace", top_k=5)
        titles = [d.get("title", "") for d in results]
        assert any("Tool Wear" in t for t in titles), (
            f"Expected 'Tool Wear Failure Mode' in top-5 results, got: {titles}"
        )

    def test_retrieve_thermal_query_hits_relevant_doc(self):
        """A thermal/heat query must retrieve the Thermal Failure Mode document."""
        from retriever.retriever import FaissRetriever
        r = FaissRetriever()
        results = r.retrieve(
            "Critical machine High risk heat dissipation failure thermal", top_k=5
        )
        titles = [d.get("title", "") for d in results]
        assert any("Thermal" in t or "Heat" in t for t in titles), (
            f"Expected thermal/heat doc in top-5 results, got: {titles}"
        )

    def test_retrieve_reset_invalidates_index(self):
        """reset() must force a rebuild and still return valid results."""
        from retriever.retriever import FaissRetriever
        r = FaissRetriever()
        _ = r.retrieve("predictive maintenance", top_k=2)
        assert r._built

        r.reset()
        assert not r._built

        results2 = r.retrieve("predictive maintenance", top_k=2)
        assert r._built
        assert len(results2) > 0


# ---------------------------------------------------------------------------
# H-3: Retrieval agent query construction
# ---------------------------------------------------------------------------

class TestH3RetrievalAgent:
    """H-3: Retrieval agent must build context-aware queries."""

    def test_build_query_includes_status(self):
        from agents.retrieval_agent import _build_query
        state = _make_full_state(status="Critical", prob=0.9, health=10.0, risk="High")
        q = _build_query(state)
        assert "Critical" in q, f"Query missing machine status: {q!r}"

    def test_build_query_includes_tool_wear_hint(self):
        """Tool wear > 180 min must add tool wear failure hint to query."""
        from agents.retrieval_agent import _build_query
        state = _make_full_state(raw_input=_CRITICAL_INPUT, status="Critical",
                                 prob=0.9, health=10.0, risk="High")
        q = _build_query(state)
        assert "tool wear failure" in q.lower(), (
            f"Expected 'tool wear failure' hint for wear=253 min, got: {q!r}"
        )

    def test_build_query_includes_hdf_hint(self):
        """temp_diff < 8.6 K AND rpm < 1380 must add heat dissipation hint."""
        from agents.retrieval_agent import _build_query
        state = _make_full_state(raw_input=_CRITICAL_INPUT, status="Critical")
        q = _build_query(state)
        assert "heat dissipation" in q.lower() or "thermal" in q.lower(), (
            f"Expected heat dissipation/thermal hint for HDF conditions, got: {q!r}"
        )

    def test_build_query_includes_osf_hint(self):
        """wear_intensity > 85% of OSF threshold must add overstrain hint."""
        from agents.retrieval_agent import _build_query
        # H-type: threshold = 11000, 253×76.6 = 19,379 >> threshold
        state = _make_full_state(raw_input=_CRITICAL_INPUT, status="Critical")
        q = _build_query(state)
        assert "overstrain" in q.lower() or "mechanical stress" in q.lower(), (
            f"Expected overstrain hint for wear×torque=19379>11000, got: {q!r}"
        )

    def test_build_query_uses_shap_contributors(self):
        """SHAP top_contributors must be mapped to terms in the query."""
        from agents.retrieval_agent import _build_query
        state = _make_full_state(status="Warning")
        state["shap_explanation"] = {
            "top_contributors": ["Torque [Nm]", "Tool wear [min]"],
            "positive_contributors": ["Torque [Nm]"],
            "negative_contributors": [],
            "explanation_text": "Test.",
            "confidence": 0.85,
        }
        q = _build_query(state)
        assert "torque" in q.lower() or "tool wear" in q.lower(), (
            f"SHAP contributors not reflected in query: {q!r}"
        )

    def test_run_retrieval_sets_citations_on_documents(self):
        """After run_retrieval, retrieved_documents must have citation fields."""
        from agents.retrieval_agent import run_retrieval
        state = _make_full_state(raw_input=_CRITICAL_INPUT, status="Critical",
                                 prob=0.9, health=10.0, risk="High")
        result = run_retrieval(state)
        docs = result.get("retrieved_documents", [])
        assert len(docs) > 0, "run_retrieval returned no documents"
        docs_with_citation = [d for d in docs if d.get("citation")]
        assert len(docs_with_citation) > 0, (
            "None of the retrieved documents have a citation field (H-3)"
        )


# ---------------------------------------------------------------------------
# H-3: Report generator citations
# ---------------------------------------------------------------------------

class TestH3ReportGeneratorCitations:
    """H-3: Markdown output must include Evidence & Citations section when docs are available."""

    def _make_report_with_citations(self) -> Dict[str, Any]:
        return {
            "prediction_summary": "Machine classified as Critical with health score 10.0.",
            "technical_explanation": "High torque and tool wear drive failure risk.",
            "primary_drivers": ["Torque [Nm]", "Tool wear [min]"],
            "retrieved_evidence": [
                {"title": "Tool Wear Failure Mode", "source": "x", "excerpt": "y",
                 "confidence": 0.85, "citation": "Boothroyd (2006). Tool Wear.",
                 "tags": ["tool", "wear"]},
            ],
            "retrieved_evidence_citations": [
                "Boothroyd (2006). Fundamentals of Machining.",
                "ISO 13373-1:2002. Condition monitoring.",
            ],
            "maintenance_recommendations": [
                {"action": "Inspect tooling", "priority": "High", "rationale": "Wear threshold."}
            ],
            "risk_assessment": {
                "risk_level": "High", "severity": "High",
                "business_impact": "High impact", "urgency": "Within 24 hours",
                "rationale": "Compounded failure signals.",
            },
            "confidence": 92.0,
            "final_recommendation": "Inspect tooling | Urgency: Within 24 hours | Confidence: 92.0%",
        }

    def test_markdown_contains_evidence_section(self):
        from reports.generator import to_markdown
        report = self._make_report_with_citations()
        md = to_markdown(report)
        assert "## Evidence & Citations" in md, (
            "Markdown report missing '## Evidence & Citations' section (H-3)"
        )

    def test_markdown_contains_citation_text(self):
        from reports.generator import to_markdown
        report = self._make_report_with_citations()
        md = to_markdown(report)
        assert "Boothroyd" in md, "Citation text not found in Markdown output (H-3)"
        assert "ISO 13373" in md, "Second citation not found in Markdown output (H-3)"

    def test_markdown_no_citations_section_when_empty(self):
        """When retrieved_evidence_citations is empty, section must be absent."""
        from reports.generator import to_markdown
        report = self._make_report_with_citations()
        report["retrieved_evidence_citations"] = []
        md = to_markdown(report)
        assert "## Evidence & Citations" not in md, (
            "Citations section should be absent when retrieved_evidence_citations is empty"
        )

    def test_report_agent_populates_citations_key(self):
        """engineering_report must have retrieved_evidence_citations key."""
        from agents.report_agent import run_report
        state = _make_full_state(status="Critical", prob=0.9, health=10.0, risk="High")
        state["retrieved_documents"] = [
            {"title": "Tool Wear Failure Mode", "source": "x", "section": None,
             "excerpt": "y", "confidence": 0.9,
             "citation": "Boothroyd (2006). Tool Wear.",
             "tags": ["tool", "wear"]},
        ]
        state["risk_assessment"] = {
            "risk_level": "High", "severity": "High",
            "business_impact": "High impact", "urgency": "Within 24 hours",
            "rationale": "Test.",
        }
        state["shap_explanation"] = {
            "top_contributors": ["Torque [Nm]"],
            "positive_contributors": ["Torque [Nm]"],
            "negative_contributors": [],
            "explanation_text": "High torque drives risk.",
            "confidence": 0.9,
        }
        state["trend_analysis"] = {
            "direction": "Worsening", "health_trend": "Worsening",
            "risk_trend": "Increasing", "summary": "Trend worsening.",
        }
        state["operational_impact"] = {
            "impact_category": "Critical", "estimated_priority": "Immediate",
            "severity_score": 9.0, "notes": "Immediate action required.",
        }
        result = run_report(state)
        report = result.get("engineering_report", {})
        assert "retrieved_evidence_citations" in report, (
            "engineering_report missing 'retrieved_evidence_citations' key (H-3)"
        )
        assert report["retrieved_evidence_citations"] == ["Boothroyd (2006). Tool Wear."], (
            f"Unexpected citations: {report['retrieved_evidence_citations']}"
        )


# ---------------------------------------------------------------------------
# H-7: Graph structure tests
# ---------------------------------------------------------------------------

class TestH7GraphStructure:
    """H-7: Graph must have correct node names and not collide with state keys."""

    def test_all_nodes_prefixed_with_node(self):
        """All node names must start with 'node_' to avoid state-key collision."""
        from graph.graph import _build_graph
        from langgraph.graph import StateGraph
        # Build the graph and inspect its internal node registry
        graph = _build_graph()
        # LangGraph stores nodes in graph.nodes dict
        node_names = list(graph.nodes.keys())
        non_prefixed = [n for n in node_names if n != "__start__" and n != "__end__"
                        and not n.startswith("node_")]
        assert not non_prefixed, (
            f"Node names without 'node_' prefix (risk state-key collision): {non_prefixed}"
        )

    def test_graph_has_all_expected_nodes(self):
        expected = {
            "node_prediction", "node_supervisor", "node_explanation",
            "node_retrieval", "node_maintenance", "node_risk", "node_trend",
            "node_operational_impact", "node_validator", "node_report",
        }
        from graph.graph import _build_graph
        graph = _build_graph()
        actual = set(graph.nodes.keys()) - {"__start__", "__end__"}
        missing = expected - actual
        assert not missing, f"Graph is missing expected nodes: {missing}"

    def test_route_from_supervisor_human_review(self):
        """Low-confidence prediction must route to human_review."""
        from graph.graph import _route_from_supervisor
        state = {
            "prediction": {"confidence": 4.0, "machine_status": "Warning"},
        }
        result = _route_from_supervisor(state)
        assert result == "human_review", f"Expected human_review, got {result!r}"

    def test_route_from_supervisor_parallel_agents(self):
        """High-confidence prediction must route to parallel_agents."""
        from graph.graph import _route_from_supervisor
        state = {
            "prediction": {"confidence": 96.0, "machine_status": "Healthy"},
        }
        result = _route_from_supervisor(state)
        assert result == "parallel_agents", f"Expected parallel_agents, got {result!r}"

    def test_route_respects_explicit_next(self):
        """If state has 'next' set by supervisor, _route_from_supervisor must honour it."""
        from graph.graph import _route_from_supervisor
        state = {
            "next": "report",
            "prediction": {"confidence": 96.0, "machine_status": "Healthy"},
        }
        result = _route_from_supervisor(state)
        assert result == "report", f"Expected 'report' (explicit next), got {result!r}"


# ---------------------------------------------------------------------------
# H-7: End-to-end critical input produces cited report
# ---------------------------------------------------------------------------

class TestH7EndToEndCritical:
    """H-7: A critical input must flow end-to-end and produce a cited, high-risk report."""

    def test_critical_input_produces_high_risk_report(self):
        from graph.graph import run_graph
        result = run_graph(_CRITICAL_INPUT, history=[])

        assert isinstance(result, dict) and len(result) > 0, (
            "run_graph returned empty dict for critical input"
        )

        pred = result.get("prediction", {})
        # Critical input has very high torque + high wear → expect high failure prob
        assert pred.get("failure_probability", 0) > 0.5, (
            f"Expected failure_probability > 0.5 for critical input, "
            f"got {pred.get('failure_probability')}"
        )

        report = result.get("engineering_report", {})
        assert report.get("prediction_summary"), "prediction_summary is empty"
        assert report.get("final_recommendation"), "final_recommendation is empty"

        # Risk should be High or Critical
        risk_level = (report.get("risk_assessment") or {}).get("risk_level", "")
        assert risk_level in ("High", "Critical"), (
            f"Expected High or Critical risk for critical input, got {risk_level!r}"
        )

    def test_end_to_end_report_contains_citations(self):
        """Engineering report must have at least one citation for a critical input."""
        from graph.graph import run_graph
        result = run_graph(_CRITICAL_INPUT, history=[])
        report = result.get("engineering_report", {})
        citations = report.get("retrieved_evidence_citations", [])
        assert isinstance(citations, list), "retrieved_evidence_citations must be a list"
        # With the expanded corpus and context-driven query, we should get citations
        assert len(citations) > 0, (
            "Engineering report has no citations for a critical input — "
            "check that retrieval_agent ran and corpus documents have 'citation' fields"
        )

    def test_healthy_input_produces_low_risk_report(self):
        from graph.graph import run_graph
        result = run_graph(_HEALTHY_INPUT, history=[])
        pred = result.get("prediction", {})
        assert pred.get("machine_status") in ("Healthy", "Warning"), (
            f"Unexpected status for healthy input: {pred.get('machine_status')!r}"
        )
        report = result.get("engineering_report", {})
        assert report.get("prediction_summary"), "prediction_summary is empty"
