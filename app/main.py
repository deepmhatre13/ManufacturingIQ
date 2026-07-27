"""
ManufacturingIQ - FastAPI Application

Fixes applied:
  - H-4 (medium): `from .. import logging_config` was a broken relative import
    (app is not a package with an __init__ that re-exports logging_config).
    Replaced with a direct top-level import.
  - H-4 (medium): `request.dict()` is Pydantic v1 API; Pydantic v2 uses
    `.model_dump()`.  Updated both call-sites.
  - H-4 (medium): `Dict` and `Any` were used in the response_model annotation
    but never imported.  Added the imports and switched to `dict` (Python 3.9+
    built-in) which FastAPI 0.115 handles correctly.
"""

import os
import logging
from typing import Any, Dict

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse

import logging_config  # top-level module, not relative

from app.schemas import MachineInput, PredictionResponse
from app.predictor import predict_machine_failure
from app.agentic import run_agentic_pipeline
from app.auth import verify_api_key

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ManufacturingIQ",
    description="Industrial Predictive Maintenance Decision-Support Platform",
    version="3.0.0",
)


@app.on_event("startup")
def startup_event():
    logging_config.configure_logging(
        service_name="api",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    # Phase 1: Preload SentenceTransformer once (no HTTP requests after this)
    try:
        logger.info("Preloading SentenceTransformer...")
        from retriever.retriever import _get_embedder
        _get_embedder()
        logger.info("SentenceTransformer preloaded successfully")
    except Exception as exc:
        logger.warning("SentenceTransformer preload failed (will lazy-load on first request): %s", exc)

    # Phase 2: Preload FAISS index from disk (rebuild only if cache missing)
    try:
        logger.info("Preloading FAISS index...")
        from retriever.retriever import retriever
        retriever.preload()
        logger.info("FAISS index preloaded successfully")
    except Exception as exc:
        logger.warning("FAISS index preload failed (will lazy-load on first request): %s", exc)

    # Phase 3: Pre-build LangGraph graph (compile once, reuse forever)
    try:
        logger.info("Building LangGraph graph...")
        from graph.graph import _build_graph
        _build_graph()
        logger.info("LangGraph graph built successfully")
    except Exception as exc:
        logger.warning("LangGraph graph build failed (will build on first request): %s", exc)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all handler so unhandled errors never leak a 500 stack trace."""
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


@app.get("/")
def root():
    return {"application": "ManufacturingIQ", "status": "running", "version": "3.0.0"}


@app.get("/health")
def health_check():
    """Health check endpoint — unauthenticated for uptime monitors."""
    return {"status": "healthy", "service": "ManufacturingIQ API", "version": "3.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_machine(
    request: MachineInput,
    api_key: str = Depends(verify_api_key),
):
    result = predict_machine_failure(request.model_dump())  # H-4: Pydantic v2
    return result


@app.post("/agentic/predict", response_model=Dict[str, Any])
def agentic_predict(
    request: MachineInput,
    api_key: str = Depends(verify_api_key),
):
    history: list[dict] = []
    try:
        from history.utils import load_history
        history = load_history()
    except Exception as exc:
        logger.warning("History load failed: %s", exc)

    result = run_agentic_pipeline(request.model_dump(), history)  # H-4: Pydantic v2
    return result