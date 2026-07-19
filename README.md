# ManufacturingIQ

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0-EC1C24.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AI-Powered Predictive Maintenance Platform** — Predict machine failures in real-time using XGBoost, with a full MLOps lifecycle managed via FastAPI and Streamlit.

> Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Google OAuth Setup](#google-oauth-setup-for-dashboard)
- [Running the Application](#running-the-application)
  - [FastAPI Backend](#1-start-fastapi-backend)
  - [Streamlit Dashboard](#2-start-streamlit-dashboard)
  - [Docker](#docker)
- [Training the Model](#training-the-model)
- [MLOps & Monitoring](#mlops--monitoring)
  - [Drift Monitoring](#drift-monitoring)
  - [Automated Retraining](#automated-retraining)
- [Agentic AI System](#agentic-ai-system)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### 🔮 Predictive Intelligence
- Real-time machine failure prediction with health scoring
- 6 engineered features for robust predictions (temperature delta, torque-speed ratio, wear intensity, etc.)
- XGBoost classifier optimized via Optuna hyperparameter tuning

### 🏭 MLOps Center
- **Drift Detection**: Monitor feature drift and model performance degradation using Evidently AI
- **Automated Retraining**: Scheduled retraining pipeline with candidate model evaluation
- **Model Registry**: Versioned model storage with promotion gates
- **Performance Metrics**: Track accuracy, precision, recall, F1 over time

### 🔐 Enterprise Security
- **Two-layer authentication**: Google OAuth for dashboard users + API key auth for service-to-service
- **Constant-time API key comparison** to prevent timing attacks
- **Optional email/domain allowlists** for dashboard access control

### 🤖 Agentic AI System
- Multi-agent architecture using LangGraph for intelligent decision support
- Agents for prediction, explanation, risk assessment, trend analysis, and report generation
- RAG-powered knowledge retrieval from maintenance corpus
- PDF report generation with automated insights

### 📊 Centralized Feature Engineering
- Single source of truth for all feature calculations
- Epsilon-guarded divisions for production safety
- Training/serving parity — same code used in notebook, API, and monitoring

### 📝 Structured Logging
- Development mode: human-readable colored console output
- Production mode: JSON-lines format for log aggregation (ELK, Datadog, etc.)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  (Google OAuth · Predictive Intelligence · MLOps Center)    │
└──────────────────────────┬──────────────────────────────────┘
                           │ X-API-Key
┌──────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend                          │
│  (/predict · /health · centralized feature engineering)     │
└──────┬───────────────┬───────────────┬──────────────────────┘
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────────────────┐
│  XGBoost    │ │  Drift      │ │  Agentic AI (LangGraph)  │
│  Model      │ │  Monitor    │ │  · Prediction Agent      │
│             │ │  (Evidently)│ │  · Explanation Agent     │
│             │ │             │ │  · Risk Agent            │
│             │ │             │ │  · Trend Agent           │
│             │ │             │ │  · Maintenance Agent     │
│             │ │             │ │  · Report Agent          │
└─────────────┘ └─────────────┘ └──────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)
- Dependencies listed in [requirements.txt](requirements.txt)

### Installation

```bash
# Clone repository
git clone https://github.com/deepmhatre13/ManufacturingIQ.git
cd ManufacturingIQ

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MANUFACTURINGIQ_API_KEYS` | Comma-separated valid API keys for backend auth | Yes (backend) | — |
| `MANUFACTURINGIQ_API_KEY` | API key for dashboard-to-backend communication | Yes (dashboard) | — |
| `API_BASE_URL` | Backend API URL | No | `http://127.0.0.1:8000` |
| `ENV` | Environment mode: `development` or `production` | No | `development` |
| `LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | No | `INFO` |
| `GEMINI_API_KEY` | Google Gemini API key (for AI agent features) | No | — |

### Google OAuth Setup for Dashboard

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new OAuth 2.0 Client ID (Web application type)
3. Add authorized redirect URIs:
   - Local: `http://localhost:8501/oauth2callback`
   - Production: `https://your-app-url.streamlit.app/oauth2callback`
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your credentials:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "REPLACE_WITH_RANDOM_SECRET"
client_id = "REPLACE_WITH_GOOGLE_CLIENT_ID"
client_secret = "REPLACE_WITH_GOOGLE_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

# Optional: restrict access to specific emails/domains
# ALLOWED_EMAILS = "user1@example.com,user2@example.com"
# ALLOWED_EMAIL_DOMAINS = "yourcompany.com"
```

---

## Running the Application

### 1. Start FastAPI Backend

```bash
# Set API key
export MANUFACTURINGIQ_API_KEYS="your-secure-api-key-here"

# Run with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Streamlit Dashboard

```bash
# Set environment variables
export MANUFACTURINGIQ_API_KEY="your-secure-api-key-here"
export API_BASE_URL="http://localhost:8000"

# Run Streamlit
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. You must log in with Google to access it.

### Docker

```bash
# Build the image
docker build -t manufacturingiq .

# Run the container
docker run -p 8000:8000 \
  -e MANUFACTURINGIQ_API_KEYS="your-secure-api-key-here" \
  manufacturingiq
```

---

## Training the Model

Open and run the notebook cells in order:

```bash
jupyter notebook notebook.ipynb
```

The notebook will:
1. Load and explore the AI4I 2020 dataset
2. Apply centralized feature engineering from `feature_engineering.engineer`
3. Train and compare multiple models (Logistic Regression, Random Forest, XGBoost)
4. Optimize XGBoost with Optuna
5. Save production model and metadata
6. Generate drift reports and monitoring dashboards

---

## MLOps & Monitoring

### Drift Monitoring

```bash
python monitoring/drift_monitor.py
```

Uses [Evidently AI](https://www.evidentlyai.com/) to detect:
- **Data drift**: Feature distribution shifts using statistical tests
- **Model drift**: Performance degradation over time

### Automated Retraining

```bash
# Run the retraining scheduler
python scheduler/auto_retrain.py
```

The retraining pipeline:
1. Trains a candidate model on recent data
2. Evaluates against the current production model
3. Promotes the candidate if it outperforms the current model
4. Logs all metrics to MLflow

---

## Agentic AI System

The platform includes a multi-agent AI system built with [LangGraph](https://langchain-ai.github.io/langgraph/):

| Agent | Responsibility |
|-------|---------------|
| **Supervisor Agent** | Orchestrates the multi-agent workflow |
| **Prediction Agent** | Generates failure predictions with confidence scores |
| **Explanation Agent** | Provides SHAP-based model explanations |
| **Risk Agent** | Assesses operational risk levels |
| **Trend Agent** | Analyzes historical performance trends |
| **Maintenance Agent** | Recommends maintenance actions |
| **Retrieval Agent** | RAG-based knowledge retrieval from maintenance corpus |
| **Report Agent** | Generates comprehensive PDF reports |
| **Decision Validator Agent** | Validates and cross-checks agent outputs |

---

## API Reference

### `GET /`
Health check (unauthenticated).

### `GET /health`
Detailed health status (unauthenticated).

### `POST /predict`
Get machine failure prediction (requires `X-API-Key` header).

**Request body:**
```json
{
  "Type": "M",
  "Air_temperature_K": 298.1,
  "Process_temperature_K": 308.6,
  "Rotational_speed_rpm": 1551,
  "Torque_Nm": 42.8,
  "Tool_wear_min": 12
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "Type": "M",
    "Air_temperature_K": 298.1,
    "Process_temperature_K": 308.6,
    "Rotational_speed_rpm": 1551,
    "Torque_Nm": 42.8,
    "Tool_wear_min": 12
  }'
```

---

## Project Structure

```
ManufacturingIQ/
├── app/                          # FastAPI backend
│   ├── main.py                   # API endpoints with auth
│   ├── predictor.py              # Model inference
│   ├── schemas.py                # Pydantic models
│   ├── auth.py                   # API key authentication
│   ├── agentic.py                # Agentic AI API endpoints
│   └── scoring.py                # Scoring utilities
├── agents/                       # LangGraph agentic AI system
│   ├── supervisor_agent.py       # Workflow orchestrator
│   ├── prediction_agent.py       # Failure prediction agent
│   ├── explanation_agent.py      # SHAP explanation agent
│   ├── risk_agent.py             # Risk assessment agent
│   ├── trend_agent.py            # Trend analysis agent
│   ├── maintenance_agent.py      # Maintenance recommendation agent
│   ├── retrieval_agent.py        # RAG knowledge retrieval agent
│   ├── report_agent.py           # PDF report generation agent
│   ├── decision_validator_agent.py  # Output validation agent
│   └── _utils.py                 # Shared agent utilities
├── dashboard/                    # Streamlit frontend
│   ├── app.py                    # Main dashboard with OAuth
│   ├── pages/                    # Dashboard pages
│   │   ├── predictive_intelligence.py
│   │   └── mlops_center.py
│   ├── components/               # Reusable UI components
│   │   ├── cards.py
│   │   ├── charts.py
│   │   ├── gauges.py
│   │   └── tables.py
│   ├── utils/                    # Dashboard utilities
│   │   ├── api.py                # Backend client
│   │   └── agentic_api.py        # Agentic AI client
│   └── assets/                   # Static assets (CSS, images)
├── data/                         # Data (versioned via DVC)
│   ├── raw/                      # Raw dataset (DVC-tracked)
│   ├── processed/                # Processed data
│   └── production/               # Production data
├── feature_engineering/          # Centralized feature engineering
│   ├── __init__.py
│   └── engineer.py               # Single source of truth for features
├── graph/                        # LangGraph workflow graph
│   └── graph.py                  # Agent orchestration graph
├── history/                      # Prediction history management
│   ├── __init__.py
│   └── utils.py
├── knowledge/                    # RAG knowledge base
│   └── corpus.py                 # Maintenance knowledge corpus
├── models/                       # Trained models (DVC-tracked)
│   ├── production_model.pkl
│   ├── feature_columns.pkl
│   ├── model_metadata.json
│   └── embeddings/               # Vector embeddings for RAG
├── monitoring/                   # Drift detection & metrics
│   ├── __init__.py
│   ├── drift_monitor.py          # Evidently-based drift detection
│   └── metrics_logger.py         # Performance metrics logging
├── reports/                      # Generated PDF reports
│   └── generator.py              # Report generation engine
├── retraining/                   # Automated model retraining
│   ├── __init__.py
│   ├── retrain.py                # Retraining orchestration
│   ├── train_candidate.py        # Candidate model training
│   ├── model_evaluator.py        # Model comparison & evaluation
│   └── promote_model.py          # Model promotion logic
├── retriever/                    # RAG document retriever
│   └── retriever.py              # Vector search retrieval
├── scheduler/                    # Scheduled tasks
│   ├── __init__.py
│   └── auto_retrain.py           # Scheduled retraining
├── state/                        # LangGraph state management
│   └── schema.py                 # State schemas
├── tests/                        # Test suite
│   ├── test_auth.py
│   ├── test_feature_engineering.py
│   ├── test_drift_monitor.py
│   ├── test_dashboard_auth_config.py
│   ├── test_critical_fixes.py
│   ├── test_high_priority_fixes.py
│   └── test_h3_h7.py
├── .streamlit/
│   ├── secrets.toml.example      # Streamlit secrets template
│   └── config.toml               # Streamlit server config
├── logging_config.py             # Structured logging configuration
├── notebook.ipynb                # Training notebook with EDA
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── .env.example                  # Environment variables template
├── .gitignore
├── .dvcignore
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
├── SECURITY.md                   # Security policy
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_feature_engineering.py -v
pytest tests/test_auth.py -v
pytest tests/test_drift_monitor.py -v
pytest tests/test_dashboard_auth_config.py -v
```

---

## Security

- **API Keys**: Loaded from `MANUFACTURINGIQ_API_KEYS` env var, compared using `secrets.compare_digest()` to prevent timing attacks
- **Never hardcode keys** in source code or commit them to version control
- **Google OAuth**: Dashboard access requires Google sign-in, with optional email/domain allowlists
- **Two auth layers**: Dashboard login (OAuth) authenticates the human user; X-API-Key authenticates the dashboard service to the backend
- **Dependency scanning**: Keep dependencies updated via `pip-audit` or Dependabot

For reporting security vulnerabilities, see [SECURITY.md](SECURITY.md).

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from UCI Machine Learning Repository
- [Evidently AI](https://www.evidentlyai.com/) for drift monitoring
- [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [Streamlit](https://streamlit.io/) for the dashboard framework