# ManufacturingIQ

AI-Powered Predictive Maintenance Platform with XGBoost, FastAPI, and Streamlit.

## Features

- **Predictive Intelligence**: Real-time machine failure prediction with health scoring
- **MLOps Center**: Model monitoring, drift detection, and automated retraining
- **Centralized Feature Engineering**: Single source of truth for all feature calculations
- **Structured Logging**: Production-ready logging with JSON/diagnostics support
- **API Authentication**: Secure API key-based access control with constant-time comparison
- **Dashboard Authentication**: Google OAuth login for Streamlit dashboard with optional email allowlists

## Architecture

```
notebook.ipynb              # Training notebook with EDA and model optimization
feature_engineering/         # Centralized feature engineering module
app/                         # FastAPI backend
  main.py                    # API endpoints with auth
  predictor.py               # Model inference
  schemas.py                 # Pydantic models
  auth.py                    # API key authentication
dashboard/                   # Streamlit frontend
  app.py                     # Main dashboard with OAuth
  utils/api.py               # Backend client
monitoring/                  # Drift detection and metrics
retraining/                  # Automated model retraining
scheduler/                   # Retraining scheduler
```

## Getting Started

### Prerequisites

- Python 3.12+
- Virtual environment (venv)
- Dependencies listed in requirements.txt

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
# Required for API authentication
MANUFACTURINGIQ_API_KEYS=changeme-dev-key

# Dashboard API key (for backend authentication from dashboard)
MANUFACTURINGIQ_API_KEY=changeme-dev-key

# API base URL (default: local FastAPI server)
API_BASE_URL=http://127.0.0.1:8000

# Environment mode: development or production
ENV=development

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

### Google OAuth Setup for Dashboard

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new OAuth 2.0 Client ID
3. Add authorized redirect URIs:
   - Local: `http://localhost:8501/oauth2callback`
   - Production: `https://your-app-url.streamlit.app/oauth2callback`
4. Download client credentials and add to `.streamlit/secrets.toml`:

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

### Running the Application

#### 1. Start FastAPI Backend

```bash
# Set API key
export MANUFACTURINGIQ_API_KEYS="your-secure-api-key-here"

# Run with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**
- `GET /` - Health check (unauthenticated)
- `GET /health` - Detailed health status (unauthenticated)
- `POST /predict` - Get machine failure prediction (requires X-API-Key header)

**Example API Request:**
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

#### 2. Start Streamlit Dashboard

```bash
# Set environment variables
export MANUFACTURINGIQ_API_KEY="your-secure-api-key-here"
export API_BASE_URL="http://localhost:8000"

# Run Streamlit
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`. You must log in with Google to access the dashboard.

### Training the Model

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

### Running Drift Monitoring

```bash
python monitoring/drift_monitor.py
```

### Automated Retraining

```bash
python scheduler/auto_retrain.py
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MANUFACTURINGIQ_API_KEYS` | Comma-separated valid API keys for backend auth | Yes (backend) | - |
| `MANUFACTURINGIQ_API_KEY` | API key for dashboard-to-backend communication | Yes (dashboard) | - |
| `API_BASE_URL` | Backend API URL | No | `http://127.0.0.1:8000` |
| `ENV` | Environment mode: `development` or `production` | No | `development` |
| `LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | No | `INFO` |

## Logging

Logging is configured via `logging_config.py`:

- **Development mode** (`ENV=development`): Human-readable colored console output with timestamps
- **Production mode** (`ENV=production`): JSON-lines format for log aggregation

Each log line includes:
- Timestamp (UTC, ISO 8601)
- Log level
- Module/logger name
- Service name (`api`, `dashboard`, `monitoring`, `retraining`, `scheduler`)
- Message and optional exception tracebacks via `logger.exception()`

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_feature_engineering.py -v
pytest tests/test_auth.py -v
pytest tests/test_drift_monitor.py -v
pytest tests/test_dashboard_auth_config.py -v
```

## Feature Engineering

The `feature_engineering/engineer.py` module is the single source of truth for all 6 engineered features:

1. **temperature_difference**: Process temperature - Air temperature (thermal delta)
2. **torque_speed_ratio**: Torque / Rotational speed (mechanical load at given speed)
3. **wear_intensity**: Tool wear × Torque (cumulative mechanical stress)
4. **machine_stress_index**: Composite normalized index (thermal + mechanical + wear factors)
5. **thermal_risk_index**: Process temperature / Air temperature (thermal loading ratio)
6. **wear_efficiency_index**: Rotational speed / (Tool wear + 1) (operational efficiency)

All divisions are epsilon-guarded (`+ 1e-6`) for production safety. This module is imported by both the notebook and the API, ensuring training/serving parity.

## Security Notes

- **API Keys**: Loaded from `MANUFACTURINGIQ_API_KEYS` env var, compared using `secrets.compare_digest()` to prevent timing attacks
- **Never hardcode keys** in source code or commit them to version control
- **Google OAuth**: Dashboard access requires Google sign-in, with optional email/domain allowlists
- **Two auth layers**: Dashboard login (OAuth) authenticates the human user; X-API-Key authenticates the dashboard service to the backend

## Project Structure

```
ManufacturingIQ/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with auth
│   ├── predictor.py         # Model inference with centralized features
│   ├── schemas.py           # Pydantic request/response models
│   └── auth.py              # API key authentication
├── dashboard/
│   ├── app.py               # Streamlit dashboard with Google OAuth
│   ├── __init__.py
│   ├── pages/
│   ├── components/
│   ├── utils/
│   │   └── api.py           # Backend client with X-API-Key header
│   └── assets/
├── feature_engineering/
│   ├── __init__.py
│   └── engineer.py          # Single source of truth for features
├── monitoring/
│   ├── __init__.py
│   ├── drift_monitor.py
│   └── metrics_logger.py
├── retraining/
│   ├── __init__.py
│   ├── retrain.py
│   ├── train_candidate.py
│   ├── model_evaluator.py
│   └── promote_model.py
├── scheduler/
│   ├── __init__.py
│   └── auto_retrain.py
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_auth.py
│   ├── test_drift_monitor.py
│   └── test_dashboard_auth_config.py
├── models/
│   ├── production_model.pkl
│   ├── feature_columns.pkl
│   └── model_metadata.json
├── logging_config.py        # Structured logging configuration
├── notebook.ipynb
├── requirements.txt
├── Dockerfile
├── .env.example
├── .streamlit/secrets.toml.example
├── .gitignore
└── README.md
```

## Known Issues & Future Work

- **Per-user rate limiting**: Currently uses shared static API keys. Future: migrate to JWT-based auth with proper identity provider.
- **Model retraining triggers**: Currently manual/scheduled. Future: add drift-based automated triggers.
- **Model explainability**: SHAP values computed in notebook only. Future: serve SHAP explanations via API endpoint.

## License

Confidential - Internal Use Only