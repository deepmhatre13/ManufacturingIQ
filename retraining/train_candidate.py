import logging
import pandas as pd
import joblib
import json
from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    roc_auc_score
)

from xgboost import (
    XGBClassifier
)

from logging_config import configure_logging

configure_logging(service_name="retraining")
logger = logging.getLogger(__name__)

try:
    data = pd.read_csv(
        "data/production/production_data_labeled.csv"
    )
    logger.info("Loaded labeled production data: %d rows", len(data))
except Exception as exc:
    logger.exception("Failed to load production data: %s", exc)
    raise

target = "Machine failure"

drop_columns = [

    "failure_probability",

    "health_score",

    "machine_status"
]

X = data.drop(
    columns=drop_columns + [target]
)

y = data[target]

X_train, X_test, y_train, y_test = (

    train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )
)

candidate_model = XGBClassifier(

    random_state=42,

    eval_metric="logloss"
)

candidate_model.fit(
    X_train,
    y_train
)

pred_probs = (

    candidate_model

    .predict_proba(
        X_test
    )[:, 1]
)

candidate_auc = (

    roc_auc_score(
        y_test,
        pred_probs
    )
)

joblib.dump(

    candidate_model,

    "models/candidate_model.pkl"
)

logger.info(
    "Candidate ROC AUC: %.4f",
    candidate_auc
)


metrics = {

    "roc_auc":
        float(candidate_auc)
}

with open(
    "models/candidate_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )