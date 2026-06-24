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

data = pd.read_csv(
    "data/production/production_data_labeled.csv"
)

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

print(
    f"\nCandidate ROC AUC: {candidate_auc:.4f}"
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