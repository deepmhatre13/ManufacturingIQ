import json

PRODUCTION_AUC = 0.94

with open(
    "models/candidate_metrics.json"
) as f:

    candidate_metrics = json.load(
        f
    )

candidate_auc = candidate_metrics[
    "roc_auc"
]

if candidate_auc > PRODUCTION_AUC:

    print(
        "\nPROMOTE CANDIDATE"
    )

else:

    print(
        "\nKEEP PRODUCTION MODEL"
    )