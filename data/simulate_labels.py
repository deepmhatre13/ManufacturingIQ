import pandas as pd
import numpy as np

production_data = pd.read_csv(
    "data/production/production_data.csv"
)

np.random.seed(42)

production_data["Machine failure"] = (

    production_data[
        "failure_probability"
    ]

    > np.random.uniform(
        0,
        1,
        len(production_data)
    )

).astype(int)

production_data.to_csv(
    "data/production/production_data_labeled.csv",
    index=False
)

print(
    production_data[
        [
            "failure_probability",
            "Machine failure"
        ]
    ].head()
)

print(
    "\nLabels Generated"
)