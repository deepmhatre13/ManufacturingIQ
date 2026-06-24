import pandas as pd


TRAINING_REFERENCE = {

    "Torque [Nm]": 40.0,

    "Tool wear [min]": 100.0,

    "Rotational speed [rpm]": 1500.0
}


def calculate_drift():

    try:

        production_data = pd.read_csv(
            "data/production/production_data.csv"
        )

    except Exception:

        return {

            "drift_detected": False,

            "message":
                "No production data"
        }

    drift_report = {}

    for feature, training_mean in (

        TRAINING_REFERENCE.items()

    ):

        production_mean = (

            production_data[
                feature
            ].mean()
        )

        drift_percentage = abs(

            (
                production_mean
                -
                training_mean
            )

            /

            training_mean

        ) * 100

        drift_report[
            feature
        ] = round(
            drift_percentage,
            2
        )

    max_drift = max(
        drift_report.values()
    )

    return {

    "drift_detected":
        bool(max_drift > 10),

    "max_drift":
        float(round(max_drift,2)),

    "details": {

        k: float(v)

        for k,v in drift_report.items()
    }
}


if __name__ == "__main__":

    print(
        calculate_drift()
    )