import os
import joblib
import pandas as pd

model = joblib.load(
    "models/production_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)


def create_features(data):

    df = pd.DataFrame([{

        "Type":
            data["Type"],

        "Air temperature [K]":
            data["Air_temperature_K"],

        "Process temperature [K]":
            data["Process_temperature_K"],

        "Rotational speed [rpm]":
            data["Rotational_speed_rpm"],

        "Torque [Nm]":
            data["Torque_Nm"],

        "Tool wear [min]":
            data["Tool_wear_min"]
    }])

    df["temperature_difference"] = (

        df["Process temperature [K]"]

        -

        df["Air temperature [K]"]
    )

    df["torque_speed_ratio"] = (

        df["Torque [Nm]"]

        /

        (
            df["Rotational speed [rpm]"]

            + 1e-6
        )
    )

    df["wear_intensity"] = (

        df["Tool wear [min]"]

        *

        df["Torque [Nm]"]
    )

    df["machine_stress_index"] = (

        df["Torque [Nm]"]

        *

        df["Rotational speed [rpm]"]
    )

    df["thermal_risk_index"] = (

        df["temperature_difference"]

        *

        df["Torque [Nm]"]
    )

    df["wear_efficiency_index"] = (

        df["Tool wear [min]"]

        /

        (
            df["Rotational speed [rpm]"]

            + 1e-6
        )
    )

    return df


def calculate_health(probability):

    health_score = (

        100

        -

        probability * 100
    )

    health_score = max(
        0,
        min(
            100,
            health_score
        )
    )

    if health_score < 20:

        status = "Critical"

    elif health_score < 50:

        status = "High Risk"

    elif health_score < 80:

        status = "Warning"

    else:

        status = "Healthy"

    return health_score, status


def log_production_data(
    feature_df,
    probability,
    health_score,
    status
):
    print("LOGGING DATA...")

    os.makedirs(
        "data/production",
        exist_ok=True
    )

    production_file = (
        "data/production/production_data.csv"
    )

    log_df = feature_df.copy()

    log_df[
        "failure_probability"
    ] = probability

    log_df[
        "health_score"
    ] = health_score

    log_df[
        "machine_status"
    ] = status

    if os.path.exists(
        production_file
    ):

        log_df.to_csv(
            production_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        log_df.to_csv(
            production_file,
            index=False
        )


def predict_machine_failure(payload):

    feature_df = create_features(
        payload
    )

    model_input = feature_df[
        feature_columns
    ]

    probability = (

        model

        .predict_proba(
            model_input
        )[0][1]
    )

    health_score, status = (

        calculate_health(
            probability
        )
    )

    log_production_data(
        feature_df,
        probability,
        health_score,
        status
    )

    return {

        "failure_probability":

            round(
                float(probability),
                4
            ),

        "health_score":

            round(
                float(health_score),
                2
            ),

        "machine_status":

            status
    }