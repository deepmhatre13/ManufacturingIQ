from fastapi import FastAPI

from app.schemas import (
    MachineInput,
    PredictionResponse
)

from app.predictor import (
    predict_machine_failure
)

app = FastAPI(

    title="ManufacturingIQ",

    description=
    "Predictive Maintenance Platform",

    version="1.0.0"
)


@app.get("/")
def root():

    return {

        "application":
            "ManufacturingIQ",

        "status":
            "running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_machine(

    request: MachineInput
):

    result = (

        predict_machine_failure(
            request.dict()
        )
    )

    return result