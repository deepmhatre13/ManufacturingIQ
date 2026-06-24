from pydantic import BaseModel

class MachineInput(BaseModel):
    Type:str
    Air_temperature_K:float
    Process_temperature_K:float
    Rotational_speed_rpm:float
    Torque_Nm:float
    Tool_wear_min:float

class PredictionResponse(BaseModel):
    failure_probability:float
    health_score:float
    machine_status:str
