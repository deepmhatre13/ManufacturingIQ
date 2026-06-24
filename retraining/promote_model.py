import shutil

shutil.copy(

    "models/candidate_model.pkl",

    "models/production_model.pkl"
)

print(
    "\nProduction Model Updated"
)