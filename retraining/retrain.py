import subprocess

from monitoring.drift_monitor import (
    calculate_drift
)

drift_report = (
    calculate_drift()
)

print(
    drift_report
)

if drift_report[
    "drift_detected"
]:

    print(
        "\nRetraining Triggered"
    )

    subprocess.run(

        [
            "python",
            "data/simulate_labels.py"
        ]
    )

    subprocess.run(

        [
            "python",
            "retraining/train_candidate.py"
        ]
    )

else:

    print(
        "\nNo Retraining Needed"
    )