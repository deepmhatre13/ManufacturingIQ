import time
import subprocess


CHECK_INTERVAL = 60


while True:

    print(
        "\nChecking Drift..."
    )

    subprocess.run(

        [
            "python",
            "retraining/retrain.py"
        ]
    )

    print(
        "\nSleeping..."
    )

    time.sleep(
        CHECK_INTERVAL
    )