"""
ManufacturingIQ - Table Components
Compact data table rendering
"""

import pandas as pd
from typing import List, Dict, Any


def prediction_history_table(data: List[Dict[str, Any]]):
    """Render a compact prediction history table - latest 5 records"""
    if not data:
        return pd.DataFrame()

    # Show only latest 5
    df = pd.DataFrame(data[:5])

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

    def color_status(val):
        if val == "Healthy":
            return "🟢"
        elif val == "Warning":
            return "🟡"
        else:
            return "🔴"

    df["status_icon"] = df["status"].apply(color_status)
    df["health_score"] = df["health_score"].apply(lambda x: f"{x:.1f}")
    df["failure_probability"] = df["failure_probability"].apply(lambda x: f"{float(x)*100:.1f}%")

    display_df = df.rename(columns={
        "timestamp": "Timestamp",
        "machine_type": "Machine",
        "health_score": "Health",
        "failure_probability": "Risk",
        "status_icon": "",
        "status": "Status"
    })

    cols = ["Timestamp", "Machine", "Health", "Risk", "", "Status"]
    display_df = display_df[cols] if all(c in display_df.columns for c in cols) else display_df

    return display_df