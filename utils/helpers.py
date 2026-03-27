from typing import Dict


ACTION_LABELS = {
    0: "Assign ICU",
    1: "Assign General",
    2: "Assign Isolation",
    3: "Keep Waiting",
    4: "Internal Transfer",
    5: "External Transfer",
}


def format_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "Average Reward": round(metrics.get("total_reward", 0.0), 2),
        "Average Waiting Time": round(metrics.get("avg_waiting_time", 0.0), 2),
        "Bed Utilization Rate": round(metrics.get("bed_utilization_rate", 0.0), 4),
        "Correct Allocations": round(metrics.get("correct_allocations", 0.0), 2),
        "Wrong Allocations": round(metrics.get("wrong_allocations", 0.0), 2),
        "Critical Delays": round(metrics.get("critical_delays", 0.0), 2),
        "Internal Transfers": round(metrics.get("internal_transfers", 0.0), 2),
        "External Transfers": round(metrics.get("external_transfers", 0.0), 2),
        "Decision Accuracy": round(metrics.get("decision_accuracy", 0.0), 4),
    }


def action_options():
    return [(action_id, label) for action_id, label in ACTION_LABELS.items()]
