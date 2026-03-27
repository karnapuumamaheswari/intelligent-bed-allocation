import csv
import os
import random
from datetime import datetime, timedelta
from typing import List


PATIENT_NAMES = [
    "Aarav",
    "Anaya",
    "Vikram",
    "Sara",
    "Kabir",
    "Ishita",
    "Rahul",
    "Meera",
    "Arjun",
    "Diya",
]


def generate_patients_csv(output_path: str = "data/patients.csv", count: int = 100) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    start_time = datetime(2026, 1, 1, 8, 0, 0)

    rows: List[dict] = []
    for patient_id in range(1, count + 1):
        severity = random.choices(
            ["critical", "moderate", "stable"],
            weights=[0.25, 0.45, 0.30],
            k=1,
        )[0]
        required_bed = random.choices(
            ["ICU", "General", "Isolation"],
            weights=[0.25, 0.60, 0.15],
            k=1,
        )[0]
        rows.append(
            {
                "patient_id": patient_id,
                "name": random.choice(PATIENT_NAMES),
                "age": random.randint(18, 85),
                "gender": random.choice(["Male", "Female"]),
                "severity": severity,
                "required_bed": required_bed,
                "arrival_time": (start_time + timedelta(minutes=15 * patient_id)).isoformat(),
                "status": "waiting",
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def generate_beds_csv(output_path: str = "data/beds.csv") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    bed_specs = [("ICU", 2), ("General", 4), ("Isolation", 1)]

    bed_id = 1
    for bed_type, count in bed_specs:
        for index in range(count):
            rows.append(
                {
                    "bed_id": bed_id,
                    "bed_type": bed_type,
                    "ward_name": f"{bed_type}_Ward_{index + 1}",
                    "status": "free",
                    "patient_id": "",
                }
            )
            bed_id += 1

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


if __name__ == "__main__":
    patient_path = generate_patients_csv()
    bed_path = generate_beds_csv()
    print(f"Generated patient data at {patient_path}")
    print(f"Generated bed data at {bed_path}")
