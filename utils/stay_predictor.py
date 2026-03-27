from typing import Dict, List

import numpy as np


class StayDurationPredictor:
    def __init__(self, samples: int = 500):
        self.coefficients = self._fit_coefficients(samples)

    def _feature_vector(self, age: int, severity: str, required_bed: str, comorbidity: int) -> np.ndarray:
        severity_code = {"stable": 0, "moderate": 1, "critical": 2}[severity]
        bed_code = {"Isolation": 0, "General": 1, "ICU": 2}[required_bed]
        return np.array(
            [
                1.0,
                age / 100.0,
                float(severity_code),
                float(bed_code),
                float(comorbidity),
                float(severity_code * bed_code),
            ],
            dtype=np.float32,
        )

    def _generate_training_rows(self, samples: int) -> List[np.ndarray]:
        rows = []
        for _ in range(samples):
            age = np.random.randint(18, 90)
            severity = np.random.choice(["stable", "moderate", "critical"], p=[0.30, 0.45, 0.25])
            required_bed = np.random.choice(["Isolation", "General", "ICU"], p=[0.15, 0.60, 0.25])
            comorbidity = int(np.random.choice([0, 1], p=[0.6, 0.4]))

            base_days = 2.0
            severity_bonus = {"stable": 0.5, "moderate": 2.0, "critical": 4.5}[severity]
            bed_bonus = {"Isolation": 1.0, "General": 1.5, "ICU": 3.5}[required_bed]
            age_bonus = age / 55.0
            comorbidity_bonus = 1.8 * comorbidity
            noise = np.random.normal(0, 0.6)
            stay_days = max(1.0, base_days + severity_bonus + bed_bonus + age_bonus + comorbidity_bonus + noise)

            features = self._feature_vector(age, severity, required_bed, comorbidity)
            rows.append(np.append(features, stay_days))
        return rows

    def _fit_coefficients(self, samples: int) -> np.ndarray:
        rows = np.array(self._generate_training_rows(samples), dtype=np.float32)
        x = rows[:, :-1]
        y = rows[:, -1]
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return coefficients

    def predict_days(self, age: int, severity: str, required_bed: str, comorbidity: int) -> int:
        features = self._feature_vector(age, severity, required_bed, comorbidity)
        prediction = float(features @ self.coefficients)
        return max(1, int(round(prediction)))

    def explain_prediction(self, patient: Dict) -> str:
        predicted_days = self.predict_days(
            age=patient["age"],
            severity=patient["severity"],
            required_bed=patient["required_bed"],
            comorbidity=int(patient.get("comorbidity", 0)),
        )
        return (
            f"Estimated stay: {predicted_days} days based on age, severity, "
            f"bed requirement, and comorbidity risk."
        )
