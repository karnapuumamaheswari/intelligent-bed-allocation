import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class HospitalEnv:
    ACTION_MEANINGS = {
        0: "Assign ICU",
        1: "Assign General",
        2: "Assign Isolation",
        3: "Keep Waiting",
        4: "Internal Transfer",
        5: "External Transfer",
    }

    def __init__(
        self,
        max_steps: int = 100,
        total_icu_beds: int = 2,
        total_general_beds: int = 4,
        total_isolation_beds: int = 1,
        hospital_name: str = "Default Hospital",
        special_bed_label: str = "Isolation",
        auto_generate_patients: bool = True,
    ):
        self.max_steps = max_steps

        self.total_icu_beds = total_icu_beds
        self.total_general_beds = total_general_beds
        self.total_isolation_beds = total_isolation_beds
        self.hospital_name = hospital_name
        self.special_bed_label = special_bed_label
        self.auto_generate_patients = auto_generate_patients

        self.action_size = 6
        self.state_size = 14

        self.reset()

    def reset(self) -> np.ndarray:
        self.time_step = 0
        self.transfer_count = 0

        self.free_icu = self.total_icu_beds
        self.free_general = self.total_general_beds
        self.free_isolation = self.total_isolation_beds

        self.waiting_queue: List[Dict] = []
        self.active_patients: List[Dict] = []
        self.completed_patients: List[Dict] = []
        self.transfer_log: List[Dict] = []
        self.bed_slots = self._initialize_bed_slots()

        self.total_reward = 0.0
        self.total_waiting_time = 0
        self.waiting_patients_processed = 0
        self.correct_allocations = 0
        self.wrong_allocations = 0
        self.external_transfers = 0
        self.internal_transfers = 0
        self.critical_delays = 0
        self.total_patients_seen = 0
        self.utilization_samples: List[float] = []

        if self.auto_generate_patients:
            initial_patients = random.randint(4, 6)
            for _ in range(initial_patients):
                self.waiting_queue.append(self._generate_patient())

        self.current_patient = self.waiting_queue[0] if self.waiting_queue else None
        self.utilization_samples.append(self._occupancy_rate())
        return self._get_state()

    def _initialize_bed_slots(self) -> Dict[str, List[Dict]]:
        return {
            "ICU": [
                {"bed_no": f"ICU-{index:02d}", "occupied": False, "patient_id": None}
                for index in range(1, self.total_icu_beds + 1)
            ],
            "General": [
                {"bed_no": f"GEN-{index:02d}", "occupied": False, "patient_id": None}
                for index in range(1, self.total_general_beds + 1)
            ],
            "Isolation": [
                {"bed_no": f"{self.special_bed_label[:3].upper()}-{index:02d}", "occupied": False, "patient_id": None}
                for index in range(1, self.total_isolation_beds + 1)
            ],
        }

    def _generate_patient(self) -> Dict:
        severity = random.choices(
            ["stable", "moderate", "critical"],
            weights=[0.25, 0.40, 0.35],
            k=1,
        )[0]

        if severity == "critical":
            required_bed = random.choices(
                ["ICU", "Isolation", "General"],
                weights=[0.75, 0.10, 0.15],
                k=1,
            )[0]
            stay = max(2, int(np.random.normal(7, 2)))
        elif severity == "moderate":
            required_bed = random.choices(
                ["General", "Isolation", "ICU"],
                weights=[0.70, 0.15, 0.15],
                k=1,
            )[0]
            stay = max(2, int(np.random.normal(5, 1.5)))
        else:
            required_bed = random.choices(
                ["General", "Isolation"],
                weights=[0.85, 0.15],
                k=1,
            )[0]
            stay = max(1, int(np.random.normal(3, 1)))

        self.total_patients_seen += 1

        return {
            "id": self.total_patients_seen,
            "name": f"Patient {self.total_patients_seen}",
            "age": random.randint(18, 85),
            "gender": random.choice(["Male", "Female"]),
            "severity": severity,
            "required_bed": required_bed,
            "comorbidity": random.choice([0, 1]),
            "waiting_time": 0,
            "remaining_stay": stay,
            "status": "waiting",
        }

    def create_patient(
        self,
        name: str,
        age: int,
        gender: str,
        severity: str,
        required_bed: str,
        comorbidity: int = 0,
        waiting_time: int = 0,
        predicted_stay: Optional[int] = None,
    ) -> Dict:
        self.total_patients_seen += 1
        return {
            "id": self.total_patients_seen,
            "name": name,
            "age": age,
            "gender": gender,
            "severity": severity,
            "required_bed": required_bed,
            "comorbidity": int(comorbidity),
            "waiting_time": waiting_time,
            "remaining_stay": predicted_stay if predicted_stay is not None else max(1, waiting_time + 3),
            "status": "waiting",
        }

    def add_patient_to_queue(self, patient: Dict, prioritize: bool = True) -> None:
        if prioritize:
            self.waiting_queue.insert(0, patient)
        else:
            self.waiting_queue.append(patient)
        self.current_patient = self.waiting_queue[0]

    def get_status_snapshot(self) -> Dict:
        waiting_critical, waiting_moderate, waiting_stable = self._waiting_counts()
        return {
            "free_icu": self.free_icu,
            "free_general": self.free_general,
            "free_isolation": self.free_isolation,
            "occupied_icu": self.total_icu_beds - self.free_icu,
            "occupied_general": self.total_general_beds - self.free_general,
            "occupied_isolation": self.total_isolation_beds - self.free_isolation,
            "occupancy_rate": self._occupancy_rate(),
            "queue_length": len(self.waiting_queue),
            "waiting_critical": waiting_critical,
            "waiting_moderate": waiting_moderate,
            "waiting_stable": waiting_stable,
            "active_patients": len(self.active_patients),
            "completed_patients": len(self.completed_patients),
            "transfers": self.transfer_count,
            "current_patient": self.current_patient,
        }

    def _severity_to_int(self, severity: str) -> int:
        return {"stable": 0, "moderate": 1, "critical": 2}[severity]

    def _bed_to_int(self, bed_type: str) -> int:
        return {"Isolation": 0, "General": 1, "ICU": 2}[bed_type]

    def _occupancy_rate(self) -> float:
        total_beds = self.total_icu_beds + self.total_general_beds + self.total_isolation_beds
        free_beds = self.free_icu + self.free_general + self.free_isolation
        return (total_beds - free_beds) / total_beds

    def _waiting_counts(self) -> Tuple[int, int, int]:
        critical = sum(1 for p in self.waiting_queue if p["severity"] == "critical")
        moderate = sum(1 for p in self.waiting_queue if p["severity"] == "moderate")
        stable = sum(1 for p in self.waiting_queue if p["severity"] == "stable")
        return critical, moderate, stable

    def _all_beds_full(self) -> bool:
        return self.free_icu == 0 and self.free_general == 0 and self.free_isolation == 0

    def _suitable_bed_available(self, required_bed: str) -> bool:
        return (
            (required_bed == "ICU" and self.free_icu > 0)
            or (required_bed == "General" and self.free_general > 0)
            or (required_bed == "Isolation" and self.free_isolation > 0)
        )

    def _get_state(self) -> np.ndarray:
        if not self.current_patient:
            if self.auto_generate_patients:
                self.current_patient = self._generate_patient()
            else:
                return np.zeros(self.state_size, dtype=np.float32)

        waiting_critical, waiting_moderate, waiting_stable = self._waiting_counts()

        return np.array(
            [
                self.free_icu / self.total_icu_beds,
                self.free_general / self.total_general_beds,
                self.free_isolation / self.total_isolation_beds,
                waiting_critical / 10.0,
                waiting_moderate / 10.0,
                waiting_stable / 10.0,
                self._severity_to_int(self.current_patient["severity"]) / 2.0,
                self._bed_to_int(self.current_patient["required_bed"]) / 2.0,
                self._occupancy_rate(),
                self.transfer_count / 10.0,
                self.time_step / self.max_steps,
                self.current_patient["waiting_time"] / 10.0,
                len(self.waiting_queue) / 12.0,
                1.0 if self._all_beds_full() else 0.0,
            ],
            dtype=np.float32,
        )

    def _allocate_bed(self, bed_type: str, patient: Dict) -> bool:
        slot = next((slot for slot in self.bed_slots[bed_type] if not slot["occupied"]), None)
        if slot is None:
            return False

        if bed_type == "ICU" and self.free_icu > 0:
            self.free_icu -= 1
        elif bed_type == "General" and self.free_general > 0:
            self.free_general -= 1
        elif bed_type == "Isolation" and self.free_isolation > 0:
            self.free_isolation -= 1
        else:
            return False

        patient["status"] = "admitted"
        patient["assigned_bed"] = bed_type
        patient["assigned_bed_no"] = slot["bed_no"]
        slot["occupied"] = True
        slot["patient_id"] = patient["id"]
        self.active_patients.append(patient)
        self.waiting_patients_processed += 1
        self.total_waiting_time += patient["waiting_time"]
        return True

    def _occupy_bed_for_transfer(self, bed_type: str, patient: Dict) -> bool:
        slot = next((slot for slot in self.bed_slots[bed_type] if not slot["occupied"]), None)
        if slot is None:
            return False

        if bed_type == "ICU" and self.free_icu > 0:
            self.free_icu -= 1
        elif bed_type == "General" and self.free_general > 0:
            self.free_general -= 1
        elif bed_type == "Isolation" and self.free_isolation > 0:
            self.free_isolation -= 1
        else:
            return False

        patient["assigned_bed"] = bed_type
        patient["assigned_bed_no"] = slot["bed_no"]
        patient["status"] = "admitted"
        slot["occupied"] = True
        slot["patient_id"] = patient["id"]
        return True

    def _release_bed(self, bed_type: str, bed_no: Optional[str] = None) -> None:
        if bed_no:
            for slot in self.bed_slots[bed_type]:
                if slot["bed_no"] == bed_no:
                    slot["occupied"] = False
                    slot["patient_id"] = None
                    break
        if bed_type == "ICU":
            self.free_icu += 1
        elif bed_type == "General":
            self.free_general += 1
        else:
            self.free_isolation += 1

    def _release_patients(self) -> None:
        remaining = []
        for patient in self.active_patients:
            patient["remaining_stay"] -= 1
            if patient["remaining_stay"] <= 0:
                self._release_bed(patient["assigned_bed"], patient.get("assigned_bed_no"))
                patient["status"] = "discharged"
                self.completed_patients.append(patient)
            else:
                remaining.append(patient)
        self.active_patients = remaining

    def _update_waiting_times(self) -> float:
        penalty = 0.0
        for patient in self.waiting_queue:
            patient["waiting_time"] += 1

            if patient["severity"] == "critical":
                penalty -= 0.8
                if patient["waiting_time"] >= 3:
                    penalty -= 1.2
            elif patient["severity"] == "moderate":
                penalty -= 0.25
            else:
                penalty -= 0.05
        return penalty

    def _add_new_patient(self) -> None:
        if not self.auto_generate_patients:
            return
        arrivals = random.choices([0, 1, 2, 3], weights=[0.1, 0.35, 0.35, 0.2])[0]
        for _ in range(arrivals):
            self.waiting_queue.append(self._generate_patient())

    def _pop_current_patient(self) -> None:
        if self.waiting_queue:
            self.waiting_queue.pop(0)
        self.current_patient = self.waiting_queue[0] if self.waiting_queue else None

    def _find_transfer_candidate(self, required_bed: str) -> Optional[Tuple[Dict, str]]:
        if required_bed == "ICU" and self.free_general > 0:
            for patient in self.active_patients:
                if patient["assigned_bed"] == "ICU" and patient["severity"] in {"stable", "moderate"}:
                    return patient, "General"

        if required_bed == "General" and self.free_isolation > 0:
            for patient in self.active_patients:
                if patient["assigned_bed"] == "General" and patient["required_bed"] == "Isolation":
                    return patient, "Isolation"

        if required_bed == "Isolation" and self.free_general > 0:
            for patient in self.active_patients:
                if patient["assigned_bed"] == "Isolation" and patient["severity"] == "stable":
                    return patient, "General"

        return None

    def _internal_transfer(self, current_patient: Dict) -> bool:
        candidate = self._find_transfer_candidate(current_patient["required_bed"])
        if not candidate:
            return False

        transfer_patient, target_bed = candidate
        previous_bed = transfer_patient["assigned_bed"]
        previous_bed_no = transfer_patient.get("assigned_bed_no")

        self._release_bed(previous_bed, previous_bed_no)
        moved = self._occupy_bed_for_transfer(target_bed, transfer_patient)
        if not moved:
            if previous_bed == "ICU":
                self.free_icu -= 1
            elif previous_bed == "General":
                self.free_general -= 1
            else:
                self.free_isolation -= 1
            for slot in self.bed_slots[previous_bed]:
                if slot["bed_no"] == previous_bed_no:
                    slot["occupied"] = True
                    slot["patient_id"] = transfer_patient["id"]
                    break
            return False
        self.transfer_count += 1
        self.internal_transfers += 1
        self.transfer_log.append(
            {
                "patient_id": transfer_patient["id"],
                "from_bed": previous_bed_no,
                "to_bed": transfer_patient.get("assigned_bed_no"),
                "transfer_type": "internal",
                "time_step": self.time_step,
            }
        )
        return self._allocate_bed(current_patient["required_bed"], current_patient)

    def _finalize_external_transfer(self, patient: Dict) -> None:
        patient["status"] = "transferred_external"
        self.external_transfers += 1
        self.transfer_count += 1
        self.waiting_patients_processed += 1
        self.total_waiting_time += patient["waiting_time"]
        self.transfer_log.append(
            {
                "patient_id": patient["id"],
                "from_bed": patient.get("assigned_bed_no"),
                "to_bed": None,
                "transfer_type": "external",
                "time_step": self.time_step,
            }
        )
        self._pop_current_patient()

    def _apply_assignment_action(self, action: int, patient: Dict) -> float:
        action_to_bed = {0: "ICU", 1: "General", 2: "Isolation"}
        chosen_bed = action_to_bed[action]
        reward = 0.0
        success = self._allocate_bed(chosen_bed, patient)

        if not success:
            return -8.0

        self._pop_current_patient()

        if chosen_bed == patient["required_bed"]:
            if chosen_bed == "ICU":
                reward += 15.0
            else:
                reward += 10.0
            self.correct_allocations += 1
            return reward

        if patient["severity"] == "critical" and chosen_bed != "ICU":
            self.wrong_allocations += 1
            return -10.0

        if patient["required_bed"] == "Isolation" and chosen_bed != "Isolation":
            self.wrong_allocations += 1
            return -10.0

        if patient["severity"] == "stable" and chosen_bed == "General":
            self.correct_allocations += 1
            return 4.0

        self.wrong_allocations += 1
        return -6.0

    def _current_metrics(self) -> Dict:
        avg_waiting_time = (
            self.total_waiting_time / self.waiting_patients_processed
            if self.waiting_patients_processed
            else 0.0
        )
        total_decisions = self.correct_allocations + self.wrong_allocations
        return {
            "total_reward": self.total_reward,
            "avg_waiting_time": avg_waiting_time,
            "bed_utilization_rate": float(np.mean(self.utilization_samples)) if self.utilization_samples else 0.0,
            "correct_allocations": self.correct_allocations,
            "wrong_allocations": self.wrong_allocations,
            "critical_delays": self.critical_delays,
            "external_transfers": self.external_transfers,
            "internal_transfers": self.internal_transfers,
            "processed_patients": self.waiting_patients_processed,
            "decision_accuracy": (self.correct_allocations / total_decisions) if total_decisions else 0.0,
        }

    def step(self, action: int):
        reward = 0.0
        done = False

        if not self.current_patient:
            if not self.waiting_queue and self.auto_generate_patients:
                self.waiting_queue.append(self._generate_patient())
            self.current_patient = self.waiting_queue[0] if self.waiting_queue else None

        if not self.current_patient and not self.auto_generate_patients:
            self._release_patients()
            self.time_step += 1
            self.utilization_samples.append(self._occupancy_rate())
            return self._get_state(), 0.0, self.time_step >= self.max_steps, self._current_metrics()

        patient = self.current_patient
        required_bed = patient["required_bed"]
        severity = patient["severity"]

        if action in {0, 1, 2}:
            reward += self._apply_assignment_action(action, patient)

        elif action == 3:
            if self._suitable_bed_available(required_bed):
                reward -= 8.0
            elif severity == "critical":
                reward += 2.0
                self.critical_delays += 1
            else:
                reward += 1.0

        elif action == 4:
            transferred = self._internal_transfer(patient)
            if transferred:
                reward += 6.0
                self.correct_allocations += 1
                self._pop_current_patient()
            else:
                reward -= 6.0

        elif action == 5:
            if self._suitable_bed_available(required_bed) or self._find_transfer_candidate(required_bed):
                reward -= 6.0
            else:
                reward += 4.0
                if severity == "critical":
                    reward += 1.0
                self._finalize_external_transfer(patient)

        else:
            reward -= 10.0

        self._release_patients()
        reward += self._update_waiting_times()

        if len(self.waiting_queue) > 10:
            reward -= (len(self.waiting_queue) - 10) * 0.3

        if len(self.waiting_queue) > 0 and self._occupancy_rate() < 0.8 and not self._all_beds_full():
            reward -= 0.5

        self._add_new_patient()

        if not self.current_patient and self.waiting_queue:
            self.current_patient = self.waiting_queue[0]
        elif not self.current_patient and not self.waiting_queue and self.auto_generate_patients:
            self.waiting_queue.append(self._generate_patient())
            self.current_patient = self.waiting_queue[0]

        self.time_step += 1
        self.total_reward += reward
        self.utilization_samples.append(self._occupancy_rate())

        if self.time_step >= self.max_steps:
            done = True

        return self._get_state(), reward, done, self._current_metrics()
