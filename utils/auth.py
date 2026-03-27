import hashlib
import json
import os
import sqlite3
from typing import Dict, Optional


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_DIR = os.path.join(BASE_DIR, "database")
AUTH_DB_PATH = os.path.join(DB_DIR, "hospital_auth.db")


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def initialize_auth_db() -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS hospitals (
            hospital_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            location TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_configs (
            hospital_id INTEGER PRIMARY KEY,
            icu_beds INTEGER NOT NULL,
            general_beds INTEGER NOT NULL,
            special_beds INTEGER NOT NULL,
            special_bed_label TEXT NOT NULL,
            has_internal_transfer_team INTEGER NOT NULL,
            has_external_transfer_partners INTEGER NOT NULL,
            is_configured INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_states (
            hospital_id INTEGER PRIMARY KEY,
            free_icu INTEGER NOT NULL,
            free_general INTEGER NOT NULL,
            free_isolation INTEGER NOT NULL,
            time_step INTEGER NOT NULL,
            transfer_count INTEGER NOT NULL,
            total_reward REAL NOT NULL,
            total_waiting_time INTEGER NOT NULL,
            waiting_patients_processed INTEGER NOT NULL,
            correct_allocations INTEGER NOT NULL,
            wrong_allocations INTEGER NOT NULL,
            external_transfers INTEGER NOT NULL,
            internal_transfers INTEGER NOT NULL,
            critical_delays INTEGER NOT NULL,
            total_patients_seen INTEGER NOT NULL,
            waiting_queue_json TEXT NOT NULL,
            active_patients_json TEXT NOT NULL,
            completed_patients_json TEXT NOT NULL,
            transfer_log_json TEXT NOT NULL,
            utilization_samples_json TEXT NOT NULL,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id)
        )
        """
    )

    seed_rows = [
        ("City Care Hospital", "citycare", _hash_password("citycare123"), "Chennai"),
        ("Sunrise Medical Center", "sunrise", _hash_password("sunrise123"), "Bengaluru"),
    ]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO hospitals (hospital_name, username, password_hash, location)
        VALUES (?, ?, ?, ?)
        """,
        seed_rows,
    )
    conn.commit()
    conn.close()


def authenticate_hospital(username: str, password: str) -> Optional[Dict]:
    initialize_auth_db()
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT hospital_id, hospital_name, username, location
        FROM hospitals
        WHERE username = ? AND password_hash = ?
        """,
        (username, _hash_password(password)),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "hospital_id": row[0],
        "hospital_name": row[1],
        "username": row[2],
        "location": row[3],
    }


def get_hospital_config(hospital_id: int) -> Optional[Dict]:
    initialize_auth_db()
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT h.hospital_name, h.location, c.icu_beds, c.general_beds, c.special_beds,
               c.special_bed_label, c.has_internal_transfer_team, c.has_external_transfer_partners,
               c.is_configured
        FROM hospitals h
        LEFT JOIN hospital_configs c ON h.hospital_id = c.hospital_id
        WHERE h.hospital_id = ?
        """,
        (hospital_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "hospital_name": row[0],
        "location": row[1],
        "icu_beds": row[2] if row[2] is not None else 2,
        "general_beds": row[3] if row[3] is not None else 4,
        "special_beds": row[4] if row[4] is not None else 1,
        "special_bed_label": row[5] if row[5] is not None else "Isolation",
        "has_internal_transfer_team": bool(row[6]) if row[6] is not None else True,
        "has_external_transfer_partners": bool(row[7]) if row[7] is not None else True,
        "is_configured": bool(row[8]) if row[8] is not None else False,
    }


def save_hospital_config(hospital_id: int, config: Dict) -> None:
    initialize_auth_db()
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO hospital_configs (
            hospital_id, icu_beds, general_beds, special_beds, special_bed_label,
            has_internal_transfer_team, has_external_transfer_partners, is_configured
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(hospital_id) DO UPDATE SET
            icu_beds = excluded.icu_beds,
            general_beds = excluded.general_beds,
            special_beds = excluded.special_beds,
            special_bed_label = excluded.special_bed_label,
            has_internal_transfer_team = excluded.has_internal_transfer_team,
            has_external_transfer_partners = excluded.has_external_transfer_partners,
            is_configured = excluded.is_configured
        """,
        (
            hospital_id,
            int(config["icu_beds"]),
            int(config["general_beds"]),
            int(config["special_beds"]),
            config["special_bed_label"],
            int(bool(config["has_internal_transfer_team"])),
            int(bool(config["has_external_transfer_partners"])),
            int(bool(config["is_configured"])),
        ),
    )
    conn.commit()
    conn.close()


def save_hospital_state(hospital_id: int, state: Dict) -> None:
    initialize_auth_db()
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO hospital_states (
            hospital_id, free_icu, free_general, free_isolation, time_step, transfer_count,
            total_reward, total_waiting_time, waiting_patients_processed, correct_allocations,
            wrong_allocations, external_transfers, internal_transfers, critical_delays,
            total_patients_seen, waiting_queue_json, active_patients_json, completed_patients_json,
            transfer_log_json, utilization_samples_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(hospital_id) DO UPDATE SET
            free_icu = excluded.free_icu,
            free_general = excluded.free_general,
            free_isolation = excluded.free_isolation,
            time_step = excluded.time_step,
            transfer_count = excluded.transfer_count,
            total_reward = excluded.total_reward,
            total_waiting_time = excluded.total_waiting_time,
            waiting_patients_processed = excluded.waiting_patients_processed,
            correct_allocations = excluded.correct_allocations,
            wrong_allocations = excluded.wrong_allocations,
            external_transfers = excluded.external_transfers,
            internal_transfers = excluded.internal_transfers,
            critical_delays = excluded.critical_delays,
            total_patients_seen = excluded.total_patients_seen,
            waiting_queue_json = excluded.waiting_queue_json,
            active_patients_json = excluded.active_patients_json,
            completed_patients_json = excluded.completed_patients_json,
            transfer_log_json = excluded.transfer_log_json,
            utilization_samples_json = excluded.utilization_samples_json
        """,
        (
            hospital_id,
            int(state["free_icu"]),
            int(state["free_general"]),
            int(state["free_isolation"]),
            int(state["time_step"]),
            int(state["transfer_count"]),
            float(state["total_reward"]),
            int(state["total_waiting_time"]),
            int(state["waiting_patients_processed"]),
            int(state["correct_allocations"]),
            int(state["wrong_allocations"]),
            int(state["external_transfers"]),
            int(state["internal_transfers"]),
            int(state["critical_delays"]),
            int(state["total_patients_seen"]),
            json.dumps(state["waiting_queue"]),
            json.dumps(state["active_patients"]),
            json.dumps(state["completed_patients"]),
            json.dumps(state["transfer_log"]),
            json.dumps(state["utilization_samples"]),
        ),
    )
    conn.commit()
    conn.close()


def load_hospital_state(hospital_id: int) -> Optional[Dict]:
    initialize_auth_db()
    conn = sqlite3.connect(AUTH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT free_icu, free_general, free_isolation, time_step, transfer_count,
               total_reward, total_waiting_time, waiting_patients_processed, correct_allocations,
               wrong_allocations, external_transfers, internal_transfers, critical_delays,
               total_patients_seen, waiting_queue_json, active_patients_json, completed_patients_json,
               transfer_log_json, utilization_samples_json
        FROM hospital_states
        WHERE hospital_id = ?
        """,
        (hospital_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "free_icu": row[0],
        "free_general": row[1],
        "free_isolation": row[2],
        "time_step": row[3],
        "transfer_count": row[4],
        "total_reward": row[5],
        "total_waiting_time": row[6],
        "waiting_patients_processed": row[7],
        "correct_allocations": row[8],
        "wrong_allocations": row[9],
        "external_transfers": row[10],
        "internal_transfers": row[11],
        "critical_delays": row[12],
        "total_patients_seen": row[13],
        "waiting_queue": json.loads(row[14]),
        "active_patients": json.loads(row[15]),
        "completed_patients": json.loads(row[16]),
        "transfer_log": json.loads(row[17]),
        "utilization_samples": json.loads(row[18]),
    }
