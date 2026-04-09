import os
import sqlite3
from typing import Dict, List

import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_DIR = os.path.join(BASE_DIR, "database")
HOSPITAL_DB_PATH = os.path.join(DB_DIR, "hospital_operations.db")


def initialize_hospital_db() -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            hospital_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            name TEXT,
            age INTEGER,
            gender TEXT,
            severity TEXT,
            required_bed TEXT,
            comorbidity INTEGER,
            waiting_time INTEGER,
            remaining_stay INTEGER,
            status TEXT,
            assigned_bed TEXT,
            PRIMARY KEY (hospital_id, patient_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS beds (
            hospital_id INTEGER NOT NULL,
            bed_no TEXT NOT NULL,
            ward_name TEXT NOT NULL,
            bed_type TEXT NOT NULL,
            status TEXT NOT NULL,
            PRIMARY KEY (hospital_id, bed_no)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transfers (
            transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            from_bed TEXT,
            to_bed TEXT,
            transfer_type TEXT NOT NULL,
            time_step INTEGER NOT NULL,
            UNIQUE(hospital_id, patient_id, transfer_type, time_step)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS allocations (
            allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            bed_no TEXT NOT NULL,
            bed_type TEXT NOT NULL,
            patient_status TEXT NOT NULL,
            time_step INTEGER NOT NULL,
            UNIQUE(hospital_id, patient_id, bed_no, patient_status, time_step)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transfer_partners (
            partner_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            partner_name TEXT NOT NULL,
            location TEXT,
            contact TEXT,
            max_daily_capacity INTEGER,
            UNIQUE(hospital_id, partner_name)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS external_transfers (
            transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            partner_name TEXT NOT NULL,
            time_step INTEGER NOT NULL,
            note TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rl_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            time_step INTEGER NOT NULL,
            recommended_action INTEGER,
            chosen_action INTEGER,
            feedback_label TEXT NOT NULL,
            feedback_score REAL,
            note TEXT,
            state_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    conn.close()


def _patient_rows(patients: List[Dict], hospital_id: int):
    rows = []
    for patient in patients:
        rows.append(
            (
                hospital_id,
                int(patient["id"]),
                patient.get("name", f"Patient {patient['id']}"),
                int(patient.get("age", 0) or 0),
                patient.get("gender", ""),
                patient.get("severity", ""),
                patient.get("required_bed", ""),
                int(patient.get("comorbidity", 0) or 0),
                int(patient.get("waiting_time", 0) or 0),
                int(patient.get("remaining_stay", 0) or 0),
                patient.get("status", ""),
                patient.get("assigned_bed", ""),
            )
        )
    return rows


def sync_hospital_snapshot(hospital_id: int, env, bed_inventory_df: pd.DataFrame) -> None:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM beds WHERE hospital_id = ?", (hospital_id,))
    bed_rows = [
        (
            hospital_id,
            row["Bed No"],
            row["Ward"],
            row["Bed Type"],
            row["Status"],
        )
        for _, row in bed_inventory_df.iterrows()
    ]
    cursor.executemany(
        """
        INSERT INTO beds (hospital_id, bed_no, ward_name, bed_type, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        bed_rows,
    )

    all_patients = env.waiting_queue + env.active_patients + env.completed_patients
    cursor.executemany(
        """
        INSERT OR REPLACE INTO patients (
            hospital_id, patient_id, name, age, gender, severity, required_bed,
            comorbidity, waiting_time, remaining_stay, status, assigned_bed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _patient_rows(all_patients, hospital_id),
    )

    transfer_rows = [
        (
            hospital_id,
            int(transfer.get("patient_id", 0)),
            transfer.get("from_bed"),
            transfer.get("to_bed"),
            transfer.get("transfer_type", ""),
            int(transfer.get("time_step", 0)),
        )
        for transfer in env.transfer_log
    ]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO transfers (
            hospital_id, patient_id, from_bed, to_bed, transfer_type, time_step
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        transfer_rows,
    )

    allocation_rows = [
        (
            hospital_id,
            int(patient["id"]),
            patient.get("assigned_bed_no", ""),
            patient.get("assigned_bed", ""),
            patient.get("status", ""),
            int(env.time_step),
        )
        for patient in (env.active_patients + env.completed_patients)
        if patient.get("assigned_bed_no")
    ]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO allocations (
            hospital_id, patient_id, bed_no, bed_type, patient_status, time_step
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        allocation_rows,
    )

    conn.commit()
    conn.close()


def fetch_hospital_patients(hospital_id: int) -> pd.DataFrame:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT patient_id AS [Patient ID], name AS [Name], age AS [Age], gender AS [Gender],
               severity AS [Severity], required_bed AS [Required Bed], waiting_time AS [Waiting Time],
               remaining_stay AS [Remaining Stay], status AS [Status], assigned_bed AS [Assigned Bed]
        FROM patients
        WHERE hospital_id = ?
        ORDER BY patient_id DESC
        """,
        conn,
        params=(hospital_id,),
    )
    conn.close()
    return df


def fetch_hospital_allocations(hospital_id: int) -> pd.DataFrame:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT allocation_id AS [Allocation ID], patient_id AS [Patient ID], bed_no AS [Bed No],
               bed_type AS [Bed Type], patient_status AS [Patient Status], time_step AS [Time Step]
        FROM allocations
        WHERE hospital_id = ?
        ORDER BY allocation_id DESC
        """,
        conn,
        params=(hospital_id,),
    )
    conn.close()
    return df


def fetch_transfer_partners(hospital_id: int) -> pd.DataFrame:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT partner_id AS [Partner ID],
               partner_name AS [Partner Name],
               location AS [Location],
               contact AS [Contact],
               max_daily_capacity AS [Daily Capacity]
        FROM transfer_partners
        WHERE hospital_id = ?
        ORDER BY partner_name
        """,
        conn,
        params=(hospital_id,),
    )
    conn.close()
    return df


def add_transfer_partner(
    hospital_id: int,
    partner_name: str,
    location: str = None,
    contact: str = None,
    max_daily_capacity: int = None,
) -> None:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO transfer_partners (
            hospital_id, partner_name, location, contact, max_daily_capacity
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(hospital_id),
            partner_name.strip(),
            location.strip() if location else None,
            contact.strip() if contact else None,
            int(max_daily_capacity) if max_daily_capacity is not None else None,
        ),
    )
    conn.commit()
    conn.close()


def remove_transfer_partner(hospital_id: int, partner_id: int) -> None:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM transfer_partners
        WHERE hospital_id = ? AND partner_id = ?
        """,
        (int(hospital_id), int(partner_id)),
    )
    conn.commit()
    conn.close()


def record_external_transfer(
    hospital_id: int,
    patient_id: int,
    partner_name: str,
    time_step: int,
    note: str = None,
) -> None:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO external_transfers (
            hospital_id, patient_id, partner_name, time_step, note
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(hospital_id),
            int(patient_id),
            partner_name.strip(),
            int(time_step),
            note.strip() if note else None,
        ),
    )
    conn.commit()
    conn.close()


def fetch_external_transfers(hospital_id: int) -> pd.DataFrame:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT transfer_id AS [Transfer ID],
               patient_id AS [Patient ID],
               partner_name AS [Partner Hospital],
               time_step AS [Time Step],
               note AS [Note],
               created_at AS [Logged At]
        FROM external_transfers
        WHERE hospital_id = ?
        ORDER BY transfer_id DESC
        """,
        conn,
        params=(hospital_id,),
    )
    conn.close()
    return df


def record_rl_feedback(
    hospital_id: int,
    patient_id: int,
    time_step: int,
    feedback_label: str,
    recommended_action: int = None,
    chosen_action: int = None,
    feedback_score: float = None,
    note: str = None,
    state_json: str = None,
) -> None:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO rl_feedback (
            hospital_id, patient_id, time_step, recommended_action, chosen_action,
            feedback_label, feedback_score, note, state_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            hospital_id,
            int(patient_id),
            int(time_step),
            recommended_action if recommended_action is None else int(recommended_action),
            chosen_action if chosen_action is None else int(chosen_action),
            feedback_label,
            feedback_score,
            note,
            state_json,
        ),
    )
    conn.commit()
    conn.close()


def fetch_rl_feedback(hospital_id: int) -> pd.DataFrame:
    initialize_hospital_db()
    conn = sqlite3.connect(HOSPITAL_DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT feedback_id AS [Feedback ID],
               patient_id AS [Patient ID],
               time_step AS [Time Step],
               recommended_action AS [Recommended Action],
               chosen_action AS [Chosen Action],
               feedback_label AS [Feedback],
               feedback_score AS [Score],
               note AS [Note],
               created_at AS [Logged At]
        FROM rl_feedback
        WHERE hospital_id = ?
        ORDER BY feedback_id DESC
        """,
        conn,
        params=(hospital_id,),
    )
    conn.close()
    return df
