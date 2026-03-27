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
