import os
import sys
import inspect
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from baseline.rule_based import rule_based_action
from env.hospital_env import HospitalEnv
from evaluation.evaluate import compare_policies
from models.dqn_model import DQN
import utils.auth as auth_store
from utils.helpers import ACTION_LABELS, format_metrics
from utils.hospital_db import (
    fetch_hospital_allocations,
    fetch_hospital_patients,
    initialize_hospital_db,
    sync_hospital_snapshot,
)
from utils.stay_predictor import StayDurationPredictor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "dqn_hospital_best.pth")


LIGHT_THEME = {
    "background": "#eef4f7",
    "panel": "#ffffff",
    "text": "#142133",
    "muted": "#617388",
    "accent": "#0f766e",
    "accent_soft": "#dff7f4",
    "accent_alt": "#ef6c3e",
    "accent_alt_soft": "#fff0e8",
    "border": "#d4e0e8",
    "shadow": "0 16px 40px rgba(19, 38, 56, 0.10)",
}

DARK_THEME = {
    "background": "#0c1720",
    "panel": "#152430",
    "text": "#eef6ff",
    "muted": "#a9bccd",
    "accent": "#49c5b6",
    "accent_soft": "#17343b",
    "accent_alt": "#ff9a6b",
    "accent_alt_soft": "#3a261f",
    "border": "#243846",
    "shadow": "0 18px 42px rgba(0, 0, 0, 0.28)",
}


@st.cache_resource
def load_stay_predictor() -> StayDurationPredictor:
    return StayDurationPredictor()


@st.cache_resource
def load_dqn_model():
    if not os.path.exists(MODEL_PATH):
        return None, "No trained DQN checkpoint found yet."

    env = HospitalEnv(max_steps=100)
    model = DQN(env.state_size, env.action_size).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError:
        return None, "Checkpoint exists but is incompatible with the current environment. Retrain the model."

    model.eval()
    return model, None


def get_theme_mode() -> str:
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"
    return st.session_state.theme_mode


def apply_theme(mode: str) -> None:
    theme = DARK_THEME if mode == "Dark" else LIGHT_THEME
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top left, {theme["accent_soft"]}, {theme["background"]} 28%);
            color: {theme["text"]};
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }}
        .stApp > header {{
            background: {theme["background"]};
        }}
        .block-container {{
            background: {theme["background"]};
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }}
        [data-testid="stSidebar"] {{
            background: {theme["panel"]};
            border-right: 1px solid {theme["border"]};
            box-shadow: {theme["shadow"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {theme["text"]} !important;
        }}
        .hero-card {{
            background: linear-gradient(145deg, {theme["panel"]}, {theme["accent_soft"]});
            border: 1px solid {theme["border"]};
            border-radius: 24px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: {theme["shadow"]};
        }}
        .hero-kicker {{
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: {theme["accent_soft"]};
            color: {theme["accent"]} !important;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .hero-title {{
            margin: 0.8rem 0 0.35rem 0;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 800;
            color: {theme["text"]} !important;
        }}
        .hero-subtitle {{
            margin: 0;
            color: {theme["muted"]} !important;
            font-size: 1rem;
        }}
        .section-card {{
            background: linear-gradient(180deg, {theme["panel"]}, {theme["accent_soft"]});
            border: 1px solid {theme["border"]};
            border-radius: 22px;
            padding: 1rem 1.15rem 1.1rem 1.15rem;
            margin: 0.75rem 0 1rem 0;
            box-shadow: {theme["shadow"]};
        }}
        .section-title {{
            margin: 0 0 0.2rem 0;
            font-size: 1.15rem;
            font-weight: 800;
            color: {theme["text"]} !important;
        }}
        .section-copy {{
            margin: 0;
            color: {theme["muted"]} !important;
            font-size: 0.95rem;
        }}
        .sidebar-card {{
            background: {theme["panel"]};
            border: 1px solid {theme["border"]};
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin: 0.5rem 0 0.9rem 0;
            box-shadow: {theme["shadow"]};
        }}
        .sidebar-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {theme["muted"]} !important;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }}
        .sidebar-value {{
            font-size: 1rem;
            font-weight: 700;
            color: {theme["text"]} !important;
            margin: 0;
        }}
        .sidebar-chip {{
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: {theme["accent_soft"]};
            color: {theme["accent"]} !important;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.35rem;
            margin-top: 0.25rem;
        }}
        .status-ribbon {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            background: {theme["panel"]};
            border: 1px solid {theme["border"]};
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin: 0.7rem 0 1rem 0;
            box-shadow: {theme["shadow"]};
        }}
        .status-ribbon-copy {{
            color: {theme["text"]} !important;
            font-weight: 700;
        }}
        .severity-badge {{
            display: inline-block;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-right: 0.4rem;
        }}
        .severity-critical {{
            background: #ffe2e0;
            color: #9f1d1d !important;
        }}
        .severity-moderate {{
            background: #fff0cf;
            color: #8a5a00 !important;
        }}
        .severity-stable {{
            background: #dff5e8;
            color: #156f42 !important;
        }}
        .decision-card {{
            background: linear-gradient(145deg, {theme["panel"]}, {theme["accent_soft"]});
            border: 1px solid {theme["border"]};
            border-left: 6px solid {theme["accent"]};
            border-radius: 22px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1rem 0;
            box-shadow: {theme["shadow"]};
        }}
        .decision-label {{
            color: {theme["muted"]} !important;
            font-size: 0.82rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 0.04em;
        }}
        .decision-value {{
            color: {theme["text"]} !important;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0.25rem 0;
        }}
        .decision-copy {{
            color: {theme["muted"]} !important;
            margin: 0;
        }}
        .tone-chip {{
            display: inline-block;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.6rem;
        }}
        .tone-urgent {{
            background: #ffe0e0;
            color: #a61b1b !important;
        }}
        .tone-allocate {{
            background: #ddf6ef;
            color: #0f766e !important;
        }}
        .tone-transfer {{
            background: #fff0dc;
            color: #9a5a00 !important;
        }}
        .tone-escalate {{
            background: #f2e8ff;
            color: #6f42c1 !important;
        }}
        .tone-wait {{
            background: #e8edf3;
            color: #4d6072 !important;
        }}
        .action-summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.8rem 0 1rem 0;
        }}
        .action-summary-card {{
            background: {theme["panel"]};
            border: 1px solid {theme["border"]};
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: {theme["shadow"]};
        }}
        .action-summary-label {{
            color: {theme["muted"]} !important;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .action-summary-value {{
            color: {theme["text"]} !important;
            font-size: 1.1rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }}
        div[data-testid="stMetric"] {{
            background: {theme["panel"]};
            border: 1px solid {theme["border"]};
            border-radius: 18px;
            padding: 0.85rem;
            box-shadow: {theme["shadow"]};
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"],
        div[data-testid="stMetric"] p,
        div[data-testid="stMetric"] div {{
            color: {theme["text"]} !important;
            opacity: 1 !important;
        }}
        div[data-testid="stForm"] {{
            background: {theme["panel"]};
            border: 1px solid {theme["border"]};
            border-radius: 22px;
            padding: 1rem;
            box-shadow: {theme["shadow"]};
        }}
        div[data-testid="stForm"] * {{
            background: transparent;
            color: {theme["text"]} !important;
        }}
        div.stButton > button, .stDownloadButton > button {{
            background: {theme["accent"]};
            color: white;
            border-radius: 12px;
            border: 1px solid transparent;
            font-weight: 700;
            min-height: 2.85rem;
            padding: 0.55rem 1rem;
            box-shadow: 0 10px 20px rgba(15, 118, 110, 0.20);
        }}
        div.stButton > button:hover {{
            opacity: 0.98;
            transform: translateY(-1px);
            box-shadow: 0 14px 24px rgba(15, 118, 110, 0.24);
        }}
        div.stButton > button[kind="secondary"] {{
            background: {theme["panel"]} !important;
            color: {theme["text"]} !important;
            border: 1px solid {theme["border"]} !important;
            box-shadow: {theme["shadow"]};
        }}
        div.stButton > button[kind="secondary"]:hover {{
            border-color: {theme["accent"]} !important;
            color: {theme["accent"]} !important;
        }}
        div[data-testid="stFormSubmitButton"] > button {{
            background: linear-gradient(135deg, {theme["accent"]}, {theme["accent_alt"]}) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 12px 24px rgba(15, 118, 110, 0.22);
        }}
        .stAlert {{
            border-radius: 18px;
            box-shadow: {theme["shadow"]};
        }}
        p, span, label, .stMarkdown, .stCaption, .stTextInput label, h1, h2, h3 {{
            color: {theme["text"]} !important;
        }}
        input, textarea, [data-baseweb="input"], [data-baseweb="base-input"] {{
            background: {theme["panel"]} !important;
            color: {theme["text"]} !important;
            border-color: {theme["border"]} !important;
            border-radius: 12px !important;
        }}
        [data-baseweb="select"] > div,
        [data-baseweb="select"] span,
        [data-baseweb="select"] input {{
            background: {theme["panel"]} !important;
            color: {theme["text"]} !important;
            border-radius: 12px !important;
        }}
        [data-testid="stSelectbox"] div,
        [data-testid="stNumberInput"] div,
        [data-testid="stTextInput"] div,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {{
            color: {theme["text"]} !important;
        }}
        button[role="tab"] {{
            color: {theme["text"]} !important;
            background: {theme["panel"]} !important;
            border-radius: 12px 12px 0 0;
            border: 1px solid {theme["border"]} !important;
        }}
        button[role="tab"][aria-selected="true"] {{
            border-bottom: 2px solid {theme["accent"]} !important;
            background: {theme["accent_soft"]} !important;
        }}
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stToggle"] label {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stSidebar"] button {{
            width: 100%;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            margin-bottom: 0.35rem;
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            background: {theme["panel"]};
            border-radius: 12px;
            box-shadow: {theme["shadow"]};
        }}
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] * {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stToolbar"] {{
            right: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">Hospital Operations Console</div>
            <div class="hero-title">{title}</div>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_card(label: str, value: str, chips: Optional[list[str]] = None) -> None:
    chip_markup = ""
    if chips:
        chip_markup = "".join(f'<span class="sidebar-chip">{chip}</span>' for chip in chips)
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">{label}</div>
            <p class="sidebar-value">{value}</p>
            {chip_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_ribbon(message: str, chips: Optional[list[str]] = None) -> None:
    chip_markup = ""
    if chips:
        chip_markup = "".join(f'<span class="sidebar-chip">{chip}</span>' for chip in chips)
    st.markdown(
        f"""
        <div class="status-ribbon">
            <div class="status-ribbon-copy">{message}</div>
            <div>{chip_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(severity: str) -> str:
    severity_value = (severity or "stable").lower()
    css_class = {
        "critical": "severity-critical",
        "moderate": "severity-moderate",
        "stable": "severity-stable",
    }.get(severity_value, "severity-stable")
    return f'<span class="severity-badge {css_class}">{severity_value}</span>'


def render_decision_card(title: str, value: str, copy: str) -> None:
    tone = action_tone(value)
    tone_class = {
        "Urgent": "tone-urgent",
        "Allocate": "tone-allocate",
        "Transfer": "tone-transfer",
        "Escalate": "tone-escalate",
        "Wait": "tone-wait",
    }[tone]
    st.markdown(
        f"""
        <div class="decision-card">
            <span class="tone-chip {tone_class}">{tone}</span>
            <div class="decision-label">{title}</div>
            <div class="decision-value">{value}</div>
            <p class="decision-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_summary(rule_action: str, dqn_action: str, stay_text: str) -> None:
    st.markdown(
        f"""
        <div class="action-summary-grid">
            <div class="action-summary-card">
                <div class="action-summary-label">Rule-Based</div>
                <div class="action-summary-value">{rule_action}</div>
            </div>
            <div class="action-summary-card">
                <div class="action-summary-label">DQN Recommendation</div>
                <div class="action-summary-value">{dqn_action}</div>
            </div>
            <div class="action-summary-card">
                <div class="action-summary-label">Predicted Stay</div>
                <div class="action-summary-value">{stay_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_tone(action_label: str) -> str:
    if "ICU" in action_label:
        return "Urgent"
    if "General" in action_label or "Isolation" in action_label or "Special" in action_label:
        return "Allocate"
    if "Internal Transfer" in action_label:
        return "Transfer"
    if "External Transfer" in action_label:
        return "Escalate"
    return "Wait"


def default_hospital_config() -> Dict:
    auth_hospital = st.session_state.get("authenticated_hospital", {})
    return {
        "hospital_name": auth_hospital.get("hospital_name", "City Care Hospital"),
        "location": auth_hospital.get("location", "Chennai"),
        "icu_beds": 2,
        "general_beds": 4,
        "special_beds": 1,
        "special_bed_label": "Isolation",
        "has_internal_transfer_team": True,
        "has_external_transfer_partners": True,
        "is_configured": False,
    }


def get_hospital_config_state() -> Dict:
    if "hospital_config" not in st.session_state:
        st.session_state.hospital_config = default_hospital_config()
    return st.session_state.hospital_config


def build_env_from_config(config: Dict) -> HospitalEnv:
    init_params = inspect.signature(HospitalEnv.__init__).parameters
    if "total_icu_beds" in init_params and "auto_generate_patients" in init_params:
        return HospitalEnv(
            max_steps=100,
            total_icu_beds=int(config["icu_beds"]),
            total_general_beds=int(config["general_beds"]),
            total_isolation_beds=int(config["special_beds"]),
            hospital_name=config["hospital_name"],
            special_bed_label=config["special_bed_label"],
            auto_generate_patients=False,
        )
    if "total_icu_beds" in init_params:
        env = HospitalEnv(
            max_steps=100,
            total_icu_beds=int(config["icu_beds"]),
            total_general_beds=int(config["general_beds"]),
            total_isolation_beds=int(config["special_beds"]),
            hospital_name=config["hospital_name"],
            special_bed_label=config["special_bed_label"],
        )
        env.auto_generate_patients = False
        env.reset()
        return env

    env = HospitalEnv(max_steps=100)
    env.total_icu_beds = int(config["icu_beds"])
    env.total_general_beds = int(config["general_beds"])
    env.total_isolation_beds = int(config["special_beds"])
    env.hospital_name = config["hospital_name"]
    env.special_bed_label = config["special_bed_label"]
    env.auto_generate_patients = False
    env.reset()
    return env


def export_env_state(env: HospitalEnv) -> Dict:
    return {
        "free_icu": env.free_icu,
        "free_general": env.free_general,
        "free_isolation": env.free_isolation,
        "time_step": env.time_step,
        "transfer_count": env.transfer_count,
        "total_reward": env.total_reward,
        "total_waiting_time": env.total_waiting_time,
        "waiting_patients_processed": env.waiting_patients_processed,
        "correct_allocations": env.correct_allocations,
        "wrong_allocations": env.wrong_allocations,
        "external_transfers": env.external_transfers,
        "internal_transfers": env.internal_transfers,
        "critical_delays": env.critical_delays,
        "total_patients_seen": env.total_patients_seen,
        "waiting_queue": env.waiting_queue,
        "active_patients": env.active_patients,
        "completed_patients": env.completed_patients,
        "transfer_log": env.transfer_log,
        "utilization_samples": env.utilization_samples,
    }


def normalize_patient_record(patient: Dict) -> Dict:
    normalized = dict(patient)
    normalized.setdefault("id", 0)
    normalized.setdefault("name", f"Patient {normalized['id']}")
    normalized.setdefault("age", 0)
    normalized.setdefault("gender", "")
    normalized.setdefault("severity", "stable")
    normalized.setdefault("required_bed", "General")
    normalized.setdefault("comorbidity", 0)
    normalized.setdefault("waiting_time", 0)
    normalized.setdefault("remaining_stay", 0)
    normalized.setdefault("status", "waiting")
    normalized.setdefault("assigned_bed", "")
    normalized.setdefault("assigned_bed_no", "")
    return normalized


def rebuild_bed_state(env: HospitalEnv) -> None:
    if hasattr(env, "_initialize_bed_slots"):
        env.bed_slots = env._initialize_bed_slots()

    env.free_icu = env.total_icu_beds
    env.free_general = env.total_general_beds
    env.free_isolation = env.total_isolation_beds

    for patient in env.active_patients:
        bed_type = patient.get("assigned_bed", "")
        bed_no = patient.get("assigned_bed_no", "")
        if not bed_type:
            continue

        if hasattr(env, "bed_slots") and bed_type in env.bed_slots:
            matched_slot = None
            if bed_no:
                matched_slot = next(
                    (slot for slot in env.bed_slots[bed_type] if slot["bed_no"] == bed_no),
                    None,
                )
            if matched_slot is None:
                matched_slot = next(
                    (slot for slot in env.bed_slots[bed_type] if not slot["occupied"]),
                    None,
                )
                if matched_slot and not bed_no:
                    patient["assigned_bed_no"] = matched_slot["bed_no"]
            if matched_slot:
                matched_slot["occupied"] = True
                matched_slot["patient_id"] = patient.get("id")

        if bed_type == "ICU":
            env.free_icu = max(0, env.free_icu - 1)
        elif bed_type == "General":
            env.free_general = max(0, env.free_general - 1)
        elif bed_type == "Isolation":
            env.free_isolation = max(0, env.free_isolation - 1)


def restore_env_state(env: HospitalEnv, state: Dict) -> HospitalEnv:
    env.time_step = state["time_step"]
    env.transfer_count = state["transfer_count"]
    env.total_reward = state["total_reward"]
    env.total_waiting_time = state["total_waiting_time"]
    env.waiting_patients_processed = state["waiting_patients_processed"]
    env.correct_allocations = state["correct_allocations"]
    env.wrong_allocations = state["wrong_allocations"]
    env.external_transfers = state["external_transfers"]
    env.internal_transfers = state["internal_transfers"]
    env.critical_delays = state["critical_delays"]
    env.total_patients_seen = state["total_patients_seen"]
    env.waiting_queue = [normalize_patient_record(patient) for patient in state["waiting_queue"]]
    env.active_patients = [normalize_patient_record(patient) for patient in state["active_patients"]]
    env.completed_patients = [normalize_patient_record(patient) for patient in state["completed_patients"]]
    env.transfer_log = state["transfer_log"]
    rebuild_bed_state(env)
    env.utilization_samples = state["utilization_samples"] or [env._occupancy_rate()]
    env.current_patient = env.waiting_queue[0] if env.waiting_queue else None
    return env


def persist_dashboard_state() -> None:
    hospital = st.session_state.get("authenticated_hospital")
    env = st.session_state.get("dashboard_env")
    if not hospital or env is None:
        return
    auth_store.save_hospital_state(hospital["hospital_id"], export_env_state(env))
    sync_hospital_snapshot(hospital["hospital_id"], env, generate_bed_inventory(env))


def get_env() -> HospitalEnv:
    config = get_hospital_config_state()
    if "dashboard_env" not in st.session_state:
        env = build_env_from_config(config)
        hospital = st.session_state.get("authenticated_hospital")
        if hospital:
            saved_state = auth_store.load_hospital_state(hospital["hospital_id"])
            if saved_state:
                env = restore_env_state(env, saved_state)
        st.session_state.dashboard_env = env
    return st.session_state.dashboard_env


def reset_env() -> None:
    st.session_state.dashboard_env = build_env_from_config(get_hospital_config_state())
    persist_dashboard_state()


def build_patient_df(patients):
    if not patients:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "ID": patient.get("id", 0),
                "Name": patient.get("name", f"Patient {patient.get('id', 0)}"),
                "Age": patient.get("age", ""),
                "Gender": patient.get("gender", ""),
                "Severity": patient.get("severity", ""),
                "Required Bed": patient.get("required_bed", ""),
                "Waiting Time": patient.get("waiting_time", 0),
                "Stay Left": patient.get("remaining_stay", 0),
                "Assigned Bed No": patient.get("assigned_bed_no", ""),
                "Status": patient.get("status", ""),
            }
            for patient in patients
        ]
    )


def style_patient_dataframe(df: pd.DataFrame):
    if df.empty:
        return df

    def severity_style(value):
        palette = {
            "critical": "background-color: #ffe0e0; color: #9f1d1d; font-weight: 700;",
            "moderate": "background-color: #fff0cf; color: #8a5a00; font-weight: 700;",
            "stable": "background-color: #dff5e8; color: #156f42; font-weight: 700;",
        }
        return palette.get(str(value).lower(), "")

    def status_style(value):
        palette = {
            "waiting": "background-color: #e8edf3; color: #4d6072; font-weight: 700;",
            "admitted": "background-color: #ddf6ef; color: #0f766e; font-weight: 700;",
            "discharged": "background-color: #f1f5f9; color: #475569; font-weight: 700;",
            "transferred_external": "background-color: #f2e8ff; color: #6f42c1; font-weight: 700;",
        }
        return palette.get(str(value).lower(), "")

    styled = df.style
    if "Severity" in df.columns:
        styled = styled.map(severity_style, subset=["Severity"])
    if "Status" in df.columns:
        styled = styled.map(status_style, subset=["Status"])
    return styled


def style_bed_inventory_dataframe(df: pd.DataFrame):
    if df.empty:
        return df

    def bed_status_style(value):
        palette = {
            "free": "background-color: #ddf6ef; color: #0f766e; font-weight: 700;",
            "occupied": "background-color: #ffe0e0; color: #9f1d1d; font-weight: 700;",
        }
        return palette.get(str(value).lower(), "")

    styled = df.style
    if "Status" in df.columns:
        styled = styled.map(bed_status_style, subset=["Status"])
    return styled


def generate_bed_inventory(env: HospitalEnv) -> pd.DataFrame:
    if hasattr(env, "bed_slots"):
        rows = []
        for bed_type, slots in env.bed_slots.items():
            ward_name = f"{env.special_bed_label if bed_type == 'Isolation' else bed_type} Ward"
            label = env.special_bed_label if bed_type == "Isolation" else bed_type
            for slot in slots:
                rows.append(
                    {
                        "Bed No": slot["bed_no"],
                        "Ward": ward_name,
                        "Bed Type": label,
                        "Status": "Occupied" if slot["occupied"] else "Free",
                        "Patient ID": slot["patient_id"] or "",
                    }
                )
        return pd.DataFrame(rows)

    rows = []
    bed_groups = [
        ("ICU", env.total_icu_beds, env.free_icu),
        ("General", env.total_general_beds, env.free_general),
        (env.special_bed_label, env.total_isolation_beds, env.free_isolation),
    ]

    for bed_type, total_count, free_count in bed_groups:
        occupied_count = total_count - free_count
        for index in range(1, total_count + 1):
            status = "Occupied" if index <= occupied_count else "Free"
            rows.append(
                {
                    "Bed No": f"{bed_type[:3].upper()}-{index:02d}",
                    "Ward": f"{bed_type} Ward",
                    "Bed Type": bed_type,
                    "Status": status,
                }
            )

    return pd.DataFrame(rows)


def build_status_snapshot(env: HospitalEnv) -> Dict:
    waiting_critical = sum(1 for patient in env.waiting_queue if patient.get("severity") == "critical")
    waiting_moderate = sum(1 for patient in env.waiting_queue if patient.get("severity") == "moderate")
    waiting_stable = sum(1 for patient in env.waiting_queue if patient.get("severity") == "stable")
    total_beds = env.total_icu_beds + env.total_general_beds + env.total_isolation_beds
    free_beds = env.free_icu + env.free_general + env.free_isolation
    occupancy_rate = (total_beds - free_beds) / total_beds if total_beds else 0.0

    return {
        "free_icu": env.free_icu,
        "free_general": env.free_general,
        "free_special": env.free_isolation,
        "occupancy_rate": occupancy_rate,
        "queue_length": len(env.waiting_queue),
        "waiting_critical": waiting_critical,
        "waiting_moderate": waiting_moderate,
        "waiting_stable": waiting_stable,
        "active_patients": len(env.active_patients),
        "completed_patients": len(env.completed_patients),
        "transfers": getattr(env, "transfer_count", 0),
        "current_patient": getattr(env, "current_patient", None),
    }


def get_next_discharge_info(env: HospitalEnv, bed_type: str) -> Optional[Dict]:
    candidates = [
        patient
        for patient in env.active_patients
        if patient.get("assigned_bed") == bed_type
    ]
    if not candidates:
        return None

    next_patient = min(candidates, key=lambda patient: patient.get("remaining_stay", 9999))
    return {
        "patient_name": next_patient.get("name", f"Patient {next_patient['id']}"),
        "remaining_stay": next_patient.get("remaining_stay", 0),
        "severity": next_patient.get("severity", ""),
        "bed_number": next_patient.get("assigned_bed_no", ""),
    }


def get_capacity_guidance(env: HospitalEnv, patient: Dict) -> Dict[str, str]:
    patient = normalize_patient_record(patient)
    required_bed = patient["required_bed"]
    free_now = (
        (required_bed == "ICU" and env.free_icu > 0)
        or (required_bed == "General" and env.free_general > 0)
        or (required_bed == "Isolation" and env.free_isolation > 0)
    )
    next_discharge = get_next_discharge_info(env, required_bed)
    transfer_candidate = env._find_transfer_candidate(required_bed) if hasattr(env, "_find_transfer_candidate") else None

    if free_now:
        free_beds = available_bed_numbers(env, required_bed)
        return {
            "status": "available_now",
            "message": f"{required_bed} bed is available now. Immediate assignment is possible. Available bed numbers: {', '.join(free_beds)}.",
        }

    if next_discharge and next_discharge["remaining_stay"] <= 2:
        return {
            "status": "wait_for_discharge",
            "message": (
                f"No {required_bed} bed is free right now, but {next_discharge['patient_name']} is expected "
                f"to discharge from {required_bed} in about {next_discharge['remaining_stay']} step(s)."
            ),
        }

    if transfer_candidate:
        return {
            "status": "internal_transfer_possible",
            "message": "No direct bed is free, but an internal transfer may create capacity inside the hospital.",
        }

    return {
        "status": "external_transfer_risk",
        "message": f"No {required_bed} bed is currently available and no near discharge is visible.",
    }


def available_bed_numbers(env: HospitalEnv, bed_type: str):
    if hasattr(env, "bed_slots"):
        return [slot["bed_no"] for slot in env.bed_slots[bed_type] if not slot["occupied"]]

    if bed_type == "ICU":
        total_count = env.total_icu_beds
        free_count = env.free_icu
        prefix = "ICU"
    elif bed_type == "General":
        total_count = env.total_general_beds
        free_count = env.free_general
        prefix = "GEN"
    else:
        total_count = env.total_isolation_beds
        free_count = env.free_isolation
        prefix = env.special_bed_label[:3].upper()

    occupied_count = total_count - free_count
    return [f"{prefix}-{index:02d}" for index in range(occupied_count + 1, total_count + 1)]


def recommended_queue_action(env: HospitalEnv, patient: Dict) -> str:
    guidance = get_capacity_guidance(env, patient)
    if guidance["status"] == "available_now":
        return "Assign Bed"
    if guidance["status"] == "wait_for_discharge":
        return "Wait for Discharge"
    if guidance["status"] == "internal_transfer_possible":
        return "Internal Transfer"
    return "External Transfer"


def create_manual_patient(
    env: HospitalEnv,
    name: str,
    age: int,
    gender: str,
    severity: str,
    required_bed: str,
    comorbidity: int,
    waiting_time: int,
    predicted_stay: int,
) -> Dict:
    if hasattr(env, "create_patient"):
        return env.create_patient(
            name=name,
            age=age,
            gender=gender,
            severity=severity,
            required_bed=required_bed,
            comorbidity=comorbidity,
            waiting_time=waiting_time,
            predicted_stay=predicted_stay,
        )

    env.total_patients_seen += 1
    return {
        "id": env.total_patients_seen,
        "name": name,
        "age": age,
        "gender": gender,
        "severity": severity,
        "required_bed": required_bed,
        "comorbidity": int(comorbidity),
        "waiting_time": waiting_time,
        "remaining_stay": predicted_stay,
        "status": "waiting",
    }


def push_patient_to_queue(env: HospitalEnv, patient: Dict) -> None:
    if hasattr(env, "add_patient_to_queue"):
        env.add_patient_to_queue(patient, prioritize=True)
        return

    env.waiting_queue.insert(0, patient)
    env.current_patient = env.waiting_queue[0]


def admit_patient_immediately(env: HospitalEnv, patient: Dict) -> bool:
    if not hasattr(env, "_allocate_bed"):
        return False

    if patient["required_bed"] == "ICU":
        return env._allocate_bed("ICU", patient)
    if patient["required_bed"] == "General":
        return env._allocate_bed("General", patient)
    return env._allocate_bed("Isolation", patient)


def auto_assign_waiting_patients(env: HospitalEnv, max_patients: Optional[int] = None) -> Dict[str, int]:
    processed = 0
    assigned = 0
    waited = 0
    internal = 0
    external = 0

    while env.waiting_queue and (max_patients is None or processed < max_patients):
        action = rule_based_action(env)
        env.step(action)
        processed += 1

        if action in {0, 1, 2}:
            assigned += 1
        elif action == 3:
            waited += 1
        elif action == 4:
            internal += 1
        elif action == 5:
            external += 1

        if action == 3 and env.current_patient and get_capacity_guidance(env, env.current_patient)["status"] != "available_now":
            break

    return {
        "processed": processed,
        "assigned": assigned,
        "waited": waited,
        "internal": internal,
        "external": external,
    }


def get_dqn_action(env: HospitalEnv, model: Optional[DQN]) -> Optional[int]:
    if model is None:
        return None
    state_tensor = torch.FloatTensor(env._get_state()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.argmax(model(state_tensor), dim=1).item()


def explain_recommendation(env: HospitalEnv, action: int) -> str:
    patient = env.current_patient
    if patient is None:
        return "No patient is waiting for a decision right now."

    patient = normalize_patient_record(patient)
    required_bed = patient["required_bed"]
    severity = patient["severity"]
    special_label = env.special_bed_label
    guidance = get_capacity_guidance(env, patient)

    if action == 0:
        return (
            f"Immediate ICU allocation is recommended because the patient is {severity} "
            f"and the required bed type is {required_bed}. {guidance['message']}"
        )
    if action == 1:
        return (
            "General bed allocation is recommended because the patient can be safely admitted "
            f"without occupying ICU capacity. {guidance['message']}"
        )
    if action == 2:
        return (
            f"{special_label} bed allocation is recommended because the patient requires "
            f"{special_label.lower()} or special handling. {guidance['message']}"
        )
    if action == 3:
        return f"Waiting is recommended. {guidance['message']}"
    if action == 4:
        return "Internal transfer is recommended to free an appropriate bed for the incoming patient while keeping care inside the hospital."
    if action == 5:
        return f"External transfer is recommended because the required bed cannot be safely provided inside the hospital at this time. {guidance['message']}"
    return "No recommendation explanation is available."


def render_hospital_setup() -> None:
    render_section_intro(
        "Hospital Setup",
        "Register hospital capacity, ward structure, and transfer capabilities before operational use.",
    )
    config = get_hospital_config_state()

    if config.get("is_configured") and not st.session_state.get("edit_hospital_config", False):
        summary_df = pd.DataFrame(
            [
                {"Field": "Hospital Name", "Value": config["hospital_name"]},
                {"Field": "Location", "Value": config["location"]},
                {"Field": "ICU Beds", "Value": config["icu_beds"]},
                {"Field": "General Beds", "Value": config["general_beds"]},
                {"Field": "Special Beds", "Value": config["special_beds"]},
                {"Field": "Special Bed Type", "Value": config["special_bed_label"]},
                {
                    "Field": "Internal Transfer Team",
                    "Value": "Available" if config["has_internal_transfer_team"] else "Not Available",
                },
                {
                    "Field": "External Transfer Partners",
                    "Value": "Available" if config["has_external_transfer_partners"] else "Not Available",
                },
            ]
        )
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.success("Hospital environment is already configured and ready for daily operations.")
        if st.button("Edit Hospital Configuration"):
            st.session_state.edit_hospital_config = True
            st.rerun()
        return

    with st.form("hospital_setup_form"):
        col1, col2 = st.columns(2)
        hospital_name = col1.text_input("Hospital Name", value=config["hospital_name"])
        location = col2.text_input("Location", value=config["location"])

        col3, col4, col5 = st.columns(3)
        icu_beds = col3.number_input("ICU Beds", min_value=1, max_value=100, value=int(config["icu_beds"]))
        general_beds = col4.number_input("General Beds", min_value=1, max_value=300, value=int(config["general_beds"]))
        special_beds = col5.number_input("Special Beds", min_value=1, max_value=100, value=int(config["special_beds"]))

        col6, col7, col8 = st.columns(3)
        special_bed_label = col6.selectbox("Special Bed Type", ["Isolation", "Special Care", "Emergency Overflow"])
        internal_team = col7.selectbox("Internal Transfer Team", ["Available", "Not Available"])
        external_partners = col8.selectbox("External Transfer Partners", ["Available", "Not Available"])

        submitted = st.form_submit_button("Register / Update Hospital")
        if submitted:
            st.session_state.hospital_config = {
                "hospital_name": hospital_name,
                "location": location,
                "icu_beds": int(icu_beds),
                "general_beds": int(general_beds),
                "special_beds": int(special_beds),
                "special_bed_label": special_bed_label,
                "has_internal_transfer_team": internal_team == "Available",
                "has_external_transfer_partners": external_partners == "Available",
                "is_configured": True,
            }
            authenticated_hospital = st.session_state.get("authenticated_hospital")
            if authenticated_hospital:
                auth_store.save_hospital_config(authenticated_hospital["hospital_id"], st.session_state.hospital_config)
            st.session_state.edit_hospital_config = False
            reset_env()
            st.success("Hospital environment registered and simulation reset with the new configuration.")

    if config.get("is_configured"):
        if st.button("Cancel Editing"):
            st.session_state.edit_hospital_config = False
            st.rerun()

    st.caption(
        "Step 1: The hospital registers its bed capacity and transfer capabilities once. "
        "Later, the same environment is reused for daily operations unless an admin updates it."
    )


def render_login() -> bool:
    auth_store.initialize_auth_db()
    initialize_hospital_db()
    render_section_intro(
        "Hospital Login",
        "Only authenticated hospitals can access their own dashboard and operational data.",
    )

    if st.session_state.get("authenticated_hospital"):
        return True

    with st.form("hospital_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            hospital = auth_store.authenticate_hospital(username, password)
            if hospital is None:
                st.error("Invalid credentials. Use a registered hospital account.")
            else:
                st.session_state["authenticated_hospital"] = hospital
                saved_config = auth_store.get_hospital_config(hospital["hospital_id"])
                base_config = default_hospital_config()
                st.session_state["hospital_config"] = {**base_config, **(saved_config or {})}
                st.session_state.pop("dashboard_env", None)
                st.rerun()

    st.info("Demo accounts: `citycare / citycare123` or `sunrise / sunrise123`")
    return False


def logout_hospital() -> None:
    persist_dashboard_state()
    st.session_state.pop("authenticated_hospital", None)
    st.session_state.pop("hospital_config", None)
    st.session_state.pop("dashboard_env", None)
    st.rerun()


def render_status(env: HospitalEnv) -> None:
    snapshot = build_status_snapshot(env)

    render_section_intro(
        "Live Hospital Status",
        "Track current capacity, active admissions, critical queue load, and immediate bed availability.",
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ICU Beds", f"{snapshot['free_icu']} free / {env.total_icu_beds}")
    col2.metric("General Beds", f"{snapshot['free_general']} free / {env.total_general_beds}")
    col3.metric(f"{env.special_bed_label} Beds", f"{snapshot['free_special']} free / {env.total_isolation_beds}")
    col4.metric("Occupancy Rate", f"{snapshot['occupancy_rate']:.0%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Waiting Queue", snapshot["queue_length"])
    col6.metric("Critical Waiting", snapshot["waiting_critical"])
    col7.metric("Active Admissions", snapshot["active_patients"])
    col8.metric("Transfers", snapshot["transfers"])

    bed_col1, bed_col2, bed_col3 = st.columns(3)
    bed_col1.caption(f"Free ICU Beds: {', '.join(available_bed_numbers(env, 'ICU')) or 'None'}")
    bed_col2.caption(f"Free General Beds: {', '.join(available_bed_numbers(env, 'General')) or 'None'}")
    bed_col3.caption(
        f"Free {env.special_bed_label} Beds: {', '.join(available_bed_numbers(env, 'Isolation')) or 'None'}"
    )

    if snapshot["queue_length"] > 0:
        render_status_ribbon(
            f"{snapshot['queue_length']} patient(s) are waiting for a decision.",
            chips=[
                f"Critical {snapshot['waiting_critical']}",
                f"Moderate {snapshot['waiting_moderate']}",
                f"Stable {snapshot['waiting_stable']}",
            ],
        )

    if snapshot["current_patient"]:
        current_patient = normalize_patient_record(snapshot["current_patient"])
        guidance = get_capacity_guidance(env, current_patient)
        st.markdown(
            f"{severity_badge(current_patient.get('severity', 'stable'))}",
            unsafe_allow_html=True,
        )
        st.info(
            f"Current decision case: {current_patient.get('name', 'Patient')} | "
            f"Severity: {current_patient.get('severity', '')} | Required bed: {current_patient.get('required_bed', '')} | "
            f"Waiting time: {current_patient.get('waiting_time', 0)} | {guidance['message']}"
        )


def render_patient_intake(env: HospitalEnv, predictor: StayDurationPredictor) -> None:
    render_section_intro(
        "Patient Intake",
        "Register new arrivals with severity, bed requirement, and risk details to trigger decision support.",
    )

    with st.form("patient_intake_form"):
        col1, col2, col3 = st.columns(3)
        name = col1.text_input("Patient Name", value="New Patient")
        age = col2.number_input("Age", min_value=0, max_value=120, value=45, step=1)
        gender = col3.selectbox("Gender", ["Male", "Female", "Other"])

        col4, col5, col6 = st.columns(3)
        severity = col4.selectbox("Severity", ["stable", "moderate", "critical"])
        required_bed = col5.selectbox("Required Bed", ["General", "ICU", "Isolation"])
        comorbidity = col6.selectbox("Comorbidity Risk", ["No", "Yes"])

        waiting_time = st.number_input("Current Waiting Time", min_value=0, max_value=48, value=0, step=1)
        submitted = st.form_submit_button("Register Incoming Patient")

        if submitted:
            predicted_days = predictor.predict_days(
                age=int(age),
                severity=severity,
                required_bed=required_bed,
                comorbidity=1 if comorbidity == "Yes" else 0,
            )
            patient = create_manual_patient(
                env,
                name=name,
                age=int(age),
                gender=gender,
                severity=severity,
                required_bed=required_bed,
                comorbidity=1 if comorbidity == "Yes" else 0,
                waiting_time=int(waiting_time),
                predicted_stay=predicted_days,
            )
            guidance = get_capacity_guidance(env, patient)
            if guidance["status"] == "available_now" and admit_patient_immediately(env, patient):
                persist_dashboard_state()
                st.success(
                    f"Patient registered and admitted immediately. Predicted hospital stay: {predicted_days} days."
                )
            else:
                push_patient_to_queue(env, patient)
                persist_dashboard_state()
                st.success(
                    f"Patient registered and added to queue. Predicted hospital stay: {predicted_days} days."
                )


def render_recommendation_panel(env: HospitalEnv, predictor: StayDurationPredictor, model: Optional[DQN], model_note: Optional[str]) -> None:
    render_section_intro(
        "Recommendation for Hospital Staff",
        "Review bed assignment, discharge-aware waiting advice, and transfer guidance before acting.",
    )

    if not env.current_patient:
        st.warning("No patient is currently waiting for a recommendation.")
        return

    current_patient = normalize_patient_record(env.current_patient)
    predicted_stay = predictor.predict_days(
        age=int(current_patient.get("age", 40)),
        severity=current_patient.get("severity", "stable"),
        required_bed=current_patient.get("required_bed", "General"),
        comorbidity=int(current_patient.get("comorbidity", 0)),
    )

    baseline_action = rule_based_action(env)
    dqn_action = get_dqn_action(env, model)

    render_decision_card(
        "Primary Recommendation",
        ACTION_LABELS[dqn_action] if dqn_action is not None else ACTION_LABELS[baseline_action],
        "This is the hospital-facing action suggestion based on the current patient, available capacity, and discharge-aware transfer logic.",
    )
    render_action_summary(
        ACTION_LABELS[baseline_action],
        ACTION_LABELS[dqn_action] if dqn_action is not None else "Unavailable",
        f"{predicted_stay} days",
    )

    if model_note:
        st.caption(model_note)

    patient_df = pd.DataFrame(
        [
            {
                "Name": current_patient.get("name", f"Patient {current_patient['id']}"),
                "Age": current_patient.get("age", ""),
                "Gender": current_patient.get("gender", ""),
                "Severity": current_patient.get("severity", ""),
                "Required Bed": current_patient.get("required_bed", ""),
                "Comorbidity": "Yes" if int(current_patient.get("comorbidity", 0)) else "No",
                "Waiting Time": current_patient.get("waiting_time", 0),
                "Predicted Stay": predicted_stay,
            }
        ]
    )
    st.table(patient_df)
    st.markdown(severity_badge(current_patient.get("severity", "stable")), unsafe_allow_html=True)

    explanation_title = "Recommended Reason"
    explanation_body = explain_recommendation(env, dqn_action if dqn_action is not None else baseline_action)
    st.markdown(f"**{explanation_title}**")
    st.write(explanation_body)

    capacity_guidance = get_capacity_guidance(env, current_patient)
    st.markdown("**Bed Availability Check**")
    st.write(capacity_guidance["message"])

    suggested_beds = available_bed_numbers(env, current_patient.get("required_bed", "General"))
    if suggested_beds:
        st.caption(f"Suggested bed numbers for this patient: {', '.join(suggested_beds)}")

    action_map = {f"{action_id} - {label}": action_id for action_id, label in ACTION_LABELS.items()}
    selected_label = st.selectbox("Hospital Decision", list(action_map.keys()))

    col1, col2, col3 = st.columns(3)
    if col1.button("Apply Selected Decision"):
        _, reward, _, info = env.step(action_map[selected_label])
        persist_dashboard_state()
        st.success(f"Decision applied. Immediate reward: {reward:.2f}")
        st.caption(f"Average waiting time is now {info['avg_waiting_time']:.2f}.")

    if col2.button("Apply Rule-Based Suggestion"):
        _, reward, _, info = env.step(baseline_action)
        persist_dashboard_state()
        st.success(f"Applied: {ACTION_LABELS[baseline_action]} | Reward: {reward:.2f}")
        st.caption(explain_recommendation(env, baseline_action))

    if col3.button("Apply DQN Suggestion"):
        if dqn_action is None:
            st.error("DQN recommendation is unavailable. Train or reload the model first.")
        else:
            _, reward, _, info = env.step(dqn_action)
            persist_dashboard_state()
            st.success(f"Applied: {ACTION_LABELS[dqn_action]} | Reward: {reward:.2f}")
            st.caption(f"Decision accuracy is now {info['decision_accuracy']:.2%}.")


def render_operations(env: HospitalEnv) -> None:
    render_section_intro(
        "Queue and Transfer Operations",
        "Process waiting patients, inspect the bed register, and review internal and external transfers.",
    )

    ops_col1, ops_col2 = st.columns(2)
    if ops_col1.button("Assign Next Waiting Patient"):
        if env.current_patient is None:
            st.warning("No patient is waiting in the queue.")
        else:
            action = rule_based_action(env)
            _, reward, _, _ = env.step(action)
            persist_dashboard_state()
            st.success(f"Processed next patient using {ACTION_LABELS[action]}. Reward: {reward:.2f}")

    if ops_col2.button("Auto Assign Free Beds"):
        summary = auto_assign_waiting_patients(env, max_patients=len(env.waiting_queue))
        persist_dashboard_state()
        st.success(
            f"Processed {summary['processed']} patient(s). "
            f"Assigned: {summary['assigned']}, Wait: {summary['waited']}, "
            f"Internal transfer: {summary['internal']}, External transfer: {summary['external']}."
        )

    queue_tab, active_tab, transfer_tab, database_tab, allocation_tab = st.tabs(
        ["Waiting Queue", "Admitted Patients", "Transfer Log", "Database Patients", "Allocation History"]
    )

    with queue_tab:
        queue_df = build_patient_df(env.waiting_queue)
        if queue_df.empty:
            st.caption("No patients are waiting.")
        else:
            queue_df["Suggested Queue Action"] = [
                recommended_queue_action(env, patient) for patient in env.waiting_queue
            ]
            queue_df["Capacity Note"] = [
                get_capacity_guidance(env, patient)["message"] for patient in env.waiting_queue
            ]
            st.dataframe(style_patient_dataframe(queue_df), width="stretch")

    with active_tab:
        active_df = build_patient_df(env.active_patients)
        if active_df.empty:
            st.caption("No patients are admitted right now.")
        else:
            st.dataframe(style_patient_dataframe(active_df), width="stretch")

        st.markdown("**Ward and Bed Register**")
        st.dataframe(style_bed_inventory_dataframe(generate_bed_inventory(env)), width="stretch")

        if env.active_patients:
            discharge_df = pd.DataFrame(
                [
                    {
                        "Patient": patient.get("name", f"Patient {patient['id']}"),
                        "Assigned Bed": patient.get("assigned_bed", ""),
                        "Severity": patient.get("severity", ""),
                        "Estimated Discharge In": patient.get("remaining_stay", 0),
                    }
                    for patient in sorted(env.active_patients, key=lambda patient: patient.get("remaining_stay", 9999))
                ]
            )
            st.markdown("**Expected Discharge Timeline**")
            st.dataframe(discharge_df, width="stretch")

    with transfer_tab:
        if env.transfer_log:
            st.dataframe(pd.DataFrame(env.transfer_log), width="stretch")
        else:
            st.caption("No transfers have been recorded yet.")

    with database_tab:
        hospital = st.session_state.get("authenticated_hospital")
        if hospital:
            patient_records_df = fetch_hospital_patients(hospital["hospital_id"])
            if patient_records_df.empty:
                st.caption("No patient records have been stored in the database yet.")
            else:
                st.dataframe(patient_records_df, width="stretch")

    with allocation_tab:
        hospital = st.session_state.get("authenticated_hospital")
        if hospital:
            allocation_df = fetch_hospital_allocations(hospital["hospital_id"])
            if allocation_df.empty:
                st.caption("No allocation history has been stored yet.")
            else:
                st.dataframe(allocation_df, width="stretch")


def render_analytics() -> None:
    render_section_intro(
        "Analytics and Model Comparison",
        "Review learning trends and compare the RL policy against the rule-based baseline.",
    )

    reward_path = os.path.join(BASE_DIR, "saved_models", "rewards_history.npy")
    waiting_path = os.path.join(BASE_DIR, "saved_models", "waiting_time_history.npy")

    if os.path.exists(reward_path) and os.path.exists(waiting_path):
        rewards = np.load(reward_path)
        waiting = np.load(waiting_path)
        curves_df = pd.DataFrame(
            {
                "Episode": np.arange(1, len(rewards) + 1),
                "Reward": rewards,
                "Waiting Time": waiting,
            }
        )
        st.line_chart(curves_df.set_index("Episode")[["Reward", "Waiting Time"]], width="stretch")
    else:
        st.info("Training curves will appear here after running `python training/train_dqn.py`.")

    if st.button("Run Policy Comparison"):
        comparison = compare_policies(episodes=30)
        comparison_df = pd.DataFrame(
            {
                policy_name: format_metrics(policy_metrics)
                for policy_name, policy_metrics in comparison.items()
            }
        )
        st.dataframe(comparison_df, width="stretch")


st.set_page_config(page_title="Hospital Bed Allocation Dashboard", layout="wide")
theme_mode = get_theme_mode()
apply_theme(theme_mode)
render_hero(
    "Intelligent Hospital Bed Allocation and Patient Transfer System",
    "A hospital-focused command center for capacity registration, patient intake, bed recommendations, queue control, and policy analytics.",
)

if not render_login():
    st.stop()

st.sidebar.title("Navigation")
authenticated_hospital = st.session_state.get("authenticated_hospital", {})
render_sidebar_card(
    "Signed In Hospital",
    authenticated_hospital.get("hospital_name", "Unknown Hospital"),
    chips=[
        authenticated_hospital.get("username", ""),
        authenticated_hospital.get("location", ""),
    ],
)
selected_theme = st.sidebar.toggle("Dark Mode", value=theme_mode == "Dark")
st.session_state.theme_mode = "Dark" if selected_theme else "Light"
apply_theme(st.session_state.theme_mode)
page = st.sidebar.radio(
    "Go to",
    [
        "Hospital Setup",
        "Patient Intake",
        "Recommendations",
        "Operations",
        "Analytics",
    ],
)

if st.sidebar.button("Logout"):
    logout_hospital()

st.sidebar.markdown("**Simulation Controls**")
if st.sidebar.button("Reset Hospital"):
    reset_env()
if st.sidebar.button("Advance One Time Step"):
    get_env().step(3)
    persist_dashboard_state()

env = get_env()
predictor = load_stay_predictor()
dqn_model, dqn_note = load_dqn_model()

st.sidebar.markdown("**Current Snapshot**")
snapshot = build_status_snapshot(env)
render_sidebar_card(
    "Current Snapshot",
    get_hospital_config_state()["hospital_name"],
    chips=[
        f"Queue {snapshot['queue_length']}",
        f"Admissions {snapshot['active_patients']}",
        f"Transfers {snapshot['transfers']}",
    ],
)

if page == "Hospital Setup":
    render_hospital_setup()
    render_status(get_env())

elif page == "Patient Intake":
    render_status(env)
    render_patient_intake(env, predictor)

elif page == "Recommendations":
    render_status(env)
    render_recommendation_panel(env, predictor, dqn_model, dqn_note)

elif page == "Operations":
    render_status(env)
    render_operations(env)

elif page == "Analytics":
    render_analytics()
