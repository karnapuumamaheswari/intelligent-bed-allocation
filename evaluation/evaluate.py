import os
import sys
from typing import Callable, Dict, List

import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from baseline.rule_based import rule_based_action
from env.hospital_env import HospitalEnv
from models.dqn_model import DQN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    keys = [
        "total_reward",
        "avg_waiting_time",
        "bed_utilization_rate",
        "correct_allocations",
        "wrong_allocations",
        "critical_delays",
        "internal_transfers",
        "external_transfers",
        "decision_accuracy",
        "processed_patients",
    ]
    return {key: float(np.mean([result.get(key, 0.0) for result in results])) for key in keys}


def _run_policy(episodes: int, policy_fn: Callable[[HospitalEnv, np.ndarray], int]) -> Dict[str, float]:
    env = HospitalEnv(max_steps=100)
    episode_summaries = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        info = {}

        while not done:
            action = policy_fn(env, state)
            state, _, done, info = env.step(action)

        episode_summaries.append(info)

    return _aggregate_metrics(episode_summaries)


def evaluate_dqn(model_path: str = None, episodes: int = 100) -> Dict[str, float]:
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "saved_models", "dqn_hospital_best.pth")

    env = HospitalEnv(max_steps=100)
    model = DQN(env.state_size, env.action_size).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run training/train_dqn.py first."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            "Saved model is incompatible with the current environment/action space. "
            "Please retrain the DQN using training/train_dqn.py."
        ) from exc
    model.eval()

    def dqn_policy(_, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = model(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    metrics = _run_policy(episodes, dqn_policy)
    print_metrics("DQN Evaluation Results", metrics)
    return metrics


def evaluate_rule_based(episodes: int = 100) -> Dict[str, float]:
    metrics = _run_policy(episodes, lambda env, _: rule_based_action(env))
    print_metrics("Rule-Based Evaluation Results", metrics)
    return metrics


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Average Reward: {metrics['total_reward']:.2f}")
    print(f"Average Waiting Time: {metrics['avg_waiting_time']:.2f}")
    print(f"Average Bed Utilization: {metrics['bed_utilization_rate']:.2%}")
    print(f"Average Correct Allocations: {metrics['correct_allocations']:.2f}")
    print(f"Average Wrong Allocations: {metrics['wrong_allocations']:.2f}")
    print(f"Average Critical Delays: {metrics['critical_delays']:.2f}")
    print(f"Average Internal Transfers: {metrics['internal_transfers']:.2f}")
    print(f"Average External Transfers: {metrics['external_transfers']:.2f}")
    print(f"Average Decision Accuracy: {metrics['decision_accuracy']:.2%}")


def compare_policies(episodes: int = 100, model_path: str = None) -> Dict[str, Dict[str, float]]:
    baseline_metrics = evaluate_rule_based(episodes=episodes)
    dqn_metrics = evaluate_dqn(model_path=model_path, episodes=episodes)

    print("\nComparison Summary")
    print("------------------")
    print(
        "Reward Improvement: "
        f"{dqn_metrics['total_reward'] - baseline_metrics['total_reward']:.2f}"
    )
    print(
        "Waiting Time Reduction: "
        f"{baseline_metrics['avg_waiting_time'] - dqn_metrics['avg_waiting_time']:.2f}"
    )
    print(
        "Utilization Gain: "
        f"{(dqn_metrics['bed_utilization_rate'] - baseline_metrics['bed_utilization_rate']) * 100:.2f}%"
    )

    return {"rule_based": baseline_metrics, "dqn": dqn_metrics}


if __name__ == "__main__":
    compare_policies()
