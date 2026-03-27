import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.hospital_env import HospitalEnv
from models.dqn_model import DQN
from models.replay_buffer import ReplayBuffer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_action(model, state, epsilon, action_size):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values, dim=1).item()


def train():
    set_seed(42)

    env = HospitalEnv(max_steps=100)

    state_size = env.state_size
    action_size = env.action_size

    q_network = DQN(state_size, action_size).to(DEVICE)
    target_network = DQN(state_size, action_size).to(DEVICE)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=20000)

    episodes = 500
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    target_update_freq = 10

    rewards_history = []
    waiting_time_history = []
    best_reward = float("-inf")

    os.makedirs("saved_models", exist_ok=True)

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = select_action(q_network, state, epsilon, action_size)
            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

                current_q = q_network(states).gather(1, actions)

                with torch.no_grad():
                    next_actions = q_network(next_states).argmax(dim=1, keepdim=True)
                    next_q = target_network(next_states).gather(1, next_actions)
                    target_q = rewards + gamma * next_q * (1 - dones)

                loss = criterion(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(episode_reward)
        waiting_time_history.append(info.get("avg_waiting_time", 0.0))

        if (episode + 1) % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(q_network.state_dict(), "saved_models/dqn_hospital_best.pth")

        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(rewards_history[-10:])
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Best: {best_reward:.2f} | "
                f"Avg(10): {recent_avg:.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    torch.save(q_network.state_dict(), "saved_models/dqn_hospital_last.pth")
    np.save("saved_models/rewards_history.npy", np.array(rewards_history, dtype=np.float32))
    np.save("saved_models/waiting_time_history.npy", np.array(waiting_time_history, dtype=np.float32))
    print("Best model saved to saved_models/dqn_hospital_best.pth")
    print("Last model saved to saved_models/dqn_hospital_last.pth")
    print("Training curves saved to saved_models/*.npy")

    return rewards_history, waiting_time_history


if __name__ == "__main__":
    train()
