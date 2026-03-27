import random
from collections import deque
from typing import Deque, Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.int64)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)