import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F

from ddpg_agent import Agent

GAMMA = 0.99            # discount factor
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LEARN_TIMESTEP = 20     # Learning interval
NUM_LEARN = 10          # Number of learning updates

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent:
    def __init__(self, state_size, action_size, random_seed, n_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        self.n_agents = n_agents
        self.agents = [Agent(state_size, action_size, random_seed) for _ in range(n_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones, timestep):
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_TIMESTEP == 0:
            for _ in range(NUM_LEARN):
                for agent in self.agents:
                    experiences = self.memory.sample()
                    agent.learn(experiences, GAMMA)

    def act(self, states):
        actions = [agent.act(np.expand_dims(state, axis=0)) for agent, state in zip(self.agents, states)]
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)