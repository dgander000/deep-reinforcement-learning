import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, state_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.size = 0
        self.state_buffer = np.zeros((self.buffer_size, state_size), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size, state_size), dtype=np.float32)
        self.action_buffer = np.zeros(self.buffer_size, dtype=np.int64)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_size, dtype=np.int64)

    def store_transition(self, state, action, reward, next_state, done):
        self.state_buffer[self.index] = state
        self.next_state_buffer[self.index] = next_state
        self.action_buffer[self.index] = action
        self.reward_buffer[self.index] = reward
        self.done_buffer[self.index] = done
        self.index += 1
        self.index = self.index % self.buffer_size
        self.size += 1
        self.size = min(self.size, self.buffer_size)

    def sample(self, batch_size):
        mem_size = min(self.size, self.buffer_size)
        minibatch = np.random.choice(mem_size, batch_size, replace=False)

        states = self.state_buffer[minibatch]
        actions = self.action_buffer[minibatch]
        rewards = self.reward_buffer[minibatch]
        next_states = self.next_state_buffer[minibatch]
        dones = self.done_buffer[minibatch]

        return states, actions, rewards, next_states, dones
