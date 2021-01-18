import numpy as np
import torch 
from deep_q_network import DeepQNetwork
from replay_buffer import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, lr, epsilon_start, epsilon_min, epsilon_decay, state_size, action_size,
                 buffer_size, batch_size, update_frequency, soft_update=False, tau=0.001, checkpoint_dir='tmp'):
        self.gamma = gamma
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.target_network_update_frequency = update_frequency
        self.soft_update = soft_update
        self.tau = tau
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(action_size)]
        self.learn_counter = 0

        self.memory = ReplayBuffer(buffer_size, self.state_size)
        self.q_local = DeepQNetwork(self.lr, self.state_size, self.action_size, 'model_q_local.pth', self.checkpoint_dir)
        self.q_target = DeepQNetwork(self.lr, self.state_size, self.action_size, 'model_q_target.pth', self.checkpoint_dir)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.tensor(state,dtype=torch.float).to(self.q_local.device)
            actions = self.q_local.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def step(self, state, action, reward, next_state, done):
        self.store_transition(state, action, reward, next_state, done)
        self.learn()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        states = torch.tensor(state).to(self.q_local.device)
        rewards = torch.tensor(reward).to(self.q_local.device)
        dones = torch.tensor(done).to(self.q_local.device)
        actions = torch.tensor(action).to(self.q_local.device)
        next_states = torch.tensor(new_state).to(self.q_local.device)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        if self.soft_update == True:
            for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        elif self.learn_counter % self.target_network_update_frequency == 0:
            self.q_target.load_state_dict(self.q_local.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_models(self):
        self.q_local.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_local.load_checkpoint()
        self.q_target.load_checkpoint()

    def learn(self):
        if self.memory.size < self.batch_size:
            return

        self.q_local.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.sample_memory()

        q_targets = self.q_target.forward(next_states).max(dim=1)[0]
        q_targets = rewards + self.gamma*q_targets*(1-dones)

        indices = np.arange(self.batch_size)
        q_expected = self.q_local.forward(states)[indices, actions]

        loss = self.q_local.loss(q_expected, q_targets).to(self.q_local.device)
        loss.backward()
        self.q_local.optimizer.step()
        self.learn_counter += 1

        self.decrement_epsilon()
        self.update_target_network()