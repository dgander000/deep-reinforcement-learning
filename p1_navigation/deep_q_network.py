import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, state_size, action_size, checkpoint_name, checkpoint_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

    def save_checkpoint(self):
        print('saving checkpoint')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('loading checkpoint')
        self.load_state_dict(torch.load(self.checkpoint_file))