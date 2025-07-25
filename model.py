import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linear_QNet(nn.Module):
    """
    Simple feedforward neural network for DQN agent.
    """
    def __init__(self, input_size, hidden_size, output_size, device=DEVICE):
        """
        Initialize the network layers and move to the specified device.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name):
        """
        Save the model weights to the specified file.
        """
        torch.save(self.state_dict(), file_name)

class QTrainer:
    """
    Handles training logic for the DQN agent.
    """
    def __init__(self, model, lr, gamma, device=DEVICE):
        """
        Initialize the trainer with model, learning rate, and discount factor.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step using the Bellman equation.
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()



