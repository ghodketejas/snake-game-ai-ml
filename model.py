"""
Neural network models and training logic for classic DQN agent.

This module provides the Linear_QNet neural network architecture and
QTrainer class for training the classic DQN agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Linear_QNet(nn.Module):
    """
    Simple feedforward neural network for DQN agent.
    
    A two-layer neural network with ReLU activation designed for
    Q-value approximation in the Snake game environment.
    """
    
    def __init__(self, input_size, hidden_size, output_size, device=DEVICE):
        """
        Initialize the neural network layers.
        
        Args:
            input_size: Number of input features (state size)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons (action size)
            device: PyTorch device for computation
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Save the model weights to a file.
        
        Args:
            file_name: Path to save the model weights
        """
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    Training logic for the DQN agent.
    
    Implements the Q-learning update rule using the Bellman equation
    with experience replay and target network concepts.
    """
    
    def __init__(self, model, lr, gamma, device=DEVICE):
        """
        Initialize the trainer with model and hyperparameters.
        
        Args:
            model: Neural network model to train
            lr: Learning rate
            gamma: Discount factor for future rewards
            device: PyTorch device for computation
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
        
        Args:
            state: Current state tensor or list of states
            action: Action taken (one-hot encoded)
            reward: Reward received
            next_state: Next state tensor or list of states
            done: Whether episode ended
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        # Handle single samples by adding batch dimension
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Get predicted Q-values for current state
        pred = self.model(state)

        # Calculate target Q-values using Bellman equation
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Perform optimization step
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()



