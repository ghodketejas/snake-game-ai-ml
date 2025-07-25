"""
smarter_model.py
Enhanced neural network models and trainers for the smarter Snake agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Enhanced Q-Network for smarter agent
class DeeperQNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, device, dropout_rate=0.2):
        super().__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Layer normalization for better training stability (works with batch size 1)
            layers.append(nn.LayerNorm(hidden_size))
            
            # LeakyReLU for better gradient flow
            layers.append(nn.LeakyReLU(0.1))
            
            # Dropout for regularization (except for last hidden layer)
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for better training
        self._initialize_weights()
        
        self.to(self.device)
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = x.to(self.device)
        return self.network(x)
    
    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='model.pth'):
        self.load_state_dict(torch.load(file_name, map_location=self.device))

# Enhanced QTrainer for the deeper network
class SmarterQTrainer:
    def __init__(self, model, lr, gamma, device, clip_grad_norm=1.0):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        
        # Use AdamW optimizer for better weight decay
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # Use Huber loss for better stability with outliers
        self.criterion = nn.HuberLoss(delta=1.0)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone().detach()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step() 

if __name__ == '__main__':
    print("This module contains model definitions and should not be run directly.")
    print("Please run 'smarter_agent.py' to train the smarter agent.") 