"""
Enhanced neural network models and trainers for smarter DQN agent.

This module provides the DeeperQNet neural network architecture and
SmarterQTrainer class implementing Double DQN with advanced features
like Huber loss, gradient clipping, and target network updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class DeeperQNet(nn.Module):
    """
    Deeper neural network for enhanced DQN agent.
    
    Features include:
    - Multiple hidden layers with configurable sizes
    - Layer normalization for training stability
    - LeakyReLU activation functions
    - Dropout for regularization
    - Xavier weight initialization
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, device, dropout_rate=0.2):
        """
        Initialize the deeper neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
            device: PyTorch device for computation
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        self.device = device
        self.dropout_rate = dropout_rate

        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            # Keep last hidden layer dropout off (common choice)
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization for better training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through the deeper network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = x.to(self.device)
        return self.network(x)

    def save(self, file_name='model.pth'):
        """
        Save the model weights to a file.
        
        Args:
            file_name: Path to save the model weights
        """
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        """
        Load model weights from a file.
        
        Args:
            file_name: Path to load the model weights from
        """
        self.load_state_dict(torch.load(file_name, map_location=self.device))


class SmarterQTrainer:
    """
    Enhanced training logic for Double DQN with advanced features.
    
    Features include:
    - Double DQN architecture with target network
    - Huber loss for robustness
    - Gradient clipping for stability
    - AdamW optimizer with weight decay
    - Learning rate scheduling
    - Soft and hard target network updates
    """
    
    def __init__(self, model, target_model, lr, gamma, device, clip_grad_norm=0.5):
        """
        Initialize the enhanced trainer.
        
        Args:
            model: Online/policy network
            target_model: Target network for stability
            lr: Learning rate
            gamma: Discount factor
            device: PyTorch device for computation
            clip_grad_norm: Maximum gradient norm for clipping
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model               # Online/policy network
        self.target_model = target_model # Target network (lagged)
        self.device = device
        self.clip_grad_norm = clip_grad_norm

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        # Step every 1000 optimizer steps; decay LR by 0.9
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20000, eta_min=1e-5)
        self.criterion = nn.HuberLoss(delta=1.0)

    @torch.no_grad()
    def _gather_target_q(self, next_state, non_final_mask):
        """
        Compute Double DQN target Q-values.
        
        Double DQN formula:
            a* = argmax_a Q_online(next_state, a)
            target = Q_target(next_state, a*)
            
        Args:
            next_state: Next state tensor
            non_final_mask: Boolean mask for non-final states
            
        Returns:
            torch.Tensor: Target Q-values for non-final states
        """
        # Online network picks the actions
        online_q_next = self.model(next_state[non_final_mask])                  # (Nnf, A)
        best_actions = online_q_next.argmax(dim=1, keepdim=True)                # (Nnf, 1)

        # Target network evaluates them
        target_q_next_all = self.target_model(next_state[non_final_mask])       # (Nnf, A)
        target_q_next = target_q_next_all.gather(1, best_actions).squeeze(1)    # (Nnf,)
        return target_q_next

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step using Double DQN with vectorized operations.
        
        Args:
            state: Current state tensor or list of states
            action: Action taken (one-hot encoded)
            reward: Reward received
            next_state: Next state tensor or list of states
            done: Whether episode ended
        """
        state = torch.tensor(np.array(state), dtype=torch.float, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.long, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=self.device)
        done = torch.tensor(np.array(done), dtype=torch.bool, device=self.device)

        # Make single samples look like a batch
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Turn one-hot actions into indices
        if action.dim() == 2 and action.size(1) > 1:
            action_idx = torch.argmax(action, dim=1, keepdim=True)        # (B,1)
        else:
            action_idx = action.view(-1, 1)                               # (B,1)

        # Q(s,a) from online network
        q_pred_all = self.model(state)                                    # (B,A)
        q_pred = q_pred_all.gather(1, action_idx).squeeze(1)              # (B,)

        # Double-DQN target
        with torch.no_grad():
            q_target = reward.clone()                                     # (B,)
            non_final = ~done
            if non_final.any():
                next_q = self._gather_target_q(next_state, non_final)     # (Nnf,)
                q_target[non_final] += self.gamma * next_q

        # Optimize
        self.optimizer.zero_grad()
        loss = self.criterion(q_pred, q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # Faster soft target update (τ = 0.05)
        self.soft_update_target(tau=0.05)

    @torch.no_grad()
    def hard_update_target(self):
        """
        Copy online network weights into target network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def soft_update_target(self, tau=0.01):
        """
        Soft update: θ_target ← τ θ_online + (1 - τ) θ_target
        
        Args:
            tau: Soft update factor (0 < tau < 1)
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    print("This module contains model definitions and should not be run directly.")
    print("Please run 'smarter_agent.py' to train the smarter agent.")