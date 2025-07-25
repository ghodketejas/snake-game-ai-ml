"""
smarter_model.py
Enhanced neural network models and trainers for the smarter Snake agent.
Implements Double DQN (online + target) with Huber loss, gradient clipping,
AdamW, and a scheduler.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class DeeperQNet(nn.Module):
    """
    Deeper neural network for the smarter agent, with normalization, dropout, and LeakyReLU.
    """
    def __init__(self, input_size, hidden_sizes, output_size, device, dropout_rate=0.2):
        """
        Build a multi-layer feedforward network with LayerNorm and Dropout.
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
            # keep last hidden layer dropout off (common choice)
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
        """
        x = x.to(self.device)
        return self.network(x)

    def save(self, file_name='model.pth'):
        """
        Save the model weights to the specified file.
        """
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        """
        Load model weights from the specified file.
        """
        self.load_state_dict(torch.load(file_name, map_location=self.device))


class SmarterQTrainer:
    """
    Handles training logic for the smarter agent's deeper network using Double DQN.
    """
    def __init__(self, model, target_model, lr, gamma, device, clip_grad_norm=1.0):
        """
        Initialize the trainer with model, target model, learning rate, discount factor, and gradient clipping.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model               # online / policy network
        self.target_model = target_model # target network (lagged)
        self.device = device
        self.clip_grad_norm = clip_grad_norm

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        # Step every 1000 optimizer steps; decay LR by 0.9
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.criterion = nn.HuberLoss(delta=1.0)

    @torch.no_grad()
    def _gather_target_q(self, next_state, non_final_mask):
        """
        Compute Double DQN target Q-values:
            a* = argmax_a Q_online(next_state, a)
            target = Q_target(next_state, a*)
        """
        # Online picks the actions
        online_q_next = self.model(next_state[non_final_mask])                  # (Nnf, A)
        best_actions = online_q_next.argmax(dim=1, keepdim=True)                # (Nnf, 1)

        # Target evaluates them
        target_q_next_all = self.target_model(next_state[non_final_mask])       # (Nnf, A)
        target_q_next = target_q_next_all.gather(1, best_actions).squeeze(1)    # (Nnf,)
        return target_q_next

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step using Double DQN with vectorized ops.
        `action` is expected to be one-hot (as in your existing pipeline).
        """
        state      = torch.tensor(np.array(state), dtype=torch.float, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=self.device)
        action     = torch.tensor(np.array(action), dtype=torch.long, device=self.device)
        reward     = torch.tensor(np.array(reward), dtype=torch.float, device=self.device)
        done       = torch.tensor(np.array(done), dtype=torch.bool, device=self.device)

        # If we got a single sample, make it batch-like
        if state.dim() == 1:
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = done.unsqueeze(0)

        batch_size = state.size(0)

        # Convert one-hot actions (shape [B, A]) to indices (shape [B, 1])
        if action.dim() == 2 and action.size(1) > 1:
            action_indices = torch.argmax(action, dim=1, keepdim=True)  # (B, 1)
        else:
            # Already indices
            action_indices = action.view(-1, 1)

        # Q(s, a) from online net
        q_pred_all = self.model(state)                        # (B, A)
        q_pred = q_pred_all.gather(1, action_indices).squeeze(1)  # (B,)

        # Targets
        with torch.no_grad():
            q_target = reward.clone()                         # (B,)
            non_final_mask = ~done
            if non_final_mask.any():
                # Double DQN target for non-terminal transitions
                target_q_next = self._gather_target_q(next_state, non_final_mask)
                q_target[non_final_mask] += self.gamma * target_q_next

        self.optimizer.zero_grad()
        loss = self.criterion(q_pred, q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

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
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    print("This module contains model definitions and should not be run directly.")
    print("Please run 'smarter_agent.py' to train the smarter agent.")
