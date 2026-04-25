import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network with LSTM for temporal patterns.

    Improvements over original:
    - Dueling architecture separates value and advantage streams
    - LSTM layers capture temporal dependencies
    - Batch normalization for stable training
    - Noisy layers for exploration (optional)
    """

    def __init__(self, num_states, num_actions, hidden_size=128, lstm_layers=2, dropout=0.2):
        super(DuelingDQN, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        # Feature extraction layers
        self.fc1 = nn.Linear(num_states, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Dueling streams
        # Value stream
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_fc2 = nn.Linear(hidden_size // 2, 1)

        # Advantage stream
        self.advantage_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage_fc2 = nn.Linear(hidden_size // 2, num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the network.

        Args:
            x: Input state tensor [batch_size, seq_len, num_states] or [batch_size, num_states]
            hidden_state: Optional LSTM hidden state

        Returns:
            Q-values, hidden_state
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            # Add sequence dimension
            x = x.unsqueeze(1)  # [batch_size, 1, num_states]

        batch_size, seq_len, _ = x.shape

        # Feature extraction
        x = x.reshape(-1, self.num_states)  # [batch_size * seq_len, num_states]
        x = self.fc1(x)

        # Only apply BatchNorm if batch_size > 1
        if x.shape[0] > 1:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.dropout1(x)
        x = x.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]

        # LSTM temporal processing
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(x)
        else:
            lstm_out, hidden_state = self.lstm(x, hidden_state)

        # Take last output
        x = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Dueling streams
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)  # [batch_size, 1]

        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)  # [batch_size, num_actions]

        # Combine using dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values, hidden_state

    def get_action(self, state, epsilon=0.0, hidden_state=None, device=None):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate
            hidden_state: LSTM hidden state
            device: Device to use (cpu or cuda)

        Returns:
            action, hidden_state
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions), hidden_state

        # Use provided device or try to infer from model parameters
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            else:
                state = state.to(device)
            q_values, hidden_state = self.forward(state, hidden_state)
            action = q_values.argmax(dim=1).item()

        return action, hidden_state


class DoubleDQNAgent:
    """
    Double DQN Agent with target network.

    Improvements:
    - Target network to stabilize training
    - Gradient clipping to prevent exploding gradients
    - Soft updates for target network
    - Learning rate scheduling
    """

    def __init__(
        self,
        num_states,
        num_actions,
        hidden_size=128,
        lstm_layers=2,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.005,
        device=None
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Networks
        self.policy_net = DuelingDQN(num_states, num_actions, hidden_size, lstm_layers).to(self.device)
        self.target_net = DuelingDQN(num_states, num_actions, hidden_size, lstm_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Note: torch.compile disabled on Windows due to Triton dependency
        # Models will still run efficiently with CUDA + mixed precision

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )

        # Mixed precision training for faster GPU computation
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        # Enable cuDNN autotuner for optimal convolution algorithms
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            # Enable TF32 for Ampere GPUs (3080 Ti) - 8x faster matmul!
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(f"[Agent] Device: {self.device}")
        print(f"[Agent] Policy network parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
        if self.scaler:
            print(f"[Agent] Mixed precision training enabled")
        if self.device.type == 'cuda':
            print(f"[Agent] cuDNN benchmark + TF32 tensor cores enabled")

    def compute_loss(self, batch, use_double_dqn=True):
        """
        Compute Double DQN loss.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
            use_double_dqn: Use Double DQN update (True) or standard DQN (False)

        Returns:
            loss tensor
        """
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors with non-blocking for async GPU transfer
        states = torch.FloatTensor(states).to(self.device, non_blocking=True)
        actions = torch.LongTensor(actions).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(next_states).to(self.device, non_blocking=True)
        dones = torch.FloatTensor(dones).to(self.device, non_blocking=True)

        # Current Q values
        current_q_values, _ = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values
        with torch.no_grad():
            if use_double_dqn:
                # Double DQN: use policy network to select action, target network to evaluate
                next_q_policy, _ = self.policy_net(next_states)
                next_actions = next_q_policy.argmax(dim=1)
                next_q_target, _ = self.target_net(next_states)
                next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values, _ = self.target_net(next_states)
                next_q_values = next_q_values.max(dim=1)[0]

            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute Huber loss (more robust to outliers than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        return loss

    def update(self, batch, weights=None):
        """
        Update the policy network.

        Args:
            batch: Training batch
            weights: Optional importance sampling weights for PER

        Returns:
            loss value, TD errors
        """
        self.policy_net.train()

        # Optimize with mixed precision if available
        self.optimizer.zero_grad()

        if self.scaler:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                loss = self.compute_loss(batch)
                if weights is not None:
                    weights = torch.FloatTensor(weights).to(self.device)
                    loss = (loss * weights).mean()

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            loss = self.compute_loss(batch)
            if weights is not None:
                weights = torch.FloatTensor(weights).to(self.device)
                loss = (loss * weights).mean()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

            self.optimizer.step()

        # Soft update target network
        self.soft_update_target_network()

        # Compute TD errors for PER
        with torch.no_grad():
            states, actions, rewards, next_states, dones = batch
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            current_q, _ = self.policy_net(states)
            current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

            next_q, _ = self.target_net(next_states)
            next_q = next_q.max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

            td_errors = torch.abs(current_q - target_q).cpu().numpy()

        return loss.item(), td_errors

    def soft_update_target_network(self):
        """Soft update: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def hard_update_target_network(self):
        """Hard update: θ_target = θ_policy"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save model checkpoint"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {filepath}")
