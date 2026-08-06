"""Deep Q-Network (DQN) Agent implementation for Dynamic Pricing RL.

Implements epsilon-greedy exploration, experience replay memory updates,
target network weight synchronization, loss computation, and checkpoint state serialization.
"""

from pathlib import Path
from typing import Tuple, List, Any, Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.logger import get_logger
from src.dqn.network import QNetwork
from src.dqn.replay_buffer import ReplayBuffer

logger = get_logger("dqn_agent")


class DQNAgent:
    """Deep Q-Network Agent for Hotel Dynamic Pricing."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        price_actions: List[float],
        max_inventory: int,
        max_days: int,
        batch_size: int,
        memory_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        min_epsilon: float,
        hidden_units: List[int],
        device: str,
        random_seed: int = 42,
    ) -> None:
        """Initializes the DQNAgent.

        Args:
            state_dim: Dimension of the state representation (5).
            action_dim: Dimension of the action selection space (len(price_actions)).
            price_actions: List of actual price values.
            max_inventory: Maximum room inventory (used for state normalization).
            max_days: Maximum booking window horizon (used for state normalization).
            batch_size: Batch size for replay training steps.
            memory_size: Replay memory capacity.
            learning_rate: Optimizer learning rate.
            gamma: Future reward discount factor.
            epsilon: Starting exploration rate.
            epsilon_decay: Decay multiplier per training episode.
            min_epsilon: Bounded minimum exploration rate.
            hidden_units: Hidden layers configuration for the networks.
            device: Target computational device ('cuda' or 'cpu').
            random_seed: Seed for reproducibility.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.price_actions = price_actions
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.random_seed = random_seed

        # Setup compute device dynamically
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        logger.info(f"DQN Agent configured on device: {self.device}")

        # Set random seeds for reproducibility
        torch.manual_seed(self.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Policy & Target Networks
        self.q_net = QNetwork(state_dim, action_dim, hidden_units).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_units).to(self.device)
        self.update_target_network()
        self.target_net.eval()  # Target network remains in evaluation mode

        # Replay Buffer
        self.memory = ReplayBuffer(memory_size)

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _preprocess_state(self, state: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """Converts raw state tuple parameters to a normalized float tensor.

        Args:
            state: Tuple containing (inventory, days_left, season_idx, month_idx, hotel_idx).

        Returns:
            torch.Tensor: Preprocessed state tensor of shape (state_dim,).
        """
        inv, days, season, month, hotel = state
        normalized = [
            float(inv) / self.max_inventory,
            float(days) / self.max_days,
            float(season) / 3.0,
            float(month) / 11.0,
            float(hotel) / 1.0,
        ]
        return torch.tensor(normalized, dtype=torch.float32, device=self.device)

    def choose_action(self, state: Tuple[int, int, int, int, int], use_epsilon: bool = True) -> int:
        """Selects action index using epsilon-greedy strategy.

        Args:
            state: Encoded environment state.
            use_epsilon: If True, applies exploration decay checks.

        Returns:
            int: Selected action index corresponding to a price action.
        """
        if use_epsilon and (random.random() < self.epsilon):
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = self._preprocess_state(state).unsqueeze(0)  # Add batch dimension -> (1, 5)
            self.q_net.eval()
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            self.q_net.train()
            return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: Tuple[int, int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int, int],
        done: bool,
    ) -> None:
        """Saves transition to the experience replay memory buffer.

        Args:
            state: Current state.
            action: Selected action index.
            reward: Step reward.
            next_state: Resulting state.
            done: Terminal status flag.
        """
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """Runs a single optimization gradient descent step on memory samples.

        Returns:
            float: Training loss value, or 0.0 if memory is insufficient.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)

        # Unpack batch and preprocess state lists to tensors
        states = torch.stack([self._preprocess_state(x[0]) for x in batch])
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack([self._preprocess_state(x[3]) for x in batch])
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute Q(s, a) for the actions chosen in current states
        self.q_net.train()
        q_values = self.q_net(states).gather(1, actions)

        # Compute Target Q-values: max_a Q_target(s', a)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + (self.gamma * max_next_q * (1 - dones))

        # Optimize policy network parameters
        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copies policy network weights to the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decays the epsilon value by the decay factor."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def predict(self, state: Tuple[int, int, int, int, int]) -> int:
        """Greedy prediction of optimal action index (no exploration).

        Args:
            state: Encoded state tuple.

        Returns:
            int: Predicted optimal action index.
        """
        return self.choose_action(state, use_epsilon=False)

    def save(self, model_filepath: Path, optimizer_filepath: Path) -> None:
        """Saves checkpoints of model weights and optimizer states.

        Args:
            model_filepath: Target path for the network file.
            optimizer_filepath: Target path for the optimizer state file.
        """
        model_filepath = Path(model_filepath)
        optimizer_filepath = Path(optimizer_filepath)
        model_filepath.parent.mkdir(parents=True, exist_ok=True)
        optimizer_filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.q_net.state_dict(), model_filepath)
        torch.save(self.optimizer.state_dict(), optimizer_filepath)
        logger.info(f"Model saved to: {model_filepath}")
        logger.info(f"Optimizer state saved to: {optimizer_filepath}")

    def load(self, model_filepath: Path, optimizer_filepath: Path) -> None:
        """Loads checkpoints of model weights and optimizer states.

        Args:
            model_filepath: Source path to the network file.
            optimizer_filepath: Source path to the optimizer state file.
        """
        model_filepath = Path(model_filepath)
        optimizer_filepath = Path(optimizer_filepath)

        if not model_filepath.exists():
            raise FileNotFoundError(f"Model state dictionary not found at {model_filepath}")
        if not optimizer_filepath.exists():
            raise FileNotFoundError(f"Optimizer state not found at {optimizer_filepath}")

        self.q_net.load_state_dict(torch.load(model_filepath, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(optimizer_filepath, map_location=self.device))
        self.update_target_network()
        logger.info(f"Successfully loaded DQN checkpoint files from {model_filepath.parent}")
