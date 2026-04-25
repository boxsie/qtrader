import numpy as np
import pickle
from pathlib import Path
from collections import deque


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.

    This is a binary tree where parent node value = sum of children values.
    Leaf nodes store the priorities, and we can efficiently sample proportional to priority.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Update parent nodes with the change in priority"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample on leaf node given priority value s"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return total priority"""
        return self.tree[0]

    def add(self, priority, data):
        """Add new experience with priority"""
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """Update priority of a sample"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get sample based on priority value s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class Memory:
    """
    Prioritized Experience Replay (PER) buffer.

    Features:
    - Samples important transitions more frequently
    - Uses TD-error as priority metric
    - Includes importance sampling weights to correct bias
    - Supports both proportional and rank-based prioritization

    Reference: Schaul et al., 2015 - Prioritized Experience Replay
    """

    def __init__(
        self,
        capacity=100000,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        epsilon=1e-6
    ):
        """
        Args:
            capacity: Maximum number of experiences to store
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction (0 = no correction, 1 = full correction)
            beta_increment: Increment beta towards 1.0 over time
            epsilon: Small constant to prevent zero priority
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.n_entries

    def add(self, experience, td_error=None):
        """
        Add experience to memory.

        Args:
            experience: (state, action, reward, next_state, done) tuple
            td_error: Optional TD-error for priority. If None, uses max priority.
        """
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + self.epsilon) ** self.alpha

        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """
        Sample batch with priorities.

        Returns:
            batch: List of experiences
            indices: Tree indices for updating priorities
            weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()

        # IS weights = (1/N * 1/P(i))^beta
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on new TD-errors.

        Args:
            indices: Tree indices to update
            td_errors: New TD-error values
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def save(self, filepath):
        """Save replay buffer to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'tree_data': self.tree.data,
            'tree_tree': self.tree.tree,
            'write_idx': self.tree.write_idx,
            'n_entries': self.tree.n_entries,
            'max_priority': self.max_priority,
            'beta': self.beta
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Memory saved to {filepath}")

    def load(self, filepath):
        """Load replay buffer from disk"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        self.tree.data = save_dict['tree_data']
        self.tree.tree = save_dict['tree_tree']
        self.tree.write_idx = save_dict['write_idx']
        self.tree.n_entries = save_dict['n_entries']
        self.max_priority = save_dict['max_priority']
        self.beta = save_dict['beta']
        print(f"Memory loaded from {filepath}")


class EpisodeMemory:
    """
    Store complete episodes for multi-step returns and sequence learning.

    This is useful for LSTM-based agents that need temporal context.
    """

    def __init__(self, max_episodes=1000):
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode = []

    def add_step(self, state, action, reward, next_state, done):
        """Add step to current episode"""
        self.current_episode.append((state, action, reward, next_state, done))

        if done:
            self.finish_episode()

    def finish_episode(self):
        """Finish current episode and add to buffer"""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def sample_sequences(self, batch_size, sequence_length):
        """
        Sample sequences of experiences.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence

        Returns:
            List of sequences, each containing sequence_length steps
        """
        sequences = []

        for _ in range(batch_size):
            # Sample random episode
            if not self.episodes:
                continue

            episode = self.episodes[np.random.randint(len(self.episodes))]

            if len(episode) < sequence_length:
                continue

            # Sample random starting point
            start_idx = np.random.randint(0, len(episode) - sequence_length + 1)
            sequence = episode[start_idx:start_idx + sequence_length]
            sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.episodes)

    def save(self, filepath):
        """Save episode buffer to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(list(self.episodes), f)
        print(f"Episode memory saved to {filepath}")

    def load(self, filepath):
        """Load episode buffer from disk"""
        with open(filepath, 'rb') as f:
            episodes = pickle.load(f)
        self.episodes = deque(episodes, maxlen=self.max_episodes)
        print(f"Episode memory loaded from {filepath}")
