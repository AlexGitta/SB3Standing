import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Add priority tracking
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        
        # Existing buffers
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, value, log_prob, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.masks[self.position] = 0.0 if done else 1.0
        
        # Set max priority for new experience
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return None
            
        # Calculate sampling probabilities
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.values[indices]),
            torch.FloatTensor(self.log_probs[indices]),
            torch.FloatTensor(self.masks[indices]),
            torch.FloatTensor(weights),
            indices
        )

    def update_priorities(self, indices, rewards):
        """Update priorities based on rewards"""
        for idx, reward in zip(indices, rewards):
            self.priorities[idx] = max(abs(reward), 1e-6)  # Avoid zero priority
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def clear(self):
        """Clear the replay buffer"""
        self.position = 0
        self.size = 0