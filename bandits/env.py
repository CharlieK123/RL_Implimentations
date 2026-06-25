import numpy as np

class BanditEnv:
    def __init__ (self, n_arms, probs=None):
        self.n_arms = n_arms
        self.probs = probs if probs is not None else np.random.rand(n_arms)  # Random probabilities for each arm

    def step(self, action):
        reward = np.random.rand() < self.probs[action]  # Reward is 1 with probability of the chosen arm
        return reward