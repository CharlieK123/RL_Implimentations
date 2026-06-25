import numpy as np


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms          # number of arms (actions) available
        self.epsilon = epsilon        # probability of exploring a random arm
        self.Q = np.zeros(n_arms)     # estimated value of each arm — start optimistic-neutral at 0
        self.N = np.zeros(n_arms)     # number of times each arm has been pulled

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)   # explore: pick a uniformly random arm
        return int(np.argmax(self.Q))               # exploit: pick the current best estimate

    def update(self, action, reward):
        # incremental sample-average update:
        #   Q[a] += (R - Q[a]) / N[a]
        # equivalent to keeping the running mean of rewards seen for that arm
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
