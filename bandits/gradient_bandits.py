import numpy as np


class GradientBandit:
    def __init__(self, n_arms, alpha=0.1):
        self.n_arms = n_arms          # number of arms (actions) available
        self.alpha = alpha            # step size / learning rate for preference updates
        self.H = np.zeros(n_arms)     # preferences, one per arm — start equal (uniform policy)
        self.baseline = 0.0           # running average reward, used as the gradient baseline
        self.t = 0                    # step counter, for updating the running average

    def softmax(self):
        exp_H = np.exp(self.H - np.max(self.H))   # subtract max for numerical stability
        return exp_H / np.sum(exp_H)

    def act(self):
        probs = self.softmax()
        return np.random.choice(self.n_arms, p=probs)   # sample an arm from the policy

    def update(self, action, reward):
        # update running-average baseline first
        self.t += 1
        self.baseline += (reward - self.baseline) / self.t

        probs = self.softmax()
        one_hot = np.zeros(self.n_arms)
        one_hot[action] = 1.0

        # gradient ascent on preferences:
        #   chosen arm:    H += alpha * (R - baseline) * (1 - pi)
        #   other arms:    H -= alpha * (R - baseline) * pi
        # both captured in one vectorised line via (one_hot - probs)
        self.H += self.alpha * (reward - self.baseline) * (one_hot - probs)
