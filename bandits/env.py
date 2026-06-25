import numpy as np

from gradient_bandits import GradientBandit
from epsilon_greedy import EpsilonGreedy


class BanditEnv:
    def __init__ (self, n_arms, probs=None):
        self.n_arms = n_arms
        self.probs = probs if probs is not None else np.random.rand(n_arms)  # Random probabilities for each arm

    def step(self, action):
        reward = np.random.rand() < self.probs[action]  # Reward is 1 with probability of the chosen arm
        return reward


def run_bandit(agent, n_arms=10, steps=2000, probs=None, seed=None):
    """Run any bandit agent (act/update interface) against the env and report how it did."""
    if seed is not None:
        np.random.seed(seed)

    env = BanditEnv(n_arms, probs=probs)

    rewards = np.zeros(steps)
    best_arm = int(np.argmax(env.probs))      # the arm the agent *should* learn to prefer
    optimal_action = np.zeros(steps)          # 1 when the agent picked the best arm

    for t in range(steps):
        action = agent.act()                  # choose an arm per the agent's policy
        reward = env.step(action)             # pull it, observe reward
        agent.update(action, reward)          # learn from the outcome

        rewards[t] = reward
        optimal_action[t] = (action == best_arm)

    print(f"True arm probabilities: {np.round(env.probs, 3)}")
    print(f"Best arm: {best_arm} (p={env.probs[best_arm]:.3f})")
    if hasattr(agent, "H"):                   # gradient bandit: report preferences + softmax policy
        print(f"Final policy:           {np.round(agent.H, 3)}, softmax: {np.round(agent.softmax(), 3)}")
    elif hasattr(agent, "Q"):                 # value-based agent (e.g. epsilon-greedy)
        print(f"Final value estimates:  {np.round(agent.Q, 3)}")
    print(f"Average reward:         {rewards.mean():.3f}")
    print(f"% optimal action (last 100 steps): {optimal_action[-100:].mean() * 100:.1f}%")

    return agent, rewards, optimal_action


def run_gradient_bandit(n_arms=10, steps=2000, alpha=0.1, probs=None, seed=None):
    agent = GradientBandit(n_arms, alpha=alpha)
    return run_bandit(agent, n_arms=n_arms, steps=steps, probs=probs, seed=seed)


def run_epsilon_greedy(n_arms=10, steps=2000, epsilon=0.1, probs=None, seed=None):
    agent = EpsilonGreedy(n_arms, epsilon=epsilon)
    return run_bandit(agent, n_arms=n_arms, steps=steps, probs=probs, seed=seed)


if __name__ == "__main__":
    probs = [0.8, 0.3, 0.9, 0.1, 0.5]  # Example probabilities for each arm

    print("=== Gradient bandit ===")
    run_gradient_bandit(seed=0, probs=probs, n_arms=len(probs), steps=200000, alpha=0.1)

    print("\n=== Epsilon-greedy ===")
    run_epsilon_greedy(seed=0, probs=probs, n_arms=len(probs), steps=200000, epsilon=0.1)