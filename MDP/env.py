import gymnasium as gym
import numpy as np


env = gym.make('FrozenLake-v1', is_slippery=False)  # deterministic environment

n_states, n_actions = env.observation_space.n, env.action_space.n

Q1 = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
Q2 = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros

ALPHA = 0.05
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
EPISODES = 50_000

for epoch in range(EPISODES):
    state, info = env.reset()
    done = False
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    total_reward = 0

    if epoch == EPISODES - 1:
        EPSILON = 0
        env.render()  # Render the environment for the last episode

    while not done:

        if np.random.rand() < EPSILON:
            action = env.action_space.sample()  # Explore: select a random action
        else:
            action = np.argmax((Q1[state]))  # Exploit: select the action with max Q-value

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


        if np.random.rand() < 0.5:
            Q_action, Q_eval = Q1, Q2
        else:
            Q_action, Q_eval = Q2, Q1


        target = reward
        if not terminated:
            best_action = np.argmax(Q1[next_state])
            target += GAMMA * Q1[next_state, best_action]

        Q1[state, action] += ALPHA * (target - Q1[state, action])

        state = next_state
        total_reward += reward
    if (epoch+1) % 1000 == 0:
        print(f"Episode: {epoch + 1}, State: {state}, Action: {action}, Reward: {total_reward}, Done: {done}")

env.close()

env = gym.make("FrozenLake-v1", is_slippery=False)

successes = 0

for _ in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(Q1[state])   # or Q if using single Q-learning
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    successes += reward

print(successes / 1000)