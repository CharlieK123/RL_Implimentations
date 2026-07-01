import gymnasium as gym
import numpy as np
from PPO_model import PPO
import torch
import time

env = gym.make("LunarLander-v3")
render_env = gym.make("LunarLander-v3", render_mode="human")

def watch_agent(env, agent, episodes=3, max_seconds=10):
    start_time = time.time()

    for _ in range(episodes):
        if time.time() - start_time >= max_seconds:
            break

        obs, _ = env.reset()
        done = False
        ep_return = 0

        while not done:
            if time.time() - start_time >= max_seconds:
                return

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                logits = agent.policy_net(obs_tensor)
                action = torch.argmax(logits).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        print(f"watched episode return: {ep_return}")

obs_dim = env.observation_space.shape[0]   # 4
act_dim = env.action_space.n               # 2

agent = PPO(
    obs_dim=obs_dim,
    act_dim=act_dim,
    hidden_dim=64,
    num_hidden=2,
    gamma=0.99,
    lmbda=0.95,
    eps=0.2,
    ent_coef=0.01,
    value_coef=0.5,
    epochs=4,
    lr=3e-4,
    minibatch=64,
)

updates = 10000
rollout_steps = 1000

for i in range(updates):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    episode_reward = 0
    episode_all_rewards = []

    obs, _ = env.reset()
    
    for _ in range(rollout_steps):
        action, log_prob, value = agent.get_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        states.append(torch.as_tensor(obs, dtype=torch.float32))
        actions.append(torch.as_tensor(action))
        rewards.append(torch.as_tensor(reward, dtype=torch.float32))
        dones.append(torch.as_tensor(done, dtype=torch.float32))
        log_probs.append(log_prob)
        values.append(value)

        obs = next_obs

        if done:
            obs, _ = env.reset()
            episode_all_rewards.append(episode_reward)
            if episode_reward == 500: print(f'500 at: {i}')
            episode_reward = 0
        
    with torch.no_grad():
        next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        next_value = agent.value_net(next_obs_tensor).squeeze(-1)

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)

    advantages, returns = agent.gae(rewards, values, dones, next_value)
    agent.update(states, actions, log_probs, advantages, returns)

    if i % 100 == 0:
        print(episode_all_rewards, sum(episode_all_rewards))
        if i % 500 == 0:
            print(i)
            watch_agent(render_env, agent)





