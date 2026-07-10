import gymnasium as gym
import numpy as np
from PPO_model import PPO
import torch
import time
from metrics import env_log

def make_env(env_id):
    def thunk():
        return gym.make(env_id)
    return thunk

env_id = "LunarLander-v3"
num_envs = 8
render_env = gym.make("LunarLander-v3", render_mode="human")

env = gym.vector.SyncVectorEnv(
    [make_env(env_id) for _ in range(num_envs)]
)



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


def update_reward_ema(done_list, ep_ema, ep_reward, a):
    for j in range(len(done_list)):

        if done_list[j]:  # env[i] has terminated and the ema should adjust
            episode_return = ep_reward[j]

            if ep_ema is np.nan:
                ep_ema = episode_return

            else:
                ep_ema = (1 - a) * ep_ema + (a * episode_return)
            ep_reward[j] = 0

    return ep_ema, ep_reward, episode_return


obs_dim = env.single_observation_space.shape[0]  # 4
act_dim = env.single_action_space.n  # 2

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
    epochs=5,
    policy_lr=3e-4,
    value_lr=3e-4,
    minibatch=64,
    multi_envs=True,
    discrete=True
)

episode_reward = np.zeros(num_envs)
latest_ep_reward = np.nan
episode_ema = np.nan
alpha = 0.02

obs, _ = env.reset()
updates = 10000
rollout_steps = 500

start = time.time()
for i in range(updates):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    for _ in range(rollout_steps):
        action, log_prob, value = agent.get_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated

        states.append(torch.as_tensor(obs, dtype=torch.float32))
        actions.append(torch.as_tensor(action))
        rewards.append(torch.as_tensor(reward, dtype=torch.float32))
        dones.append(torch.as_tensor(done, dtype=torch.float32))
        log_probs.append(log_prob)
        values.append(value)

        episode_reward += reward
        obs = next_obs

        # eval
        if True in done:  # check to see if at least one env terminated
            episode_ema, episode_reward, latest_ep_reward = update_reward_ema(done, episode_ema, episode_reward, alpha)
            env_log(episode_ema, latest_ep_reward)

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

    if i % 10 == 0:
        print(f'iteration: {i}, ema: {episode_ema}, noisy episode: {latest_ep_reward} \n')



final_agent = agent
