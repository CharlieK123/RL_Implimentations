import gymnasium as gym
import numpy as np
from PPO_model_revised import PPO
import torch
import time
from pathlib import Path
from metrics import env_log, reset_history
from pretrain_recorder import RecordingNormalizeObservation, StateRecorder

# JEPA pretraining-data collection (STATE-ONLY). Recording happens as a side effect of
# training: a short contiguous slice of states (STEPS_PER_SHARD steps) is dumped every
# RECORD_EVERY_UPDATES updates, giving fine temporal coverage of the random->expert
# curriculum. Set RECORD_PRETRAIN_DATA = False to train without saving anything.
RECORD_PRETRAIN_DATA = True
TOTAL_UPDATES = 20_000
RECORD_EVERY_UPDATES = 10        # capture a slice this often (in PPO updates)
STEPS_PER_SHARD = 256            # contiguous steps per slice (out of the 2048-step rollout)
MAX_SHARDS = None                # optional cap on total slices (None = until training ends)
PRETRAIN_OUT_DIR = Path(__file__).parent / "pretrain_data" / "states"


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

num_envs = 16


env_id = 'HumanoidStandup-v5'


def run():
    obs, _ = env.reset()
    updates = TOTAL_UPDATES
    rollout_steps = 2048

    # Tracks which envs were done on the previous step. Under gymnasium NEXT_STEP
    # autoreset, the step immediately after a done is a reset step whose action was
    # ignored, so we flag it as a "dummy" transition. Persists across rollouts
    # because obs (and the underlying env state) is continuous across updates.
    prev_done = np.zeros(num_envs, dtype=bool)

    episode_reward = np.zeros(num_envs)
    latest_ep_reward = np.nan
    episode_ema = np.nan
    alpha = 0.02

    reset_history()   # clear any previous run's metrics so reader shows only THIS run
    start = time.time()
    for i in range(updates):
        states, actions, rewards, terminateds, truncateds, dummies, log_probs, values = [], [], [], [], [], [], [], []

        # Record this whole rollout to the pretraining dataset iff it's a chosen update.
        recording = recorder is not None and recorder.should_record(i)
        if recording:
            recorder.start_rollout()

        for _ in range(rollout_steps):
            # Record the RAW (un-normalised) state for THIS step, stashed by
            # RecordingNormalizeObservation on the previous step / reset. prev_done is the
            # dummy (autoreset) flag. Slice ends once STEPS_PER_SHARD states are captured.
            if recording and not recorder.slice_complete():
                recorder.record_step(raw_obs=obs_norm.last_raw_obs.copy(), dummy=prev_done)

            action, raw_action, log_prob, value, mean, std = agent.get_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            states.append(torch.as_tensor(obs, dtype=torch.float32))
            actions.append(torch.as_tensor(raw_action))
            rewards.append(torch.as_tensor(reward, dtype=torch.float32))
            # terminated vs truncated kept separate: truncation must still bootstrap.
            terminateds.append(torch.as_tensor(terminated, dtype=torch.float32))
            truncateds.append(torch.as_tensor(truncated, dtype=torch.float32))
            # this step is a NEXT_STEP autoreset dummy iff the env was done last step.
            dummies.append(torch.as_tensor(prev_done, dtype=torch.float32))
            log_probs.append(log_prob)
            values.append(value)

            prev_done = done
            episode_reward += reward
            obs = next_obs

            # eval
            if True in done:  # check to see if at least one env terminated
                if "episode" in info:
                    finished = info["_episode"]

                    for raw_return in info["episode"]["r"][finished]:
                        if np.isnan(episode_ema):
                            episode_ema = raw_return
                        else:
                            episode_ema = (1 - alpha) * episode_ema + alpha * raw_return

                        latest_ep_reward = raw_return
                        env_log(episode_ema, latest_ep_reward)

        if recording:
            recorder.flush_shard(i)
            recorder.save_obs_rms(obs_norm.obs_rms)  # keep stats current if interrupted

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            next_value = agent.value_net(next_obs_tensor).squeeze(-1)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        terminateds = torch.stack(terminateds)
        truncateds = torch.stack(truncateds)
        dummies = torch.stack(dummies)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantages, returns = agent.gae(rewards, values, terminateds, next_value, truncateds)

        agent.update(states, actions, log_probs, advantages, returns, dummies, obs_norm, rew_norm)

        if i % 10 == 0:
            print(f'iteration: {i}, ema: {episode_ema}, noisy episode: {latest_ep_reward}')
            print(torch.mean(std, dim=0))
            print(torch.mean(mean, dim=0))
            #print(f'run mean: {agent.return_norm.running_mean}, run std: {agent.return_norm.std}, count: {agent.return_norm.count} \n')
            fig = torch.quantile(values.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=value.device))
            print(f'value 5 fig: {fig}, val var: {torch.var(values)}')
            print(f'Total timesteps: {rollout_steps * (i+1) * num_envs:,}, time for update: {time.time()-start:.3f} sec')
            start = time.time()
            print('-------------------------\n\n')
            print("returns:", returns.mean().item(), returns.std().item())
            print("values :", values.mean().item(), values.std().item())
            #print(np.mean(obs, axis=0))

    if recorder is not None:
        recorder.finalize(obs_norm.obs_rms)


if __name__ == "__main__":

    def make_env(env_id):
        def thunk():
            return gym.make(env_id)

        return thunk


    render_env = gym.make(env_id, render_mode="human")

    env = gym.vector.AsyncVectorEnv(
        [make_env(env_id) for _ in range(num_envs)]
    )

    env = gym.wrappers.vector.RecordEpisodeStatistics(env)

    # RecordingNormalizeObservation stashes the raw (pre-normalisation) obs so the dataset
    # stores unnormalised states (behaviour is otherwise identical to the plain wrapper).
    obs_norm = RecordingNormalizeObservation(env)

    env = gym.wrappers.vector.TransformObservation(
        obs_norm,
        lambda obs: obs.clip(-10, 10),
        obs_norm.single_observation_space,
    )

    rew_norm = gym.wrappers.vector.NormalizeReward(env, gamma=0.99)

    env = gym.wrappers.vector.TransformReward(
        rew_norm,
        lambda r: r.clip(-10, 10),
    )
    obs_dim = env.single_observation_space.shape[0]  # 4
    act_dim = env.single_action_space.shape[0]  # 2

    agent = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=256,
        num_hidden=2,
        gamma=0.99,
        lmbda=0.95,
        eps=0.2,
        ent_coef=0.001,
        value_coef=0.5,
        epochs=10,
        policy_lr=1e-4,
        value_lr=1e-4,
        minibatch=256,
        multi_envs=True,
        discrete=False,
        decay_actor_lr=(1e-4, 1e-5, 500),
        decay_critic_lr=(1e-4, 1e-5, 500),
        decay_ent=(0.001, 0.000, 500),
        kl_target=0.03
    )

    recorder = (
        StateRecorder(
            out_dir=PRETRAIN_OUT_DIR,
            num_envs=num_envs,
            obs_dim=obs_dim,
            env_id=env_id,
            record_every=RECORD_EVERY_UPDATES,
            steps_per_shard=STEPS_PER_SHARD,
            max_shards=MAX_SHARDS,
        )
        if RECORD_PRETRAIN_DATA
        else None
    )

    run()


