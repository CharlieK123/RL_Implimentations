import pickle
import time
import threading
import queue
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from PPO_model import PPO


BASE = Path(__file__).parent
HISTORY_FILE = BASE / "training_history.pkl"
CKPT_FILE = BASE / "policy.pt"

"""
CHANGE THE ID BELOW TO THE NEW ENVIRONMENT NAME FROM GYMNASIUM DOCS
"""
ENV_ID = "LunarLander-v3"

"""
THESE NN HYPERPARAMS MUST MATCH WHAT YOU CHOOSE IN THE ENV FILE
YOU CAN FIND THE OBS AND ACT DIM OF A GIVEN ENV AT THE TOP OF THE DOC ON GYMNASIUM 

also if you get a shape error it will usually say what the needed shape is 
"""
OBS_DIM = 8
ACT_DIM = 4

HIDDEN_DIM = 256
NUM_HIDDEN = 2
DISCRETE = True  # make sure this is right

# these dont matter
GAMMA = 0.99
LAMBDA = 0.95
EPS = 0.2
ENT_COEF = 0.005
VALUE_COEF = 0.5
EPOCHS = 5
LR = 1e-4
MINIBATCH = 256


def load_history():
    with open(HISTORY_FILE, "rb") as f:
        return pickle.load(f)


def safe(history, key):
    return history.get(key, [])


def make_agent():
    return PPO(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_hidden=NUM_HIDDEN,
        gamma=GAMMA,
        lmbda=LAMBDA,
        eps=EPS,
        ent_coef=ENT_COEF,
        value_coef=VALUE_COEF,
        epochs=EPOCHS,
        policy_lr=LR,
        value_lr=LR,
        minibatch=MINIBATCH,
        multi_envs=False,
        discrete=DISCRETE,
    )


def load_checkpoint(agent):
    if not CKPT_FILE.exists():
        print(f"No checkpoint found at: {CKPT_FILE}")
        return False, None

    ckpt = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
    print("Loaded checkpoint:", CKPT_FILE.resolve())
    print("Checkpoint keys:", list(ckpt.keys()))

    if "policy_state_dict" in ckpt:
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
    elif "policy" in ckpt:
        agent.policy_net.load_state_dict(ckpt["policy"])
    else:
        print("Checkpoint has no policy weights.")
        return False, None

    if "value_state_dict" in ckpt:
        agent.value_net.load_state_dict(ckpt["value_state_dict"])
    elif "value" in ckpt:
        agent.value_net.load_state_dict(ckpt["value"])

    if not agent.discrete and "log_std" in ckpt:
        agent.log_std.data.copy_(ckpt["log_std"])

    if "obs_rms" in ckpt:
        obs_stats = {
            "mean": np.asarray(ckpt["obs_rms"]["mean"]),
            "var": np.asarray(ckpt["obs_rms"]["var"]),
            "count": ckpt["obs_rms"].get("count", 1.0),
        }
    elif "obs_mean" in ckpt and "obs_var" in ckpt:
        obs_stats = {
            "mean": np.asarray(ckpt["obs_mean"]),
            "var": np.asarray(ckpt["obs_var"]),
            "count": ckpt.get("obs_count", 1.0),
        }
    else:
        print("No obs normalization stats found. Using raw observations.")
        obs_stats = None

    return True, obs_stats


def normalize_obs(obs, obs_stats):
    if obs_stats is None:
        return obs

    obs = (obs - obs_stats["mean"]) / np.sqrt(obs_stats["var"] + 1e-8)
    return np.clip(obs, -10.0, 10.0)


def watch_agent(env, agent, obs_stats, episodes=3, max_seconds=30):
    start_time = time.time()

    for ep in range(episodes):
        if time.time() - start_time >= max_seconds:
            break

        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            if time.time() - start_time >= max_seconds:
                return

            obs_input = normalize_obs(obs, obs_stats)
            obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32)

            with torch.no_grad():
                if agent.discrete:
                    logits = agent.policy_net(obs_tensor)
                    action = torch.argmax(logits).item()
                else:
                    mean = agent.policy_net(obs_tensor)
                    action = torch.tanh(mean).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_return += float(reward)
            ep_len += 1

        print(f"Watched episode {ep + 1}: return={ep_return:.2f}, len={ep_len}")


def watch_current_policy():
    agent = make_agent()
    loaded, obs_stats = load_checkpoint(agent)

    if not loaded:
        return

    env = gym.make(ENV_ID, render_mode="human")

    try:
        watch_agent(env, agent, obs_stats, episodes=3, max_seconds=30)
    finally:
        env.close()


def terminal_listener(cmd_queue):
    while True:
        try:
            cmd = input("Type w + Enter to watch current policy: ").strip().lower()
            cmd_queue.put(cmd)
        except EOFError:
            break


def plot_series(ax, x, y, title, xlabel, ylim=None, alpha=1.0):
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    if y is None or len(y) == 0:
        return

    if x is not None and len(x) > 0:
        ax.plot(x[:len(y)], y, alpha=alpha)
    else:
        ax.plot(range(len(y)), y, alpha=alpha)

    if ylim is not None:
        ax.set_ylim(*ylim)


def main():
    cmd_queue = queue.Queue()

    threading.Thread(
        target=terminal_listener,
        args=(cmd_queue,),
        daemon=True,
    ).start()

    plt.ion()

    fig = plt.figure(figsize=(13, 13))
    gs = GridSpec(4, 2, figure=fig)

    ax_policy = fig.add_subplot(gs[0, 0])
    ax_value = fig.add_subplot(gs[0, 1])
    ax_ema = fig.add_subplot(gs[1, 0])
    ax_return = fig.add_subplot(gs[1, 1])
    ax_ev = fig.add_subplot(gs[2, 0])
    ax_entropy = fig.add_subplot(gs[2, 1])
    ax_clip = fig.add_subplot(gs[3, 0])
    ax_kl = fig.add_subplot(gs[3, 1])

    axes = [
        ax_policy,
        ax_value,
        ax_ema,
        ax_return,
        ax_ev,
        ax_entropy,
        ax_clip,
        ax_kl,
    ]

    plt.show(block=False)

    while plt.fignum_exists(fig.number):
        try:
            while not cmd_queue.empty():
                cmd = cmd_queue.get_nowait()

                if cmd == "w":
                    watch_current_policy()

            history = load_history()

            updates = safe(history, "update")
            policy_loss = safe(history, "policy_loss")
            value_loss = safe(history, "value_loss")
            episode_ema = safe(history, "episode_ema")
            episode_return = safe(history, "episode_return")
            ev = safe(history, "ev")
            entropy = safe(history, "entropy")
            clip = safe(history, "clip")
            kl = safe(history, "kl")

            for ax in axes:
                ax.clear()

            plot_series(ax_policy, updates, policy_loss, "Policy loss", "Update")
            plot_series(ax_value, updates, value_loss, "Value loss", "Update")

            plot_series(ax_ema, None, episode_ema, "Episode EMA", "Episode")
            plot_series(
                ax_return,
                None,
                episode_return,
                "Episode return",
                "Episode",
                alpha=0.35,
            )

            plot_series(
                ax_ev,
                updates,
                ev,
                "Explained variance",
                "Update",
                ylim=(-0.2, 1.05),
            )

            plot_series(ax_entropy, updates, entropy, "Entropy", "Update")

            plot_series(
                ax_clip,
                updates,
                clip,
                "Clip fraction",
                "Update",
                ylim=(0.0, 1.0),
            )

            plot_series(ax_kl, updates, kl, "Approx KL", "Update")

            ax_kl.axhline(0.01, linestyle="--", linewidth=1)
            ax_kl.axhline(0.03, linestyle="--", linewidth=1)
            ax_kl.axhline(0.05, linestyle="--", linewidth=1)

            plt.tight_layout()
            plt.pause(1)

        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            print("Waiting for valid history...")
            time.sleep(1)

        except KeyboardInterrupt:
            break

    plt.close(fig)


if __name__ == "__main__":
    main()