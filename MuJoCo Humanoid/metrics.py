import torch
import pickle

history = {
    "update": [],
    "episode_ema": [],
    "episode_return": [],
    "policy_loss": [],
    "value_loss": [],
    "entropy": [],
    "ev": []
}


def ppo_log(i, p_loss, v_loss, ent, ev):
    history['update'].append(i)
    history['policy_loss'].append(p_loss)
    history['value_loss'].append(v_loss)
    history['entropy'].append(ent)
    history['ev'].append(ev)

def env_log(ema, ep):
    history['episode_ema'].append(ema)
    history['episode_return'].append(ep)



def save_history(path="training_history.pkl"):
    with open(path, "wb") as f:
        pickle.dump(history, f)

def load_history(path="training_history.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)