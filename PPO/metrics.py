import os
import pickle
from pathlib import Path

# Absolute path so it doesn't matter what directory a training run is launched
# from -- the history always lives next to this file, which is where reader.py
# looks for it. (Previously this was a relative path, so a run launched from a
# different cwd wrote its history somewhere reader never read.)
HISTORY_PATH = Path(__file__).parent / "training_history.pkl"

history = {
    "update": [],
    "episode_ema": [],
    "episode_return": [],
    "policy_loss": [],
    "value_loss": [],
    "entropy": [],
    "ev": [],
    'kl': [],
    'clip': []
}


def reset_history(path=HISTORY_PATH):
    """Wipe in-memory metrics AND the on-disk file.

    Call this ONCE at the very start of a training run. Without it, the file
    still holds the PREVIOUS run's data until this run reaches its first save,
    so reader.py shows stale numbers (e.g. "9700 updates" while your fresh run
    has done 100). Deleting the file makes reader show "waiting" until this run
    writes its own first save.
    """
    for key in history:
        history[key].clear()
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def ppo_log(i, p_loss, v_loss, ent, ev, kl, clip):
    history['update'].append(i)
    history['policy_loss'].append(p_loss)
    history['value_loss'].append(v_loss)
    history['entropy'].append(ent)
    history['ev'].append(ev)
    history['kl'].append(kl)
    history['clip'].append(clip)




def env_log(ema, ep):
    history['episode_ema'].append(ema)
    history['episode_return'].append(ep)



def save_history(path=HISTORY_PATH):
    with open(path, "wb") as f:
        pickle.dump(history, f)

def load_history(path=HISTORY_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)
