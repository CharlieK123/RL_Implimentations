from pathlib import Path

import numpy as np


PATH = Path(
    r"C:\Users\charl\PycharmProjects\jepa_test\data"
    r"\stackcube_jepa.npz"
)

data = np.load(PATH)

for key in data.files:
    print(f"{key:20s} {data[key].shape} {data[key].dtype}")

states = data["states"]
actions = data["actions"]
next_states = data["next_states"]
episode_ids = data["episode_ids"]
timesteps = data["timesteps"]

assert len(states) == len(actions) == len(next_states)
assert len(states) == len(episode_ids) == len(timesteps)
assert states.shape[1] == 48
assert actions.shape[1] == 8
assert np.isfinite(states).all()
assert np.isfinite(actions).all()
assert np.isfinite(next_states).all()

# Verify transitions never wrap between episodes.
same_episode = episode_ids[:-1] == episode_ids[1:]
consecutive_time = timesteps[1:] == timesteps[:-1] + 1

print("\nTransitions:", len(states))
print("Episodes:", len(np.unique(episode_ids)))
print("State range:", states.min(), states.max())
print("Action range:", actions.min(), actions.max())
print(
    "Valid consecutive pairs:",
    np.sum(same_episode & consecutive_time),
)