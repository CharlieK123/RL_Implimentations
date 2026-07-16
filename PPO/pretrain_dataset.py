"""
Dataset loader for the HumanoidStandup JEPA pretraining corpus (STATE-ONLY).

Wraps the .npz slices written by `env_continous.py`'s recorder in a
`torch.utils.data.Dataset` that yields fixed-length windows of STATES which never cross
an episode boundary or a gymnasium NEXT_STEP autoreset step.

- seq_len=1 (default): each sample is a pair
      state      [obs_dim]   (s_t)
      next_state [obs_dim]   (s_{t+1})
  i.e. the one-step target a state-predictive JEPA learns.

- seq_len=k: each sample is a length-(k+1) trajectory of states
      states [k+1, obs_dim]   (s_t ... s_{t+k})
  for a multi-step JEPA context/target split.

Usage:
    from torch.utils.data import DataLoader
    ds = HumanoidJEPADataset("pretrain_data/states", seq_len=1)
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HumanoidJEPADataset(Dataset):
    def __init__(self, out_dir, seq_len=1, normalize=False):
        self.out_dir = Path(out_dir)
        self.seq_len = int(seq_len)
        assert self.seq_len >= 1

        self.shard_paths = sorted(self.out_dir.glob("*.npz"))
        # obs_rms.npz is metadata, not a slice.
        self.shard_paths = [p for p in self.shard_paths if p.name != "obs_rms.npz"]
        if not self.shard_paths:
            raise FileNotFoundError(f"No .npz slices found in {self.out_dir}")

        # Optional normalization stats (raw obs -> what the policy saw).
        self.mean = self.std = None
        if normalize:
            rms_path = self.out_dir / "obs_rms.npz"
            if not rms_path.exists():
                raise FileNotFoundError(f"normalize=True but {rms_path} is missing")
            rms = np.load(rms_path)
            self.mean = rms["mean"].astype(np.float32)
            self.std = np.sqrt(rms["var"].astype(np.float32) + 1e-8)

        # Lazily-opened slice arrays, keyed by index (mmap so we don't load all at once).
        self._cache = {}

        # Flat index of valid window starts: (shard_idx, env_idx, t). A length-(L+1)
        # window obs[t .. t+L] is valid iff dummy[t .. t+L] are all False — this keeps both
        # endpoints of every s_i -> s_{i+1} pair inside one episode (dummy[k+1] marks that
        # the episode ended at step k, so no window ever crosses a reset).
        self._index = []
        for s_idx, path in enumerate(self.shard_paths):
            with np.load(path) as d:
                dummy = d["dummy"]  # [T, N]
            T, N = dummy.shape
            L = self.seq_len
            for env in range(N):
                for t in range(T - L):
                    if not dummy[t : t + L + 1, env].any():
                        self._index.append((s_idx, env, t))

        self._index = np.asarray(self._index, dtype=np.int64)

    def _shard(self, s_idx):
        arr = self._cache.get(s_idx)
        if arr is None:
            arr = np.load(self.shard_paths[s_idx], mmap_mode="r")
            self._cache[s_idx] = arr
        return arr

    def _maybe_normalize(self, obs):
        if self.mean is None:
            return obs
        return np.clip((obs - self.mean) / self.std, -10.0, 10.0)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        s_idx, env, t = (int(x) for x in self._index[idx])
        d = self._shard(s_idx)
        L = self.seq_len

        obs = np.asarray(d["obs"][t : t + L + 1, env], dtype=np.float32)  # [L+1, obs_dim]
        obs = self._maybe_normalize(obs)

        if L == 1:
            return {
                "state": torch.from_numpy(obs[0]),
                "next_state": torch.from_numpy(obs[1]),
            }
        return {"states": torch.from_numpy(obs)}  # [L+1, obs_dim]


def _smoke_test():
    out_dir = Path(__file__).parent / "pretrain_data" / "states"

    ds = HumanoidJEPADataset(out_dir, seq_len=1)
    print(f"seq_len=1: {len(ds):,} state pairs across {len(ds.shard_paths)} slices")
    sample = ds[0]
    print("  state:", tuple(sample["state"].shape),
          "next_state:", tuple(sample["next_state"].shape))

    ds5 = HumanoidJEPADataset(out_dir, seq_len=5)
    print(f"seq_len=5: {len(ds5):,} windows")
    s5 = ds5[0]
    print("  states:", tuple(s5["states"].shape))


if __name__ == "__main__":
    _smoke_test()
