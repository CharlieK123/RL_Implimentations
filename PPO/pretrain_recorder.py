"""
State recorder for building a JEPA pretraining dataset from a PPO run.

The idea (see the plan): instead of rolling out a frozen policy — which only ever
visits the narrow "expert standup" path — we log STATES while PPO trains. A learning
policy is stochastic by construction and sweeps random -> competent, so the resulting
corpus spans the whole spectrum of body dynamics for free. That variety is exactly what
a state-space world model / JEPA wants.

This records STATES ONLY (no actions/rewards). Two subtleties it handles:

1. RAW observations. The training env normalises observations with a running mean/var
   that DRIFTS during training, so a stored normalised vector means different things at
   different updates. We must store the un-normalised obs. gymnasium's
   `NormalizeObservation.observations()` is handed the raw obs before it normalises, so
   we subclass it and stash a copy. This is exact (no lossy inversion of the [-10,10]
   clip) and does not change training dynamics.

2. Fine temporal sampling with intact adjacency. A JEPA needs consecutive s_t -> s_{t+1},
   so we can NOT keep "every Nth timestep". Instead we grab a short CONTIGUOUS slice
   (`steps_per_shard` steps) from the start of the rollout every `record_every` updates.
   Each slice is internally contiguous (usable for prediction); slices are spread finely
   across training so the corpus tracks the random -> expert curriculum closely.

The only bookkeeping stored alongside the states is a single `dummy` bool per step (the
gymnasium NEXT_STEP autoreset marker). It is enough to reconstruct episode boundaries:
`dummy[t+1]` is true exactly when the episode ended at step t, so a window is valid iff
none of its steps are dummies (see pretrain_dataset.py).
"""

import json
import uuid
from pathlib import Path

import numpy as np
import gymnasium as gym


class RecordingNormalizeObservation(gym.wrappers.vector.NormalizeObservation):
    """NormalizeObservation that remembers the last RAW (pre-normalisation) obs batch.

    `observations()` receives the raw obs, updates the running stats, and returns the
    normalised obs. We copy the raw batch before delegating so the training loop can read
    the true environment observation for step t.
    """

    def observations(self, observations):
        # copy: the wrapper/env may reuse the underlying buffer on the next step.
        self.last_raw_obs = np.asarray(observations, dtype=np.float32).copy()
        return super().observations(observations)


class StateRecorder:
    """Buffers raw states and flushes short contiguous slices as .npz shards.

    A "shard" == one recorded slice, stored time-major with per-env columns:
        obs    float16 [T, N, obs_dim]   raw (un-normalised) observations
        dummy  bool    [T, N]            gymnasium NEXT_STEP autoreset step

    A slice is captured every `record_every` updates (fine temporal sampling of the
    random -> expert curriculum), taking the first `steps_per_shard` steps of that
    rollout (contiguous, so s_t -> s_{t+1} adjacency survives down each env column).
    """

    def __init__(
        self,
        out_dir,
        num_envs,
        obs_dim,
        env_id,
        record_every=10,
        steps_per_shard=256,
        max_shards=None,
        store_dtype=np.float16,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.env_id = env_id
        self.record_every = record_every
        self.steps_per_shard = steps_per_shard
        self.max_shards = max_shards
        self.store_dtype = store_dtype

        # Per-step buffers for the slice currently being recorded (None = not recording).
        self._buf = None

        # Bookkeeping for an honest end-of-run summary (no silent truncation).
        self.shards_written = 0
        self.states_written = 0

        # Write the (static) schema now so the dataset is self-describing even if
        # training is interrupted before finalize().
        self._write_schema()

    def should_record(self, update_idx):
        """True iff a slice should be captured from this update's rollout."""
        if self.max_shards is not None and self.shards_written >= self.max_shards:
            return False
        return (update_idx % self.record_every) == 0

    def start_rollout(self):
        """Begin buffering a fresh slice."""
        self._buf = {"obs": [], "dummy": []}

    def record_step(self, raw_obs, dummy):
        """Buffer one env-step of states. Stops once the slice reaches steps_per_shard."""
        if self._buf is None:
            return
        if len(self._buf["obs"]) >= self.steps_per_shard:
            return  # slice for this rollout is complete; ignore the rest
        self._buf["obs"].append(np.asarray(raw_obs, dtype=self.store_dtype))
        self._buf["dummy"].append(np.asarray(dummy, dtype=bool))

    def slice_complete(self):
        """True once the current slice has captured steps_per_shard steps."""
        return self._buf is not None and len(self._buf["obs"]) >= self.steps_per_shard

    def flush_shard(self, update_idx):
        """Write the buffered slice to a UUID-named .npz and clear the buffer."""
        if self._buf is None or len(self._buf["obs"]) == 0:
            self._buf = None
            return

        obs = np.stack(self._buf["obs"])      # [T, N, obs_dim]
        dummy = np.stack(self._buf["dummy"])  # [T, N]

        path = self.out_dir / f"{uuid.uuid4()}.npz"
        np.savez_compressed(
            path,
            obs=obs,
            dummy=dummy,
            update_idx=np.int64(update_idx),
            env_id=np.array(self.env_id),
        )

        self.shards_written += 1
        self.states_written += obs.shape[0] * obs.shape[1]
        self._buf = None
        cap = f"/{self.max_shards}" if self.max_shards is not None else ""
        print(
            f"[recorder] slice {self.shards_written}{cap} (update {update_idx}) "
            f"-> {path.name}  T={obs.shape[0]} N={obs.shape[1]}  "
            f"total states={self.states_written:,}"
        )

    def save_obs_rms(self, obs_rms):
        """Persist the current obs running mean/var/count. Cheap; call after each flush
        so the normalization stats stay current even if training is interrupted."""
        if obs_rms is None:
            return
        np.savez(
            self.out_dir / "obs_rms.npz",
            mean=np.asarray(obs_rms.mean, dtype=np.float32),
            var=np.asarray(obs_rms.var, dtype=np.float32),
            count=np.asarray(obs_rms.count, dtype=np.float64),
        )

    def _write_schema(self):
        schema = {
            "arrays": {
                "obs": f"{self.store_dtype().dtype.name} [T, N, {self.obs_dim}]  raw (un-normalised) observations (states)",
                "dummy": "bool [T, N]  gymnasium NEXT_STEP autoreset step (episode-boundary marker)",
                "update_idx": "int64 []  PPO update this slice came from (0=start of training)",
                "env_id": "str []  gymnasium env id",
            },
            "layout": (
                "STATE-ONLY dataset. One .npz per recorded slice. Time-major with per-env "
                "columns so s_t -> s_{t+1} adjacency is preserved down each env stream. Axis "
                "N is the parallel env index; episodes in different columns are unrelated and "
                "interleave over training."
            ),
            "window_rule": (
                "A valid length-(L+1) window obs[t .. t+L] down one env column exists iff "
                "dummy[t .. t+L] are ALL False. dummy[k+1] is true exactly when the episode "
                "ended at step k, so this keeps every window inside one episode and never "
                "crosses an autoreset. (No terminated/truncated arrays are needed — dummy "
                "encodes the same boundaries.)"
            ),
            "obs_normalization": (
                "obs is RAW. obs_rms.npz holds the running mean/var/count from the latest "
                "flush; apply (obs - mean)/sqrt(var + 1e-8) then clip(-10, 10) to match what "
                "the policy saw, or use raw obs directly for the world model."
            ),
            "notes": (
                "States logged DURING PPO training, so low-update_idx slices are near-random "
                "behaviour and later slices are competent standups. A short contiguous slice "
                "is captured every `record_every` updates (fine temporal sampling), NOT by "
                "time-subsampling within a rollout (which would break s_t -> s_{t+1})."
            ),
        }
        with open(self.out_dir / "schema.json", "w") as f:
            json.dump(schema, f, indent=2)

    def finalize(self, obs_rms=None):
        """Refresh obs_rms.npz and print an end-of-run summary."""
        self.save_obs_rms(obs_rms)
        print(
            f"[recorder] done: {self.shards_written} slices, "
            f"{self.states_written:,} states written to {self.out_dir}. "
            f"(one {self.steps_per_shard}-step slice every {self.record_every} updates.)"
        )
