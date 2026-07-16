
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Change this import to wherever your JEPA class and train function live.
from model2 import JEPA, train


DATA_PATH = Path(
    r"C:\Users\charl\PycharmProjects\jepa_test\data"
    r"\stackcube_jepa.npz"
)

HORIZON = 5
BATCH_SIZE = 1024
EPOCHS = 100


class StackCubeDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        horizon: int = 1,
    ) -> None:
        super().__init__()

        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        data = np.load(path)

        if not bool(data["states_are_normalized"]):
            raise ValueError("Expected normalized states")

        if not bool(data["actions_are_normalized"]):
            raise ValueError("Expected normalized actions")

        self.states = torch.from_numpy(
            data["states"]
        ).float()

        self.actions = torch.from_numpy(
            data["actions"]
        ).float()

        self.episode_state_starts = np.asarray(
            data["episode_state_starts"],
            dtype=np.int64,
        )

        self.episode_state_ends = np.asarray(
            data["episode_state_ends"],
            dtype=np.int64,
        )

        self.episode_action_starts = np.asarray(
            data["episode_action_starts"],
            dtype=np.int64,
        )

        self.episode_action_ends = np.asarray(
            data["episode_action_ends"],
            dtype=np.int64,
        )

        self.episode_lengths = np.asarray(
            data["episode_lengths"],
            dtype=np.int32,
        )

        self.horizon = horizon

        self.sample_episode_ids: list[int] = []
        self.sample_timesteps: list[int] = []

        self._validate_storage()
        self._build_sample_index()

    def _validate_storage(self) -> None:
        episode_count = len(self.episode_lengths)

        metadata_arrays = [
            self.episode_state_starts,
            self.episode_state_ends,
            self.episode_action_starts,
            self.episode_action_ends,
        ]

        if any(
            len(array) != episode_count
            for array in metadata_arrays
        ):
            raise ValueError(
                "Episode metadata arrays have mismatched lengths"
            )

        for episode_id in range(episode_count):
            state_start = int(
                self.episode_state_starts[episode_id]
            )
            state_end = int(
                self.episode_state_ends[episode_id]
            )

            action_start = int(
                self.episode_action_starts[episode_id]
            )
            action_end = int(
                self.episode_action_ends[episode_id]
            )

            episode_length = int(
                self.episode_lengths[episode_id]
            )

            state_count = state_end - state_start
            action_count = action_end - action_start

            if action_count != episode_length:
                raise ValueError(
                    f"Episode {episode_id}: "
                    f"expected {episode_length} actions, "
                    f"got {action_count}"
                )

            if state_count != episode_length + 1:
                raise ValueError(
                    f"Episode {episode_id}: "
                    f"expected {episode_length + 1} states, "
                    f"got {state_count}"
                )

    def _build_sample_index(self) -> None:
        for episode_id, episode_length in enumerate(
            self.episode_lengths
        ):
            episode_length = int(episode_length)

            if episode_length < self.horizon:
                continue

            valid_sample_count = (
                episode_length
                - self.horizon
                + 1
            )

            for timestep in range(valid_sample_count):
                self.sample_episode_ids.append(
                    episode_id
                )
                self.sample_timesteps.append(
                    timestep
                )

        if not self.sample_episode_ids:
            raise RuntimeError(
                f"No valid samples for horizon={self.horizon}"
            )

    def __len__(self) -> int:
        return len(self.sample_episode_ids)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        episode_id = self.sample_episode_ids[index]
        timestep = self.sample_timesteps[index]

        state_start = int(
            self.episode_state_starts[episode_id]
        )

        action_start = int(
            self.episode_action_starts[episode_id]
        )

        state_t_index = (
            state_start
            + timestep
        )

        future_state_index = (
            state_start
            + timestep
            + self.horizon
        )

        first_action_index = (
            action_start
            + timestep
        )

        final_action_index = (
            first_action_index
            + self.horizon
        )

        state_t = self.states[state_t_index]

        action_sequence = self.actions[
            first_action_index:final_action_index
        ]

        state_tk = self.states[
            future_state_index
        ]

        if action_sequence.shape != (
            self.horizon,
            self.actions.shape[1],
        ):
            raise RuntimeError(
                "Incorrect action sequence shape: "
                f"got {action_sequence.shape}"
            )

        # Existing action encoder expects a flat vector.
        #
        # [horizon, action_dim]
        # becomes
        # [horizon * action_dim]
        flattened_actions = action_sequence.flatten()

        return (
            state_t,
            flattened_actions,
            state_tk,
        )


def main() -> None:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = StackCubeDataset(
        path=DATA_PATH,
        horizon=HORIZON,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=device == "cuda",
        drop_last=True,
    )

    state, actions, future_state = next(
        iter(loader)
    )

    print(f"Device:             {device}")
    print(f"Horizon:            {HORIZON}")
    print(f"Dataset samples:    {len(dataset):,}")
    print(f"State batch:        {state.shape}")
    print(f"Action batch:       {actions.shape}")
    print(f"Future state batch: {future_state.shape}")

    action_dim = 8
    flattened_action_dim = (
        HORIZON * action_dim
    )

    model = JEPA(
        latent_dim=48,
        encoder_params=(
            2,      # transformer blocks
            48,     # residual / latent dimension
            128,    # transformer FFN hidden dimension
            4,      # attention heads
        ),
        projection_params=(
            2,      # predictor hidden layers
            128,    # predictor hidden dimension
        ),
        momentum=0.995,
        ac=(
            flattened_action_dim,
            32,
            1,
            64,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.0,
    )

    train(
        model=model,
        epochs=EPOCHS,
        loader=loader,
        optim=optimizer,
        device=device,
    )


if __name__ == "__main__":
    main()

