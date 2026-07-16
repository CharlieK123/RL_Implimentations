from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


INPUT_PATH = Path(
    r"C:\Users\charl\.maniskill\demos\StackCube-v1\motionplanning"
    r"\trajectory.state.pd_joint_pos.physx_cpu.h5"
)

OUTPUT_PATH = Path(
    r"C:\Users\charl\PycharmProjects\jepa_test\data"
    r"\stackcube_jepa.npz"
)


ACTION_LOW = np.array(
    [
        -2.8973,
        -1.7628,
        -2.8973,
        -3.0718,
        -2.8973,
        -0.0175,
        -2.8973,
        -1.0,
    ],
    dtype=np.float32,
)

ACTION_HIGH = np.array(
    [
        2.8973,
        1.7628,
        2.8973,
        -0.0698,
        2.8973,
        3.7525,
        2.8973,
        1.0,
    ],
    dtype=np.float32,
)


def trajectory_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError as error:
        raise ValueError(
            f"Unexpected trajectory name: {name}"
        ) from error


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(
            "Expected actions with shape [T, action_dim], "
            f"got {actions.shape}"
        )

    if actions.shape[1] != ACTION_LOW.shape[0]:
        raise ValueError(
            f"Expected action dimension {ACTION_LOW.shape[0]}, "
            f"got {actions.shape[1]}"
        )

    action_range = ACTION_HIGH - ACTION_LOW

    if np.any(action_range <= 0):
        raise ValueError(
            "Every action high bound must exceed its low bound"
        )

    normalized_actions = (
        2.0 * (actions - ACTION_LOW) / action_range
        - 1.0
    )

    return np.clip(
        normalized_actions,
        -1.0,
        1.0,
    ).astype(np.float32)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_states: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []

    episode_state_starts: list[int] = []
    episode_state_ends: list[int] = []

    episode_action_starts: list[int] = []
    episode_action_ends: list[int] = []

    episode_lengths: list[int] = []
    episode_success: list[bool] = []
    source_episode_ids: list[int] = []

    state_offset = 0
    action_offset = 0
    skipped_trajectories = 0

    with h5py.File(INPUT_PATH, "r") as file:
        trajectory_names = sorted(
            file.keys(),
            key=trajectory_sort_key,
        )

        for trajectory_name in trajectory_names:
            trajectory = file[trajectory_name]

            source_episode_id = trajectory_sort_key(
                trajectory_name
            )

            if (
                "obs" not in trajectory
                or "actions" not in trajectory
            ):
                print(
                    f"Skipping {trajectory_name}: "
                    "missing obs or actions"
                )
                skipped_trajectories += 1
                continue

            observations_node = trajectory["obs"]
            actions_node = trajectory["actions"]

            if not isinstance(
                observations_node,
                h5py.Dataset,
            ):
                print(
                    f"Skipping {trajectory_name}: "
                    "obs is not a flat dataset"
                )
                skipped_trajectories += 1
                continue

            if not isinstance(
                actions_node,
                h5py.Dataset,
            ):
                print(
                    f"Skipping {trajectory_name}: "
                    "actions is not a dataset"
                )
                skipped_trajectories += 1
                continue

            observations = np.asarray(
                observations_node[:],
                dtype=np.float32,
            )

            raw_actions = np.asarray(
                actions_node[:],
                dtype=np.float32,
            )

            if observations.ndim != 2:
                print(
                    f"Skipping {trajectory_name}: "
                    "expected observations with shape "
                    f"[T+1, state_dim], got {observations.shape}"
                )
                skipped_trajectories += 1
                continue

            if raw_actions.ndim != 2:
                print(
                    f"Skipping {trajectory_name}: "
                    "expected actions with shape "
                    f"[T, action_dim], got {raw_actions.shape}"
                )
                skipped_trajectories += 1
                continue

            num_states = observations.shape[0]
            num_actions = raw_actions.shape[0]

            if num_states != num_actions + 1:
                print(
                    f"Skipping {trajectory_name}: "
                    "expected T+1 states and T actions, "
                    f"got {num_states} states and "
                    f"{num_actions} actions"
                )
                skipped_trajectories += 1
                continue

            if num_actions < 1:
                print(
                    f"Skipping {trajectory_name}: empty episode"
                )
                skipped_trajectories += 1
                continue

            if not np.isfinite(observations).all():
                print(
                    f"Skipping {trajectory_name}: "
                    "non-finite observations"
                )
                skipped_trajectories += 1
                continue

            if not np.isfinite(raw_actions).all():
                print(
                    f"Skipping {trajectory_name}: "
                    "non-finite actions"
                )
                skipped_trajectories += 1
                continue

            if np.any(
                raw_actions < ACTION_LOW - 1e-4
            ):
                print(
                    f"Skipping {trajectory_name}: "
                    "action below controller lower bound"
                )
                skipped_trajectories += 1
                continue

            if np.any(
                raw_actions > ACTION_HIGH + 1e-4
            ):
                print(
                    f"Skipping {trajectory_name}: "
                    "action above controller upper bound"
                )
                skipped_trajectories += 1
                continue

            actions = normalize_actions(raw_actions)

            episode_state_starts.append(state_offset)
            episode_action_starts.append(action_offset)

            episode_states.append(observations)
            episode_actions.append(actions)

            state_offset += num_states
            action_offset += num_actions

            episode_state_ends.append(state_offset)
            episode_action_ends.append(action_offset)

            episode_lengths.append(num_actions)
            source_episode_ids.append(source_episode_id)

            if "success" in trajectory:
                success_values = np.asarray(
                    trajectory["success"][:]
                )
                success = bool(success_values.any())
            else:
                success = False

            episode_success.append(success)

    if not episode_states:
        raise RuntimeError(
            "No valid trajectories were found"
        )

    raw_states = np.concatenate(
        episode_states,
        axis=0,
    ).astype(np.float32)

    actions = np.concatenate(
        episode_actions,
        axis=0,
    ).astype(np.float32)

    state_mean = raw_states.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    state_std = raw_states.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    # Do not amplify nearly constant features.
    state_std = np.where(
        state_std < 1e-4,
        1.0,
        state_std,
    ).astype(np.float32)

    states = (
        (raw_states - state_mean)
        / state_std
    ).astype(np.float32)

    if not np.isfinite(states).all():
        raise RuntimeError(
            "Normalized states contain non-finite values"
        )

    if not np.isfinite(actions).all():
        raise RuntimeError(
            "Normalized actions contain non-finite values"
        )

    episode_state_starts_array = np.asarray(
        episode_state_starts,
        dtype=np.int64,
    )

    episode_state_ends_array = np.asarray(
        episode_state_ends,
        dtype=np.int64,
    )

    episode_action_starts_array = np.asarray(
        episode_action_starts,
        dtype=np.int64,
    )

    episode_action_ends_array = np.asarray(
        episode_action_ends,
        dtype=np.int64,
    )

    episode_lengths_array = np.asarray(
        episode_lengths,
        dtype=np.int32,
    )

    episode_success_array = np.asarray(
        episode_success,
        dtype=np.bool_,
    )

    source_episode_ids_array = np.asarray(
        source_episode_ids,
        dtype=np.int32,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        states=states,
        actions=actions,
        episode_state_starts=episode_state_starts_array,
        episode_state_ends=episode_state_ends_array,
        episode_action_starts=episode_action_starts_array,
        episode_action_ends=episode_action_ends_array,
        episode_lengths=episode_lengths_array,
        episode_success=episode_success_array,
        source_episode_ids=source_episode_ids_array,
        state_mean=state_mean,
        state_std=state_std,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        states_are_normalized=np.array(
            True,
            dtype=np.bool_,
        ),
        actions_are_normalized=np.array(
            True,
            dtype=np.bool_,
        ),
    )

    print("\nConversion complete")
    print(f"Input:               {INPUT_PATH}")
    print(f"Output:              {OUTPUT_PATH}")
    print(
        f"Valid episodes:      "
        f"{len(episode_lengths_array):,}"
    )
    print(
        f"Skipped episodes:    "
        f"{skipped_trajectories:,}"
    )
    print(
        f"Total states:        "
        f"{states.shape[0]:,}"
    )
    print(
        f"Total actions:       "
        f"{actions.shape[0]:,}"
    )
    print(
        f"State dimension:     "
        f"{states.shape[1]}"
    )
    print(
        f"Action dimension:    "
        f"{actions.shape[1]}"
    )
    print(
        f"Shortest episode:    "
        f"{episode_lengths_array.min():,}"
    )
    print(
        f"Longest episode:     "
        f"{episode_lengths_array.max():,}"
    )
    print(
        f"Average length:      "
        f"{episode_lengths_array.mean():.2f}"
    )
    print(
        f"Successful episodes: "
        f"{episode_success_array.sum():,}"
    )

    print("\nNormalized state statistics")
    print(
        f"State mean:          "
        f"{states.mean():.6f}"
    )
    print(
        f"State std:           "
        f"{states.std():.6f}"
    )
    print(
        f"State minimum:       "
        f"{states.min():.6f}"
    )
    print(
        f"State maximum:       "
        f"{states.max():.6f}"
    )

    print("\nNormalized action statistics")
    print(
        f"Action minimum:      "
        f"{actions.min():.6f}"
    )
    print(
        f"Action maximum:      "
        f"{actions.max():.6f}"
    )

    print(
        f"\nFile size:           "
        f"{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MB"
    )


if __name__ == "__main__":
    main()