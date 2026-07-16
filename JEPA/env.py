import gymnasium as gym
import mani_skill

env = gym.make(
    "StackCube-v1",
    obs_mode="state",
    reward_mode="dense",
)

env = gym.make(
    "StackCube-v1",
    control_mode="pd_joint_pos"
)

print(env.action_space.low)
print(env.action_space.high)

print("Observation space", env.observation_space)