import torch
import numpy as np
import gymnasium


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




