import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
from Norm import RunningNorm
from feed_forward import FFN
import pickle
from metrics import ppo_log, save_history

from pathlib import Path
import os
import torch

CKPT = Path(__file__).parent / "ppo_policy.pt"
TMP_CKPT = Path(__file__).parent / "ppo_policy.tmp"

def save_policy(agent):
    torch.save({
        "policy": agent.policy_net.state_dict(),
        "value": agent.value_net.state_dict(),
        "log_std": agent.log_std.detach() if not agent.discrete else None,
        "discrete": agent.discrete,
    }, TMP_CKPT)

    os.replace(TMP_CKPT, CKPT)

class PPO:
    def __init__(self,
            obs_dim,
            act_dim,
            hidden_dim,
            num_hidden,
            gamma,
            lmbda,
            eps,
            ent_coef,
            value_coef,
            epochs,
            lr,
            minibatch,
            multi_envs,
            discrete=True):

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.multienv = multi_envs
        self.discrete = discrete

        self.policy_net = FFN(obs_dim, act_dim, hidden_dim, num_hidden)
        self.value_net = FFN(obs_dim, 1, hidden_dim, num_hidden)
        #self.return_norm = RunningNorm()

        if not self.discrete:
            self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
            self.policy_optim = optim.Adam(list(self.policy_net.parameters()) + [self.log_std], lr=lr)
        else:
            self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr)

        self.i = 0

    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if self.discrete:
            with torch.no_grad():
                logits = self.policy_net(obs)
                dist = Categorical(logits=logits)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.value_net(obs).squeeze(-1)
                #value = value * self.return_norm.std + self.return_norm.running_mean
        else:
            with torch.no_grad():
                mean = self.policy_net(obs)
                std = self.log_std.clamp(-2.0, 0.0).exp().expand_as(mean)

                dist = Normal(mean, std)
                raw_action = dist.rsample()
                action = torch.tanh(raw_action)

                log_prob = dist.log_prob(raw_action).sum(-1)
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)

                value = self.value_net(obs).squeeze(-1)
                #value = value * self.return_norm.std + self.return_norm.running_mean


        if self.discrete:
            if not self.multienv:
                return action.item(), log_prob, value
            else:
                return action.numpy(), log_prob, value
        return action.numpy(), raw_action, log_prob, value, mean, std

    def td_error(self, r, v_s, v_s1, done):
        # discounted TD error. NOTE TD = r if end of episode
        return r + ((1 - done) * self.gamma * v_s1) - v_s

    def gae(self, rewards, values, dones, next_value):

        values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        gae = torch.zeros_like(next_value)

        advantages = torch.zeros_like(rewards)

        for t in reversed(range(rewards.shape[0])):

            td = self.td_error(rewards[t], values[t], values[t+1], dones[t])

            gae = td + self.lmbda * self.gamma * (1-dones[t]) * gae
            advantages[t] = gae


        returns = advantages + values[:-1]

        return advantages, returns


    def update(self, states, actions, old_log_probs, advantages, returns):
        p_loss, v_loss, ent = [], [], []

        if self.multienv:
            T, N = old_log_probs.shape

            states = states.reshape(T * N, states.shape[-1])
            old_log_probs = old_log_probs.reshape(T * N)
            advantages = advantages.reshape(T * N)
            returns = returns.reshape(T * N)

            if self.discrete:
                actions = actions.reshape(T * N).long()
            else:
                actions = actions.reshape(T * N, actions.shape[-1]).float()

        else:
            if self.discrete:
                actions = actions.long()
            else:
                actions = actions.float()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = states.shape[0]
        #self.return_norm.update(returns)

        # explained variance
        with torch.no_grad():
            values = self.value_net(states).squeeze(-1)
            #norm_returns = self.return_norm.normalize(returns)

        ev = 1 - torch.var(returns - values) / (
                torch.var(returns) + 1e-8
        )
        if self.i % 10 == 0:
            print(f'Explained Variance: {ev}')


        if not self.discrete and self.i % 10 == 0:
            #real_values = values * self.return_norm.std + self.return_norm.running_mean
            returns.std().item()
            print(f'mse: {torch.mean(torch.square(returns - values)).item()}, rmse: {torch.sqrt(torch.mean((returns - values) ** 2)).item()}')
            print("used std    :", self.log_std.clamp(-2.0, 0.0).exp().detach().cpu().numpy())

        for epoch in range(self.epochs):
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.minibatch):
                end = start + self.minibatch
                mb_idx = indices[start:end]

                batch_states = states[mb_idx]
                batch_actions = actions[mb_idx]
                batch_old_log = old_log_probs[mb_idx]
                batch_adv = advantages[mb_idx]
                batch_returns = returns[mb_idx]

                # policy
                out = self.policy_net(batch_states)
                if self.discrete:
                    dist = Categorical(logits=out)

                    new_log = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                else:
                    std = self.log_std.clamp(-2.0, 0.0).exp().expand_as(out)

                    dist = Normal(out, std)
                    squashed_actions = torch.tanh(batch_actions)

                    new_log = dist.log_prob(batch_actions).sum(-1)
                    new_log -= torch.log(1 - squashed_actions.pow(2) + 1e-6).sum(-1)

                    entropy = dist.entropy().sum(-1).mean()


                values = self.value_net(batch_states).squeeze(-1)

                r = torch.exp(new_log - batch_old_log)
                policy_loss = -torch.min(r * batch_adv, torch.clamp(r, 1 - self.eps, 1 + self.eps) * batch_adv).mean()
                value_loss = nn.functional.mse_loss(values, batch_returns)

                total_loss = policy_loss + value_loss * self.value_coef - entropy * self.ent_coef

                self.policy_optim.zero_grad()
                self.value_optim.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)

                self.policy_optim.step()
                self.value_optim.step()

                p_loss.append(policy_loss.item())
                v_loss.append(value_loss.item())
                ent.append(entropy.item())

        p_loss = sum(p_loss) / len(p_loss)
        v_loss = sum(v_loss) / len(v_loss)
        ent = sum(ent) / len(ent)

        ppo_log(self.i, p_loss, v_loss, ent, ev.item())

        self.i += 1
        if self.i % 100 == 0:
            save_history()
            save_policy(self)


