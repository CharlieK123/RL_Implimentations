import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
from runningreturn import RunningNorm

from feedforward2 import FFN
import pickle
from metrics import ppo_log, save_history

from pathlib import Path
import os
import torch

CKPT = Path(__file__).parent / "ppo_policy.pt"
TMP_CKPT = Path(__file__).parent / "ppo_policy.tmp"

def LinearDecay(agent, inital, final, update):
    progress = min(agent.i / update, 1.0)
    return inital + (final - inital) * progress



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
            policy_lr,
            value_lr,
            minibatch,
            multi_envs,
            discrete=True,
            decay_actor_lr=False,
            decay_critic_lr=False,
            decay_ent=False,
            kl_target=None):

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.multienv = multi_envs
        self.discrete = discrete
        self.decay_actor_lr = decay_actor_lr
        self.decay_critic_lr = decay_critic_lr
        self.decay_ent = decay_ent
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.kl_target = kl_target

        self.policy_net = FFN(obs_dim, act_dim, hidden_dim, num_hidden)
        self.value_net = FFN(obs_dim, 1, hidden_dim, num_hidden, actor=False)
        #self.return_norm = RunningNorm()

        if not self.discrete:
            self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
            self.policy_optim = optim.Adam(list(self.policy_net.parameters()) + [self.log_std], lr=self.policy_lr, eps=1e-5)
        else:
            self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr, eps=1e-5)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1e-5)

        self.i = 0

    def save_policy(self, path, obs_norm=None, rew_norm=None):
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
        }

        if obs_norm is not None:
            checkpoint["obs_rms"] = {
                "mean": obs_norm.obs_rms.mean.copy(),
                "var": obs_norm.obs_rms.var.copy(),
                "count": obs_norm.obs_rms.count,
            }

        if rew_norm is not None:
            checkpoint["rew_rms"] = {
                "mean": rew_norm.return_rms.mean.copy(),
                "var": rew_norm.return_rms.var.copy(),
                "count": rew_norm.return_rms.count,
            }

        torch.save(checkpoint, path)

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
                std = self.log_std.clamp(-2, 2).exp().expand_as(mean)

                dist = Normal(mean, std)
                raw_action = dist.rsample()
                squashed = torch.tanh(raw_action)
                action = squashed

                log_prob = dist.log_prob(raw_action).sum(-1)
                log_prob -= torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)

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

    def gae(self, rewards, values, terminateds, next_value, truncateds=None):

        if truncateds is None:
            truncateds = torch.zeros_like(terminateds)

        values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        gae = torch.zeros_like(next_value)

        advantages = torch.zeros_like(rewards)

        for t in reversed(range(rewards.shape[0])):

            nonterminal = 1.0 - terminateds[t]
            cut = 1.0 - torch.clamp(terminateds[t] + truncateds[t], max=1.0)

            td = rewards[t] + self.gamma * nonterminal * values[t + 1] - values[t]

            gae = td + self.lmbda * self.gamma * cut * gae
            advantages[t] = gae


        returns = advantages + values[:-1]

        return advantages, returns


    def update(self, states, actions, old_log_probs, advantages, returns, dummies=None, obs_env=None, norm_env=None):
        p_loss, v_loss, ent, kl, clip = [], [], [], [], []

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

        if dummies is not None:
            valid = (dummies.reshape(-1) < 0.5)
            states = states[valid]
            actions = actions[valid]
            old_log_probs = old_log_probs[valid]
            advantages = advantages[valid]
            returns = returns[valid]

        if self.i % 10 == 0:
            print(
                f"Adv | mean={advantages.mean():.3f} std={advantages.std():.3f} abs={advantages.abs().mean():.3f} min={advantages.min():.3f} max={advantages.max():.3f}")
            print(torch.quantile(
                advantages,
                torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=advantages.device)
            ))
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
            print("used std    :", self.log_std.clamp(-2, 2).exp().detach().cpu().numpy())

        # decay (initial_lr, final, updates)
        if self.decay_actor_lr:
                inital_actor_lr, final_actor_lr, actor_updates = self.decay_actor_lr
                self.policy_lr = LinearDecay(self, inital_actor_lr, final_actor_lr, actor_updates)

                inital_critic_lr, final_critic_lr, critic_updates = self.decay_critic_lr
                self.value_lr = LinearDecay(self, inital_critic_lr, final_critic_lr, critic_updates)

                for param_group in self.policy_optim.param_groups:
                    param_group["lr"] = self.policy_lr

                for param_group in self.value_optim.param_groups:
                    param_group["lr"] = self.value_lr

        if self.decay_ent:
                initial_ent, final_ent, updates = self.decay_ent
                self.ent_coef = LinearDecay(self, initial_ent, final_ent, updates)

        early_stop = False
        for epoch in range(self.epochs):
            indices = torch.randperm(batch_size)

            if early_stop:  # early kl stopping
                break

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
                    std = self.log_std.clamp(-2.0, 2).exp().expand_as(out)

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

                def slop():
                    if not self.discrete and self.i % 10 == 0 and epoch == 0 and start == 0:
                        policy_g = torch.autograd.grad(
                            policy_loss,
                            self.log_std,
                            retain_graph=True
                        )[0]

                        entropy_g = torch.autograd.grad(
                            -self.ent_coef * entropy,
                            self.log_std,
                            retain_graph=True
                        )[0]

                        print("policy dL/dlogstd :", policy_g.detach().cpu().numpy().mean())
                        print("entropy dL/dlogstd:", entropy_g.detach().cpu().numpy().mean())
                        print("sum              :", (policy_g + entropy_g).detach().cpu().numpy().mean())
                        print("dims pushing std up:", (policy_g + entropy_g < 0).float().mean().item())


                slop()
                ratio = r
                with torch.no_grad():
                    clip_fraction = ((ratio < 1 - self.eps) | (ratio > 1 + self.eps)).float().mean()

                    log_ratio = new_log - batch_old_log
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()


                if self.kl_target is not None and approx_kl.item() >= 1.5 * self.kl_target:
                    early_stop = True
                    print(
                        f"Early stopping at epoch {epoch} due to reaching "
                        f"max kl: {approx_kl.item():.4f} (limit {1.5 * self.kl_target:.4f})"
                    )
                    break



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
                kl.append(approx_kl.item())
                clip.append(clip_fraction.item())

        # guard against early stop div 0
        if len(p_loss) == 0:
            p_loss = v_loss = ent = kl = clip = 0.0
        else:
            p_loss = sum(p_loss) / len(p_loss)
            v_loss = sum(v_loss) / len(v_loss)
            ent = sum(ent) / len(ent)
            kl = sum(kl) / len(kl)
            clip = sum(clip) / len(clip)

        ppo_log(self.i, p_loss, v_loss, ent, ev.item(), kl, clip)

        self.i += 1
        if self.i % 10 == 0:
            print(f'actor LR: {self.policy_lr:f}, critic LR: {self.value_lr:f}, Ent Coef: {round(self.ent_coef, 4)}')
            save_history()
            self.save_policy(
                "policy.pt",
                obs_norm=obs_env,
                rew_norm=norm_env,
            )


