import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from feedforward import FFN

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
            minibatch):

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch = minibatch


        self.policy_net = FFN(obs_dim, act_dim, hidden_dim, num_hidden)
        self.value_net = FFN(obs_dim, 1, hidden_dim, num_hidden)

        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr)

        self.trajectories = []
    
    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            logits = self.policy_net(obs)
            dist = Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value_net(obs).squeeze(-1)

        return action, log_prob, value
    

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
        

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = states.shape[0]

        # accumulate per-minibatch diagnostics so we can return the epoch averages
        stats = {"policy_loss": [], "value_loss": [], "entropy": [],
                 "approx_kl": [], "clip_frac": []}

        for _ in range(self.epochs):
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
                logits = self.policy_net(batch_states)
                dist = Categorical(logits=logits)

                new_log = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

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
