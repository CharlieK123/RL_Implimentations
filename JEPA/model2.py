import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from functions import effective_rank, batch_collapse_metrics
import numpy as np


class Transformer(nn.Module):
    def __init__(self, blocks, residual_dim, hidden_dim, att_heads):
        super().__init__()

        self.blocks = blocks
        self.dim = residual_dim

        if residual_dim % att_heads != 0:
            raise ValueError("residual_dim must be divisible by att_heads")

        # when called a new instance is created
        ffn = lambda: nn.Sequential(nn.Linear(residual_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, residual_dim))
        att = lambda: nn.MultiheadAttention(self.dim, att_heads, batch_first=True)
        norm = lambda: nn.RMSNorm(self.dim, eps=1e-6)

        self.attention = nn.ModuleList([att() for _ in range(blocks)])
        self.ffn = nn.ModuleList([ffn() for _ in range(blocks)])
        self.norm = nn.ModuleList([norm() for _ in range(blocks * 2)])

    def block(self, x, i):
        norm_1 = self.norm[2 * i]
        norm_2 = self.norm[2 * i + 1]
        attention = self.attention[i]
        feedforward = self.ffn[i]

        norm_out = norm_1(x)
        att_out, _ = attention(norm_out, norm_out, norm_out, need_weights=False)

        x = x + att_out

        norm_out = norm_2(x)
        ff_out = feedforward(norm_out)

        x = x + ff_out

        return x

    def forward(self, x):
        # x -> (batch, seq, dim)
        for i in range(self.blocks):
            x = self.block(x, i)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, h_layers, h_dim):
        super().__init__()

        layers = []

        # input layer
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.Tanh())

        # hidden layers
        for i in range(h_layers - 1):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.GELU())

        # output linear layer
        layers.append(nn.Linear(h_dim, out_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def stackcube_state(s, n_qpos=9):
    """
    Splits a flat StackCube-v1 state vector [..., D] into its component
    objects. Returns them independently (no dict, no stacking).
    """
    i = n_qpos
    qpos            = s[..., 0:i]
    qvel            = s[..., i:2*i]
    tcp_pose        = s[..., 2*i:2*i+7]
    cubeA_pose      = s[..., 2*i+7:2*i+14]
    cubeB_pose      = s[..., 2*i+14:2*i+21]
    tcp_to_cubeA    = s[..., 2*i+21:2*i+24]
    tcp_to_cubeB    = s[..., 2*i+24:2*i+27]
    cubeA_to_cubeB  = s[..., 2*i+27:2*i+30]

    return qpos, qvel, tcp_pose, cubeA_pose, cubeB_pose, tcp_to_cubeA, tcp_to_cubeB, cubeA_to_cubeB


class JEPA(nn.Module):
    def __init__(self, latent_dim, encoder_params, projection_params=None, momentum=0.995, ac=False, n_qpos=9):
        super().__init__()
        #  blocks, residual_dim, hidden_dim, att_heads = encoder_params
        #  hidden_layers, hidden_dim = projection_params
        if latent_dim != encoder_params[1]:
            raise ValueError(f"Latent dim and Encoder residual dim don't match. {latent_dim} and {encoder_params[1]}")

        self.encoder = Transformer(*encoder_params)
        self.target_encoder = deepcopy(self.encoder)

        self.target_encoder.requires_grad_(False)
        self.momentum = momentum
        self.ac = ac is not False

        self.n_qpos = n_qpos
        predictor_dim = latent_dim

        # per-object input dims, matching stackcube_state's output order
        self.obj_dims = [n_qpos, n_qpos, 7, 7, 7, 3, 3, 3]

        # each object gets embedded to the FULL latent_dim, since each becomes
        # one token fed into the transformer encoder (residual_dim == latent_dim)
        self.embedding_networks = nn.ModuleList(
            [FFN(dim, latent_dim, 1, 128) for dim in self.obj_dims]
        )

        if ac is not False:
            # ac = (act_dim, embed_dim, hidden_layers, hidden_dim)
            self.action_encoder = FFN(*ac)
            predictor_dim = ac[1] + latent_dim

        if projection_params is not None:
            self.predictor = FFN(predictor_dim, latent_dim, *projection_params)
        else:
            self.predictor = nn.Linear(predictor_dim, latent_dim)

    def embed_object(self, state):
        """
        state: [batch, 1, D] flat StackCube state.
        returns: [batch, n_objects, latent_dim] — one token per object.
        """
        objects = stackcube_state(state, n_qpos=self.n_qpos)
        embedded = [net(obj) for net, obj in zip(self.embedding_networks, objects)]
        return torch.cat(embedded, dim=1)

    def forward(self, state_t, state_tk, action_t=None):

        if state_t.ndim == 2:
            state_t = state_t.unsqueeze(1)

        if state_tk.ndim == 2:
            state_tk = state_tk.unsqueeze(1)

        # use current state to predict future latent state
        state_t_embedded = self.embed_object(state_t)          # [batch, n_objects, latent_dim]
        latent_t = self.encoder(state_t_embedded)              # [batch, n_objects, latent_dim]
        latent_t = latent_t.mean(dim=1)                        # pool object tokens -> [batch, latent_dim]

        if action_t is not None:
            action_embedding = self.action_encoder(action_t)   # [batch, action_embed_dim]
            latent_t = torch.cat([latent_t, action_embedding], dim=-1)

        latent_tk_hat = self.predictor(latent_t)                # [batch, latent_dim]

        # get the true latent future state with stop gradient target encoder
        with torch.no_grad():
            state_tk_embedded = self.embed_object(state_tk)
            latent_tk = self.target_encoder(state_tk_embedded).mean(dim=1)  # [batch, latent_dim]

        return latent_tk_hat, latent_tk

    @torch.no_grad()
    def update_target_params(self):
        for new_params, old_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            # works to make target new params an EMA of the true encoder.
            # theta_t = (m * theta_t-1) + (1 - m)(theta_t)
            # lerp is Linear Interpolate between the old params and the new params with weight 1-m
            # it ultimately is the same operation as the EMA above
            old_params.lerp_(new_params, weight=1.0 - self.momentum)


def train(model, epochs, loader, optim, device="cuda"):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        totals = {
            "loss": 0.0,
            "latent_std": 0.0,
            "latent_abs": 0.0,
            "prediction_cosine": 0.0,
            "identity_cosine": 0.0,
            "state_cosine": 0.0,
            "grad_norm": 0.0,
            'offdiag': 0.0,
            'effrank_enc': 0.0,
            'effrank_tar': 0.0
        }

        total_samples = 0

        for states, actions, next_states in loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)

            batch_size = states.shape[0]

            z_hat, z = model(states, next_states, actions)

            z_hat_normalized = F.normalize(z_hat, dim=-1)
            z_normalized = F.normalize(z, dim=-1)

            loss = F.smooth_l1_loss(z_hat_normalized, z_normalized)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            model.update_target_params()

            # eval metrics
            with torch.no_grad():
                states_unsq = states.unsqueeze(1) if states.ndim == 2 else states
                next_states_unsq = next_states.unsqueeze(1) if next_states.ndim == 2 else next_states

                online_latent = model.encoder(model.embed_object(states_unsq)).mean(dim=1)
                future_target = model.target_encoder(model.embed_object(next_states_unsq)).mean(dim=1)

                latent_std = online_latent.std(dim=0, unbiased=False).mean()
                latent_abs = online_latent.abs().mean()

                prediction_cosine = F.cosine_similarity(z_hat, z, dim=-1).mean()
                identity_cosine = F.cosine_similarity(online_latent, future_target, dim=-1).mean()
                state_cosine = F.cosine_similarity(states, next_states, dim=-1).mean()

                batch_offdiag_sim = batch_collapse_metrics(online_latent)
                eff_rank = effective_rank(online_latent)
                target_eff_rank = effective_rank(future_target)  # check target encoder separately too

            totals["loss"] += loss.item() * batch_size
            totals["latent_std"] += latent_std.item() * batch_size
            totals["latent_abs"] += latent_abs.item() * batch_size
            totals["prediction_cosine"] += prediction_cosine.item() * batch_size
            totals["identity_cosine"] += identity_cosine.item() * batch_size
            totals["state_cosine"] += state_cosine.item() * batch_size
            totals["grad_norm"] += float(grad_norm) * batch_size
            totals['effrank_enc'] += eff_rank * batch_size
            totals['effrank_tar'] += target_eff_rank * batch_size
            totals['offdiag'] += batch_offdiag_sim * batch_size

            total_samples += batch_size

        avg = {name: value / total_samples for name, value in totals.items()}

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={avg['loss']:.6f} | "
            f"predict future latent sim={avg['prediction_cosine']:.4f} | "
            f"latent sim={avg['identity_cosine']:.4f} | "
            f"state sim={avg['state_cosine']:.4f}"
        )

        print(
            f"latent_std={avg['latent_std']:.4f} | "
            f"latent_abs={avg['latent_abs']:.4f} | "
            f"grad_norm={avg['grad_norm']:.4f} | "
            f"eff_rank_enc={avg['effrank_enc']:.2f} | "
            f"eff_rank_tar={avg['effrank_tar']:.2f} | "
            f"offdiag_sim={avg['offdiag']:.4f}"
        )

        print('-----------------------------\n')