import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

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


class JEPA(nn.Module):
    def __init__(self, latent_dim, encoder_params, projection_params=None, momentum=0.995):
        super().__init__()
         #  blocks, residual_dim, hidden_dim, att_heads = encoder_params
         #  hidden_layers, hidden_dim = projection_params

        if latent_dim != encoder_params[1]:
            raise ValueError(f"Latent dim and Encoder residual dim don't match. {latent_dim} and {encoder_params[1]}")

        self.encoder = Transformer(*encoder_params)
        self.target_encoder = deepcopy(self.encoder)

        self.target_encoder.requires_grad_(False)
        self.momentum = momentum

        if projection_params is not None:
            self.predictor = FFN(latent_dim, latent_dim, *projection_params)
        else:
            self.predictor = nn.Linear(latent_dim, latent_dim)


    def forward(self, state_t, state_tk):
        # use current state to predict future latent state
        latent_t = self.encoder(state_t)
        latent_tk_hat = self.predictor(latent_t)

        # get the true latent future state with stop gradient target encoder
        with torch.no_grad():
            latent_tk = self.target_encoder(state_tk)

        return latent_tk_hat, latent_tk

    @torch.no_grad()
    def update_target_params(self):
        for new_params, old_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            # works to make target new params an EMA of the true encoder.
            # theta_t = (m * theta_t-1) + (1 - m)(theta_t)
            # lerp is Linear Interpolate between the old params and the new params with weight 1-m
            # it ultimately is the same operation as the EMA above
            old_params.lerp_(new_params, weight=1.0-self.momentum)