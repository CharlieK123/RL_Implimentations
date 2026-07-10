import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden, out_act='linear', actor=True):
        super().__init__()

        layers = []
        activations = {'linear': nn.Identity(), 'softmax': nn.Softmax(dim=-1)}

        # input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(activations[out_act])

        self.network = nn.Sequential(*layers)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

        self.network.apply(init_weights)

        final_gain = 0.01 if actor else 1
        final = self.network[-2]  # Last Linear layer
        nn.init.orthogonal_(final.weight, gain=final_gain)
        nn.init.zeros_(final.bias)

    def forward(self, x):
        return self.network(x)