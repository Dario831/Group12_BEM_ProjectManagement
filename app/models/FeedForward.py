# models/FeedForward.py
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_in, hidden, n_cls):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
