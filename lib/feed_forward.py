#!/usr/bin/env python3
import torch

class FeedForward(torch.nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        hiddenshape = n_embed
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, hiddenshape),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hiddenshape, n_embed),
        )

    def forward(self, x):
        return self.net(x)
