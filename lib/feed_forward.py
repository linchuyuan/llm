#!/usr/bin/env python3
import torch
from lib.config import dropout

class FeedForward(torch.nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        hiddenshape = n_embed * 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, hiddenshape),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hiddenshape, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
