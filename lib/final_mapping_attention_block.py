#!/usr/bin/env python
import torch
import math

from lib.single_head import Head
from lib.feed_forward import FeedForward

class FinalMappingAttentionBlock(torch.nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.map_reduce = Head(n_embed, n_embed, None, masked=False)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed, eps=1e-6)
        self.ln2 = torch.nn.LayerNorm(n_embed, eps=1e-6)

    def forward(self, index, memory):
        index = index + self.ln1(self.map_reduce(index, memory))
        index = index + self.ln2(self.ffwd(index))
        return index