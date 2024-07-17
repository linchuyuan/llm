#!/usr/bin/env python3

import torch
import pdb
from lib.single_head import Head

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_embed, head_size, block_size, masked):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            Head(head_size, n_embed, block_size, masked) for _ in range(n_head) ])
        self.proj = torch.nn.Linear(head_size * n_head, n_embed)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, index, memory):
        x = torch.cat([h(index, memory) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
