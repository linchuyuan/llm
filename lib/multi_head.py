#!/usr/bin/env python3

import torch
import pdb
from lib.single_head import Head
from lib.config import dropout

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_embed, head_size, block_size, masked, top_k_ratio=0.1):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            Head(head_size, n_embed, block_size, masked, top_k_ratio) for _ in range(n_head) ])
        self.proj = torch.nn.Linear(head_size * n_head, n_embed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, index, memory):
        x = torch.cat([h(index, memory) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
