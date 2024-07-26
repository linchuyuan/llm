#!/usr/bin/env python
import torch
import pdb
import math

class Head(torch.nn.Module):
    def __init__(self, head_size, n_embed, block_size, masked):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.q = torch.nn.Linear(n_embed, head_size, bias=False)
        self.k = torch.nn.Linear(n_embed, head_size, bias=False)
        self.v = torch.nn.Linear(n_embed, head_size, bias=False)
        self.masked = masked
        if self.masked:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, index, memory):
        if memory is None:
            memory = index

        _, T, _ = index.shape
        q = self.q(index) # B, T, head_size
        # print("q weights %s" % torch.sum(self.q.weight))
        k = self.k(memory) # B, T, head_size
        w = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_size) # [B, T, head_size] @ [B, head_size, T] = [B, T, T]
        if self.masked:
            w = w.masked_fill(self.tril[:self.block_size, :self.block_size] == 0, float('-inf')) # B, T, T
        w = torch.nn.functional.softmax(w, dim=-1) # (B, T, T)
        v = self.v(memory) # B, T, head_size
        out = w @ v
        return out
