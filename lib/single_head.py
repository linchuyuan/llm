#!/usr/bin/env python
import torch
import pdb
import math

from lib.config import dropout

class Head(torch.nn.Module):
    def __init__(self, head_size, n_embed, block_size, masked, top_k_ratio=0.1):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.q = torch.nn.Linear(n_embed, head_size, bias=False)
        self.k = torch.nn.Linear(n_embed, head_size, bias=False)
        self.v = torch.nn.Linear(n_embed, head_size, bias=False)
        self.masked = masked
        self.top_k_ratio = top_k_ratio  # Ratio for dynamic top-K selection
        self.dropout = torch.nn.Dropout(dropout)
        if self.masked:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, index, memory=None):
        if memory is None:
            memory = index
        B, T, _ = index.shape
        self.top_k = int(self.top_k_ratio * T)

        # Linear projections
        q = self.q(index)  # [B, T, head_size]
        k = self.k(memory)  # [B, T, head_size]
        v = self.v(memory)  # [B, T, head_size]

        # Norm-based top-K selection
        q_norms = torch.norm(q, dim=-1)  # [B, T]
        top_k_indices = torch.topk(q_norms, self.top_k, dim=1).indices  # [B, top_k]

        # Gather top-K queries, keys, and values
        q_topk = torch.gather(q, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.head_size))  # [B, top_k, head_size]
        k_topk = torch.gather(k, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.head_size))  # [B, top_k, head_size]
        v_topk = torch.gather(v, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.head_size))  # [B, top_k, head_size]

        # Scaled dot-product attention on top-K queries using einsum for efficiency
        w = torch.einsum('b i d, b j d -> b i j', q_topk, k_topk) / math.sqrt(self.head_size)  # [B, top_k, top_k]
        w = w - w.max(dim=-1, keepdim=True).values  # Stabilize softmax calculation

        if self.masked and self.training:
            mask = torch.ones(self.block_size, self.block_size, device=w.device).tril()[:self.top_k, :self.top_k]
            w = w.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout on attention weights
        w = torch.nn.functional.softmax(w, dim=-1)  # [B, top_k, top_k]
        w = self.dropout(w)

        # Compute output with weighted values
        out_topk = w @ v_topk  # [B, top_k, head_size]

        # Initialize full output with zeros and scatter top-K values back
        output = torch.zeros(B, T, self.head_size, device=out_topk.device)
        output.scatter_add_(1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.head_size), out_topk)

        return output
