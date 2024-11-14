#!/usr/bin/env python3
import torch
import pdb
from lib.multi_head import MultiHeadAttention
from lib.feed_forward import FeedForward
from torch.nn import LayerNorm

class DecoderBlock(torch.nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head
        self.attention_masked = MultiHeadAttention(
            n_head, n_embed, head_size, block_size, masked=True)
        self.attention_unmasked = MultiHeadAttention(
            n_head, n_embed, head_size, block_size, masked=False)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = LayerNorm(n_embed, eps=1e-6)
        self.ln2 = LayerNorm(n_embed, eps=1e-6)
        self.ln3 = LayerNorm(n_embed, eps=1e-6)
        self.ln_memory = LayerNorm(n_embed, eps=1e-6)

    # using args so it is compatible with nn.sequential
    def forward(self, args: tuple):
        index, memory = args[0], args[1]
        index_norm = self.ln1(index)
        index = index + self.attention_masked(index_norm, index_norm)

        index_norm = self.ln2(index)
        memory_norm = self.ln_memory(memory)
        index = index + self.attention_unmasked(index_norm, memory_norm)

        index = index + self.ffwd(self.ln3(index))

        return (index, memory)
