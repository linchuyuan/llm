#!/usr/bin/env python3
import torch
import pdb
from lib.multi_head import MultiHeadAttention
from lib.feed_forward import FeedForward

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
        self.ln1 = torch.nn.LayerNorm(n_embed, eps=1e-6)
        self.ln2 = torch.nn.LayerNorm(n_embed, eps=1e-6)
        self.ln3 = torch.nn.LayerNorm(n_embed, eps=1e-6)

    # using args so it is compatible with nn.sequential
    def forward(self, args: tuple):
        index, memory = args[0], args[1]
        index = index + self.attention_masked(
                self.ln1(index), None)
        index = index + self.attention_unmasked(
                self.ln2(index), memory)
        index = index + self.ffwd(self.ln3(index))
        return (index, memory)
