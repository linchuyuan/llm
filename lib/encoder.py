#!/usr/bin/env python3
import torch

from lib.multi_head import MultiHeadAttention
from lib.feed_forward import FeedForward

class EncoderBlock(torch.nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, n_embed, head_size, block_size, masked=False)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed, eps=1e-6)
        self.ln2 = torch.nn.LayerNorm(n_embed, eps=1e-6)

    def forward(self, index):
        index = index + self.sa(self.ln1(index), memory=None)
        index = index + self.ffwd(self.ln2(index))
        return index
