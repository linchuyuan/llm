 #!/usr/bin/env python3
import torch

from lib.multi_head import MultiHeadAttention
from lib.feed_forward import FeedForward
from torch.nn import LayerNorm

class EncoderBlock(torch.nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, block_size, masked=False):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(
            n_head, n_embed, head_size, block_size, masked=masked)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = LayerNorm(n_embed, eps=1e-6)
        self.ln2 = LayerNorm(n_embed, eps=1e-6)

    def forward(self, index):
        index_norm = self.ln1(index)
        index = index + self.sa(index_norm, index_norm)
        index_norm = self.ln2(index)
        index = index + self.ffwd(index_norm)
        return index
