#!/usr/bin/env python3
import torch

class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create matrix of positional embeddings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input tensor
        # x = x / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise RuntimeError("Input sequence length exceeds maximum positional encoding length.")

        # Add positional encodings for each time step in input sequence
        x = x + self.pe[:seq_len, :]
        return x
