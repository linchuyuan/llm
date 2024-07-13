#!/usr/bin/env python3
import torch
import os
import matplotlib.pyplot as plt
import pdb

from model.decoder_only_informer import DecoderOnlyInformer
from lib.encoder import EncoderBlock
from lib.decoder import DecoderBlock
from lib.sinusoidal_position_embedding import SinusoidalPositionalEmbedding
from lib.config import Config

class EncoderDecoderInformer(DecoderOnlyInformer):

    def __init__(self, config : Config):
        super(DecoderOnlyInformer, self).__init__()
        self.config = config

        # encoder block
        self.encoder_embedding = torch.nn.Linear(
                config.n_features, config.n_embed).to(self.config.cuda0)
        self.encoder_position_embedding_table = SinusoidalPositionalEmbedding(
                config.n_embed, config.n_encoder_block_size).to(self.config.cuda0)
        self.encoder_blocks = torch.nn.Sequential(*[EncoderBlock(
            config.n_embed, config.n_encoder_head,
            config.n_encoder_block_size) for _ in range(config.n_encoder_layer)]).to(config.cuda0)

        # decoder block
        self.decoder_embedding = torch.nn.Linear(
                config.n_features, config.n_embed).to(self.config.cuda1)
        self.decoder_position_embedding_table = SinusoidalPositionalEmbedding(
                config.n_embed, config.n_decoder_block_size).to(self.config.cuda1)
        self.decoder_blocks = torch.nn.Sequential(*[DecoderBlock(
            config.n_embed, config.n_decoder_head,
            config.n_decoder_block_size, masked=False) for _ in range(config.n_decoder_layer)]).to(config.cuda1)

        # final mapping
        self.final_linear1 = torch.nn.Linear(config.n_embed, config.n_features).to(self.config.cuda1)
        # self.layernorm = torch.nn.LayerNorm(config.n_features, eps=1e-6).to(self.config.cuda1)
        # self.final_linear2 = torch.nn.Linear(config.n_features, config.n_features).to(self.config.cuda1)
        self.loss = torch.nn.MSELoss()

    def forward(self, index, targets):
        # B, T
        # print("tokens is %s, \ntargets is %s" % (index, targets))
        B, T, C = index.shape

        # encoder
        encoder_logits = self.encoder_embedding(index)
        encoder_logits = self.encoder_position_embedding_table(encoder_logits)
        memory = self.encoder_blocks(encoder_logits)

        # decoder
        decoder_logits = self.decoder_embedding(targets)
        decoder_logits = self.decoder_position_embedding_table(decoder_logits)
        decoder_logits, _ = self.decoder_blocks((decoder_logits, memory.to(decoder_logits.device)))

        # final mapping
        logits = self.final_linear1(decoder_logits)
        # logits = self.layernorm(logits)
        # logits = self.final_linear2(logits)
        loss = self.loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, index, target, checkpoint_path=None):
        checkpoint = None
        print("Generating from path %s" %(checkpoint_path))
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            if torch.cuda.device_count() == 1:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.config.cuda0))
            else:
                checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise SystemError("No checkpoint available.")

        tgt = target.clone().detach().to(target.device)
        tgt[:,-self.config.token_offset:,:] = 0
        logits, loss = self.forward(index, tgt)
        return logits
