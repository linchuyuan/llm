#!/usr/bin/env python3
import torch
import os
import matplotlib.pyplot as plt
import pdb
import time

from model.decoder_only_informer import DecoderOnlyInformer
from lib.encoder import EncoderBlock
from lib.decoder import DecoderBlock
from lib.sinusoidal_position_embedding import SinusoidalPositionalEmbedding
from lib.config import Config

class EncoderDecoderInformer(torch.nn.Module):

    def __init__(self, config : Config):
        super().__init__()
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
            config.n_decoder_block_size) for _ in range(config.n_decoder_layer)]).to(config.cuda1)

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
        tgt = targets.clone()
        tgt[:,-self.config.token_offset:,:] = 0
        decoder_logits = self.decoder_embedding(tgt)
        decoder_logits = self.decoder_position_embedding_table(decoder_logits)
        decoder_logits, _ = self.decoder_blocks((decoder_logits, memory.to(decoder_logits.device)))

        # final mapping
        logits = self.final_linear1(decoder_logits)
        # logits = self.layernorm(logits)
        # logits = self.final_linear2(logits)
        loss = self.loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def estimate_loss(self, get_batch, batch_size,
                      src_block_size, tgt_block_size, eval_iters=200):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(batch_size, src_block_size, tgt_block_size,  split)
                logits, loss = self.forward(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out

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

        tgt = target.clone()
        tgt[:,-self.config.token_offset:,:] = 0
        logits, loss = self.forward(index, tgt)
        return logits

    def train_and_update(self, get_batch, batch_size,  lr,
                         epoch, src_block_size, tgt_block_size=None,
                         eval_interval=1e3, checkpoint_path=None):
        if tgt_block_size is None:
            tgt_block_size = src_block_size
        start_time = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        checkpoint = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print("Resume from %s..." % checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            try:
                self.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                del checkpoint
                torch.cuda.empty_cache()
            except:
                print("Unable to restore from previous checkpoint, restaring...")
        else:
            print("Clean run starts")
        print("starting")
        for i in range(int(epoch)):
            if i % eval_interval == 0 and i != 0:
                losses = self.estimate_loss(get_batch, batch_size, src_block_size, tgt_block_size)
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                scheduler.step(losses['val'])  # Step the scheduler with the validation loss

            x, y = get_batch(batch_size, src_block_size, tgt_block_size)
            logits, loss = self.forward(x, y)
            print("Loop %s, %s, speed %s b/s" % (
                i, loss.item(), round(i / (time.time() - start_time), 2)), end='\r')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()