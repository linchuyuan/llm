#!/usr/bin/env python3
import torch
import os
import matplotlib.pyplot as plt
import pdb
import time

from model.decoder_only_informer import DecoderOnlyInformer
from lib.encoder import EncoderBlock
from lib.decoder import DecoderBlock
from lib.final_mapping_attention_block import FinalMappingAttentionBlock
from lib.sinusoidal_position_embedding import SinusoidalPositionalEmbedding
from lib.token_embedding import TokenEmbedding
from lib.config import Config

class EncoderDecoderInformer(torch.nn.Module):

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        # encoder block
        self.encoder_embedding = TokenEmbedding(
                config.n_features, config.n_embed).to(self.config.cuda0)
        self.encoder_position_embedding_table = SinusoidalPositionalEmbedding(
                config.n_embed, config.n_encoder_block_size).to(self.config.cuda0)
        self.encoder_blocks = torch.nn.Sequential(*[EncoderBlock(
            config.n_embed, config.n_encoder_head,
            config.n_encoder_block_size) for _ in range(config.n_encoder_layer)]).to(config.cuda0)


        # decoder block
        self.decoder_embedding = TokenEmbedding(
                config.n_features, config.n_embed).to(self.config.cuda1)
        self.decoder_position_embedding_table = SinusoidalPositionalEmbedding(
                config.n_embed, config.n_target_data_size).to(self.config.cuda1)

        self.final_self_attention = FinalMappingAttentionBlock(config.n_embed).to(config.cuda1)
        self.decoder_blocks = torch.nn.Sequential(*[DecoderBlock(
            config.n_embed, config.n_decoder_head,
            config.n_predict_block_size) for _ in range(config.n_decoder_layer)]).to(config.cuda1)


        # final mapping
        self.final_linear1 = torch.nn.Linear(config.n_embed, config.n_features).to(self.config.cuda1)
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
        pred = targets[:, -self.config.n_predict_block_size:, :].clone().detach()
        targets = targets[:, :self.config.n_decoder_block_size, :]
        decoder_logits = self.decoder_embedding(targets)
        decoder_logits = self.decoder_position_embedding_table(decoder_logits)
        decoder_logits = self.final_self_attention(
            decoder_logits[:, -self.config.n_predict_block_size:, :], decoder_logits)

        decoder_out, _ = self.decoder_blocks((decoder_logits, memory.to(decoder_logits.device)))

        # final mapping
        logits = self.final_linear1(decoder_out)
        # logits = logits[:, -self.config.n_predict_block_size:, :]

        loss = self.loss(logits, pred)
        return logits, loss

    @torch.no_grad()
    def estimate_loss(self, get_batch, eval_iters=200):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(self.config.batch_size,
                    self.config.n_encoder_block_size,
                    self.config.n_target_data_size, split)
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

        logits, loss = self.forward(index, target)
        return logits

    def train_and_update(self, get_batch, epoch, eval_interval):

        start_time = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        checkpoint = None
        checkpoint_path = self.config.informerCheckpointPath()
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
            print("Clean run starts %s " % checkpoint_path)
        print("starting")
        for i in range(int(epoch)):
            if i % eval_interval == 0 and i != 0:
                losses = self.estimate_loss(get_batch)
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                scheduler.step(losses['val'])  # Step the scheduler with the validation loss

            x, y = get_batch(self.config.batch_size,
                self.config.n_encoder_block_size,
                self.config.n_target_data_size)
            logits, loss = self.forward(x, y)
            print("Loop %s, %s, speed %s b/s" % (
                i, round(loss.item(), 2), round(i / (time.time() - start_time), 2)), end='\r')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()