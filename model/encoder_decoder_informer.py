#!/usr/bin/env python3
import torch
import os
import matplotlib.pyplot as plt
import pdb
import time

from lib.encoder import EncoderBlock
from lib.decoder import DecoderBlock
from lib.final_mapping_attention_block import FinalMappingAttentionBlock
from lib.sinusoidal_position_embedding import SinusoidalPositionalEmbedding
from lib.token_embedding import TokenEmbedding
from lib.config import Config
from lib.data import DataFrame

class EncoderDecoderInformer(torch.nn.Module):

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        # encoder block
        # self.encoder_linear = torch.nn.Linear(self.config.n_features,
        #         self.config.n_features).to(self.config.cuda0)
        self.encoder_ln = torch.nn.LayerNorm(
                self.config.n_features).to(self.config.cuda0)
        self.encoder_embedding = TokenEmbedding(
                config.n_features, config.n_embed).to(self.config.cuda0)
        self.encoder_position_embedding_table = SinusoidalPositionalEmbedding(
                config.n_embed, 5000).to(self.config.cuda0)
        self.encoder_blocks = torch.nn.Sequential(*[EncoderBlock(
            config.n_embed, config.n_encoder_head,
            config.n_encoder_block_size)
            for _ in range(config.n_encoder_layer)]).to(self.config.cuda0)


        # decoder block
        # self.decoder_linear = torch.nn.Linear(self.config.n_features,
        #         self.config.n_features).to(self.config.cuda1)
        self.decoder_ln = torch.nn.LayerNorm(self.config.n_features).to(self.config.cuda1)
        self.decoder_embedding = TokenEmbedding(
                config.n_features, config.n_embed).to(self.config.cuda1)
        self.decoder_position_embedding_table = SinusoidalPositionalEmbedding(
                config.n_embed, 5000).to(self.config.cuda1)
        self.decoder_blocks = torch.nn.Sequential(*[DecoderBlock(
            config.n_embed, config.n_decoder_head,
            config.n_decoder_block_size + config.n_predict_block_size)
            for _ in range(config.n_decoder_layer)]).to(self.config.cuda1)


        # final mapping
        self.final_linear1 = torch.nn.Linear(config.n_embed, config.n_features).to(self.config.cuda1)

    def forward(self, index, targets):
        # B, T
        # print("tokens is %s, \ntargets is %s" % (index, targets))
        B, T, C = index.shape

        # encoder
        # index = self.encoder_linear(index)
        index = self.encoder_ln(index)
        encoder_logits = self.encoder_embedding(index)
        encoder_logits = self.encoder_position_embedding_table(encoder_logits)
        memory = self.encoder_blocks(encoder_logits)

        # decoder
        # pred = targets[:, -self.config.n_predict_block_size:, :].clone().detach()
        # decoder_in = targets[:, :self.config.n_decoder_block_size, :].clone().detach()
        decoder_in = targets.clone().detach()
        decoder_in[:,-self.config.n_predict_block_size:,:] = 1
        # decoder_in = self.decoder_linear(decoder_in)
        decoder_in = self.decoder_ln(decoder_in)
        decoder_logits = self.decoder_embedding(decoder_in)
        decoder_logits = self.decoder_position_embedding_table(decoder_logits)

        decoder_out, _ = self.decoder_blocks((decoder_logits, memory.to(decoder_logits.device)))

        # final mapping
        logits = self.final_linear1(decoder_out)
        return logits[:,-self.config.n_predict_block_size:, :5]
        # return logits

@torch.no_grad()
def estimate_loss(model, config, criterion, get_batch, eval_iters=2):
    out = {}
    model.eval()
    for split in ['training', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(config.batch_size,
                config.n_encoder_block_size,
                config.n_decoder_block_size,
                config.n_predict_block_size,
                split)
            logits = model.forward(x, y)
            loss = criterion(logits, y[:,-config.n_predict_block_size:,:5])
            # loss = criterion(logits, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate(model, config,  index, target, checkpoint_path=None, step=1):
    checkpoint = None
    model.eval()
    print("Generating from path %s" %(checkpoint_path))
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if torch.cuda.device_count() == 1:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(config.cuda0))
        else:
            checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise SystemError("No checkpoint available.")
    for _ in range(step):
        buf = DataFrame.padOnes(config.n_predict_block_size, target)
        # tgt_size = config.n_decoder_block_size + config.n_predict_block_size
        # pred = model.forward(index, buf[:, -tgt_size:, :])
        pred = model.forward(index, buf)
        target = torch.concatenate((target, pred), dim=1)
    return target

def train_and_update(model, config, get_batch, epoch, eval_interval):
    start_time = time.time()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    checkpoint = None
    checkpoint_path = config.informerCheckpointPath()
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print("Resume from %s..." % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
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
            losses = estimate_loss(model, config, criterion, get_batch)
            print(f"step {i}: train loss {losses['training']:.4f}, val loss {losses['val']:.4f}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            scheduler.step(losses['val'])  # Step the scheduler with the validation loss

        x, y = get_batch(config.batch_size,
            config.n_encoder_block_size,
            config.n_decoder_block_size,
            config.n_predict_block_size)
        logits = model.forward(x, y)
        loss = criterion(logits, y[:,-config.n_predict_block_size:,:5])
        # loss = criterion(logits, y)
        print("Loop %s, %s, speed %s b/s" % (
            i, round(loss.item(), 2), round(i / (time.time() - start_time), 2)), end='\r')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
