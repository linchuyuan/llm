#!/usr/bin/env python3
import torch
import os
import pdb
import time
from collections import OrderedDict

from lib.encoder import EncoderBlock
from lib.decoder import DecoderBlock
from lib.final_mapping_attention_block import FinalMappingAttentionBlock
from lib.sinusoidal_position_embedding import SinusoidalPositionalEmbedding
from lib.token_embedding import TokenEmbedding
from lib.config import Config, dropout
from lib.data import DataFrame
from lib.layer_norm import LayerNorm
from lib.temporal_embedding import TemporalEmbedding

_pred_start = 0
_pred_end = 5
class EncoderDecoderInformer(torch.nn.Module):

    def __init__(self, config : Config):
        super().__init__()
        self.config = config

        # encoder block
        self.dropout = torch.nn.Dropout(dropout)
        self.encoder_linear = torch.nn.Linear(self.config.n_features,
            self.config.n_features).to(self.config.cuda0)
        self.encoder_ln = LayerNorm(
            self.config.n_encoder_block_size).to(self.config.cuda0)
        self.encoder_embedding = TokenEmbedding(
            config.n_features, config.n_embed).to(self.config.cuda0)
        self.encoder_position_embedding_table = SinusoidalPositionalEmbedding(
            config.n_embed, 5000).to(self.config.cuda0)
        self.encoder_temporal_embedding = TemporalEmbedding(
            config.n_embed).to(self.config.cuda0)
        self.encoder_blocks = torch.nn.Sequential(*[EncoderBlock(
            config.n_embed, config.n_encoder_head,
            config.n_encoder_block_size)
            for _ in range(config.n_encoder_layer)]).to(self.config.cuda0)


        # decoder block
        self.decoder_linear = torch.nn.Linear(self.config.n_features,
            self.config.n_features).to(self.config.cuda1)
        self.decoder_ln = LayerNorm(
            self.config.n_decoder_block_size+self.config.n_predict_block_size).to(
                self.config.cuda1)
        self.decoder_embedding = TokenEmbedding(
            config.n_features, config.n_embed).to(self.config.cuda1)
        self.decoder_position_embedding_table = SinusoidalPositionalEmbedding(
            config.n_embed, 5000).to(self.config.cuda1)
        self.decoder_temporal_embedding = TemporalEmbedding(
            config.n_embed).to(self.config.cuda1)
        self.decoder_blocks = torch.nn.Sequential(*[DecoderBlock(
            config.n_embed, config.n_decoder_head,
            config.n_decoder_block_size + config.n_predict_block_size)
            for _ in range(config.n_decoder_layer)]).to(self.config.cuda1)


        # final mapping
        self.final_linear1 = torch.nn.Linear(
            config.n_embed, config.n_features).to(self.config.cuda1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            # torch.nn.init.kaiming_normal_(
            #     module.weight,mode='fan_in',nonlinearity='leaky_relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv1d):
            # torch.nn.init.kaiming_normal_(
            #     module.weight, mode='fan_in', nonlinearity='leaky_relu')
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, index, index_mark, targets, targets_mark):
        # B, T
        B, T, C = index.shape

        # dropout
        index = self.dropout(index)
        targets = self.dropout(targets)

        # encoder
        index = self.encoder_linear(index)
        index = self.encoder_ln(index)
        encoder_logits = self.encoder_embedding(index)
        encoder_logits = self.encoder_position_embedding_table(encoder_logits)
        encoder_logits = encoder_logits + self.encoder_temporal_embedding(index_mark)
        memory = self.encoder_blocks(encoder_logits)

        # decoder
        # pred = targets[:, -self.config.n_predict_block_size:, :].clone().detach()
        # decoder_in = targets[:, :self.config.n_decoder_block_size, :].clone().detach()
        decoder_in = targets.clone().detach()
        decoder_in[:,-self.config.n_predict_block_size:,:] = 0
        decoder_in = decoder_in.to(self.decoder_linear.weight.device)
        targets_mark = targets_mark.to(decoder_in.device)
        decoder_in = self.decoder_linear(decoder_in)
        decoder_in = self.decoder_ln(decoder_in)
        decoder_logits = self.decoder_embedding(decoder_in)
        decoder_logits = self.decoder_position_embedding_table(decoder_logits)
        decoder_logits = decoder_logits + self.decoder_temporal_embedding(targets_mark)

        decoder_out, _ = self.decoder_blocks(
            (decoder_logits, memory.to(decoder_logits.device)))

        # final mapping
        logits = self.final_linear1(decoder_out)
        return logits[:,-self.config.n_predict_block_size:, :]
        # return logits

@torch.no_grad()
def estimate_loss(model, config, criterion, get_batch, eval_iters=10):
    out = {}
    model.eval()
    for split in ['training', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, x_mark, y, y_mark= get_batch(config.batch_size,
                config.n_encoder_block_size,
                config.n_decoder_block_size,
                config.n_predict_block_size,
                split)
            logits = model.forward(x, x_mark, y, y_mark)
            y = y.to(logits.device)
            loss = criterion(logits[:,-config.n_predict_block_size:, _pred_start:_pred_end],
                y[:,-config.n_predict_block_size:,_pred_start:_pred_end])
            # loss = criterion(logits, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def removeModulePrefidx(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict

@torch.no_grad()
def generate(model, config,  index, index_mark, target, target_mark, 
             checkpoint_path=None, step=1):
    checkpoint = None
    model.eval()
    print("Generating from path %s" %(checkpoint_path))
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if torch.cuda.device_count() == 1:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(config.cuda0))
        elif torch.cuda.device_count() == 0:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(config.cpu()))
        else:
            checkpoint = torch.load(checkpoint_path)
        state_dict = removeModulePrefidx(checkpoint["model_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise SystemError("No checkpoint available.")
    for _ in range(step):
        buf = DataFrame.padOnes(config.n_predict_block_size, target)
        buf_mark = DataFrame.genTimestamp(
            target_mark[-1,-1], config.n_predict_block_size)
        buf_mark = torch.concat(
            (target_mark, buf_mark.to(target_mark.device)), dim=1).long()
        pred = model.forward(index, index_mark, buf, buf_mark)
        target = target.to(pred.device)
        target = torch.concatenate(
            (target[:,:,_pred_start:_pred_end], pred[:,:,_pred_start:_pred_end]), dim=1)
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
        if i % eval_interval == 0:
            losses = estimate_loss(model, config, criterion, get_batch)
            print(f"step {i}: train loss {losses['training']:.4f}, val loss {losses['val']:.4f}")
            if i % 200 == 0:
                print("checkpointing")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
            scheduler.step(losses['val'])  # Step the scheduler with the validation loss
        x, x_mark, y, y_mark = get_batch(config.batch_size,
            config.n_encoder_block_size,
            config.n_decoder_block_size,
            config.n_predict_block_size)
        logits = model.forward(x, x_mark, y, y_mark)
        y = y.to(logits.device)
        loss = criterion(logits[:,-config.n_predict_block_size:, _pred_start:_pred_end],
            y[:,-config.n_predict_block_size:,_pred_start:_pred_end])
        print("Loop %s, %s, speed %s b/s" % (
            i, round(loss.item(), 2), round(i / (time.time() - start_time), 2)), end='\r')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
