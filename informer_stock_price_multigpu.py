#!/usr/bin/env python3
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import os
from lib.data import DataFrame
from lib.config import Config
from model.encoder_decoder_informer import EncoderDecoderInformer

run_predict = input("run predict (y/n, default n)?:")
if not run_predict:
    run_predict = 'n'

config = Config(
    tickers = [
        "SMCI"
    ],
    token_offset = 4,
    batch_size = 16,
    lr = 1e-3,
    epoch = 2000,
)

informer_config = Config(
    n_embed = 64,
    n_encoder_block_size = 8,
    n_encoder_head = 2,
    n_encoder_layer = 1,
    n_decoder_block_size = 8,
    n_decoder_head = 2,
    n_decoder_layer = 2,
    batch_size = config.batch_size,
    token_offset = config.token_offset,
)

def predict(model, data, src_block_size, tgt_block_size, ix=0, checkpoint_path=None):
    x, y = data.getInputWithIx(src_block_size, tgt_block_size, ix)
    predict = model.generate(x, y, checkpoint_path=checkpoint_path)
    return predict

data = DataFrame(
    config.tickers,
    config.cuda0,
    config.cuda1,
    config.token_offset)
x, y = data.getBatch(
    config.batch_size,
    src_block_size=informer_config.n_encoder_block_size,
    tgt_block_size=informer_config.n_decoder_block_size)
_, _, informer_config.n_features = x.shape
model = EncoderDecoderInformer(informer_config)

if run_predict == 'y':
    ix = 2000
    x = predict(model, data, informer_config.n_encoder_block_size,
                informer_config.n_decoder_block_size,
                ix=ix, checkpoint_path=informer_config.informerCheckpointPath())
    predicted_data = data.raw()[:,1].clone().detach().to(x.device)
    predicted_data = predicted_data[:-config.token_offset]
    predicted_data = torch.concatenate((
        predicted_data[ix:ix+informer_config.n_encoder_block_size],
        x[0,-informer_config.token_offset:,1].flatten()), dim=0)
    plt.plot(
        data.raw()[:,1].cpu().numpy()[ix:ix + len(predicted_data)], label="Actual")
    plt.plot(predicted_data.cpu().numpy(), label="Predicted")
    plt.legend()
    plt.show()
else:
    model.train_and_update(
            data.getBatch, config.batch_size, config.lr,
            config.epoch, informer_config.n_encoder_block_size,
            informer_config.n_decoder_block_size,
            checkpoint_path=informer_config.informerCheckpointPath())
