#!/usr/bin/env python3
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import os
from lib.data import DataFrame
from lib.config import Config
from model.decoder_only_informer import DecoderOnlyInformer

run_predict = input("run predict (y/n, default n)?:")
if not run_predict:
    run_predict = 'n'

max_gen_token = 32
batch_size = 16
block_size = 8
lr = 1e-3
epoch = 2000
decoder_config = Config(
    tickers = [
        "SMCI"
    ],
    token_offset = 1,
    block_size = block_size,
    cuda_block_split = 1,
    n_embed = 128,
    n_head = 2,
    n_layer = 2,
)

def predict(model, block_size, data, ix=0, checkpoint_path=None):
    # x, y = data.getBatch(1, block_size)
    x, y = data.getInputWithIx(block_size, 0)
    predict = model.generate(x, max_gen_token=max_gen_token,
        checkpoint_path=checkpoint_path, targets=y)
    return predict

data = DataFrame(
        decoder_config.tickers,
        decoder_config.cuda0,
        decoder_config.cuda1,
        decoder_config.token_offset)
x, y = data.getBatch(batch_size, block_size)
_, _, decoder_config.n_features = x.shape
model = DecoderOnlyInformer(decoder_config)

if run_predict == 'y':
    ix = 2000
    x = predict(model, block_size, data, 
                ix=ix, checkpoint_path=decoder_config.decoderOnlyInformerCheckpointPath())

    # x = x.reshape(block_size)
    predicted_data = data.raw()[:,1].clone().detach().to(x.device)
    predicted_data = predicted_data[:-decoder_config.token_offset]
    predicted_data = torch.concatenate((predicted_data[ix:ix+block_size], x[:,block_size:,1].flatten()), dim=0)
    plt.plot(data.raw()[:,1].cpu().numpy()[ix:ix + len(predicted_data)], label="Actual")
    plt.plot(predicted_data.cpu().numpy(), label="Predicted")
    plt.legend()
    plt.show()
else:
    model.train_and_update(
            data.getBatch, batch_size, lr, epoch,
            block_size, checkpoint_path=decoder_config.decoderOnlyInformerCheckpointPath())
