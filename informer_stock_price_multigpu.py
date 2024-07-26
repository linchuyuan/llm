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
        "SMCI",
    ],
    batch_size = 8,
    lr = 1e-5,
    epoch = 5001,
    eval_interval = 1e3,
)

informer_config = Config(
    n_embed = 2000,
    n_encoder_block_size = 4000,
    n_encoder_head = 1,
    n_encoder_layer = 1,
    n_decoder_block_size = 2000,
    n_decoder_head = 1,
    n_decoder_layer = 1,
    n_predict_block_size = 100,
    lr = config.lr,
    batch_size = config.batch_size,
)

informer_config.n_target_data_size = \
    informer_config.n_predict_block_size + \
    informer_config.n_decoder_block_size

def predict(model, data, config, ix=0, checkpoint_path=None):
    x, y = data.getInputWithIx(config.n_encoder_block_size,
        config.n_target_data_size, ix)
    predict = model.generate(x, y,
        checkpoint_path=checkpoint_path)
    return predict

data = DataFrame(
    config.tickers,
    config.cuda0,
    config.cuda1)

x, y = data.getBatch(
    config.batch_size,
    src_block_size=informer_config.n_encoder_block_size,
    tgt_block_size=informer_config.n_target_data_size)
_, _, informer_config.n_features = x.shape

model = EncoderDecoderInformer(informer_config)

if run_predict == 'y':
    ix = 100
    x = predict(model, data, informer_config,
                ix=ix, checkpoint_path=informer_config.informerCheckpointPath())
    predicted_data = data.raw()[:,1].clone().detach().to(x.device)
    start = ix+informer_config.n_encoder_block_size - informer_config.n_decoder_block_size
    end = start + informer_config.n_decoder_block_size
    predicted_data = torch.concatenate((
        predicted_data[start:end],
        x[0,:,1].flatten()), dim=0)
    plt.plot(predicted_data.cpu().numpy(), label="Predicted")
    plt.plot(
        data.raw()[:,1].cpu().numpy()[start:start + len(predicted_data)], label="Actual")
    plt.legend()
    plt.show()
else:
    model.train_and_update(data.getBatch, config.epoch, config.eval_interval)
