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
    epoch = 10001,
    eval_interval = 1e3,
)

informer_config = Config(
    n_embed = 2000,
    n_encoder_block_size =4000,
    n_encoder_head = 1,
    n_encoder_layer = 1,
    n_decoder_block_size = 100,
    n_decoder_head = 1,
    n_decoder_layer = 1,
    n_predict_block_size = 100,
    lr = config.lr,
    batch_size = config.batch_size,
)

def predict(model, data, config, ix=0, checkpoint_path=None):
    x, y = data.getInputWithIx(config.n_encoder_block_size,
        config.n_decoder_block_size, config.n_predict_block_size, ix)
    predict = model.generate(x, y,
        checkpoint_path=checkpoint_path)
    return y, predict

data = DataFrame(
    config.tickers,
    config.cuda0,
    config.cuda1)

x, y = data.getBatch(
    config.batch_size,
    src_block_size=informer_config.n_encoder_block_size,
    tgt_block_size=informer_config.n_decoder_block_size,
    pred_block_size=informer_config.n_predict_block_size)
_, _, informer_config.n_features = x.shape

model = EncoderDecoderInformer(informer_config)

if run_predict == 'y':
    ix = 0
    tgt, pred = predict(model, data, informer_config,
                ix=ix, checkpoint_path=informer_config.informerCheckpointPath())
    predict_feature_ix = 1
    predicted_line = tgt.clone().detach()
    predicted_line[:,-informer_config.n_predict_block_size:,:] = pred
    plt.plot(predicted_line[0,:,predict_feature_ix].flatten().cpu().numpy(), label="Predicted")
    plt.plot(tgt[0,:,predict_feature_ix].flatten().cpu().numpy(), label="Actual")
    plt.legend()
    plt.show()
else:
    model.train_and_update(data.getBatch, config.epoch, config.eval_interval)