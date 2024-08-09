#!/usr/bin/env python3
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import os
from lib.data import DataFrame
from lib.config import Config
from model.encoder_decoder_informer import \
        EncoderDecoderInformer, \
        train_and_update, \
        generate

run_predict = input("run predict (y/n, default n)?:")
if not run_predict:
    run_predict = 'n'
predict_feature_ix = 1
config = Config(
    tickers = [
        "SMCI",
    ],
    batch_size = 6,
    lr = 1e-5,
    epoch = 10001,
    eval_interval = 1e2,
)

informer_config = Config(
    n_embed = 3600,
    n_encoder_block_size = 1000,
    n_encoder_head = 1,
    n_encoder_layer = 1,
    n_decoder_block_size = 600,
    n_decoder_head = 1,
    n_decoder_layer = 1,
    n_predict_block_size = 100,
    lr = config.lr,
    batch_size = config.batch_size,
)

def predict(model, data, config, ix=0, step=1, checkpoint_path=None):
    x, y = data.getInputWithIx(config.n_encoder_block_size,
        config.n_decoder_block_size, config.n_predict_block_size, ix)
    predict = generate(model, informer_config, x, y,
        checkpoint_path=checkpoint_path, step=step)
    return predict

data = DataFrame(
    config.tickers,
    config.cuda0)

x, y = data.getBatch(
    config.batch_size,
    src_block_size=informer_config.n_encoder_block_size,
    tgt_block_size=informer_config.n_decoder_block_size,
    pred_block_size=informer_config.n_predict_block_size)
_, _, informer_config.n_features = x.shape

model = EncoderDecoderInformer(informer_config)
model = torch.nn.DataParallel(model)
model.to(informer_config.cuda0)

if run_predict == 'y':
    ix = 300
    step = 4
    pred = predict(model, data, informer_config,
        ix=ix, step=step,
        checkpoint_path=informer_config.informerCheckpointPath())
    plt.plot(pred[0,:,predict_feature_ix].flatten().cpu().numpy(), label="Predicted")
    actual_start = ix + informer_config.n_encoder_block_size - informer_config.n_decoder_block_size
    plt.plot(
        data.raw()[actual_start:actual_start + len(pred[0]), predict_feature_ix].flatten().cpu().numpy(),
        label="Actual")
    plt.legend()
    plt.show()
elif run_predict == 'p':
    x, y = data.getLatest(informer_config.n_encoder_block_size,
            informer_config.n_decoder_block_size,
            informer_config.n_predict_block_size)
    predict = generate(model, informer_config, x, y, 
            checkpoint_path=informer_config.informerCheckpointPath())
    y[:, -informer_config.n_predict_block_size:,:] = predict
    plt.plot(y[0, :, predict_feature_ix].flatten().numpy())
    plt.legend()
    plt.show()
else:
    train_and_update(model, informer_config, data.getBatch, config.epoch, config.eval_interval)
