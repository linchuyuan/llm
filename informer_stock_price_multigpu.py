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
predict_feature_ix = 0
config = Config(
    tickers = [
        "SMCI",
        "NVDA",
        "SPY",
        "AVGO",
        "NVDU",
        "SOXL",
        "UVIX",
        "QQQ",
        "TQQQ",
        "QYLD",
        "XYLD",
        "RYLD",
    ],
    batch_size = 8,
    lr = 5e-6,
    epoch = 3001,
    eval_interval = 1e2,
)

informer_config = Config(
    n_embed = 2000,
    n_encoder_block_size = 800,
    n_encoder_head = 10,
    n_encoder_layer = 2,
    n_decoder_block_size = 400,
    n_decoder_head = 10,
    n_decoder_layer = 2,
    n_predict_block_size = 200,
    lr = config.lr,
    batch_size = config.batch_size,
)

def predict(model, data, config, ix=0, step=1, checkpoint_path=None):
    x, x_mark, y, y_mark = data.getInputWithIx(config.n_encoder_block_size,
        config.n_decoder_block_size, config.n_predict_block_size, ix)
    predict = generate(model, informer_config, x, x_mark, y, y_mark, 
        checkpoint_path=checkpoint_path, step=step)
    return predict

data = DataFrame(
    config.tickers,
    config.cuda0)

x, x_mark, y, y_mark = data.getBatch(
    config.batch_size,
    src_block_size=informer_config.n_encoder_block_size,
    tgt_block_size=informer_config.n_decoder_block_size,
    pred_block_size=informer_config.n_predict_block_size)
_, _, informer_config.n_features = x.shape

model = EncoderDecoderInformer(informer_config)
# model = torch.nn.DataParallel(model)
# model.to(informer_config.cuda0)

if run_predict == 'y':
    ix = 7000
    criterion = torch.nn.MSELoss()
    raw, raw_mark = data.raw()
    pred = predict(model, data, informer_config,
        ix=ix, checkpoint_path=informer_config.informerCheckpointPath())
    plt.plot(pred[0,:,predict_feature_ix].flatten().cpu().numpy(), label="Predicted_%s" % (ix))
    actual_start = ix + informer_config.n_encoder_block_size - informer_config.n_decoder_block_size
    actual = raw[actual_start:actual_start + len(pred[0]), predict_feature_ix]
    plt.plot(actual.flatten().cpu().numpy(), label="Actual")
    plt.legend()

    pred = pred[:,:,predict_feature_ix]
    loss = criterion(pred.flatten().to(config.cuda0), actual.flatten().to(config.cuda0))
    print("Predict loss is ", loss.item())
    plt.show()
elif run_predict == 'p':
    x, x_mark, y, y_mark = data.getLatest(informer_config.n_encoder_block_size,
        informer_config.n_decoder_block_size)
    # x = data.raw()[:5000].unsqueeze(0)
    predict = generate(model, informer_config, x, x_mark, y, y_mark, 
        checkpoint_path=informer_config.informerCheckpointPath())
    predict = predict.cpu()
    plt.plot(predict[0, :, predict_feature_ix].flatten().numpy())
    plt.legend()
    plt.show()
else:
    train_and_update(model, informer_config, data.getBatch, config.epoch, config.eval_interval)
