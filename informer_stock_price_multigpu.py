#!/usr/bin/env python3
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import os
from lib.data import DataFrame
from lib.polygon import getOptionTickers, parseOptionTicker
from lib.data_manager import DataFrameManager
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
        "NVDA",
        "SPY",
        "AVGO",
        "NVDU",
        "SOXL",
        "UVIX",
        "QQQ",
        "TQQQ",
        "TSLA",
        "MSFT",
        "GOOGL",
        "META",
        "SNAP",
    ],
    batch_size = 1,
    lr = 5e-4,
    epoch = 55001,
    eval_interval = 5e1,
    option_low = 500,
    option_high = 700,
)

config = Config(
    config = config,
    n_embed = 4000,
    n_encoder_head = 10,
    n_encoder_layer = 2,
    n_decoder_block_size = 1000,
    n_decoder_head = 10,
    n_decoder_layer = 1,
    n_predict_block_size = 200,
)

option_tickers = getOptionTickers('META')
filtered_option_ticker = list()
if option_tickers:
    for ticker in option_tickers:
        _, _, _, strike = parseOptionTicker(ticker)
        if strike >= config.option_low and strike <= config.option_high:
            filtered_option_ticker.append(ticker)
    print("Option size is ", len(filtered_option_ticker))

o_config = Config(
    config = config,
    tickers = filtered_option_ticker,
)

o_data = DataFrame(
    o_config.tickers,
    config.cuda0,
    is_option=True,
    db_file="option_db.pickle",
)
s_data = DataFrame(
    config.tickers,
    config.cuda0,
    is_option=False,
    db_file="db.pickle",
)

data = DataFrameManager(
    s_data, o_data, config.n_decoder_block_size,
    config.n_predict_block_size)

x, x_mark, x_ticker, y, y_mark = data.getBatch(
    config.batch_size,
    src_block_size=0,
    tgt_block_size=config.n_decoder_block_size,
    pred_block_size=config.n_predict_block_size)

_, _, config.n_decoder_features = y.shape
_, config.n_encoder_block_size, config.n_encoder_features = x.shape
config.n_unique_ticker = o_data.n_unique_ticker

def predict(model, data, config, ix=0, step=1, checkpoint_path=None, pad=True):
    x, x_mark, x_ticker, y, y_mark = data.getInputWithIx(
        config.n_decoder_block_size, config.n_predict_block_size, ix)
    predict = generate(model, config, x, x_mark, x_ticker, y, y_mark,
        checkpoint_path=checkpoint_path, step=step, require_pad=pad)
    return predict

model = EncoderDecoderInformer(config)
# model = torch.nn.DataParallel(model)
# model.to(config.cuda0)

if run_predict == 'y':
    ix = int(input("Enter idx: "))
    criterion = torch.nn.MSELoss()
    raw, raw_mark = data.raw()
    pred = predict(model, data, config,
        ix=ix, checkpoint_path=config.informerCheckpointPath(), pad=False)
    plt.plot(pred[0,:,predict_feature_ix].flatten().cpu().numpy(), label="Predicted_%s" % (ix))
    actual = raw[ix-config.n_decoder_block_size-config.n_predict_block_size:ix]
    plt.plot(actual[:, 0].flatten().cpu().numpy(), label="Actual")
    plt.legend()

    loss = criterion(
        pred[:,-config.n_predict_block_size:,0:1].flatten().to(config.cuda0),
        actual[-config.n_predict_block_size:,0:1].flatten().to(config.cuda0))
    print("Predict loss is ", loss.item())
    plt.show()
elif run_predict == 'p':
    x, x_mark, x_ticker, y, y_mark = data.getLatest(
        config.n_decoder_block_size)
    predict = generate(model, config, x, x_mark, x_ticker, y, y_mark, 
        checkpoint_path=config.informerCheckpointPath())
    predict = predict.cpu()
    plt.plot(predict[0, :, predict_feature_ix].flatten().numpy())
    plt.legend()
    plt.show()
else:
    train_and_update(model, config, data.getBatch, config.epoch, config.eval_interval)
